"""Fit a code player to a trained network's agent, in rounds.

Loads weights saved by ``zero_scratch`` (32 channels, 4 blocks): the teacher. The first round's
games are the teacher against itself, `LineEvaluator` and the search alone. Each later round's
are the last fit against the teacher and against itself, whose mistakes reach positions the
teacher alone doesn't (DAgger). The new positions are judged by the teacher's search, the
constants fit to every lesson so far, printed with how they do on games kept out, and the fit
is set against `ThreatPlayer`, `LineEvaluator` and the teacher on the games players are scored
on (`EVAL_SEED`), on every core. Writes ``{prefix}_lessons.npz`` and, per round,
``{prefix}_round<k>.json``.

Run with ``uv run python run.py in_a_row.code_fit weights.pt [starts] [prefix] [rounds]``:
by default 150 starts a pair of players a round, prefix ``code``, and 3 rounds.
"""

import time
from pathlib import Path

from examples.games import CORES
from examples.games.in_a_row import (
    EVAL_SEED,
    MOVE_FEATURES,
    SEARCH_ALONE,
    VALUE_FEATURES,
    FittedEvaluator,
    InARow,
    InARowPosition,
    Lessons,
    LineEvaluator,
    ThreatPlayer,
    distinct_positions,
    judge,
)
from examples.games.in_a_row.fit import fit
from gridzero.agents import AlphaZeroPlayer, Evaluator
from gridzero.agents.net import NetEvaluator
from gridzero.game import Player, Record, ladder, series

from .zero_check import saved_net

SEED = 5_000_000  # the lessons' starts, clear of EVAL_SEED, and their judges' ties
PLAYED_AT = 64  # simulations a decision in the lessons' games
JUDGED_AT = 256  # simulations a position when judged
LADDER_STARTS = 50  # starts of each round's ladder, at most
LADDER_AT = 64  # its simulations a decision


def main(weights: str, starts: str = "150", prefix: str = "code", rounds: str = "3") -> None:
    """Fit in rounds, printing the constants and how each fit does.

    Parameters
    ----------
    weights : str
        The teacher's saved weights.
    starts : str, default "150"
        Starts of each pair of players a round, each played once per rotation of the seats.
    prefix : str, default "code"
        Every file written starts with it.
    rounds : str, default "3"
        How many rounds of lessons and fits.

    Raises
    ------
    FileExistsError
        If a run already used the prefix: ``code_report`` reads every round it finds.
    """
    folder, name = Path(prefix).parent, Path(prefix).name
    if [*folder.glob(f"{name}_round*.json"), *folder.glob(f"{name}_lessons.npz")]:
        raise FileExistsError(f"{prefix} has been used: give the new run a prefix of its own")
    game = InARow()
    net = saved_net(game, weights)
    teacher = NetEvaluator(net)
    lessons: Lessons | None = None
    fitted: FittedEvaluator | None = None
    for round_ in range(int(rounds)):
        start = time.perf_counter()
        records = _games(game, teacher, fitted, int(starts), SEED + round_ * int(starts))
        owner, ply, games = distinct_positions(game, records)
        new = judge(
            game,
            teacher,
            owner,
            ply,
            games,
            simulations=JUDGED_AT,
            seed=SEED + round_,
            source=round_,
            workers=CORES,
        )
        before = 0 if lessons is None else len(lessons)
        lessons = new if lessons is None else lessons + new
        fitted, score = fit(game, lessons, SEED)
        print(
            f"Round {round_}: {len(records)} games, {len(lessons) - before} new positions, "
            f"{len(lessons)} in all, {time.perf_counter() - start:.0f} s. On games kept out: "
            f"value correlation {score.value_correlation:.3f}, prior agrees "
            f"{score.agreement:.3f}, regret {score.regret:.4f}.",
            flush=True,
        )
        print(f"  value {_named(VALUE_FEATURES, fitted.value)}, at 0 {fitted.intercept:+.2f}")
        print(f"  prior {_named(MOVE_FEATURES, fitted.prior)}", flush=True)
        fitted.save(f"{prefix}_round{round_}.json")
        lessons.save(f"{prefix}_lessons.npz")
        rivals: dict[str, Player[InARow, InARowPosition]] = {
            "threat": ThreatPlayer(),
            "lines": _agent(LineEvaluator(), LADDER_AT),
            "teacher": _agent(teacher, LADDER_AT),
        }
        scores = ladder(
            game,
            _agent(fitted, LADDER_AT),
            rivals,
            min(LADDER_STARTS, int(starts)),
            EVAL_SEED,
            workers=CORES,
        )
        text = ", ".join(f"{name} {s.won}-{s.lost}-{s.drawn}" for name, s in scores.items())
        print(f"  at {LADDER_AT} simulations against {text} (wins-losses-draws).", flush=True)


def _agent(
    evaluator: Evaluator[InARow, InARowPosition], simulations: int
) -> AlphaZeroPlayer[InARow, InARowPosition]:
    """Return a search player that draws noise: the start is fixed, so games would repeat."""
    return AlphaZeroPlayer(evaluator, simulations=simulations, considered=7, noise=True)


def _games(
    game: InARow,
    teacher: Evaluator[InARow, InARowPosition],
    fitted: FittedEvaluator | None,
    starts: int,
    seed: int,
) -> list[Record[InARowPosition]]:
    """Return the games of a round: the teacher's in the first, the last fit's after."""
    master = _agent(teacher, PLAYED_AT)
    rivals: list[Evaluator[InARow, InARowPosition]] = (
        [teacher, LineEvaluator(), SEARCH_ALONE] if fitted is None else [teacher, fitted]
    )
    first = master if fitted is None else _agent(fitted, PLAYED_AT)
    records: list[Record[InARowPosition]] = []
    for rival in rivals:
        records += series(
            game,
            [("first", first), ("rival", _agent(rival, PLAYED_AT))],
            starts,
            seed,
            workers=CORES,
        )
    return records


def _named(names: tuple[str, ...], weights: tuple[float, ...]) -> str:
    """Return weights by name, as text."""
    return ", ".join(f"{n} {w:+.2f}" for n, w in zip(names, weights, strict=True))
