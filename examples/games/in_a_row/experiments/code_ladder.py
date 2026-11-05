"""Set code players against a trained network's agent, at equal simulations and equal time.

Loads weights saved by ``zero_scratch`` (32 channels, 4 blocks), the teacher, and constants
saved by ``code_fit``. Prints what the teacher and the code evaluators make of each first
column, against perfect play (`FIRST_MOVES`); the time a simulation takes on each evaluator;
then, at each budget, a ladder of four search players (the search alone, `LineEvaluator`, the
fit, the teacher) against each other, the uniform player and `ThreatPlayer`; and last, the fit
and the lines with the simulations the teacher's budget would buy them, against the teacher. On
the games players are scored on (`EVAL_SEED`), on every core. Every search player draws Gumbel
noise: the start of Connect Four is fixed, so players without noise would play one game over
and over.

Run with ``uv run python run.py in_a_row.code_ladder weights.pt fit.json [starts]
[simulations...]``: by default 50 starts and 16, 64 and 256 simulations.
"""

import time

import numpy as np

from examples.games import CORES
from examples.games.in_a_row import (
    EVAL_SEED,
    FIRST_MOVES,
    SEARCH_ALONE,
    FittedEvaluator,
    InARow,
    InARowPosition,
    LineEvaluator,
    ThreatPlayer,
    opening,
)
from gridzero.agents import AlphaZeroPlayer, Evaluator, UniformPlayer
from gridzero.agents.net import NetEvaluator
from gridzero.game import Player, ladder

from .zero_check import saved_net

SEED = 1  # the draws of the timed decisions
TIMED = 20  # decisions timed per evaluator, from the start, at 64 simulations


def main(weights: str, fit: str, starts: str = "50", *simulations: str) -> None:
    """Print the openings, the cost of a simulation, and ladders at each budget.

    Parameters
    ----------
    weights : str
        The teacher's saved weights.
    fit : str
        The constants saved by ``code_fit``.
    starts : str, default "50"
        Starts against each rival, each played once per rotation of the seats.
    *simulations : str
        The budgets, in simulations a decision; by default 16, 64 and 256.
    """
    game = InARow()
    net = saved_net(game, weights)
    evaluators: dict[str, Evaluator[InARow, InARowPosition]] = {
        "search": SEARCH_ALONE,
        "lines": LineEvaluator(),
        "fit": FittedEvaluator.load(fit),
        "teacher": NetEvaluator(net),
    }
    print(f"perfect play after each first column: {list(FIRST_MOVES)}")
    for name in ("lines", "fit", "teacher"):
        first = opening(game, evaluators[name])
        print(
            f"{name}: after each first column {[round(v, 2) for v in first.after]}, "
            f"opens in column {first.column}.",
            flush=True,
        )
    costs = {}
    for name, evaluator in evaluators.items():
        costs[name] = _cost(game, AlphaZeroPlayer(evaluator, simulations=64, considered=7))
        print(f"{name}: {costs[name] * 1e6:.0f} us a simulation at 64 simulations.", flush=True)
    budgets = [int(s) for s in simulations or ("16", "64", "256")]
    for budget in budgets:
        players = {name: _agent(evaluator, budget) for name, evaluator in evaluators.items()}
        names = list(players)
        for i, name in enumerate(names):
            rivals: dict[str, Player[InARow, InARowPosition]] = {
                "uniform": UniformPlayer(),
                "threat": ThreatPlayer(),
                **{other: players[other] for other in names[i + 1 :]},
            }
            _print_ladder(f"{budget} simulations, {name}", game, players[name], rivals, starts)
    for budget in budgets:
        for name in ("lines", "fit"):
            bought = max(1, round(budget * costs["teacher"] / costs[name]))
            _print_ladder(
                f"Equal time: {name} at {bought} simulations",
                game,
                _agent(evaluators[name], bought),
                {f"teacher at {budget}": _agent(evaluators["teacher"], budget)},
                starts,
            )


def _agent(
    evaluator: Evaluator[InARow, InARowPosition], simulations: int
) -> AlphaZeroPlayer[InARow, InARowPosition]:
    """Return a search player that draws noise, considering every column."""
    return AlphaZeroPlayer(evaluator, simulations=simulations, considered=7, noise=True)


def _print_ladder(
    label: str,
    game: InARow,
    player: Player[InARow, InARowPosition],
    rivals: dict[str, Player[InARow, InARowPosition]],
    starts: str,
) -> None:
    """Play a ladder and print its scores."""
    start = time.perf_counter()
    scores = ladder(game, player, rivals, int(starts), EVAL_SEED, workers=CORES)
    text = ", ".join(f"{rival} {s.won}-{s.lost}-{s.drawn}" for rival, s in scores.items())
    print(
        f"{label} against {text} (wins-losses-draws), {time.perf_counter() - start:.0f} s.",
        flush=True,
    )


def _cost(game: InARow, player: AlphaZeroPlayer[InARow, InARowPosition]) -> float:
    """Return the seconds a simulation takes over the first `TIMED` decisions of a game."""
    rng = np.random.default_rng(SEED)
    position = game.start(EVAL_SEED)
    decisions = 0
    start = time.perf_counter()
    while not position.over and decisions < TIMED:
        action = player(game, game.player(position), position, game.legal(position), rng)
        position = game.step(position, action)
        decisions += 1
    return (time.perf_counter() - start) / (decisions * player.simulations)
