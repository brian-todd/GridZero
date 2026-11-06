"""What the rules do: how games end, who wins by which style, and which rosters pay.

Measures any version of the objective (`objective.py`) with greedy players, so that two versions
compare directly: run it before and after a change, and compare the JSON files. It reports:

- **self-play at each ``hold_rounds``:** how games end, how long they run, how many units die,
  flips, how often the first capturer wins, and team 0's share of decided games (each game with
  dice of its own),
- **styles, at the default rules:** a point-seeker, a fighter that ignores the point, a capper
  that barely fights, and a sitter, against each other,
- **the engineer, at each ``hold_rounds``:** greedy's default roster against the same roster with
  an archer for the engineer, and how much engineers work: its worth depends on the objective,
- **rosters, at the default rules:** every roster the caps allow against greedy's default roster
  (the same dice both ways), with what a slot of each kind is worth over a footsoldier.

It writes everything to ``{prefix}_rules.json``, and prints it as tables. Greedy players give a
lower bound on the units whose worth is in how well they are used, such as archers and engineers:
check by watching games before changing a rule on the strength of these numbers.

Run with ``uv run python run.py tactics.rules [starts] [prefix] [hold_rounds...]``: by default 120
starts and hold lengths of 4, 5, 6 and 8, about 2 minutes on 24 threads.
"""

import itertools
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TypedDict

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    SitterPlayer,
    Tactics,
    TacticsPosition,
    default_roster,
    summarize,
)
from gridzero.game import Player, Record, series

type TacticsPlayer = Player[Tactics, TacticsPosition]


class Figures(TypedDict):
    """What a set of games shows, in plain numbers (`figures`)."""

    games: int
    rounds: float
    rounds_p50: float
    rounds_p99: float
    rounds_max: int
    endings: dict[str, float]
    dead: float
    flips: float
    first_capturer_won: float
    team_0: float
    shots: float
    shots_hit: float
    built: float
    broke: float


class Duel(Figures):
    """`Figures` of a match, with its record from the first player's side (`duel`)."""

    won: int
    lost: int


class Swap(Duel):
    """A `Duel` of an engineer's roster and an archer's, and those rosters (`engineer_swap`)."""

    with_engineer: list[int]
    with_archer: list[int]


class Rosters(TypedDict):
    """Every roster against one, and what a slot is worth (`rosters`)."""

    against: list[int]
    scores: list[tuple[list[int], float]]
    per_slot: dict[str, float]


class Report(TypedDict):
    """All that `main` writes to the JSON file."""

    starts: int
    hold_rounds: int
    capture_by: int
    max_rounds: int
    self_play: dict[str, Figures]
    engineer: dict[str, Swap]
    styles: dict[str, Duel]
    rosters: Rosters


def figures(game: Tactics, records: Sequence[Record[TacticsPosition]]) -> Figures:
    """Return what a set of games shows, in plain numbers.

    Parameters
    ----------
    game : Tactics
        The rules they were played by.
    records : sequence of Record
        The games, at least one.

    Returns
    -------
    Figures
        Plain numbers for a JSON file: ``games``, ``rounds`` (mean), ``rounds_p50``,
        ``rounds_p99`` and ``rounds_max``, ``endings`` as shares of games, ``dead`` (mean units
        lost a game), ``flips`` (mean), ``first_capturer_won`` (share of games), ``team_0``
        (team 0's share of the decided games), ``shots`` (mean deferred attacks a game) with the
        share that hit, ``shots_hit``, and ``built`` and ``broke`` (mean tiles a game).
    """
    summary = summarize(game, records)
    games = summary.tally.games
    rounds = [record.final.round for record in records]
    wins = summary.tally.seat_wins
    return {
        "games": games,
        "rounds": summary.rounds,
        "rounds_p50": float(np.percentile(rounds, 50)),
        "rounds_p99": float(np.percentile(rounds, 99)),
        "rounds_max": max(rounds),
        "endings": {why: n / games for why, n in summary.endings.items()},
        "dead": float(summary.losses.sum() / games),
        "flips": summary.flips / games,
        "first_capturer_won": summary.first_capturer_won / games,
        "team_0": float(wins[0] / max(wins.sum(), 1)),
        "shots": summary.shots / games,
        "shots_hit": summary.shots_hit / max(summary.shots, 1),
        "built": summary.built / games,
        "broke": summary.broke / games,
    }


def duel(
    game: Tactics,
    first: TacticsPlayer,
    second: TacticsPlayer,
    starts: int,
    workers: int,
) -> Duel:
    """Play two players, each from both seats with the same dice, and return their figures.

    Parameters
    ----------
    game : Tactics
        The rules.
    first, second : Player
        The players.
    starts : int
        Starts, each played twice with the sides swapped.
    workers : int
        Games played at once.

    Returns
    -------
    Duel
        `figures`, with ``won`` and ``lost`` from ``first``'s side.
    """
    records = series(game, [("a", first), ("b", second)], starts, EVAL_SEED, workers=workers)
    wins = summarize(game, records).tally.wins
    return Duel(**figures(game, records), won=wins["a"], lost=wins["b"])


def styles(game: Tactics, starts: int, workers: int) -> dict[str, Duel]:
    """Play the pure styles against each other.

    Parameters
    ----------
    game : Tactics
        The rules.
    starts : int
        Starts a match, each played twice.
    workers : int
        Games played at once.

    Returns
    -------
    dict of str to Duel
        `duel`'s figures by match.
    """
    seeker = GreedyPlayer()
    fighter = GreedyPlayer(point=0.0)
    capper = GreedyPlayer(point=3.0, attack=0.2, kill=0.2, advance=0.0)
    sitter = SitterPlayer()
    matches: dict[str, tuple[TacticsPlayer, TacticsPlayer]] = {
        "seeker v fighter": (seeker, fighter),
        "capper v fighter": (capper, fighter),
        "capper v seeker": (capper, seeker),
        "seeker v sitter": (seeker, sitter),
        "fighter v sitter": (fighter, sitter),
        "sitter v sitter": (sitter, sitter),
    }
    return {name: duel(game, a, b, starts, workers) for name, (a, b) in matches.items()}


def engineer_swap(game: Tactics, starts: int, workers: int) -> Swap:
    """Play greedy's default roster against the same roster with an archer for the engineer.

    Parameters
    ----------
    game : Tactics
        The rules, with a kind that builds and a kind that defers.
    starts : int
        Starts, each played twice with the sides swapped.
    workers : int
        Games played at once.

    Returns
    -------
    Swap
        `duel`'s figures, ``won`` and ``lost`` from the engineer's side, with the rosters.

    Raises
    ------
    ValueError
        If the rules have no such kinds, or no engineer in the default roster.
    """
    builds = next((k for k, stats in enumerate(game.units) if stats.builds), None)
    defers = next((k for k, stats in enumerate(game.units) if stats.defers), None)
    if builds is None or defers is None:
        raise ValueError("the rules need a kind that builds and a kind that defers")
    roster = default_roster(game.caps, game.team_size)
    if roster[builds] == 0 or roster[defers] >= game.caps[defers]:
        raise ValueError(f"roster {roster} has no engineer to swap for an archer")
    swapped = list(roster)
    swapped[builds] -= 1
    swapped[defers] += 1
    result = duel(
        game, GreedyPlayer(roster=roster), GreedyPlayer(roster=tuple(swapped)), starts, workers
    )
    return Swap(**result, with_engineer=list(roster), with_archer=swapped)


def rosters(
    game: Tactics, starts: int, workers: int, base: tuple[int, ...] | None = None
) -> Rosters:
    """Play every roster the caps allow against one roster, and fit what each kind is worth.

    Parameters
    ----------
    game : Tactics
        The rules.
    starts : int
        Starts per roster, each played twice with the sides swapped.
    workers : int
        Games played at once.
    base : tuple of int, optional
        The roster to play against; by default `default_roster`.

    Returns
    -------
    Rosters
        ``against``: ``base``; ``scores``: each roster's share of the points against it, best
        first, as ``(roster, share)``, none if the caps allow no other; ``per_slot``: points of
        100 a slot of each kind is worth over the first kind, by least squares, by kind's name.
        All 0 with no other roster to tell them from.
    """
    opponent = base or default_roster(game.caps, game.team_size)
    every = [
        roster
        for roster in itertools.product(*(range(cap + 1) for cap in game.caps))
        if sum(roster) == game.team_size and roster != opponent
    ]
    scored = []
    for roster in every:
        result = duel(
            game, GreedyPlayer(roster=roster), GreedyPlayer(roster=opponent), starts, workers
        )
        scored.append((roster, result["won"] / result["games"]))
    scored.sort(key=lambda item: -item[1])
    x = np.array([np.subtract(r, opponent) for r, _ in scored], dtype=np.float64)
    x = x.reshape(len(scored), len(game.units))[:, 1:]
    y = np.array([2 * share - 1 for _, share in scored], dtype=np.float64)
    fit = np.linalg.lstsq(x, y, rcond=None)[0]
    names = [stats.name for stats in game.units[1:]]
    return {
        "against": list(opponent),
        "scores": [(list(r), share) for r, share in scored],
        "per_slot": {n: float(50 * c) for n, c in zip(names, fit, strict=True)},
    }


def _table(rows: Mapping[str, Figures | Duel]) -> str:
    """Return figures as a Markdown table, a row per label."""
    lines = [
        "| | Rounds | p99 | Most | Held | Killed | Deadline | Cap | Dead | Flips | First won "
        "| Team 0 | Shots | Hit | Built | Broke | Won-lost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, f in rows.items():
        e = f["endings"]
        record = f"{f['won']}-{f['lost']}" if "won" in f and "lost" in f else ""
        lines.append(
            f"| {label} | {f['rounds']:.1f} | {f['rounds_p99']:.0f} | {f['rounds_max']} "
            f"| {e['held']:.0%} | {e['killed']:.0%} | {e['deadline']:.0%} | {e['cap']:.0%} "
            f"| {f['dead']:.1f} | {f['flips']:.1f} | {f['first_capturer_won']:.0%} "
            f"| {f['team_0']:.0%} | {f['shots']:.1f} | {f['shots_hit']:.0%} "
            f"| {f['built']:.1f} | {f['broke']:.1f} | {record} |"
        )
    return "\n".join(lines)


def main(starts: str = "120", prefix: str = "tactics", *hold_rounds: str) -> None:
    """Measure the rules, print the tables, and write them to ``{prefix}_rules.json``.

    Parameters
    ----------
    starts : str, default "120"
        Starts of each series, each played twice; rosters play a tenth of them, at least 1.
    prefix : str, default "tactics"
        Where the numbers go: ``{prefix}_rules.json``.
    *hold_rounds : str
        Rounds in a row to win, one self-play series each: 4, 5, 6 and 8 if none.
    """
    n = int(starts)
    holds = [int(h) for h in hold_rounds] or [4, 5, 6, 8]
    game = Tactics()
    began = time.perf_counter()
    selfplay: dict[str, Figures] = {}
    for hold in holds:
        rules = replace(game, hold_rounds=hold)
        records = series(
            rules,
            [("a", GreedyPlayer()), ("b", GreedyPlayer())],
            n,
            EVAL_SEED,
            same_dice=False,
            workers=CORES,
        )
        selfplay[f"hold {hold}"] = figures(rules, records)
    print(f"### Greedy against itself, by rounds to hold\n\n{_table(selfplay)}")

    swaps = {f"hold {h}": engineer_swap(replace(game, hold_rounds=h), n, CORES) for h in holds}
    print(f"\n### Engineer against archer, by rounds to hold\n\n{_table(swaps)}")

    played = styles(game, n, CORES)
    print(f"\n### Styles, {game.hold_rounds} rounds to hold\n\n{_table(played)}")

    fitted = rosters(game, max(n // 10, 1), CORES)
    top = fitted["scores"]
    print(f"\n### Rosters against {fitted['against']}\n")
    print("Best: " + ", ".join(f"{r} {s:.0%}" for r, s in top[:6]))
    print("Worst: " + ", ".join(f"{r} {s:.0%}" for r, s in top[-4:]))
    print("A slot over a footsoldier, points of 100: " + str(fitted["per_slot"]))

    report: Report = {
        "starts": n,
        "hold_rounds": game.hold_rounds,
        "capture_by": game.capture_by,
        "max_rounds": game.max_rounds,
        "self_play": selfplay,
        "engineer": swaps,
        "styles": played,
        "rosters": fitted,
    }
    path = f"{prefix}_rules.json"
    with open(path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=1)
    print(f"\nWrote {path}, in {(time.perf_counter() - began) / 60:.1f} minutes.")
