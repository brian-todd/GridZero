"""What the rules do: how games end, who wins by which style, and which rosters pay.

Measures any version of the objective (`objective.py`) with greedy players or with a code player,
fielding the rosters it is given as greedy does, so that two versions, or two players, compare
directly: run it before and after a change, and compare the JSON files. It reports:

- **self-play at each ``hold_rounds``:** how games end, how long they run, how many units die,
  flips, how often the first capturer wins, team 0's share of decided games (each game with dice
  of its own), and what each kind of unit did,
- **styles, at the default rules (greedy only):** a point-seeker, a fighter that ignores the point,
  a capper that barely fights, and a sitter, against each other,
- **its own picks, at the default rules (code player only):** the roster it fields when free to
  choose, in self-play, and against itself on greedy's default roster,
- **the engineer, at each ``hold_rounds``:** greedy's default roster against the same roster with
  an archer for the engineer, and how much engineers work: its worth depends on the objective,
- **rosters, at the default rules:** every roster the caps allow against greedy's default roster
  (the same dice both ways), with what a slot of each kind is worth over a footsoldier.

It writes everything to ``{prefix}_rules.json`` after each part, and prints it as tables. Greedy
players give a lower bound on the units whose worth is in how well they are used, such as archers
and engineers: check by watching games before changing a rule on the strength of these numbers.

Run with ``uv run python run.py tactics.rules [weights] [starts] [prefix] [simulations]
[hold_rounds...]``: by default greedy players, 120 starts and hold lengths of 4, 5, 6 and 8, about
2 minutes on 24 threads. A code player (``hand``, or constants saved by ``tactics.code_fit``) plays
at 128 simulations, about 2 seconds a game on the full board: 96 starts take over an hour.
"""

import collections
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TypedDict

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    HAND_PICKED,
    CodeEvaluator,
    Fielded,
    GreedyPlayer,
    SitterPlayer,
    Tactics,
    TacticsPosition,
    default_roster,
    summarize,
)
from gridzero.agents import AlphaZeroPlayer
from gridzero.game import Player, Record, replay, series

type TacticsPlayer = Player[Tactics, TacticsPosition]

type Fielding = Callable[[tuple[int, ...] | None], TacticsPlayer]
"""Makes the player being measured, fielding a roster, or greedy's default one for None."""


class KindFigures(TypedDict):
    """What one kind of unit did, a game (`figures`)."""

    placed: float
    dealt: float
    taken: float
    kills: float
    losses: float


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
    kinds: dict[str, KindFigures]


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


class OwnPicks(TypedDict):
    """A player free to choose its roster (`own_picks`)."""

    picked: dict[str, int]
    self_play: Figures
    against_default: Duel


class Report(TypedDict):
    """All that `main` writes to the JSON file."""

    starts: int
    hold_rounds: int
    capture_by: int
    max_rounds: int
    self_play: dict[str, Figures]
    engineer: dict[str, Swap]
    styles: dict[str, Duel]
    own_picks: OwnPicks | None
    rosters: Rosters | None


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
        share that hit, ``shots_hit``, ``built`` and ``broke`` (mean tiles a game), and
        ``kinds``: by kind's name, what its units did, a game.
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
        "kinds": {
            stats.name: {
                "placed": float(summary.placed[k] / games),
                "dealt": float(summary.damage_dealt[k] / games),
                "taken": float(summary.damage_taken[k] / games),
                "kills": float(summary.kills[k] / games),
                "losses": float(summary.losses[k] / games),
            }
            for k, stats in enumerate(game.units)
        },
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


def greedy_fielding(roster: tuple[int, ...] | None) -> TacticsPlayer:
    """Return a greedy player fielding a roster (see `Fielding`)."""
    return GreedyPlayer(roster=roster)


def code_fielding(play: TacticsPlayer) -> Fielding:
    """Return what makes a player field a roster as greedy does, playing on as itself."""
    return lambda roster: Fielded(play, GreedyPlayer(roster=roster))


def own_picks(
    game: Tactics, free: TacticsPlayer, make: Fielding, starts: int, workers: int
) -> OwnPicks:
    """Play a player free to choose its roster against itself, and against itself on the default.

    Parameters
    ----------
    game : Tactics
        The rules.
    free : Player
        The player, choosing its roster.
    make : Fielding
        Makes the same player fielding a roster.
    starts : int
        Starts of each series, each played twice.
    workers : int
        Games played at once.

    Returns
    -------
    OwnPicks
        ``picked``: teams by the roster they fielded in self-play, the most common first, keyed by
        its text; ``self_play``: those games' figures; ``against_default``: `duel`'s figures, from
        the side choosing its roster.
    """
    records = series(
        game, [("a", free), ("b", free)], starts, EVAL_SEED, same_dice=False, workers=workers
    )
    picked: collections.Counter[str] = collections.Counter()
    for record in records:
        units = next(p for p in replay(game, record) if not p.deploying).state.units
        for team in range(2):
            picked[
                str(
                    [
                        int(((units.kind == k) & (units.team == team)).sum())
                        for k in range(len(game.units))
                    ]
                )
            ] += 1
    return {
        "picked": dict(picked.most_common()),
        "self_play": figures(game, records),
        "against_default": duel(game, free, make(None), starts, workers),
    }


def engineer_swap(
    game: Tactics, starts: int, workers: int, make: Fielding = greedy_fielding
) -> Swap:
    """Play the default roster against the same roster with an archer for the engineer.

    Parameters
    ----------
    game : Tactics
        The rules, with a kind that builds and a kind that defers.
    starts : int
        Starts, each played twice with the sides swapped.
    workers : int
        Games played at once.
    make : Fielding, default `greedy_fielding`
        Makes the players.

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
    roster = default_roster(game.caps, game.team_size, game.units)
    if roster[builds] == 0 or roster[defers] >= game.caps[defers]:
        raise ValueError(f"roster {roster} has no engineer to swap for an archer")
    swapped = list(roster)
    swapped[builds] -= 1
    swapped[defers] += 1
    result = duel(game, make(roster), make(tuple(swapped)), starts, workers)
    return Swap(**result, with_engineer=list(roster), with_archer=swapped)


def rosters(
    game: Tactics,
    starts: int,
    workers: int,
    base: tuple[int, ...] | None = None,
    make: Fielding = greedy_fielding,
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
    make : Fielding, default `greedy_fielding`
        Makes the players.

    Returns
    -------
    Rosters
        ``against``: ``base``; ``scores``: each roster's share of the points against it, best
        first, as ``(roster, share)``, none if the caps allow no other; ``per_slot``: points of
        100 a slot of each kind is worth over the first kind, by least squares, by kind's name.
        All 0 with no other roster to tell them from.
    """
    opponent = base or default_roster(game.caps, game.team_size, game.units)
    every = [
        roster
        for roster in itertools.product(*(range(cap + 1) for cap in game.caps))
        if sum(s.places_for(n) for s, n in zip(game.units, roster, strict=True)) == game.team_size
        and roster != opponent
    ]
    scored = []
    for roster in every:
        result = duel(game, make(roster), make(opponent), starts, workers)
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


def _kinds(rows: Mapping[str, Figures]) -> str:
    """Return what each kind of unit did as a Markdown table, a row per label and kind."""
    lines = [
        "| | Kind | Placed | Dealt | Taken | Kills | Losses |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, f in rows.items():
        lines += [
            f"| {label} | {kind} | {k['placed']:.1f} | {k['dealt']:.1f} | {k['taken']:.1f} "
            f"| {k['kills']:.1f} | {k['losses']:.1f} |"
            for kind, k in f["kinds"].items()
        ]
    return "\n".join(lines)


def _write(path: str, report: Report) -> None:
    """Write what has been measured so far, so a run stopped early keeps it."""
    with open(path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=1)


def main(
    weights: str = "greedy",
    starts: str = "120",
    prefix: str = "tactics",
    simulations: str = "128",
    *hold_rounds: str,
) -> None:
    """Measure the rules, print the tables, and write them to ``{prefix}_rules.json``.

    Parameters
    ----------
    weights : str, default "greedy"
        Who plays: ``greedy``, or a code player on `HAND_PICKED` (``hand``) or on constants saved
        by ``tactics.code_fit``.
    starts : str, default "120"
        Starts of each series, each played twice; rosters play a tenth of them, at least 1.
    prefix : str, default "tactics"
        Where the numbers go: ``{prefix}_rules.json``.
    simulations : str, default "128"
        The code player's simulations a decision.
    *hold_rounds : str
        Rounds in a row to win, one self-play series each: 4, 5, 6 and 8 if none.
    """
    n = int(starts)
    holds = [int(h) for h in hold_rounds] or [4, 5, 6, 8]
    game = Tactics()
    path = f"{prefix}_rules.json"
    make = greedy_fielding
    free: TacticsPlayer | None = None
    if weights != "greedy":
        evaluator = HAND_PICKED if weights == "hand" else CodeEvaluator.load(weights)
        free = AlphaZeroPlayer(evaluator, simulations=int(simulations))
        make = code_fielding(free)
    report: Report = {
        "starts": n,
        "hold_rounds": game.hold_rounds,
        "capture_by": game.capture_by,
        "max_rounds": game.max_rounds,
        "self_play": {},
        "engineer": {},
        "styles": {},
        "own_picks": None,
        "rosters": None,
    }
    began = time.perf_counter()
    selfplay = report["self_play"]
    for hold in holds:
        rules = replace(game, hold_rounds=hold)
        records = series(
            rules,
            [("a", make(None)), ("b", make(None))],
            n,
            EVAL_SEED,
            same_dice=False,
            workers=CORES,
        )
        selfplay[f"hold {hold}"] = figures(rules, records)
        _write(path, report)
    print(f"### Against itself, by rounds to hold\n\n{_table(selfplay)}\n\n{_kinds(selfplay)}")

    swaps = report["engineer"]
    for hold in holds:
        swaps[f"hold {hold}"] = engineer_swap(replace(game, hold_rounds=hold), n, CORES, make)
        _write(path, report)
    print(f"\n### Engineer against archer, by rounds to hold\n\n{_table(swaps)}", flush=True)

    if free is None:
        report["styles"] = styles(game, n, CORES)
        _write(path, report)
        print(f"\n### Styles, {game.hold_rounds} rounds to hold\n\n{_table(report['styles'])}")
    else:
        own = own_picks(game, free, make, n, CORES)
        report["own_picks"] = own
        _write(path, report)
        print(f"\n### Its own picks, {game.hold_rounds} rounds to hold\n\n{own['picked']}\n")
        print(_table({"self-play": own["self_play"], "v default": own["against_default"]}))
        print(_kinds({"self-play": own["self_play"]}), flush=True)

    fitted = rosters(game, max(n // 10, 1), CORES, make=make)
    report["rosters"] = fitted
    _write(path, report)
    top = fitted["scores"]
    print(f"\n### Rosters against {fitted['against']}\n")
    print("Best: " + ", ".join(f"{r} {s:.0%}" for r, s in top[:6]))
    print("Worst: " + ", ".join(f"{r} {s:.0%}" for r, s in top[-4:]))
    print("A slot over a footsoldier, points of 100: " + str(fitted["per_slot"]))
    print(f"\nWrote {path}, in {(time.perf_counter() - began) / 60:.1f} minutes.")
