"""What fog changes: how games go, what a scout is worth, and what a planner loses.

Under fog, the default, a side sees the other's units only near its own, and is shown the rest
where it last saw them (`Tactics.view`); ``fog=False`` hides nothing. Players are scored on the
same starts in the open and under fog (`EVAL_SEED`), each played once per rotation of the seats,
on every core. It prints:

- **greedy against itself**, in the open and under fog: how games end, how long they run, team
  0's share, and how much of the other side a side doesn't see when it decides;
- **scouting:** greedy fielding a second knight in place of two footsoldiers, against greedy's
  default roster, in the open, under fog, and under fog with knights that see no further than a
  footsoldier: what a knight's eyes are worth to this player;
- **a planner**, given a code player: its record against greedy in the open and under fog, deploying
  as greedy does. Under fog its search plans on what it believes (`Tactics.view`).

Run with ``uv run python run.py tactics.fog [weights] [starts] [size] [simulations]``: by default
no code player, 120 starts on 15x15 (`small_rules`; any other size is the full rules), and 128
simulations a decision for the code player (``hand``, or constants saved by ``tactics.code_fit``).
Greedy's parts take seconds; a code player's take about a minute per 48 starts on 15x15.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    HAND_PICKED,
    SEARCH_C_SCALE,
    CodeEvaluator,
    Fielded,
    GreedyPlayer,
    Tactics,
    TacticsPosition,
    default_roster,
    small_rules,
    summarize,
)
from gridzero.agents import AlphaZeroPlayer
from gridzero.game import Record, replay, series

UNSEEN_GAMES = 24  # the games whose every decision is looked at for what a side doesn't see


def main(
    weights: str = "", starts: str = "120", size: str = "15", simulations: str = "128"
) -> None:
    """Print what fog changes for greedy, for a second knight, and for a code player if given.

    Parameters
    ----------
    weights : str, default ""
        A code player to measure too: constants saved by ``tactics.code_fit``, or ``hand``. By
        default none.
    starts : str, default "120"
        Starts of each series, each played once per rotation of the seats.
    size : str, default "15"
        The side of the board: 15 for `small_rules`, else the full rules on that size.
    simulations : str, default "128"
        The code player's simulations a decision.
    """
    side, n = int(size), int(starts)
    rules = small_rules() if side == 15 else dataclasses.replace(Tactics(), shape=(side, side))
    base, fogged = (dataclasses.replace(rules, fog=fog) for fog in (False, True))
    greedy = GreedyPlayer()
    for name, game in (("In the open", base), ("Under fog", fogged)):
        records = series(game, [("a", greedy), ("b", greedy)], n, EVAL_SEED, workers=CORES)
        shown = summarize(game, records)
        games = len(records)
        ends = ", ".join(f"{k} {v / games:.0%}" for k, v in shown.endings.items() if v)
        print(
            f"{name}, greedy against itself: {games} games of {shown.rounds:.1f} rounds and "
            f"{shown.tally.plies:.0f} decisions, ended {ends}; team 0 won "
            f"{shown.tally.seat_wins[0] / games:.0%}; a side deciding didn't see "
            f"{unseen(game, records[:UNSEEN_GAMES]):.0%} of the other's units.",
            flush=True,
        )
    roster = default_roster(base.caps, base.team_size, base.units)
    scouts = two_knights(base, roster)
    if scouts is not None:
        level = min(stats.sight for stats in base.units)
        blind = tuple(dataclasses.replace(stats, sight=level) for stats in base.units)
        variants = {
            "in the open": base,
            "under fog": fogged,
            f"under fog, every kind seeing {level:g}": dataclasses.replace(fogged, units=blind),
        }
        for name, game in variants.items():
            pair = [("two", GreedyPlayer(roster=scouts)), ("default", GreedyPlayer(roster=roster))]
            records = series(game, pair, n, EVAL_SEED, workers=CORES)
            won = sum(r.winner == "two" for r in records)
            print(
                f"Greedy with {scouts} against {roster}, {name}: {won}-{len(records) - won} "
                f"({won / len(records):.0%}).",
                flush=True,
            )
    if weights:
        evaluator = HAND_PICKED if weights == "hand" else CodeEvaluator.load(weights)
        search = AlphaZeroPlayer(evaluator, simulations=int(simulations), c_scale=SEARCH_C_SCALE)
        for name, game in (("in the open", base), ("under fog", fogged)):
            pair = [("code", Fielded(search, greedy)), ("greedy", greedy)]
            records = series(game, pair, n, EVAL_SEED, workers=CORES)
            won = sum(r.winner == "code" for r in records)
            print(
                f"The code player at {simulations} simulations against greedy, {name}: "
                f"{won}-{len(records) - won}.",
                flush=True,
            )


def two_knights(game: Tactics, roster: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return ``roster`` with one more of the kind that sees furthest, paid for in the commonest.

    Parameters
    ----------
    game : Tactics
        The rules.
    roster : tuple of int
        How many of each kind a side fields.

    Returns
    -------
    tuple of int, or None
        The roster with a further unit of the kind of the longest `UnitStats.sight`, and as many
        fewer of its most numerous other kind as places that takes. None if the caps, or the
        units to give up, don't allow it.
    """
    scout = int(np.argmax([stats.sight for stats in game.units]))
    stats = game.units[scout]
    cost = stats.places_for(roster[scout] + 1) - stats.places_for(roster[scout])
    others = [k for k in range(len(roster)) if k != scout]
    paying = max(others, key=lambda k: roster[k])
    each = game.units[paying]
    given = next(
        (
            m
            for m in range(1, roster[paying] + 1)
            if each.places_for(roster[paying]) - each.places_for(roster[paying] - m) == cost
        ),
        None,
    )
    if roster[scout] >= game.caps[scout] or given is None:
        return None
    changed = list(roster)
    changed[scout] += 1
    changed[paying] -= given
    return tuple(changed)


def unseen(game: Tactics, records: Sequence[Record[TacticsPosition]]) -> float:
    """Return the mean share of the other side's units that the side deciding doesn't see.

    Parameters
    ----------
    game : Tactics
        The rules the games were played by.
    records : sequence of Record
        Games, at least one with a decision in battle.

    Returns
    -------
    float
        Over every decision in battle: the units of the other side on tiles out of the sight of
        the mover's own (`Tactics.sees`), over those it has. In the open too, where it is what
        fog would hide.
    """
    shares = []
    for record in records:
        for position in replay(game, record):
            if position.deploying or position.over:
                continue
            state = position.state
            team = game.player(position)
            theirs = state.occupied & (state.units.team != team)
            shares.append(1.0 - game.sees(position, team)[theirs].mean())
    return float(np.mean(shares))
