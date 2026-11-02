"""What a set of games of Tactics shows.

Games are played by `game.play` and `game.series`. `summarize` counts wins, draws and lengths
(`game.tally`), rounds, and the damage and kills of each kind of unit, and `balance` scores them
as rules for self-play: how fair the sides were (`game.fairness`), and how evenly the kinds of
unit do damage. Players are scored on the boards from `EVAL_SEED` on, in every experiment, so
their scores compare.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.game import Fairness, Record, Tally, fairness, tally

from .events import Attacked, Died
from .game import Tactics
from .state import TacticsPosition

EVAL_SEED = 2_000_000
"""The first start of every series that scores players: boards no training run plays."""


@dataclass(frozen=True, eq=False)
class Summary:
    """What a set of games shows.

    Parameters
    ----------
    tally : Tally
        Games, wins by name and by team (seat), draws, and the mean number of decisions.
    rounds : float
        The mean number of rounds per game.
    damage_dealt, damage_taken : ndarray of int64, shape (n_kinds,)
        Hit points removed by and from each kind of unit, over all games.
    kills, losses : ndarray of int64, shape (n_kinds,)
        Units killed by each kind (the killing blow), and units of each kind killed.
    """

    tally: Tally
    rounds: float
    damage_dealt: NDArray[np.int64]
    damage_taken: NDArray[np.int64]
    kills: NDArray[np.int64]
    losses: NDArray[np.int64]


def summarize(game: Tactics, records: Sequence[Record[TacticsPosition]]) -> Summary:
    """Count what a set of games shows.

    Parameters
    ----------
    game : Tactics
        The rules the games were played by.
    records : sequence of Record
        The games, at least one.

    Returns
    -------
    Summary
        Wins, draws, lengths, and damage and kills by kind of unit.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("summarize needs at least one game")
    n_kinds = len(game.units)
    dealt = np.zeros(n_kinds, dtype=np.int64)
    taken = np.zeros(n_kinds, dtype=np.int64)
    kills = np.zeros(n_kinds, dtype=np.int64)
    losses = np.zeros(n_kinds, dtype=np.int64)
    for record in records:
        last: Attacked | None = None
        for event in record.events:
            if isinstance(event, Attacked):
                dealt[event.attacker_kind] += event.damage
                taken[event.target_kind] += event.damage
                last = event
            elif isinstance(event, Died):
                losses[event.kind] += 1
                if last is not None and last.target == event.uid:
                    kills[last.attacker_kind] += 1
    return Summary(
        tally=tally(records),
        rounds=float(np.mean([record.final.round for record in records])),
        damage_dealt=dealt,
        damage_taken=taken,
        kills=kills,
        losses=losses,
    )


@dataclass(frozen=True, eq=False)
class Balance:
    """How fair and lively a set of games was, as the rules' scores for self-play.

    Parameters
    ----------
    fairness : Fairness
        How fair the sides were: team 0's edge over the decided games, and the draws.
    impact : ndarray of float64, shape (n_kinds,)
        Hit points removed per unit of each kind deployed, both sides together. NaN for a kind
        that no side deploys. What a unit does for others, such as an aura, doesn't count.
    evenness : float
        The smallest impact of a deployed kind over the largest, from 0 to 1: 1 when every kind
        removes as many hit points per unit, and when no unit removes any.
    """

    fairness: Fairness
    impact: NDArray[np.float64]
    evenness: float


def balance(game: Tactics, summary: Summary) -> Balance:
    """Score a set of games: how fair the sides were, and how much each kind does.

    Parameters
    ----------
    game : Tactics
        The rules the games were played by.
    summary : Summary
        What the games showed, from `summarize` with the same rules.

    Returns
    -------
    Balance
        The scores.

    Raises
    ------
    ValueError
        If ``summary`` counts another number of kinds of unit than ``game`` has.
    """
    n_kinds = len(game.units)
    if summary.damage_dealt.shape != (n_kinds,):
        raise ValueError(
            f"the summary counts {summary.damage_dealt.shape[0]} kinds of unit, "
            f"the rules have {n_kinds}"
        )
    deployed = 2 * summary.tally.games * np.array(game.counts, dtype=np.int64)
    impact = np.full(n_kinds, np.nan)
    np.divide(summary.damage_dealt, deployed, out=impact, where=deployed > 0)
    present = impact[deployed > 0]
    top = present.max(initial=0.0)
    return Balance(
        fairness=fairness(summary.tally),
        impact=impact,
        evenness=float(present.min() / top) if top > 0 else 1.0,
    )
