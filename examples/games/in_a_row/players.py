"""Scripted players of k in a row: what an agent must beat after a uniform one."""

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.game import Action

from .game import PLACE, InARow
from .state import InARowPosition


@dataclass(frozen=True)
class ThreatPlayer:
    """Wins when it can, blocks the other player's win, and never hands one over.

    In order: a stone that completes a line; else one on a cell where the other player would
    complete one; else, at random, a stone after which the other player can't win at once
    (with gravity, not the cell under the other's winning cell); else any legal stone. Choices
    within a rule are drawn uniformly from the player's generator.
    """

    def __call__(
        self,
        game: InARow,
        player: int,
        position: InARowPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the action the rules above pick (see `game.Player`)."""
        moves = np.flatnonzero(legal[PLACE])
        after = [game.step(position, (PLACE, int(m))) for m in moves]
        for choices in (
            [m for m, a in zip(moves, after, strict=True) if a.winner == player],
            _wins_of_other(game, player, position, moves),
            [m for m, a in zip(moves, after, strict=True) if not _can_win(game, a)],
            list(moves),
        ):
            if choices:
                return Action(PLACE, int(choices[rng.integers(len(choices))]))
        raise ValueError("no action is legal")


def _wins_of_other(
    game: InARow, player: int, position: InARowPosition, moves: NDArray[np.int64]
) -> list[np.int64]:
    """Return the moves with which the other player would win, were it its turn."""
    passed = dataclasses.replace(position, ply=position.ply + 1)
    other = 1 - player
    return [m for m in moves if game.step(passed, (PLACE, int(m))).winner == other]


def _can_win(game: InARow, position: InARowPosition) -> bool:
    """Return whether the player to move can complete a line at once."""
    if position.over:
        return False
    me = game.player(position)
    moves = np.flatnonzero(game.legal(position)[PLACE])
    return any(game.step(position, (PLACE, int(m))).winner == me for m in moves)
