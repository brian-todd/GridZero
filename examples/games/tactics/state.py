"""What a game of Tactics holds: units as fields per cell, the board, actions and positions.

Units are fields per cell in a `NestedState`, since one unit per cell is a rule of the game: an
empty tile holds ``kind == -1``. Moves and swaps permute cells in every field of `Unit`. The few
values that aren't per cell (the seed, the count of steps and rounds, the result, who holds the
point) are kept by a `TacticsPosition` beside the state.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from gridzero.core import Field, LatticeSpace, Nested, NestedState, State, derived
from gridzero.game import Position

from .events import Event


class Unit(NestedState):
    """The fields of the unit on each tile, all at their defaults on an empty one."""

    uid = Field(np.int8, default=-1)
    """Identity across moves, from 0 in order of deployment; -1 on an empty tile."""
    kind = Field(np.int8, default=-1)
    """Index into the game's `UnitStats`; -1 on an empty tile."""
    team = Field(np.int8, default=-1)
    """0 or 1; -1 on an empty tile."""
    hp = Field(np.int16)
    """Hit points left, at least 1 for a unit on the board."""
    facing = Field(np.int8)
    """Direction faced, 0 to 5 (see `core.LatticeSpace.directions`)."""
    initiative = Field(np.int16)
    """Place in this round's order of turns: the lowest unit not `done` acts."""
    moved = Field(np.int8)
    """Tiles moved this round."""
    swapped = Field(np.bool_)
    """Has swapped this round, which uses up its movement."""
    attacked = Field(np.bool_)
    """Has attacked, deferred its attack or worked on a tile this round."""
    faced = Field(np.bool_)
    """Has turned since its last move, attack or swap, so may not turn again until then."""
    done = Field(np.bool_)
    """Has had its turn this round."""
    engaged = Field(np.bool_)
    """Was attacked this round: later attackers get a bonus."""
    watch = Field(np.int32, default=-1)
    """The tile an archer deferred its attack onto, or -1."""


class TacticsState(State):
    """A board of Tactics and the units on it."""

    space: LatticeSpace
    """The board: a 2-D lattice, hexes in every game so far."""
    wall = Field(np.bool_)
    """Blocks movement and sight. Engineers change it, so it isn't static."""
    rubble = Field(np.bool_)
    """Blocks movement only. Engineers change it, so it isn't static."""
    home = Field(np.int8, static=True)
    """The team that deploys on the tile, or -1."""
    point = Field(np.bool_, static=True)
    """The tile the sides fight over (`objective`)."""
    units = Nested(Unit)

    @derived(wall, rubble)
    def passable(self) -> NDArray[np.bool_]:
        """Tiles that units may stand on and cross when empty: neither wall nor rubble."""
        return ~(self.wall | self.rubble)

    @derived(units)
    def occupied(self) -> NDArray[np.bool_]:
        """Tiles that hold a unit."""
        return self.units.kind >= 0


class Verb(IntEnum):
    """What an action does. The cell it names is its target.

    ``FACE + d`` turns to direction ``d``, 0 to 5, and ``PLACE + k`` deploys a unit of kind
    ``k``, so a game with ``K`` kinds has ``PLACE + K`` verbs. Verbs without a target (`FACE`,
    `END`) name the acting unit's own cell.
    """

    MOVE = 0
    """Move to the cell, along the cheapest route."""
    ATTACK = 1
    """Attack the unit on the cell."""
    DEFER = 2
    """Defer the attack onto the cell: an archer's passive."""
    SWAP = 3
    """Swap places with the friendly unit on the cell."""
    FACE = 4
    """Turn to direction 0; ``FACE + d`` turns to direction ``d``."""
    END = 10
    """End the unit's turn."""
    BUILD = 11
    """Raise the adjacent tile one step: open to rubble, rubble to wall."""
    BREAK = 12
    """Lower the adjacent tile one step: wall to rubble, rubble to open."""
    PLACE = 13
    """Deploy a unit of kind 0 on the cell; ``PLACE + k`` deploys kind ``k``."""


class Hold(NamedTuple):
    """Who holds the point, and for how many rounds in a row (`objective`)."""

    owner: int = -1
    """The team that last had a unit on the point at a round's end, or -1 if none ever has."""
    clock: int = 0
    """Round ends in a row the owner has held it, the capture included."""


NOBODY = Hold()
"""The point before anyone has captured it."""


@dataclass(frozen=True, eq=False)
class TacticsPosition(Position[TacticsState, Event]):
    """A complete position of a game of Tactics: a `game.Position` with the round.

    Positions are immutable values. `Tactics.step` makes a new one, so a search or a replay is
    just positions. Two positions compare equal only if they are the same object.

    Parameters
    ----------
    state : TacticsState
        The board and the units on it.
    seed : int
        The root of every random draw in the game. A step draws from a stream keyed by ``ply``.
    ply : int, default 0
        Decisions taken so far.
    over : bool, default False
        Whether the game has ended.
    winner : int or None, default None
        The team that won, once the game is over; None for a draw or a game still running.
    drew : bool, default False
        Whether the step that made this position drew dice: started a round or rolled an
        attack. A step that did not gives the same position under any seed.
    events : tuple of Event, default ()
        What the step that made this position did, in order (see `events`).
    round : int, default 0
        0 during deployment, then the round being played, from 1.
    hold : Hold, default NOBODY
        Who holds the point.
    """

    round: int = 0
    hold: Hold = NOBODY

    @property
    def deploying(self) -> bool:
        """Whether the players are still deploying their units."""
        return self.round == 0


@dataclass(frozen=True, eq=False)
class Units:
    """The units on a board, one entry each, ordered by uid.

    A read-only list of what the fields per cell hold, for turn order, text boards and
    metrics. All arrays have shape ``(n_units,)``.

    Parameters
    ----------
    cell : ndarray of int64
        Where each unit stands.
    uid, kind, team, hp, facing : ndarray of int64
        The unit's fields of the same names.
    """

    cell: NDArray[np.int64]
    uid: NDArray[np.int64]
    kind: NDArray[np.int64]
    team: NDArray[np.int64]
    hp: NDArray[np.int64]
    facing: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.cell.size)


def units(state: TacticsState) -> Units:
    """Return the units on a board, ordered by uid.

    Parameters
    ----------
    state : TacticsState
        The board.

    Returns
    -------
    Units
        One entry per unit.
    """
    cells = np.flatnonzero(state.occupied)
    cells = cells[np.argsort(state.units.uid[cells], kind="stable")]

    def column(values: NDArray[np.integer]) -> NDArray[np.int64]:
        out = values[cells].astype(np.int64)
        out.setflags(write=False)
        return out

    cells.setflags(write=False)
    return Units(
        cell=cells,
        uid=column(state.units.uid),
        kind=column(state.units.kind),
        team=column(state.units.team),
        hp=column(state.units.hp),
        facing=column(state.units.facing),
    )
