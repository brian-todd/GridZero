"""What a game of k in a row holds: who owns each cell, and positions with what each step did."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from gridzero.core import Field, LatticeSpace, State
from gridzero.game import Position


class InARowState(State):
    """The board: the player whose stone is on each cell."""

    space: LatticeSpace  # lines and columns follow the lattice's coordinates
    owner = Field(np.int8, default=-1)
    """0 or 1, the player whose stone is there; -1 on a free cell."""


class Placed(NamedTuple):
    """A player placed a stone."""

    player: int
    cell: int


@dataclass(frozen=True, eq=False)
class InARowPosition(Position[InARowState, Placed]):
    """A position of k in a row: whose turn it is follows from the ply (player 0 first)."""
