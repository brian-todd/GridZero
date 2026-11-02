"""The rules of k in a row: `InARow`, a family of games as data.

Two players take turns placing a stone on a square board. The first with ``length`` stones in a
straight line (along a row, a column or a diagonal) wins, and a full board without one is a
draw. With ``gravity``, a stone falls to the lowest free cell of its column, as in Connect Four;
without it, any free cell may be taken, as in tic-tac-toe.

The board is a `LatticeSpace` of shape ``(columns, rows)``: axis 0 is the column and axis 1 the
height, row 0 at the bottom. An action ``(PLACE, cell)`` names the cell where the stone lands.

It is `agents.Learnable`: a player sees its own stones, the other's, and whether it moved first
(parity matters in games of lines), on the board as it is, which is both players' frame.
"""

import functools
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import Frame
from gridzero.core import LatticeSpace, check_int
from gridzero.core._arrays import read_only
from gridzero.game import Action, check_action

from .state import InARowPosition, InARowState, Placed

PLACE = 0
"""The one verb: place a stone on the cell named."""


@dataclass(frozen=True)
class InARow:
    """The rules of k in a row, as an immutable value and a `game.Game` (see the module).

    ``InARow()`` is Connect Four's rules, and ``InARow((3, 3), 3, gravity=False)`` is
    tic-tac-toe.

    Parameters
    ----------
    shape : tuple of (int, int), default (7, 6)
        Columns and rows, each at least 1.
    length : int, default 4
        Stones in a line that win, at least 1, and no more than fit along a column or a row.
    gravity : bool, default True
        Whether a stone falls to the lowest free cell of its column.

    Raises
    ------
    TypeError
        If ``shape`` is not a tuple of two integers, ``length`` not an integer, or ``gravity``
        not a bool.
    ValueError
        If a side or ``length`` is below 1, or no line of ``length`` fits on the board.
    """

    shape: tuple[int, int] = (7, 6)
    length: int = 4
    gravity: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise TypeError(f"shape must be a tuple of two integers, got {self.shape!r}")
        for side in self.shape:
            check_int(side, "a side of shape", minimum=1)
        check_int(self.length, "length", minimum=1)
        if not isinstance(self.gravity, bool):
            raise TypeError(f"gravity must be a bool, got {type(self.gravity).__name__}")
        if self.length > max(self.shape):
            raise ValueError(f"no line of {self.length} fits on a board of shape {self.shape}")

    @property
    def n_players(self) -> int:
        """Two players: 0 moves first."""
        return 2

    @property
    def n_verbs(self) -> int:
        """One verb, `PLACE`."""
        return 1

    @property
    def space(self) -> LatticeSpace:
        """The board: squares of shape ``(columns, rows)``, without wrapping."""
        return _space(self.shape)

    @property
    def lines(self) -> NDArray[np.int64]:
        """Every line of ``length`` cells, shape (n_lines, length), read-only.

        Lines run along rows, columns and both diagonals, in lattice coordinates, so none
        wraps from one column into the next. Ordered by direction, then by first cell.
        """
        return _lines(self.shape, self.length)[0]

    @property
    def planes(self) -> tuple[str, ...]:
        """The planes of an observation: the player's stones, the other's, and "first".

        "first" is 1 on every cell for player 0, who moves first, and 0 for player 1.
        """
        return ("own", "other", "first")

    def start(self, seed: SupportsIndex) -> InARowPosition:
        """Return the empty board (see `game.Game`): the same for every seed.

        Parameters
        ----------
        seed : int
            The game's root seed, at least 0. The rules draw nothing from it.

        Returns
        -------
        InARowPosition
            Ply 0, every cell free.
        """
        return InARowPosition(InARowState(self.space), check_int(seed, "seed", minimum=0))

    def player(self, position: InARowPosition) -> int:
        """Return the player to move: 0 on even plies, 1 on odd ones.

        Raises
        ------
        ValueError
            If the game is over.
        """
        if position.over:
            raise ValueError("the game is over")
        return position.ply % 2

    def view(self, position: InARowPosition, player: int) -> InARowPosition:
        """Return the position itself: nothing is hidden."""
        return position

    def legal(self, position: InARowPosition) -> NDArray[np.bool_]:
        """Return the cells a stone may be placed on (see `game.Game`).

        Returns
        -------
        ndarray of bool, shape (1, n_cells)
            With gravity, the lowest free cell of each column that isn't full; without it,
            every free cell. All False once the game is over.
        """
        owner = position.state.owner
        mask = np.zeros((1, owner.size), dtype=np.bool_)
        if position.over:
            return mask
        if not self.gravity:
            mask[0] = owner < 0
            return mask
        heights = (self.space.grid_view(owner) >= 0).sum(axis=1)  # stones stack from row 0
        columns = np.flatnonzero(heights < self.shape[1])
        mask[0, self.space.ravel(np.stack([columns, heights[columns]], axis=-1))] = True
        return mask

    def step(
        self, position: InARowPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> InARowPosition:
        """Return the position after placing a stone (see `game.Game`).

        The game ends when the stone completes a line, which wins, or fills the board, which
        is a draw unless it also completes a line.

        Raises
        ------
        TypeError
            If the verb or the cell is not an integer.
        ValueError
            If the game is over or the action is illegal.
        IndexError
            If the cell is not a cell id.
        """
        verb, cell = check_action(position, action, self.n_verbs)
        if not self.legal(position)[verb, cell]:
            raise ValueError(f"illegal action {Action(verb, cell)} at ply {position.ply}")
        state = position.state
        player = position.ply % 2
        owner = state.owner.copy()
        owner[cell] = player
        lines, through = _lines(self.shape, self.length)
        won = bool((owner[lines[through[cell]]] == player).all(axis=1).any())
        return InARowPosition(
            state.replace({InARowState.owner: owner}),
            position.seed,
            ply=position.ply + 1,
            over=won or bool((owner >= 0).all()),
            winner=player if won else None,
            events=(Placed(player, cell),),
        )

    def observe(self, position: InARowPosition, player: int) -> NDArray[np.float32]:
        """Return what a player sees of a position (see `agents.Learnable`).

        Parameters
        ----------
        position : InARowPosition
            A position of this game.
        player : int
            The player looking, 0 or 1.

        Returns
        -------
        ndarray of float32, shape (3, n_cells)
            The planes of `planes`, 0 or 1, on the board as it is.
        """
        owner = position.state.owner
        first = np.full(owner.size, player == 0)
        return np.stack([owner == player, owner == 1 - player, first]).astype(np.float32)

    def frame(self, player: int) -> Frame:
        """Return a player's frame: the board as it is, for both players.

        A half turn would put the board upside down, against gravity, so neither player's
        view is turned.
        """
        return _frame(self.space.n_cells)


@functools.cache
def _space(shape: tuple[int, int]) -> LatticeSpace:
    """Return the board of a shape, made once."""
    return LatticeSpace(shape, periodic=False)


@functools.cache
def _frame(n_cells: int) -> Frame:
    """Return the frame that is the board itself, made once per size."""
    return Frame.board(1, n_cells)


@functools.cache
def _lines(
    shape: tuple[int, int], length: int
) -> tuple[NDArray[np.int64], tuple[NDArray[np.int64], ...]]:
    """Return the lines of a board, (n_lines, length), and the lines through each cell.

    The directions are the vertex offsets of the squares, one of each opposite pair: those
    pointing to higher columns, or up the same column.
    """
    space = _space(shape)
    offsets = space.offsets("vertex")
    forward = offsets[(offsets[:, 0] > 0) | ((offsets[:, 0] == 0) & (offsets[:, 1] > 0))]
    coords = space.coords
    found = []
    for d in forward:
        cells = coords[:, None, :] + np.arange(length)[:, None] * d  # (n_cells, length, 2)
        inside = ((cells >= 0) & (cells < np.array(shape))).all(axis=(1, 2))
        found.append(space.ravel(cells[inside]))
    lines = np.concatenate(found).reshape(-1, length)
    through = tuple(
        read_only(np.flatnonzero((lines == cell).any(axis=1))) for cell in range(space.n_cells)
    )
    return read_only(lines), through
