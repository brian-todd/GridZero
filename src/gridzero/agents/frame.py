"""What a learning agent needs from a game: observations, and each player's frame.

A network that plays both sides of a two-player game learns faster if each side sees the
board as its own: its home in the same corner, its units as "own". A game gives that through
`Learnable`: an observation per player, planes of values per cell already in the player's
frame, and the `Frame` that maps actions between the board and that frame, so that a network's
choice in its frame goes back onto the board. `Observable` asks only that of a game's views, for
a learner that plays from views alone, and `Revealable` adds the whole position as a player
would see it with nothing hidden, for a learner that learns to guess what is hidden.
"""

from dataclasses import dataclass
from typing import Protocol, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import LatticeSpace, Space, State
from gridzero.core._arrays import check_cell, check_int, int_array, read_only
from gridzero.game import Action, Game, Position


@dataclass(frozen=True, eq=False, init=False)
class Frame:
    """A player's frame: where each action of the board goes, as permutations of cells and verbs.

    Action ``(v, c)`` in the frame is action ``(verbs[v], cells[c])`` on the board. A player
    who sees the board turned half a turn, as team 1 does in Tactics, has ``cells =
    space.opposite()``, and verbs that name directions turned too (`Frame.opposite`).

    Parameters
    ----------
    cells : array_like of int, shape (n_cells,)
        A permutation of the cell ids: frame cell ``i`` shows board cell ``cells[i]``.
    verbs : array_like of int, shape (n_verbs,)
        A permutation of the verbs: frame verb ``v`` is board verb ``verbs[v]``.

    Raises
    ------
    TypeError
        If an array is not of integers.
    ValueError
        If an array is not 1-D or not a permutation.
    """

    cells: NDArray[np.int64]
    verbs: NDArray[np.int64]
    _cells_back: NDArray[np.int64]
    _verbs_back: NDArray[np.int64]
    identity: bool
    """Whether the frame is the board itself: then `into` and `back` return their input."""

    def __init__(self, cells: ArrayLike, verbs: ArrayLike) -> None:
        arrays = []
        for name, values in (("cells", cells), ("verbs", verbs)):
            array = int_array(values, name)
            if array.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
            if not np.array_equal(np.sort(array), np.arange(array.size)):
                raise ValueError(f"{name} must be a permutation of 0 .. {array.size - 1}")
            back = np.empty_like(array)
            back[array] = np.arange(array.size)
            arrays.append((read_only(array.copy()), read_only(back)))
        (cells_, cells_back), (verbs_, verbs_back) = arrays
        set_ = object.__setattr__
        set_(self, "cells", cells_)
        set_(self, "verbs", verbs_)
        set_(self, "_cells_back", cells_back)
        set_(self, "_verbs_back", verbs_back)
        same = np.array_equal(cells_, np.arange(cells_.size))
        set_(self, "identity", same and np.array_equal(verbs_, np.arange(verbs_.size)))

    @staticmethod
    def board(n_verbs: SupportsIndex, n_cells: SupportsIndex) -> Frame:
        """Return the frame that is the board itself, for a player who sees it as it is.

        Parameters
        ----------
        n_verbs, n_cells : int
            The numbers of verbs and of cells, at least 0.

        Returns
        -------
        Frame
            Identity permutations.
        """
        n_verbs = check_int(n_verbs, "n_verbs", minimum=0)
        n_cells = check_int(n_cells, "n_cells", minimum=0)
        return Frame(np.arange(n_cells), np.arange(n_verbs))

    @staticmethod
    def opposite(
        space: LatticeSpace, n_verbs: SupportsIndex, directions: SupportsIndex | None = None
    ) -> Frame:
        """Return the frame of a player who sees the board turned half a turn.

        For two players facing each other across the board, as in Tactics: cells go through
        `LatticeSpace.opposite`, and verbs that name directions (`LatticeSpace.directions`)
        turn half way round, ``directions + d`` to ``directions + (d + n // 2) % n`` for the
        ``n`` directions. The frame is its own inverse.

        Parameters
        ----------
        space : LatticeSpace
            The board, which must have opposite cells.
        n_verbs : int
            The number of verbs, at least 0.
        directions : int, optional
            The verb of direction 0, followed by one verb per direction. None if no verb names a
            direction: verbs are then unchanged.

        Returns
        -------
        Frame
            A new frame.

        Raises
        ------
        TypeError
            If ``n_verbs`` or ``directions`` is not an integer.
        ValueError
            If the lattice has no opposite cells, or (with ``directions``) no directions, or
            the verbs of the directions don't fit in ``n_verbs``.
        """
        verbs = np.arange(check_int(n_verbs, "n_verbs", minimum=0))
        if directions is not None:
            first = check_int(directions, "directions", minimum=0)
            n = len(space.directions())
            if first + n > verbs.size:
                raise ValueError(
                    f"{n} direction verbs from verb {first} don't fit in {verbs.size} verbs"
                )
            verbs[first : first + n] = first + (np.arange(n) + n // 2) % n
        return Frame(space.opposite(), verbs)

    def into[T: np.generic](self, values: NDArray[T]) -> NDArray[T]:
        """Return values over the board's actions in this frame.

        Parameters
        ----------
        values : ndarray, shape (..., n_verbs, n_cells)
            Values per action, such as a mask of legal actions. Leading axes are kept.

        Returns
        -------
        ndarray, shape (..., n_verbs, n_cells)
            ``out[..., v, c] = values[..., verbs[v], cells[c]]``: a new array, or ``values``
            itself for the identity.

        Raises
        ------
        ValueError
            If ``values`` does not end in shape ``(n_verbs, n_cells)``.
        """
        return self._gather(values, self.verbs, self.cells)

    def back[T: np.generic](self, values: NDArray[T]) -> NDArray[T]:
        """Return values over this frame's actions back on the board: the inverse of `into`.

        Parameters
        ----------
        values : ndarray, shape (..., n_verbs, n_cells)
            Values per action in this frame, such as a network's scores.

        Returns
        -------
        ndarray, shape (..., n_verbs, n_cells)
            ``back(into(x))`` is ``x``. A new array, or ``values`` itself for the identity.

        Raises
        ------
        ValueError
            If ``values`` does not end in shape ``(n_verbs, n_cells)``.
        """
        return self._gather(values, self._verbs_back, self._cells_back)

    def action_into(self, action: tuple[SupportsIndex, SupportsIndex]) -> Action:
        """Return a board action as this frame names it.

        Parameters
        ----------
        action : Action or tuple of int
            ``(verb, cell)`` on the board.

        Returns
        -------
        Action
            The same action in this frame.

        Raises
        ------
        TypeError
            If the verb or cell is not an integer.
        IndexError
            If the verb or cell is out of range.
        """
        verb = _check_verb(action[0], self.verbs.size)
        cell = check_cell(action[1], self.cells.size)
        return Action(int(self._verbs_back[verb]), int(self._cells_back[cell]))

    def action_back(self, action: tuple[SupportsIndex, SupportsIndex]) -> Action:
        """Return an action of this frame on the board: the inverse of `action_into`.

        Parameters
        ----------
        action : Action or tuple of int
            ``(verb, cell)`` in this frame.

        Returns
        -------
        Action
            The same action on the board.

        Raises
        ------
        TypeError
            If the verb or cell is not an integer.
        IndexError
            If the verb or cell is out of range.
        """
        verb = _check_verb(action[0], self.verbs.size)
        cell = check_cell(action[1], self.cells.size)
        return Action(int(self.verbs[verb]), int(self.cells[cell]))

    def _gather[T: np.generic](
        self, values: NDArray[T], verbs: NDArray[np.int64], cells: NDArray[np.int64]
    ) -> NDArray[T]:
        """Return ``values[..., verbs[:, None], cells]``, checking the shape."""
        if values.shape[-2:] != (verbs.size, cells.size):
            raise ValueError(
                f"values must end in shape ({verbs.size}, {cells.size}), got {values.shape[-2:]}"
            )
        if self.identity:
            return values
        return values[..., verbs[:, None], cells]


class Observable[P: Position[State, object], V](Game[P, V], Protocol):
    """A game whose views a learner reads: observations as planes per cell, and frames.

    A learner that plays from views alone, such as `agents.muzero.MuZeroPlayer`, needs nothing
    more. Its positions all lie on one space (`space`), which a network's convolutions follow.
    """

    @property
    def space(self) -> Space:
        """The space every position of the game lies on."""
        ...

    @property
    def planes(self) -> tuple[str, ...]:
        """The names of an observation's planes, in order."""
        ...

    def observe(self, view: V, player: int, /) -> NDArray[np.float32]:
        """Return what a player sees, as planes per cell in its frame.

        Parameters
        ----------
        view : V
            What the game shows the player (``game.view(position, player)``), of a position
            not over.
        player : int
            The player looking.

        Returns
        -------
        ndarray of float32, shape (len(planes), n_cells)
            Values in [0, 1], so that they can be stored as bytes (``round(255 * x)``).
            Cell ``c`` of the planes is board cell ``frame(player).cells[c]``.
        """
        ...

    def frame(self, player: int, /) -> Frame:
        """Return a player's frame, the same for every position of the game.

        Parameters
        ----------
        player : int
            The player.

        Returns
        -------
        Frame
            Where the board's actions go in the player's frame.
        """
        ...


class Revealable[P: Position[State, object], V](Observable[P, V], Protocol):
    """An observable game that can show a player the whole position: what a belief learns.

    A learner trained with what is hidden from it, but never shown it in play, learns to guess
    it (`agents.muzero_net.WorldNet`'s belief).
    """

    @property
    def revealed(self) -> tuple[str, ...]:
        """The names of the planes `reveal` returns, in order."""
        ...

    def reveal(self, position: P, player: int, /) -> NDArray[np.float32]:
        """Return the whole position as a player would see it with nothing hidden.

        Parameters
        ----------
        position : P
            A position of this game.
        player : int
            The player it is shown to.

        Returns
        -------
        ndarray of float32, shape (len(revealed), n_cells)
            Values in [0, 1], in the player's frame, as `observe` gives them.
        """
        ...


class Learnable[P: Position[State, object]](Observable[P, P], Protocol):
    """A game a searching learner can play: its view is its position, on a 2-D lattice.

    Learners that search and train on positions (`agents.zero`) need the game to show each
    player the whole position (perfect information), so that `Observable.observe` reads a
    position. Its positions all lie on one 2-D lattice (`space`), which `agents.net.PolicyNet`'s
    lattice convolutions follow.
    """

    @property
    def space(self) -> LatticeSpace:
        """The lattice every position of the game lies on."""
        ...


def _check_verb(verb: object, n_verbs: int) -> int:
    """Return ``verb`` as an int after checking that it numbers one of ``n_verbs`` verbs.

    Raises TypeError if it is not an integer, and IndexError if it is outside ``[0, n_verbs)``.
    """
    index = check_int(verb, "a verb")
    if not 0 <= index < n_verbs:
        raise IndexError(f"verb {index} is out of range [0, {n_verbs})")
    return index
