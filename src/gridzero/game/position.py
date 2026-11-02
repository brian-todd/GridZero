"""Positions of a game and the actions between them."""

import dataclasses
from dataclasses import dataclass
from typing import Generic, NamedTuple, SupportsIndex, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from gridzero.core import State
from gridzero.core._arrays import check_cell, check_int, check_real

_UNKEYED = frozenset({"state", "seed", "drew", "events"})
"""Fields of `Position` that `Position.key` doesn't read as values: the state is read by its
arrays, and the rest say how a position was reached, not what it is."""

# Declared covariant, so that a game's position is a `Position[State, object]`. The syntax of
# `class Position[S: State, E]` can't declare variance, and from Python 3.13 a frozen dataclass
# gets `__replace__(*, state: S, events: tuple[E, ...])`, from which type checkers infer
# invariance. Positions are never changed in place, so covariance is sound.
S_co = TypeVar("S_co", bound=State, covariant=True)
E_co = TypeVar("E_co", covariant=True)


class Action(NamedTuple):
    """One decision: a verb and the cell it names."""

    verb: int
    """Which kind of action, from 0 to the game's ``n_verbs - 1``."""
    cell: int
    """The cell it names. A verb without a target names a fixed cell of the game's choosing."""


@dataclass(frozen=True, eq=False)
class Position(Generic[S_co, E_co]):  # noqa: UP046 (variance, above)
    """A complete position of a game: the state and what isn't per cell.

    Positions are immutable values. A game's ``step`` makes a new one, so a search tree or a
    replay is just positions. Two positions compare equal only if they are the same object;
    `key` compares their values. A game adds what isn't per cell by subclassing, such as a round
    number, and names the subclass ``*Position``.

    Parameters
    ----------
    state : State
        The board and what is on it.
    seed : int
        The root of every random draw in the game. A step draws only from the stream
        ``core.Context(seed=seed, spawn_key=(1, ply))``. Streams ``(0, ...)`` are the game's
        own, such as its board's, and ``(2, ...)`` and above belong to the harness (`play`,
        `series`) and to players.
    ply : int, default 0
        Decisions taken so far.
    over : bool, default False
        Whether the game has ended.
    winner : int or None, default None
        The player who won, once the game is over; None for a draw or a game still running.
    drew : bool, default False
        Whether the step that made this position drew from its stream (``Context.drew``). A
        step that did not gives the same position under any seed.
    events : tuple, default ()
        What the step that made this position did, in order, in the game's own types.
    """

    state: S_co
    seed: int
    ply: int = 0
    over: bool = False
    winner: int | None = None
    drew: bool = False
    events: tuple[E_co, ...] = ()

    def key(self) -> bytes:
        """Return bytes that tell positions apart by their values.

        They hold the state's arrays, in field order, and every other field but ``seed``,
        ``drew`` and ``events``, which say how the position was reached: the ply, the result,
        and the fields a subclass adds. Those must be values whose ``repr`` tells them apart,
        such as integers, booleans, strings and tuples of them. Positions with equal keys are
        the same position of one game.

        Returns
        -------
        bytes
            The key.
        """
        state = self.state
        rest = tuple(
            getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in _UNKEYED
        )
        return b"".join(state[f].tobytes() for f in state) + repr(rest).encode()


def check_action(
    position: Position[State, object],
    action: tuple[SupportsIndex, SupportsIndex],
    n_verbs: int,
) -> Action:
    """Return an action as an `Action` of ints, checked as every game's ``step`` begins.

    Checks that the game isn't over, that the verb and the cell are integers, that the cell is a
    cell of the position's state, and that the verb is one of the game's. Whether the action is
    legal stays the game's to check, often more cheaply than through its whole mask.

    Parameters
    ----------
    position : Position
        The position the action is taken in.
    action : tuple of (int, int)
        ``(verb, cell)``.
    n_verbs : int
        The game's number of verbs.

    Returns
    -------
    Action
        The verb and the cell, as Python ints.

    Raises
    ------
    TypeError
        If the verb or the cell is not an integer.
    ValueError
        If the game is over or the verb is not in ``[0, n_verbs)``.
    IndexError
        If the cell is not a cell id.
    """
    if position.over:
        raise ValueError("the game is over")
    verb = check_int(action[0], "verb")
    cell = check_cell(action[1], position.state.n_cells)
    if not 0 <= verb < n_verbs:
        raise ValueError(f"verb must be in [0, {n_verbs}), got {verb}")
    return Action(verb, cell)


def winner(scores: ArrayLike, tolerance: float = 0.0) -> int | None:
    """Return the player with the highest score, or None if another's is as high.

    The result every game decides by a score per player, such as cells or area held. Scores
    within ``tolerance`` of the highest count as equal to it, for floats that agree but for
    rounding, so a game whose scores are float sums can still draw. With no tolerance, scores
    are compared exactly, integers past ``2**53`` too.

    Parameters
    ----------
    scores : array_like of real, shape (n_players,)
        Each player's score, at least one, finite.
    tolerance : float, default 0.0
        How far below the highest score a score still ties it, at least 0.

    Returns
    -------
    int or None
        The player whose score alone is the highest, or None for a draw.

    Raises
    ------
    TypeError
        If ``scores`` doesn't hold real numbers, or ``tolerance`` is not a real number.
    ValueError
        If ``scores`` is not one-dimensional or is empty, a score is not finite, or
        ``tolerance`` is negative or not finite.
    """
    values = np.asarray(scores)
    real = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)
    if not real:
        raise TypeError(f"scores must hold real numbers, got dtype {values.dtype}")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"scores must have shape (n_players,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    tolerance = check_real(tolerance, "tolerance", 0.0)
    top = values.max()
    best = np.flatnonzero(values >= top - tolerance if tolerance else values == top)
    return int(best[0]) if best.size == 1 else None
