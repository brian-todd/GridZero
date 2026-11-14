"""What a game's rules as arrays provide: the protocol searches and self-play will run on."""

from typing import Protocol

import jax
from jax.typing import ArrayLike

from gridzero.core import State
from gridzero.game import Action, Position


class ArrayGame[P: Position[State, object], S, D](Protocol):
    """A game's rules once more, as JAX functions of one state of fixed shape.

    The game itself (a `game.Game`) stays the reference: players, ladders, pages and tests
    read it. Its array rules make the same moves on a state ``S`` of arrays, a tuple of them
    with the same shapes in every position, so that they compile and run for many games at
    once (``jax.vmap``). ``P`` is the game's position type and ``D`` a step's dice: the arrays
    of its random draws.

    What they promise, and the game's tests check: converted to arrays, the position after any
    legal action is `apply` of that action with the dice the game rolled; `legal` is the game's
    mask; and `observe` is its observation, bit for bit. `step` rolls its own dice, JAX's, so
    it plays the same game without repeating any one game of a seed.

    Actions are **local**: an integer below ``n_actions``, small enough to keep per node of a
    search. What a local action means depends on the position (a target near the unit to
    move); `action` and `local` map it to the game's ``(verb, cell)`` and back.

    Attributes
    ----------
    n_actions : int
        How many local actions there are.
    """

    @property
    def n_actions(self) -> int:
        """How many local actions there are."""
        ...

    def to_arrays(self, position: P) -> S:
        """Return a position as a state of arrays.

        Parameters
        ----------
        position : P
            A position of the game.

        Returns
        -------
        S
            NumPy arrays, which JAX takes as they are. Equal positions give equal arrays.
        """
        ...

    def to_position(self, state: S) -> P:
        """Return a state of arrays as a position: the inverse of `to_arrays`.

        Parameters
        ----------
        state : S
            One state.

        Returns
        -------
        P
            A position whose arrays are ``state``'s. It carries no events. Played on by the
            game, it rolls the game's dice, not `step`'s.
        """
        ...

    def player(self, state: S) -> jax.Array:
        """Return the player who decides next.

        Parameters
        ----------
        state : S
            A state not over.

        Returns
        -------
        Array of int32, shape ()
            The player.
        """
        ...

    def legal(self, state: S) -> jax.Array:
        """Return the mask of legal local actions.

        Parameters
        ----------
        state : S
            One state.

        Returns
        -------
        Array of bool, shape (n_actions,)
            All False once the game is over.
        """
        ...

    def draws(self, state: S) -> D:
        """Return the dice of the step a state takes next, from its own stream.

        Parameters
        ----------
        state : S
            One state.

        Returns
        -------
        D
            Every draw a step may need. The same state gives the same dice.
        """
        ...

    def apply(self, state: S, action: ArrayLike, dice: D) -> S:
        """Return the state after a legal action, given the step's dice.

        Parameters
        ----------
        state : S
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``.
        dice : D
            The step's draws.

        Returns
        -------
        S
            The next state, one ply later.
        """
        ...

    def step(self, state: S, action: ArrayLike) -> S:
        """Return the state after a legal action: ``apply(state, action, draws(state))``.

        Parameters
        ----------
        state : S
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``.

        Returns
        -------
        S
            The next state, one ply later.
        """
        ...

    def observe(self, state: S, player: ArrayLike) -> jax.Array:
        """Return what a player sees of a state, as the game's ``observe`` lays it out.

        Parameters
        ----------
        state : S
            One state.
        player : Array of int, shape ()
            The player.

        Returns
        -------
        Array of float32, shape (n_planes, n_cells)
            The planes, in the player's frame.
        """
        ...

    def action(self, position: P, local: int) -> Action:
        """Return the game's action for a local action in a position.

        Parameters
        ----------
        position : P
            A position not over.
        local : int
            A local action, legal in ``position``.

        Returns
        -------
        Action
            ``(verb, cell)``.
        """
        ...

    def local(self, position: P, action: tuple[int, int]) -> int:
        """Return the local action of a game's action in a position: the inverse of `action`.

        Parameters
        ----------
        position : P
            A position not over.
        action : Action or tuple of int
            ``(verb, cell)``, legal in ``position``.

        Returns
        -------
        int
            The local action.
        """
        ...

    def dice(self, before: P, after: P) -> D:
        """Return the dice the game rolled in a step, read from the position it made.

        Parameters
        ----------
        before, after : P
            A position and the one a step made of it.

        Returns
        -------
        D
            Dice with which `apply` makes ``after`` of ``before``.
        """
        ...
