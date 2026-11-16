"""What a game's rules as arrays provide: the protocols searches and self-play run on."""

from typing import Protocol

import jax
from jax.typing import ArrayLike

from gridzero.core import State
from gridzero.game import Action, Position


class ArrayRules[S, D](Protocol):
    """A game's rules as JAX functions of one state of fixed shape: what runs in batches.

    The state ``S`` is a tuple of arrays with the same shapes in every position, so that the
    rules compile (``jax.jit``) and run for many games at once (``jax.vmap``). ``D`` is a
    step's dice: the arrays of its random draws. A step is `apply`, a pure function of its
    dice; `step` rolls them from the state's own stream, JAX's.

    Actions are **local**: an integer below ``n_actions``, small enough to keep per node of a
    search. What a local action means depends on the position (a target near the unit to
    move); `framed` maps it to where a network scores it.

    A search reads a step's chances from `outcomes`: the states a step may lead to, each with
    its exact probability, so that it can weigh a hit and a miss without rolling.

    A game that hides something plays its real state, and shows each player a state of its own
    (`view`), as `game.Game.view` does.

    Attributes
    ----------
    n_actions : int
        How many local actions there are.
    n_outcomes : int
        The most outcomes `outcomes` tells apart for one step: 1 for a game without dice.
    """

    @property
    def n_actions(self) -> int:
        """How many local actions there are."""
        ...

    @property
    def n_outcomes(self) -> int:
        """The most outcomes `outcomes` tells apart for one step."""
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

    def over(self, state: S) -> jax.Array:
        """Return whether the game has ended.

        Parameters
        ----------
        state : S
            One state.

        Returns
        -------
        Array of bool, shape ()
            True once no action is legal.
        """
        ...

    def winner(self, state: S) -> jax.Array:
        """Return the player that won.

        Parameters
        ----------
        state : S
            One state.

        Returns
        -------
        Array of int32, shape ()
            The winner, or -1: while the game runs, and for a draw once it is over.
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

    def roll(self, key: ArrayLike) -> D:
        """Return the dice of one step, drawn from a key.

        Parameters
        ----------
        key : Array of uint32, shape (2,)
            A generator's key (`gridzero.batched.root`, `gridzero.batched.stream`).

        Returns
        -------
        D
            Every draw a step may need. The same key gives the same dice.
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
            `roll` of the state's own key for this step. The same state gives the same dice.
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

    def outcomes(self, state: S, action: ArrayLike, dice: D) -> tuple[S, jax.Array]:
        """Return every way a step may go, each with its probability.

        The draws a search should weigh (an attack that hits or misses) are told apart: one
        state per outcome. Draws with too many outcomes to list (an order of turns) are read
        from ``dice``, as `apply` reads them.

        Parameters
        ----------
        state : S
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``.
        dice : D
            A step's draws, for what the outcomes do not tell apart.

        Returns
        -------
        states : S
            The next states, stacked: every array has a leading axis of ``n_outcomes``.
        chances : Array of float32, shape (n_outcomes,)
            The probability of each, summing to 1, and 0 for an outcome that cannot happen
            (whose state means nothing). A step without such draws has one outcome, the
            first, of probability 1.
        """
        ...

    def view(self, state: S, player: ArrayLike) -> S:
        """Return what a player is shown of a state, as the game's ``view`` shows it.

        A game that hides something (Tactics under fog) shows a position of what the player
        believes, which plays on as a game hiding nothing: what a search plays. A game that
        hides nothing shows the state itself.

        Parameters
        ----------
        state : S
            One state.
        player : Array of int, shape ()
            The player.

        Returns
        -------
        S
            The state shown, in the same shapes.
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

    def framed(self, state: S) -> jax.Array:
        """Return where a network scores each local action: its index in the mover's frame.

        A network of the game (`gridzero.agents.net.PolicyNet`) scores every ``(verb, cell)``
        in the frame of the player to move (`gridzero.agents.frame.Frame`). This is the map
        from local actions into those scores.

        Parameters
        ----------
        state : S
            A state not over.

        Returns
        -------
        Array of int32, shape (n_actions,)
            ``verb * n_cells + cell`` of each local action, as the mover's frame names it.
            Any index in range for a local action that is not legal.
        """
        ...


class ArrayGame[P: Position[State, object], S, D](ArrayRules[S, D], Protocol):
    """A game's rules once more, as arrays, and how they map onto the game they copy.

    The game itself (a `game.Game`) stays the reference: players, ladders, pages and tests
    read it. Its array rules (`ArrayRules`) make the same moves on a state ``S`` of arrays.
    ``P`` is the game's position type.

    What they promise, and the game's tests check: converted to arrays, the position after any
    legal action is `apply` of that action with the dice the game rolled, and one of its
    `outcomes`; `legal` is the game's mask; `observe` is its observation, bit for bit; and
    `framed` names each action as the game's frames do. `step` rolls its own dice, so it
    plays the same game without repeating any one game of a seed.

    `action` and `local` map a local action to the game's ``(verb, cell)`` and back.
    """

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
