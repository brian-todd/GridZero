"""What a game and a player are: the protocols the harness plays by."""

from typing import Protocol, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import State

from .position import Action, Position


class Game[P: Position[State, object], V](Protocol):
    """The rules of a game: an immutable value that holds no position.

    Steps are pure: the same position and action always give the same next position, which
    draws only from its stream (see `Position`) and says whether it did (``Position.drew``).
    Players decide in turns, one decision a step. ``P`` is the game's position type and ``V``
    what a player is shown of one: the position itself when nothing is hidden.

    Attributes
    ----------
    n_players : int
        How many players, numbered from 0.
    n_verbs : int
        How many kinds of action: the first axis of `legal`'s mask.
    """

    @property
    def n_players(self) -> int:
        """How many players, numbered from 0."""
        ...

    @property
    def n_verbs(self) -> int:
        """How many kinds of action."""
        ...

    def start(self, seed: SupportsIndex) -> P:
        """Return the first position of a game of this seed.

        Parameters
        ----------
        seed : int
            The game's root seed, non-negative: ``Position.seed``.

        Returns
        -------
        P
            Ply 0.
        """
        ...

    def player(self, position: P) -> int:
        """Return the player who decides next.

        Parameters
        ----------
        position : P
            A position not over.

        Returns
        -------
        int
            From 0 to ``n_players - 1``.
        """
        ...

    def view(self, position: P, player: int) -> V:
        """Return what a player is shown of a position.

        Parameters
        ----------
        position : P
            A position of this game.
        player : int
            The player looking.

        Returns
        -------
        V
            The position itself for a game with nothing hidden.
        """
        ...

    def legal(self, position: P) -> NDArray[np.bool_]:
        """Return the mask of legal actions: the actions `step` accepts.

        Parameters
        ----------
        position : P
            A position of this game.

        Returns
        -------
        ndarray of bool, shape (n_verbs, n_cells)
            ``mask[verb, cell]`` is True where ``Action(verb, cell)`` is legal. All False once
            the game is over.
        """
        ...

    def step(self, position: P, action: tuple[SupportsIndex, SupportsIndex]) -> P:
        """Return the position after one action, leaving ``position`` unchanged.

        Parameters
        ----------
        position : P
            A position not over.
        action : Action or tuple of int
            ``(verb, cell)``, legal in ``position``.

        Returns
        -------
        P
            The next position, one ply later.

        Raises
        ------
        ValueError
            If the game is over or the action is illegal (IndexError for a cell that isn't one,
            TypeError for a verb or cell that isn't an integer).
        """
        ...


class Player[G, V](Protocol):
    """Picks one legal action, from what it is shown.

    ``G`` is the game it plays, such as a game's own class, and ``V`` what that game shows a
    player. It is told which player it is, so it never asks the rules (``game.player`` needs a
    position, which a view may not be). The arguments are positional, so a player may name them
    as it likes, such as ``position`` for a game whose view is its position. Players are values,
    frozen dataclasses of their parameters, so that a series can send them to other processes.
    """

    def __call__(
        self, game: G, player: int, view: V, legal: NDArray[np.bool_], rng: np.random.Generator, /
    ) -> Action:
        """Return an action.

        Parameters
        ----------
        game : G
            The rules.
        player : int
            Which player decides: ``game.player(position)``, from 0 to ``n_players - 1``.
        view : V
            What the game shows this player of the position: ``game.view(position, player)``.
        legal : ndarray of bool, shape (n_verbs, n_cells)
            ``game.legal(position)``.
        rng : numpy.random.Generator
            The player's own stream for this decision, so a game replays exactly.

        Returns
        -------
        Action
            An action that ``legal`` allows.
        """
        ...
