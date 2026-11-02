"""The simplest player of any game: a legal action, uniformly at random."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.game import Action


@dataclass(frozen=True)
class UniformPlayer:
    """Picks one of the legal actions uniformly at random, whatever the game and the view.

    A baseline for any game: what a learner must beat first. Games may have random players
    of their own that know more, such as Tactics' ``RandomPlayer``, which ends a unit's turn
    only now and then.
    """

    def __call__(
        self,
        game: object,
        player: int,
        view: object,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return a legal action, each with the same chance (see `game.Player`).

        Raises
        ------
        ValueError
            If no action is legal.
        """
        actions = np.flatnonzero(legal)
        if actions.size == 0:
            raise ValueError("no action is legal")
        verb, cell = divmod(int(actions[rng.integers(actions.size)]), legal.shape[-1])
        return Action(verb, cell)
