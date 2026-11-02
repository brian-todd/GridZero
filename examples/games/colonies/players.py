"""Players of Colonies, beside `agents.UniformPlayer`, which plays any game.

Both read only their view, so they play fairly under fog.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.core import Adjacency, pull
from gridzero.game import Action

from .game import GATHER, HOLD, Colonies, ColoniesView


@dataclass(frozen=True)
class Settler:
    """Never attacks: holds while its land fills, then gathers toward the seen land with most room.

    Room is a cell's capacity minus everyone's people in it. Once the player's people reach
    a share ``full`` of the capacity of the cells they are in, it gathers toward the legal
    cell with the most room, the lowest id among equals; it holds while they don't, or when
    no seen land has room.

    Parameters
    ----------
    full : float, default 0.8
        How full its own cells must be before it moves on.
    """

    full: float = 0.8

    def __call__(
        self,
        game: Colonies,
        player: int,
        view: ColoniesView,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the order (see the class)."""
        own = view.people[player]
        room = np.where(legal[GATHER], view.capacity - view.people.sum(axis=0), -np.inf)
        if room.max() <= 0 or own.sum() < self.full * view.capacity[own > 0].sum():
            return Action(HOLD, 0)
        return Action(GATHER, int(np.argmax(room)))


@dataclass(frozen=True)
class Raider:
    """Attacks the enemies it outnumbers, and otherwise settles as `Settler` does.

    A seen cell is prey when the player's people in it and its neighbors outnumber the
    enemies in it by ``edge`` times. The raider gathers toward the prey with the most
    enemies, the lowest id among equals.

    Parameters
    ----------
    edge : float, default 1.5
        How many times the enemies its people nearby must be.
    full : float, default 0.8
        As for `Settler`, when there is no prey.
    """

    edge: float = 1.5
    full: float = 0.8

    def __call__(
        self,
        game: Colonies,
        player: int,
        view: ColoniesView,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the order (see the class)."""
        own = view.people[player]
        enemies = view.people.sum(axis=0) - own
        near = own + pull(own, Adjacency().compile(game.space))
        prey = legal[GATHER] & (enemies > 0) & (near > self.edge * enemies)
        if prey.any():
            return Action(GATHER, int(np.argmax(np.where(prey, enemies, -1.0))))
        return Settler(self.full)(game, player, view, legal, rng)
