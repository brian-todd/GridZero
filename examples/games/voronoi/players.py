"""Players of the Voronoi game, beside `agents.UniformPlayer`, which plays any game."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.game import Action

from .game import PLACE, TIE, Voronoi, VoronoiPosition


@dataclass(frozen=True)
class Greedy:
    """Places the stone that leaves it furthest ahead of its strongest rival, looking no further.

    Each free land cell is scored by the area the player would then hold minus the most any
    other player would; ties, within `TIE`, go to a cell drawn from the player's generator. On
    the last stone of a game of two players that is a perfect move: the best result it can
    reach.
    """

    def __call__(
        self,
        game: Voronoi,
        player: int,
        position: VoronoiPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the placement with the largest lead (see the class)."""
        cells = np.flatnonzero(legal[PLACE])
        leads = np.empty(cells.size)
        for i, cell in enumerate(cells.tolist()):
            areas = game.areas(game.step(position, (PLACE, cell)).state)
            leads[i] = areas[player] - np.delete(areas, player).max()
        best = cells[leads >= leads.max() - TIE * game.space.measures.sum()]
        return Action(PLACE, int(rng.choice(best)))
