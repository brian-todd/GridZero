"""The Voronoi game's board in colors, one per cell, to paint on a globe (`render.globe`)."""

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

from .game import VoronoiState

SEA = (0.62, 0.78, 0.92, 1.0)
"""The color of the sea."""
NOBODY = (0.75, 0.75, 0.72, 1.0)
"""The color of land nobody holds."""
STONE = 0.55
"""How much darker a cell with a stone is than its holder's color."""
PLAYERS = colormaps["tab10"](np.arange(1, 10))[:, :3]
"""Each player's color, RGB of float64 of shape (9, 3): Matplotlib's "tab10" from its second,
so that none is the sea's blue. Players past the ninth take them again from the first."""


def colors(state: VoronoiState) -> NDArray[np.float64]:
    """Return a color per cell: the sea, land nobody holds, or its holder's color.

    Players take the colors of `PLAYERS`, and cells with a stone are darker.

    Parameters
    ----------
    state : VoronoiState
        A state of the game.

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        RGBA colors in [0, 1].
    """
    out = np.tile(np.asarray(NOBODY), (state.n_cells, 1))
    out[~state.land] = SEA
    held = state.territory >= 0
    out[held, :3] = PLAYERS[state.territory[held] % len(PLAYERS)]
    stones = state.stone >= 0
    out[stones, :3] = PLAYERS[state.stone[stones] % len(PLAYERS)] * STONE
    return out
