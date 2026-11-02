"""Colonies' board in colors, one per cell, to paint on a globe (`render.globe`)."""

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

from .game import ColoniesState, ColoniesView

SEA = (0.62, 0.78, 0.92, 1.0)
"""The color of the sea."""
EMPTY = (0.75, 0.75, 0.72, 1.0)
"""The color of land nobody lives on."""
FOG = 0.45
"""How bright a cell a player doesn't see is, against one it sees."""
PLAYERS = colormaps["tab10"](np.arange(1, 10))[:, :3]
"""Each player's color, RGB of float64 of shape (9, 3): Matplotlib's "tab10" from its second,
so that none is the sea's blue. Players past the ninth take them again from the first."""


def colors(state: ColoniesState) -> NDArray[np.float64]:
    """Return a color per cell: the sea, empty land, or the color of the most people there.

    Players take the colors of `PLAYERS`, blended with the empty land's color by how full the
    cell is: its color alone at its capacity or beyond.

    Parameters
    ----------
    state : ColoniesState
        A state of the game.

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        RGBA colors in [0, 1].
    """
    out = np.tile(np.asarray(EMPTY), (state.n_cells, 1))
    out[~state.land] = SEA
    total = state.people.sum(axis=0)
    lived = state.land & (total > 0)
    fill = np.minimum(total[lived] / state.capacity[lived], 1.0)[:, None]
    most = PLAYERS[np.argmax(state.people[:, lived], axis=0) % len(PLAYERS)]
    out[lived, :3] = fill * most + (1 - fill) * out[lived, :3]
    return out


def view_colors(state: ColoniesState, view: ColoniesView) -> NDArray[np.float64]:
    """Return a color per cell of what a player is shown: the people it sees, the rest in fog.

    Parameters
    ----------
    state : ColoniesState
        The state the view was taken of, for the map.
    view : ColoniesView
        What the player is shown (`Colonies.view`).

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        RGBA colors in [0, 1]: as `colors` with only the people in ``view``, and the cells the
        player doesn't see darkened to `FOG` of their brightness.
    """
    people = np.zeros_like(state.people)
    people[: len(view.people)] = view.people
    out = colors(state.replace({ColoniesState.people: people}))
    out[~view.seen, :3] *= FOG
    return out
