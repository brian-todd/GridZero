"""Pictures of Tactics: the board as hexagons, and the units on it.

Tiles are drawn as polygons rather than rasters, so a picture stays sharp at any size: open
ground, walls, rubble and the two homes. Each unit is a disc in its team's color with its kind's
letter, a tick toward the direction it faces, and its hit points. A tile an archer watches gets
a cross in the archer's color.
"""

import numpy as np
from matplotlib.axes import Axes

from gridzero import render

from .game import Tactics
from .state import TacticsPosition

TEAM_COLORS = ((0.75, 0.22, 0.17), (0.17, 0.35, 0.63))
"""The colors of team 0 and team 1, as RGB."""

_GROUND = (0.87, 0.83, 0.71)
_WALL = (0.22, 0.22, 0.24)
_RUBBLE = (0.58, 0.49, 0.39)


def draw(game: Tactics, position: TacticsPosition, ax: Axes | None = None) -> Axes:
    """Draw a position: the board, the units, their facing and hit points, and watched tiles.

    Parameters
    ----------
    game : Tactics
        The rules, for the kinds' letters.
    position : TacticsPosition
        The position to draw.
    ax : matplotlib.axes.Axes, optional
        Where to draw. By default, on a new figure of its own, made without `pyplot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on, with equal aspect and no frame. Its first collection holds one
        hexagon per tile.
    """
    state = position.state
    centers = state.space.centers
    colors = np.tile(_GROUND, (state.n_cells, 1))
    for team, color in enumerate(TEAM_COLORS):
        colors[state.home == team] = 0.6 * np.array(_GROUND) + 0.4 * np.array(color)
    colors[state.rubble] = _RUBBLE
    colors[state.wall] = _WALL
    ax = render.cells(state.space, colors, ax)
    u = state.units
    cells = np.flatnonzero(state.occupied)
    if cells.size:
        ax.scatter(
            centers[cells, 0],
            centers[cells, 1],
            s=110,
            c=[TEAM_COLORS[t] for t in u.team[cells]],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    directions = state.space.directions() @ state.space.basis.T  # unit vectors of the plane
    for c in cells:
        x, y = centers[c]
        dx, dy = 0.45 * directions[u.facing[c]]
        ax.plot([x, x + dx], [y, y + dy], color="black", linewidth=1.2, zorder=4)
        letter = game.units[u.kind[c]].name[0].upper()
        ax.text(x, y, letter, ha="center", va="center", fontsize=7, color="white", zorder=5)
        ax.text(x, y - 0.42, str(u.hp[c]), ha="center", va="top", fontsize=5, zorder=5)
    for archer in np.flatnonzero(state.occupied & (u.watch >= 0)):
        x, y = centers[u.watch[archer]]
        ax.scatter([x], [y], marker="x", s=40, color=TEAM_COLORS[u.team[archer]], zorder=3)
    return ax
