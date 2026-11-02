"""The Voronoi game on a planet: the third game, and the first off a lattice.

Players take turns placing stones on the land of an icosphere, a fixed number each; each land
cell belongs to the player whose stone is fewest moves away over land, and the most land area
wins. ``Voronoi(players=3)`` is the same game for three. It tests `game` where the other games
don't: a `core.MeshSpace` board, cells of unequal areas, any number of players, and steps that
draw nothing.

Game
----
Voronoi, PLACE, TIE, board
    The rules, as an immutable value and a `game.Game`, their one verb, how close two areas
    must be to be equal, and the board of a frequency.

State
-----
VoronoiState, VoronoiPosition, Placed
    The land, the stones and the territory each holds, a position, and what a step did.

Players
-------
Greedy
    Places the stone that leaves it furthest ahead, looking no further.

Strength
--------
EVAL_SEED
    The games players are scored on (by `game.ladder`).

Pictures
--------
colors, PLAYERS
    A color per cell, for `render.globe`, and each player's.
"""

from examples.games.voronoi.game import (
    PLACE,
    TIE,
    Placed,
    Voronoi,
    VoronoiPosition,
    VoronoiState,
    board,
)
from examples.games.voronoi.match import EVAL_SEED
from examples.games.voronoi.picture import PLAYERS, colors
from examples.games.voronoi.players import Greedy

__all__ = [
    "EVAL_SEED",
    "PLACE",
    "PLAYERS",
    "TIE",
    "Greedy",
    "Placed",
    "Voronoi",
    "VoronoiPosition",
    "VoronoiState",
    "board",
    "colors",
]
