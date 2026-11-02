"""Worlds for simulations and games to build on: planets, and terrain for lattices.

Random fields are functions of position, so one seed gives the same continents at every
resolution. Questions about cells that need only a space, such as `core.regions`,
`core.border` and `core.random_points`, live in `core`, and so do the values cells hold: a
game board is a `core.State` on a 2-D `core.LatticeSpace`.

Planet, generate_planet
    The terrain and climate of a planet, on a spherical mesh or projected onto a 2-D lattice
    (the planet's map), with land, coasts and landmasses, and places in longitude and
    latitude.
generate_terrain, terrain_colors
    Heights of continents and seas on a 2-D lattice, for games that want them, and their
    colors.
noise
    Random smooth fields at points of a sphere or a plane.
"""

from gridzero.worlds.noise import noise
from gridzero.worlds.planet import Planet, generate_planet
from gridzero.worlds.terrain import generate_terrain, terrain_colors

__all__ = [
    "Planet",
    "generate_planet",
    "generate_terrain",
    "noise",
    "terrain_colors",
]
