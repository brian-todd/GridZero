"""Terrain for games on lattices: heights of continents and seas, and their colors."""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import LatticeSpace
from gridzero.core._arrays import check_int, check_real, finite_floats
from gridzero.worlds._grid import _grid_place
from gridzero.worlds._palette import (
    _GRASS,
    _HILLS,
    _ROCK,
    _SNOW,
    _mix,
    _sea_colors,
)
from gridzero.worlds.noise import _relief

_HIGHLAND = 0.5
"""Elevation at which the lowland has turned to hills, and mountains begin."""


def generate_terrain(
    space: LatticeSpace, *, seed: SupportsIndex = 0, land: float = 0.3, roughness: float = 0.6
) -> NDArray[np.float64]:
    """Generate terrain for a lattice from a seed: continents and seas, hills and mountains.

    Height is the sum of two `noise` fields: continents, with wavelengths from the size of the
    lattice down to about a quarter of it, plus half as much detail at smaller scales. Sea
    level is set so that land covers the fraction ``land`` of the cells. Features keep their
    size relative to the lattice, so it holds about as many continents as a planet, however
    many cells it has.

    Where the lattice wraps around, so does the terrain: height repeats along the axes that
    wrap, so land and sea continue across the edge without a seam.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of squares, hexagons or triangles, such as the board of a game.
    seed : int, default 0
        Seed of every random draw. The same seed gives the same terrain, and the same
        continents on a lattice with more or fewer cells, of the same tiling and proportions.
    land : float, default 0.3
        Fraction of the cells above sea level, in [0, 1]. The actual fraction differs from it
        by at most half a cell.
    roughness : float, default 0.6
        Amplitude ratio between successive octaves of detail, positive. Higher values give
        more ragged coasts and more islands.

    Returns
    -------
    ndarray of float64, shape (n_cells,)
        Height relative to sea level, in [-1, 1]: -1 at the deepest point of the sea and 1 at
        the highest peak. Land is where it is above 0. A game may keep it in a field of its
        state, or turn it into terrain of its own.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, ``seed`` is not an integer, or ``land`` or
        ``roughness`` is not a real number.
    ValueError
        If ``space`` is not 2-D, ``seed`` is negative, ``land`` is not in [0, 1], or
        ``roughness`` is not positive.
    """
    space = _check_terrain_space(space)
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    land = check_real(land, "land", 0.0, 1.0)
    scale = np.sqrt(4 * np.pi / (space.n_cells * space.measures[0]))  # to a planet's area
    wrapped = np.array(space.periodic)
    periods = scale * (space.basis * np.array(space.shape)).T[wrapped]  # (n_wrapped, 2)
    elevation, _ = _relief(
        scale * _grid_place(space),
        rng,
        areas=space.measures,
        land=land,
        roughness=roughness,
        periods=periods,
    )
    return elevation


def terrain_colors(elevation: ArrayLike) -> NDArray[np.float64]:
    """Return colors for terrain: seas by depth, and land by height.

    Parameters
    ----------
    elevation : array_like of float, shape (n_cells,)
        Height relative to sea level, in [-1, 1], as `generate_terrain` makes it. Land is
        where it is above 0.

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        Opaque RGBA colors in [0, 1]: blue seas, darker where deeper; green lowland, turning
        to tan hills by elevation 0.5, to rock on the mountains and to snow on the highest
        peaks.

    Raises
    ------
    TypeError
        If ``elevation`` does not hold numbers.
    ValueError
        If ``elevation`` is not 1-D, holds NaN or infinite values, or leaves [-1, 1].
    """
    elevation = finite_floats(elevation, "elevation")
    if elevation.ndim != 1:
        raise ValueError(f"elevation must be 1-D, one value per cell, got shape {elevation.shape}")
    if elevation.size and not (elevation.min() >= -1 and elevation.max() <= 1):
        raise ValueError("elevation must lie in [-1, 1]")
    height = elevation[:, None]
    ground = _mix(_GRASS, _HILLS, height / _HIGHLAND)
    ground = _mix(ground, _ROCK, (height - _HIGHLAND) / 0.25)
    ground = _mix(ground, _SNOW, (height - 0.85) / 0.15)
    rgb = np.where(height > 0, ground, _sea_colors(elevation))
    return np.column_stack([rgb, np.ones(len(rgb))])


def _check_terrain_space(space: object) -> LatticeSpace:
    """Return ``space`` after checking that terrain can be made for it."""
    if not isinstance(space, LatticeSpace):
        raise TypeError(f"terrain needs a 2-D LatticeSpace, got {type(space).__name__}")
    if space.dim != 2:
        raise ValueError(f"terrain needs a 2-D lattice, got one with {space.dim} axes")
    return space
