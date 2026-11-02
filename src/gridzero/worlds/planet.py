"""Planets: the terrain and climate of a planet, on a sphere or projected onto a lattice."""

import functools
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Self, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import LatticeSpace, MeshSpace, border, random_points, regions
from gridzero.core._arrays import (
    check_int,
    check_real,
    finite_floats,
    frozen_copy,
    read_only,
)
from gridzero.core.geometry import lon_lat, sphere_points
from gridzero.core.points import _cell_reach, _draw_in_cells
from gridzero.worlds._grid import _grid_place
from gridzero.worlds._palette import (
    _FOREST,
    _GRASS,
    _ROCK,
    _SAND,
    _SEA_ICE,
    _SNOW,
    _mix,
    _sea_colors,
)
from gridzero.worlds.noise import _relief, noise

# Climate thresholds, on the [0, 1] scales of `Planet.temperature` and `Planet.moisture`.
_ICE = 0.2  # land colder than this is ice
_WET = 0.6  # land this wet is forest


@dataclass(frozen=True, eq=False, init=False)
class Planet:
    """The terrain and climate of a planet, on a sphere or projected onto a lattice.

    A planet lies on a spherical mesh, such as ``icosphere(32)``, or is projected onto a 2-D
    lattice, its map, in the equirectangular projection: x runs over the longitudes, once
    around the planet along axis 0, and y over the latitudes, from the south pole at the bottom
    of axis 1 to the north pole at the top. A map may wrap around along either axis. Wrapping
    along y joins the two poles.

    A planet holds static values per cell, from which its land, coasts and landmasses follow.
    Places on a planet are longitudes and latitudes, on a mesh and a map alike. A planet is
    immutable, so copying it returns the same object. Its arrays are read-only, and derived
    values are computed once.

    Parameters
    ----------
    space : MeshSpace or LatticeSpace
        A spherical mesh, or a 2-D lattice of squares or hexagons.
    elevation : array_like of float, shape (n_cells,)
        Height relative to sea level, in [-1, 1]: -1 at the deepest point of the sea and 1 at
        the highest peak. Land is where it is above 0.
    temperature : array_like of float, shape (n_cells,)
        Warmth, in [0, 1]. Land colder than 0.2 is ice.
    moisture : array_like of float, shape (n_cells,)
        Wetness, in [0, 1], from desert to forest at 0.6 and wetter.

    Raises
    ------
    TypeError
        If ``space`` is neither a `MeshSpace` nor a `LatticeSpace`, or a value array does not
        hold numbers.
    ValueError
        If ``space`` is a mesh that is not spherical or a lattice that is not 2-D or is of
        triangles, or a value array has the wrong shape, holds NaN or infinite values, or
        leaves its range.
    """

    space: MeshSpace | LatticeSpace
    elevation: NDArray[np.float64]
    temperature: NDArray[np.float64]
    moisture: NDArray[np.float64]

    def __init__(
        self,
        space: MeshSpace | LatticeSpace,
        elevation: ArrayLike,
        temperature: ArrayLike,
        moisture: ArrayLike,
    ) -> None:
        object.__setattr__(self, "space", _check_planet_space(space))
        self._store("elevation", elevation, -1.0)
        self._store("temperature", temperature, 0.0)
        self._store("moisture", moisture, 0.0)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor, so unpickled arrays are checked and read-only again.
        return type(self), tuple(getattr(self, field.name) for field in fields(self))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def _store(self, name: str, given: ArrayLike, low: float) -> None:
        """Check the value array ``name``, then store it as a read-only float64 copy.

        ``given`` must be finite numbers of shape (n_cells,), in ``[low, 1]``.
        """
        n_cells = self.space.n_cells
        values = finite_floats(given, name)
        if values.shape != (n_cells,):
            raise ValueError(f"{name} must have shape ({n_cells},), got {values.shape}")
        if values.size and not (values.min() >= low and values.max() <= 1):
            raise ValueError(f"{name} must lie in [{low}, 1]")
        object.__setattr__(self, name, frozen_copy(values, np.float64))

    @functools.cached_property
    def land(self) -> NDArray[np.bool_]:
        """Cells above sea level: bool, shape (n_cells,)."""
        return read_only(self.elevation > 0)

    @functools.cached_property
    def coast(self) -> NDArray[np.bool_]:
        """Land cells that share a side with the sea: bool, shape (n_cells,)."""
        return read_only(border(self.space, self.land))

    @functools.cached_property
    def landmasses(self) -> NDArray[np.int64]:
        """The landmass of each land cell, and -1 at sea: int64, shape (n_cells,).

        Landmasses are the `core.regions` of the land: land cells connected through shared
        sides, numbered from 0 by area, largest first. On a map that wraps around, a landmass
        continues across the edge.
        """
        return read_only(regions(self.space, self.land))

    def colors(self) -> NDArray[np.float64]:
        """Return terrain colors: seas by depth, and land by moisture, height and cold.

        Returns
        -------
        ndarray of float64, shape (n_cells, 4)
            Opaque RGBA colors in [0, 1]: blue seas, darker where deeper and pale where frozen;
            land from sand through grass to forest as it gets wetter, turning to rock on high
            ground and to snow on ice.
        """
        elevation, temperature = self.elevation[:, None], self.temperature[:, None]
        sea = np.where(temperature < _ICE / 2, _SEA_ICE, _sea_colors(self.elevation))
        wet = self.moisture[:, None] / _WET
        ground = np.where(
            wet < 0.5, _mix(_SAND, _GRASS, wet / 0.5), _mix(_GRASS, _FOREST, wet / 0.5 - 1)
        )
        ground = _mix(ground, _ROCK, (elevation - 0.4) / 0.4)
        ground = _mix(ground, _SNOW, (_ICE + 0.05 - temperature) / 0.1)
        rgb = np.where(self.land[:, None], ground, sea)
        return np.column_stack([rgb, np.ones(len(rgb))])

    def cell_at(self, lon: float, lat: float) -> int:
        """Return the cell at a place: the one containing it.

        On the map of a planet, it is the cell that shows the place. Places nearer a pole than
        the centers of the first or last row of the map belong to that row.

        Parameters
        ----------
        lon, lat : float
            Longitude and latitude in degrees, as in `core.geometry.sphere_points`.

        Returns
        -------
        int
            Cell id.

        Raises
        ------
        TypeError
            If ``lon`` or ``lat`` is not a real number.
        ValueError
            If ``lon`` or ``lat`` is not finite, or ``lat`` is not in [-90, 90].
        """
        lon = check_real(lon, "the longitude")
        lat = check_real(lat, "the latitude", -90.0, 90.0)
        if isinstance(self.space, MeshSpace):
            return int(self.space.locate(sphere_points(lon, lat)))
        return int(_map_cells(self.space, np.array([lon]), np.array([lat]))[0])

    def random_place(
        self,
        rng: np.random.Generator,
        size: int | None = None,
        *,
        where: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return places drawn at random, evenly over the planet or some of its cells.

        Each cell gets places in proportion to its area on the planet, also on the planet's
        map. Pass a place to `cell_at` to get its cell, as in
        ``planet.cell_at(*planet.random_place(rng))``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of the places.
        size : int, optional
            How many places to draw, at least 0. By default, one.
        where : array_like of bool, shape (n_cells,), optional
            The cells the places must lie in, such as ``planet.land``. By default, every cell.

        Returns
        -------
        ndarray of float64, shape (2,) or (size, 2)
            One place, or ``size`` of them: longitude and latitude in degrees.

        Raises
        ------
        TypeError
            If ``rng`` is not a `numpy.random.Generator`, ``size`` is not an integer, or
            ``where`` is not boolean.
        ValueError
            If ``size`` is negative, or ``where`` has the wrong shape or holds no cell.
        """
        if isinstance(self.space, MeshSpace):
            lon, lat = lon_lat(random_points(self.space, rng, size, where=where))
            return np.stack([np.mod(lon + 180, 360) - 180, lat], axis=-1)
        boxes = self._map_box_areas
        return _draw_in_cells(rng, size, where, boxes, self._in_map_boxes, self._map_cells_of, 2)

    @functools.cached_property
    def _map_box_areas(self) -> NDArray[np.float64]:
        """Return the area on the planet of the box around each cell of its map: (n_cells,)."""
        west, east, south, north = self._map_boxes
        return np.radians(east - west) * (np.sin(np.radians(north)) - np.sin(np.radians(south)))

    def _in_map_boxes(
        self, rng: np.random.Generator, cells: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Return a place drawn evenly over the box around each cell of the map: (len(cells), 2).

        Evenly by the planet's area: longitudes evenly, and the sines of latitudes.
        """
        west, east, south, north = (bound[cells] for bound in self._map_boxes)
        lon = rng.uniform(west, east)
        sin_lat = rng.uniform(np.sin(np.radians(south)), np.sin(np.radians(north)))
        lat = np.degrees(np.arcsin(sin_lat))
        return np.column_stack([np.mod(lon + 180, 360) - 180, lat])

    def _map_cells_of(self, places: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return the cell of the map at each place, as `cell_at` finds it: (n_places,)."""
        return _map_cells(self._map, places[:, 0], places[:, 1])

    @property
    def _map(self) -> LatticeSpace:
        """The lattice of a planet's map, for what only maps have."""
        if not isinstance(self.space, LatticeSpace):
            raise TypeError("a planet on a sphere has no map")
        return self.space

    @functools.cached_property
    def _map_boxes(self) -> tuple[NDArray[np.float64], ...]:
        """Return a box of longitudes and latitudes around each cell of a planet's map.

        Its west, east, south and north bounds in degrees: four arrays of shape (n_cells,).
        """
        space = self._map
        lengths = np.array(space.shape) * np.diag(space.basis)
        place = _grid_place(space)
        lon = 360 * place[:, 0] / lengths[0] - 180
        lat = 180 * place[:, 1] / lengths[1] - 90
        half_lon, half_lat = _cell_reach(space) * [360, 180] / lengths
        # Rows lie evenly from pole to pole, so the boxes of the first and last rows reach the
        # poles, as their cells do.
        south, north = np.maximum(lat - half_lat, -90.0), np.minimum(lat + half_lat, 90.0)
        return lon - half_lon, lon + half_lon, south, north


def generate_planet(
    space: MeshSpace | LatticeSpace,
    *,
    seed: SupportsIndex = 0,
    land: float = 0.3,
    roughness: float = 0.6,
) -> Planet:
    """Generate a planet from a seed: continents, seas, and a climate that follows them.

    On a spherical mesh, the planet is a globe. On a 2-D lattice, it is projected onto the
    lattice as its map (see `Planet`), with the same continents as on a mesh for the same
    seed. A lattice of shape ``(2 * n, n)`` keeps square cells square at the equator.

    Height is the sum of two `noise` fields: continents, with wavelengths from the size of the
    planet down to about a quarter of it, plus half as much detail at smaller scales. Sea level
    is set so that land covers the fraction ``land`` of the surface. Temperature falls toward
    the poles and with height, with a little noise. Moisture varies with a third field and
    falls toward the middle of large continents.

    Parameters
    ----------
    space : MeshSpace or LatticeSpace
        A spherical mesh, such as ``icosphere(32)``, or a 2-D lattice of squares or hexagons.
    seed : int, default 0
        Seed of every random draw. The same seed gives the same planet, and on another mesh or
        map the same continents.
    land : float, default 0.3
        Fraction of the planet's surface above sea level, in [0, 1]. The actual fraction
        differs from it by less than the area of one cell. Land near the poles takes up more
        of a map than of the planet.
    roughness : float, default 0.6
        Amplitude ratio between successive octaves of detail, positive. Higher values give
        more ragged coasts and more islands.

    Returns
    -------
    Planet

    Raises
    ------
    TypeError
        If ``space`` is neither a `MeshSpace` nor a `LatticeSpace`, ``seed`` is not an integer,
        or ``land`` or ``roughness`` is not a real number.
    ValueError
        If ``space`` is a mesh that is not spherical or a lattice that is not 2-D or is of
        triangles, ``seed`` is negative, ``land`` is not in [0, 1], or ``roughness`` is not
        positive.
    """
    space = _check_planet_space(space)
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    land = check_real(land, "land", 0.0, 1.0)
    points, areas = _on_sphere(space) if isinstance(space, MeshSpace) else _on_map(space)
    elevation, continents = _relief(
        points, rng, areas=areas, land=land, roughness=roughness, periods=None
    )
    weather = noise(points, rng, scales=(1.0, 32.0), roughness=0.7)
    cold = (1 - np.hypot(points[:, 0], points[:, 1])) / 0.8  # 1 - cos(latitude), stretched
    temperature = 1 - cold - 0.6 * np.maximum(elevation, 0.0) + 0.05 * weather
    moisture = 0.55 + 0.2 * weather - 0.3 * continents
    return Planet(space, elevation, np.clip(temperature, 0, 1), np.clip(moisture, 0, 1))


def _check_planet_space(space: object) -> MeshSpace | LatticeSpace:
    """Return ``space`` after checking that a planet can lie on it."""
    if isinstance(space, LatticeSpace):
        if space.dim != 2:
            raise ValueError(f"a planet's map needs a 2-D lattice, got one with {space.dim} axes")
        if space.tiling == "triangle":
            raise ValueError("a planet's map needs a lattice of squares or hexagons, got triangles")
        return space
    if isinstance(space, MeshSpace):
        if space.metric != "spherical":
            raise ValueError(
                f"a planet needs a spherical mesh, got one with metric {space.metric!r}"
            )
        return space
    raise TypeError(
        "a planet needs a spherical MeshSpace, such as icosphere(32), or a 2-D LatticeSpace "
        f"for its map, got {type(space).__name__}"
    )


def _on_sphere(space: MeshSpace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return where each cell of a spherical mesh lies on the unit sphere, and its area.

    Shapes (n_cells, 3) and (n_cells,), float64.
    """
    return space.centers / np.linalg.norm(space.centers, axis=1, keepdims=True), space.measures


def _on_map(space: LatticeSpace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return where each cell of a planet's map lies on the planet, and the area it stands for.

    Longitude grows along x, once around the planet over the length of axis 0. Latitude grows
    along y, from -90 to 90 degrees over the length of axis 1. Shapes (n_cells, 3) and
    (n_cells,), float64: points on the unit sphere, and areas on it.
    """
    place = _grid_place(space)
    lengths = np.array(space.shape) * np.diag(space.basis)  # the basis is upper triangular
    lon = 360 * place[:, 0] / lengths[0] - 180
    lat = 180 * place[:, 1] / lengths[1] - 90
    half_row = np.radians(90 / space.shape[1])
    band = np.sin(np.radians(lat) + half_row) - np.sin(np.radians(lat) - half_row)
    return sphere_points(lon, lat), band  # cells of a row share its band equally


def _map_cells(
    space: LatticeSpace, lon: NDArray[np.float64], lat: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Return the cell of a planet's map that shows each place: int64, shape (n_places,).

    The inverse of `_on_map`, with latitudes kept between the centers of the first and the
    last row. Copies of the map once around apart along x tile the plane, so exactly one of a
    place's three copies, once around apart, lies on the map, whichever way the lattice wraps.
    """
    lengths = np.array(space.shape) * np.diag(space.basis)
    shift = space.basis.sum(axis=1) / 2  # from a cell's center to its place
    x = (np.mod(lon + 180, 360) / 360) * lengths[0] - shift[0]
    y = (lat + 90) / 180 * lengths[1] - shift[1]
    y = np.clip(y, 0.0, (space.shape[1] - 1) * space.basis[1, 1])
    q = np.linalg.solve(space.basis, np.stack([x, y]))[0]  # in steps along axis 0
    x += (np.mod(q + 0.5, space.shape[0]) - 0.5 - q) * space.basis[0, 0]  # into the rhombus
    copies = np.stack([x, x + lengths[0], x - lengths[0]], axis=-1)  # (n_places, 3)
    cells = space.locate(np.stack([copies, np.broadcast_to(y[:, None], copies.shape)], axis=-1))
    return cells[np.arange(len(cells)), np.argmax(cells >= 0, axis=1)]
