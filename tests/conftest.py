"""Shared fixtures and factories for the test suite.

Tests get their inputs from fixtures and factories defined here rather than by monkeypatching.
Pass dependencies such as random generators explicitly. Spaces are immutable, so factories may
cache and share them between tests.
"""

from collections.abc import Callable, Iterator

import matplotlib
import numpy as np
import pytest
from scipy.spatial import SphericalVoronoi

from demo import DemoState
from gridzero.core import (
    LatticeSpace,
    MeshSpace,
    Space,
    State,
    Tiling,
    icosphere,
    implementation,
)
from gridzero.worlds import Planet

# Draw off screen whatever the session: with a display Matplotlib picks Tk, whose objects
# collected outside the main thread raise warnings, which are errors here.
matplotlib.use("Agg")

type IcosphereFactory = Callable[..., MeshSpace]
type LatticeFactory = Callable[..., LatticeSpace]
type StateFactory = Callable[..., State]
type PlanetFactory = Callable[..., Planet]


NOT_INTEGERS = {
    "float": 1.5,
    "integral-float": 2.0,
    "bool": True,
    "numpy-bool": np.True_,
    "string": "2",
    "none": None,
    "float-array": np.array(2.0),
}
"""Values that are not integers. Every integer parameter must reject them with TypeError."""

NOT_REALS = {
    "bool": True,
    "numpy-bool": np.True_,
    "string": "0.5",
    "none": None,
    "complex": 0.5 + 0j,
    "float-array": np.array(0.5),
}
"""Values that are not real numbers. Every real parameter must reject them with TypeError."""

NOT_CELL_IDS = {
    "floats": ([0.0, 1.5], TypeError),
    "integral-floats": ([0.0, 1.0], TypeError),
    "bools": ([True, False], TypeError),
    "strings": (["0", "1"], TypeError),
    "beyond-int64": (np.array([0, 2**64 - 1], dtype=np.uint64), ValueError),
}
"""Arrays that are not cell ids, with the error each must raise.

Floats are never truncated and bools never read as 0 and 1. A uint64 too large for int64 raises
instead of wrapping around to a negative id, which could pass for the -1 of an unused slot.
"""


@pytest.fixture(params=list(NOT_INTEGERS.values()), ids=list(NOT_INTEGERS))
def not_an_integer(request: pytest.FixtureRequest) -> object:
    """Each kind of value that is not an integer, in turn."""
    return request.param


@pytest.fixture(params=list(NOT_REALS.values()), ids=list(NOT_REALS))
def not_a_real(request: pytest.FixtureRequest) -> object:
    """Each kind of value that is not a real number, in turn."""
    return request.param


@pytest.fixture(params=list(NOT_CELL_IDS.values()), ids=list(NOT_CELL_IDS))
def not_cell_ids(request: pytest.FixtureRequest) -> tuple[np.ndarray, type[Exception]]:
    """Each kind of array that is not cell ids, in turn, with the error it must raise."""
    values, error = request.param
    return np.asarray(values), error


@pytest.fixture(params=["reference", "numba"])
def each_impl(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run the test once on the reference path and once on the Numba fast paths.

    Test modules of operations with fast paths use it for every test, through
    ``pytestmark = pytest.mark.usefixtures("each_impl")``, so both paths meet every behavior.
    """
    with implementation(request.param):
        yield request.param


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministically seeded random generator.

    Each test gets a fresh generator, so test order never changes results.
    """
    return np.random.default_rng(12345)


@pytest.fixture
def make_lattice() -> LatticeFactory:
    """Return a factory for small lattice spaces (default: 5x5 periodic squares)."""

    def factory(shape=(5, 5), tiling: Tiling = "hypercubic", periodic=True) -> LatticeSpace:
        return LatticeSpace(shape=shape, tiling=tiling, periodic=periodic)

    return factory


@pytest.fixture
def make_state() -> StateFactory:
    """Return a factory for small states of `demo.Demo` (default: on a 5x5 periodic lattice).

    ``make_state(space=None, step=0, **values)`` sets the fields given by attribute name, for
    brevity. Others are zero, except the static ``terrain``, which defaults to ``cell id % 3``.
    """

    def factory(space: Space | None = None, step: int = 0, **values) -> State:
        space = LatticeSpace((5, 5)) if space is None else space
        values.setdefault("terrain", np.arange(space.n_cells) % 3)
        return DemoState(
            space, {getattr(DemoState, name): v for name, v in values.items()}, step=step
        )

    return factory


@pytest.fixture(scope="session")
def make_icosphere() -> IcosphereFactory:
    """Return a caching factory for icosphere planets (default: frequency 2, 42 cells)."""
    cache: dict[tuple[int, float], MeshSpace] = {}

    def factory(frequency: int = 2, radius: float = 1.0) -> MeshSpace:
        key = (frequency, radius)
        if key not in cache:
            cache[key] = icosphere(frequency, radius)
        return cache[key]

    return factory


@pytest.fixture
def make_uneven_sphere() -> IcosphereFactory:
    """Return a caching factory for spherical meshes whose cells differ widely in size and shape.

    ``make_uneven_sphere(n_cells=40, seed=0)`` builds the spherical Voronoi cells of
    ``n_cells`` random sites. With 40 cells, the largest has more than 4 times the area of the
    smallest.
    """
    cache: dict[tuple[int, int], MeshSpace] = {}

    def factory(n_cells: int = 40, seed: int = 0) -> MeshSpace:
        if (n_cells, seed) not in cache:
            sites = np.random.default_rng(seed).normal(size=(n_cells, 3))
            sites /= np.linalg.norm(sites, axis=1, keepdims=True)
            voronoi = SphericalVoronoi(sites)
            voronoi.sort_vertices_of_regions()
            sizes = [len(region) for region in voronoi.regions]
            cache[n_cells, seed] = MeshSpace(
                centers=sites,
                measures=voronoi.calculate_areas(),
                vertices=voronoi.vertices,
                polygon_indptr=np.concatenate([[0], np.cumsum(sizes)]),
                polygon_indices=np.concatenate(voronoi.regions),
                metric="spherical",
            )
        return cache[n_cells, seed]

    return factory


@pytest.fixture
def make_caps_planet(make_icosphere: IcosphereFactory) -> PlanetFactory:
    """Return a factory for planets with land on two polar caps, warm and wet everywhere.

    ``make_caps_planet(frequency=4, north=0.3, south=0.3)`` puts land where the unit vector to a
    cell's center has ``z > north`` or ``z < -south``, so the caps are separated by a band of
    sea. The land is warm and wet, so people can live anywhere on it.
    """

    def factory(frequency: int = 4, north: float = 0.3, south: float = 0.3) -> Planet:
        planet = make_icosphere(frequency)
        z = planet.centers[:, 2]
        elevation = np.where(z > 0, z - north, -z - south) / 2
        warm_and_wet = np.ones(planet.n_cells)
        return Planet(planet, elevation, warm_and_wet, warm_and_wet)

    return factory


@pytest.fixture
def make_strips_map() -> PlanetFactory:
    """Return a factory for planets' maps with land on both sides of the edge x = 0.

    ``make_strips_map(shape=(12, 8), tiling="hypercubic", periodic=True)`` puts land on the
    cells whose first lattice coordinate is in the first or the last quarter of its axis, and
    sea between. A lattice that wraps around along that axis joins the two strips of land into
    one landmass. The land is warm and wet, so people can live anywhere on it.
    """

    def factory(shape=(12, 8), tiling: Tiling = "hypercubic", periodic=True) -> Planet:
        grid = LatticeSpace(shape, tiling=tiling, periodic=periodic)
        column, quarter = grid.coords[:, 0], shape[0] // 4
        land = (column < quarter) | (column >= shape[0] - quarter)
        warm_and_wet = np.ones(grid.n_cells)
        return Planet(grid, np.where(land, 0.5, -0.5), warm_and_wet, warm_and_wet)

    return factory


@pytest.fixture
def make_square_mesh() -> Callable[[int, int], MeshSpace]:
    """Return a factory for a planar mesh of unit squares, small enough to check by hand.

    ``make_square_mesh(rows, cols)`` builds ``rows * cols`` unit squares on a
    ``(rows + 1) x (cols + 1)`` grid of vertices. Cell ``r * cols + c`` is centered at
    ``(c + 0.5, r + 0.5)``.
    """

    def factory(rows: int = 2, cols: int = 2) -> MeshSpace:
        vertex_id = np.arange((rows + 1) * (cols + 1)).reshape(rows + 1, cols + 1)
        ys, xs = np.mgrid[0 : rows + 1, 0 : cols + 1]
        r, c = np.mgrid[0:rows, 0:cols]
        r, c = r.ravel(), c.ravel()
        polygons = np.stack(
            [vertex_id[r, c], vertex_id[r, c + 1], vertex_id[r + 1, c + 1], vertex_id[r + 1, c]],
            axis=1,
        )
        return MeshSpace(
            centers=np.stack([c + 0.5, r + 0.5], axis=1),
            measures=np.ones(rows * cols),
            vertices=np.stack([xs.ravel(), ys.ravel()], axis=1),
            polygon_indptr=np.arange(0, 4 * rows * cols + 1, 4),
            polygon_indices=polygons.ravel(),
        )

    return factory


SPACE_CASES: dict[str, Callable[[], Space]] = {
    "square-periodic": lambda: LatticeSpace((5, 5)),
    "square-fixed": lambda: LatticeSpace((4, 6), periodic=False),
    "cube-mixed": lambda: LatticeSpace((3, 4, 5), periodic=(True, False, True)),
    "tesseract-periodic": lambda: LatticeSpace((3, 3, 3, 3)),
    "hex-periodic": lambda: LatticeSpace((4, 5), tiling="hex"),
    "hex-fixed": lambda: LatticeSpace((4, 5), tiling="hex", periodic=False),
    "triangle-periodic": lambda: LatticeSpace((6, 4), tiling="triangle"),
    "triangle-fixed": lambda: LatticeSpace((5, 3), tiling="triangle", periodic=False),
    "icosphere-f1": lambda: icosphere(1),
    "icosphere-f3": lambda: icosphere(3),
}


@pytest.fixture(params=sorted(SPACE_CASES))
def any_space(request: pytest.FixtureRequest) -> Space:
    """Each kind of space in turn, for tests of properties every space must satisfy."""
    return SPACE_CASES[request.param]()
