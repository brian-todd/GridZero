import copy
import functools
import pickle

import numpy as np
import pytest
from scipy.spatial import KDTree
from scipy.stats import chisquare

from gridzero.core import LatticeSpace, Tiling, icosphere, regions
from gridzero.core.geometry import lon_lat, sphere_points
from gridzero.worlds import Planet, generate_planet

MAP_CASES = [
    (tiling, periodic)
    for tiling in ("hypercubic", "hex")
    for periodic in ((True, True), (True, False), (False, True), (False, False))
]


def map_grid(tiling: Tiling = "hypercubic", periodic=(True, False), scale=1):
    """A lattice twice as wide as it is high, or about that for hexagons."""
    shape = (32 * scale, 16 * scale) if tiling == "hypercubic" else (32 * scale, 18 * scale)
    return LatticeSpace(shape, tiling=tiling, periodic=periodic)


CELL_AT_CASES = [
    *(
        (tiling, (32, 16) if tiling == "hypercubic" else (32, 18), periodic)
        for tiling, periodic in MAP_CASES
    ),
    *(("hex", (6, 20), periodic) for periodic in ((True, False), (False, False))),
]
"""Maps to find cells on. On the tall hex grids, rows shift by more than once around."""


def blank_map(tiling, shape, periodic):
    """The map of a planet that is all sea, for finding cells quickly."""
    grid = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    return Planet(grid, np.full(grid.n_cells, -0.5), np.ones(grid.n_cells), np.ones(grid.n_cells))


def band_areas(grid):
    """The share of a planet's surface each cell of its map stands for, from its row."""
    rows = grid.shape[1]
    edges = np.radians(np.linspace(-90, 90, rows + 1))
    band = np.diff(np.sin(edges)) / 2 / grid.shape[0]  # (rows,), adding up to 1
    return band[grid.coords[:, 1]]


def cell_places(grid):
    """The longitude and latitude each cell of a planet's map shows at its center."""
    lengths = np.array(grid.shape) * np.diag(grid.basis)
    corner = grid.centers + grid.basis.sum(axis=1) / 2  # the grid spans [0, lengths)
    return -180 + 360 * corner[:, 0] / lengths[0], -90 + 180 * corner[:, 1] / lengths[1]


def showing(grid, lon, lat, cells=None):
    """The cell whose center is nearest each place on the map, once around along x."""
    cells = np.arange(grid.n_cells) if cells is None else cells
    lengths = np.array(grid.shape) * np.diag(grid.basis)
    cell_lon, cell_lat = (values[cells] for values in cell_places(grid))
    dx = np.mod((lon[:, None] - cell_lon) / 360 + 0.5, 1) - 0.5  # in turns
    dy = (lat[:, None] - cell_lat) / 180
    return cells[np.argmin((dx * lengths[0]) ** 2 + (dy * lengths[1]) ** 2, axis=1)]


@functools.cache
def generated(on):
    """A generated planet with several landmasses, on a sphere or on a map. It is immutable."""
    space = icosphere(8) if on == "sphere" else LatticeSpace((48, 24), periodic=(True, False))
    return generate_planet(space, seed=0)


@pytest.fixture(params=["sphere", "map"])
def any_planet(request):
    return generated(request.param)


@pytest.mark.parametrize("land", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_land_covers_the_requested_fraction_of_the_surface(make_icosphere, land):
    planet = make_icosphere(8)
    world = generate_planet(planet, seed=2, land=land)
    covered = planet.measures[world.land].sum() / planet.measures.sum()
    assert abs(covered - land) <= planet.measures.max() / planet.measures.sum() / 2


@pytest.mark.parametrize("tiling", ["hypercubic", "hex"])
@pytest.mark.parametrize("land", [0.05, 0.3, 0.7])
def test_land_on_the_map_of_a_planet_covers_the_requested_fraction_of_the_planet(tiling, land):
    grid = map_grid(tiling)
    areas = band_areas(grid)
    world = generate_planet(grid, seed=2, land=land)
    assert abs(areas[world.land].sum() - land) <= areas.max() / 2


def test_elevation_reaches_minus_1_at_the_deepest_sea_and_1_at_the_highest_peak(make_icosphere):
    world = generate_planet(make_icosphere(8), seed=0)
    assert world.elevation.min() == -1
    assert world.elevation.max() == 1


@pytest.mark.parametrize(("tiling", "periodic"), MAP_CASES)
def test_elevation_on_a_map_reaches_minus_1_and_1(tiling, periodic):
    world = generate_planet(map_grid(tiling, periodic), seed=0)
    assert world.elevation.min() == -1
    assert world.elevation.max() == 1


@pytest.mark.parametrize("on", ["sphere", "map"])
def test_a_planet_repeats_with_its_seed_and_changes_with_it(make_icosphere, on):
    space = make_icosphere(8) if on == "sphere" else map_grid("hex", (True, True))
    first = generate_planet(space, seed=5)
    again = generate_planet(space, seed=5)
    for name in ("elevation", "temperature", "moisture"):
        np.testing.assert_array_equal(getattr(again, name), getattr(first, name))
    assert not np.array_equal(generate_planet(space, seed=6).land, first.land)


@pytest.mark.parametrize("seed", [0, 1])
def test_one_seed_puts_land_in_the_same_places_at_two_resolutions(make_icosphere, seed):
    # Only cells along the coasts may differ: 4-11% of the surface at these resolutions.
    coarse, fine = make_icosphere(8), make_icosphere(16)
    _, nearest = KDTree(coarse.centers).query(fine.centers)
    land = generate_planet(coarse, seed=seed).land[nearest]
    agree = np.average(generate_planet(fine, seed=seed).land == land, weights=fine.measures)
    assert agree > 0.85


@pytest.mark.parametrize("seed", [0, 1])
def test_the_map_of_a_planet_shows_the_planets_continents(make_icosphere, seed):
    planet, grid = make_icosphere(16), map_grid(scale=2)  # 2,562 cells, and 64 x 32
    land = generate_planet(planet, seed=seed).land[planet.locate(sphere_points(*cell_places(grid)))]
    agree = np.average(generate_planet(grid, seed=seed).land == land, weights=band_areas(grid))
    assert agree > 0.85


@pytest.mark.parametrize("tiling", ["hypercubic", "hex"])
def test_one_seed_puts_land_in_the_same_places_on_two_maps(tiling):
    coarse, fine = map_grid(tiling, scale=2), map_grid(tiling, scale=4)  # 87-89% at half that
    half = fine.basis.sum(axis=1) / 2  # from a cell's center to its corner, in lattice steps
    under = coarse.locate((fine.centers + half) / 2 - half)  # the coarse cell over each fine one
    land = generate_planet(coarse, seed=0).land[under]
    assert np.mean(generate_planet(fine, seed=0).land == land) > 0.85  # 0.93


def test_the_poles_are_colder_than_the_tropics(make_icosphere):
    planet = make_icosphere(16)
    world = generate_planet(planet, seed=0)
    _, lat = lon_lat(planet.centers)
    polar, tropical = np.abs(lat) > 60, np.abs(lat) < 30
    assert world.temperature[polar].max() < world.temperature[tropical].min()


@pytest.mark.parametrize("tiling", ["hypercubic", "hex"])
def test_the_poles_of_a_map_are_colder_than_its_tropics(tiling):
    grid = map_grid(tiling, scale=2)
    world = generate_planet(grid, seed=0)
    _, lat = cell_places(grid)
    polar, tropical = (np.abs(lat) > 60) & ~world.land, (np.abs(lat) < 30) & ~world.land
    assert world.temperature[polar].max() < world.temperature[tropical].min()


@pytest.mark.parametrize("seed", [0, 1])
def test_more_land_puts_every_place_further_inland_and_drier(make_icosphere, seed):
    # Moisture falls toward the middle of continents, but the weather can outweigh that in any
    # one place. Lowering the sea under the same weather moves every place inland.
    planet = make_icosphere(8)
    wet, dry = (generate_planet(planet, seed=seed, land=land).moisture for land in (0.3, 0.6))
    assert (dry <= wet).all()
    assert (wet - dry).mean() > 0.05  # 0.11-0.19 on seeds 0-4


def test_landmasses_are_numbered_by_area_largest_first(make_caps_planet):
    world = make_caps_planet(8, north=0.2, south=0.6)
    z = world.space.centers[:, 2]
    np.testing.assert_array_equal(world.landmasses, np.select([z > 0.2, z < -0.6], [0, 1], -1))


def test_land_coasts_and_landmasses_follow_from_elevation(any_planet):
    land = any_planet.elevation > 0
    np.testing.assert_array_equal(any_planet.land, land)
    np.testing.assert_array_equal(any_planet.landmasses, regions(any_planet.space, land))
    graph = any_planet.space.adjacency()
    beside_the_sea = [
        land[cell] and not land[graph.neighbors(cell)].all()
        for cell in range(any_planet.space.n_cells)
    ]
    np.testing.assert_array_equal(any_planet.coast, beside_the_sea)
    assert any_planet.coast.any()
    assert any_planet.landmasses.max() > 0
    for derived in (any_planet.land, any_planet.coast, any_planet.landmasses):
        assert not derived.flags.writeable


@pytest.mark.parametrize("tiling", ["hypercubic", "hex"])
@pytest.mark.parametrize("wraps", [True, False])
def test_landmasses_continue_across_the_edges_a_map_wraps_around(make_strips_map, tiling, wraps):
    world = make_strips_map(tiling=tiling, periodic=(wraps, True))
    assert world.landmasses.max() + 1 == (1 if wraps else 2)


def test_terrain_colors_are_opaque_with_blue_seas_pale_where_frozen(any_planet):
    colors = any_planet.colors()
    assert colors.shape == (any_planet.space.n_cells, 4)
    assert colors.min() >= 0
    assert colors.max() <= 1
    np.testing.assert_array_equal(colors[:, 3], 1.0)
    sea, temperature = ~any_planet.land, any_planet.temperature
    open_sea, frozen = sea & (temperature > 0.1), sea & (temperature < 0.1)
    assert open_sea.any()
    assert frozen.any()
    assert (colors[open_sea, 2] > colors[open_sea, 0]).all()
    assert (colors[frozen, :3] > 0.75).all()


def test_cell_at_finds_the_cell_around_a_place(make_caps_planet):
    world = make_caps_planet()
    planet = world.space
    lon, lat = lon_lat(planet.centers)
    found = [world.cell_at(a, b) for a, b in zip(lon, lat, strict=True)]
    np.testing.assert_array_equal(found, np.arange(planet.n_cells))


@pytest.mark.parametrize(("tiling", "shape", "periodic"), CELL_AT_CASES)
def test_cell_at_on_a_map_finds_each_cell_at_its_own_place_once_around(tiling, shape, periodic):
    world = blank_map(tiling, shape, periodic)
    lon, lat = cell_places(world.space)
    for turns in (0, 1, -2):
        found = [world.cell_at(a + 360 * turns, b) for a, b in zip(lon, lat, strict=True)]
        np.testing.assert_array_equal(found, np.arange(world.space.n_cells))


@pytest.mark.parametrize(("tiling", "shape", "periodic"), CELL_AT_CASES)
def test_cell_at_on_a_map_finds_the_cell_that_shows_a_place(rng, tiling, shape, periodic):
    world = blank_map(tiling, shape, periodic)
    _, cell_lat = cell_places(world.space)
    lon = rng.uniform(-540, 540, 600)
    lat = rng.uniform(cell_lat.min(), cell_lat.max(), 600)  # between the first and last rows
    found = [world.cell_at(a, b) for a, b in zip(lon, lat, strict=True)]
    np.testing.assert_array_equal(found, showing(world.space, lon, lat))


@pytest.mark.parametrize(("tiling", "shape", "periodic"), CELL_AT_CASES)
def test_places_beyond_the_first_and_last_rows_of_a_map_belong_to_them(
    rng, tiling, shape, periodic
):
    world = blank_map(tiling, shape, periodic)
    grid = world.space
    assert isinstance(grid, LatticeSpace)
    _, cell_lat = cell_places(grid)
    for pole, row in ((90.0, grid.shape[1] - 1), (-90.0, 0)):
        last = cell_lat.max() if pole > 0 else cell_lat.min()  # the latitude of the row
        lon = rng.uniform(-180, 180, 200)
        lat = np.concatenate(
            [np.full(100, pole), rng.uniform(min(pole, last), max(pole, last), 100)]
        )
        found = [world.cell_at(a, b) for a, b in zip(lon, lat, strict=True)]
        row_cells = np.flatnonzero(grid.coords[:, 1] == row)
        on_row = np.full(200, cell_lat[row_cells[0]])
        np.testing.assert_array_equal(found, showing(grid, lon, on_row, row_cells))


@pytest.mark.parametrize("on", ["sphere", "map-square", "map-hex"])
def test_random_places_spread_evenly_over_the_planet(make_icosphere, rng, on):
    space = {
        "sphere": make_icosphere(4),
        "map-square": map_grid("hypercubic", (False, False)),
        "map-hex": map_grid("hex", (False, False)),
    }[on]
    lon, lat = generate_planet(space, seed=0).random_place(rng, 20_000).T
    assert lon.min() >= -180
    assert lon.max() < 180
    np.testing.assert_allclose(np.mean(np.abs(lat) < 30), 0.5, atol=0.02)  # sin(30 degrees)
    np.testing.assert_allclose(np.mean(lat > 60), (1 - np.sqrt(3) / 2) / 2, atol=0.01)
    np.testing.assert_allclose(np.mean(np.abs(lon) < 90), 0.5, atol=0.02)


@pytest.mark.parametrize("on", ["sphere", "map-square", "map-hex"])
def test_random_places_spread_evenly_within_cells_too(make_icosphere, rng, on):
    # Few, large cells, and the places counted in the cells of a finer icosphere.
    space = {
        "sphere": make_icosphere(1),
        "map-square": LatticeSpace((8, 4), periodic=(True, False)),
        "map-hex": LatticeSpace((8, 5), tiling="hex", periodic=False),
    }[on]
    world = Planet(space, np.zeros(space.n_cells), np.ones(space.n_cells), np.ones(space.n_cells))
    fine = make_icosphere(8)
    cells = fine.locate(sphere_points(*world.random_place(rng, 30_000).T))
    counts = np.bincount(cells, minlength=fine.n_cells)
    assert chisquare(counts, 30_000 * fine.measures / fine.measures.sum()).pvalue > 1e-3


def test_random_places_spread_evenly_over_cells_of_any_size_and_shape(
    make_icosphere, make_uneven_sphere, rng
):
    space = make_uneven_sphere(40, seed=0)
    assert space.measures.max() > 4 * space.measures.min()
    world = Planet(space, np.zeros(40), np.ones(40), np.ones(40))
    points = sphere_points(*world.random_place(rng, 40_000).T)
    expected = 40_000 * space.measures / space.measures.sum()
    assert chisquare(np.bincount(space.locate(points), minlength=40), expected).pvalue > 1e-3
    fine = make_icosphere(8)  # and within the cells
    counts = np.bincount(fine.locate(points), minlength=fine.n_cells)
    assert chisquare(counts, 40_000 * fine.measures / fine.measures.sum()).pvalue > 1e-3


def test_random_places_on_a_planet_spread_over_its_cells_by_area(make_caps_planet, rng):
    world = make_caps_planet(4)  # 162 cells, land on two caps
    for where in (None, world.land):
        allowed = np.ones(world.space.n_cells, bool) if where is None else where
        places = world.random_place(rng, 30_000, where=where)
        cells = world.space.locate(sphere_points(*places.T))
        counts = np.bincount(cells, minlength=world.space.n_cells)
        area = world.space.measures * allowed
        assert chisquare(counts[allowed], 30_000 * area[allowed] / area.sum()).pvalue > 1e-3


@pytest.mark.parametrize("where", ["everywhere", "land", "one cell"])
def test_random_places_lie_in_the_cells_of_where(any_planet, rng, where):
    n_cells = any_planet.space.n_cells
    if where == "everywhere":
        places, allowed = any_planet.random_place(rng, 300), np.ones(n_cells, bool)
    else:
        allowed = any_planet.land.copy() if where == "land" else np.arange(n_cells) == 7
        places = any_planet.random_place(rng, 300, where=allowed)
    assert places.shape == (300, 2)
    assert places.dtype == np.float64
    cells = [any_planet.cell_at(*place) for place in places]
    assert allowed[cells].all()


def test_random_places_repeat_with_the_generator_and_come_one_or_many(any_planet):
    many = any_planet.random_place(np.random.default_rng(4), 5)
    np.testing.assert_array_equal(any_planet.random_place(np.random.default_rng(4), 5), many)
    assert not np.array_equal(any_planet.random_place(np.random.default_rng(5), 5), many)
    one = any_planet.random_place(np.random.default_rng(4))
    assert one.shape == (2,)
    assert isinstance(any_planet.cell_at(*one), int)
    assert any_planet.random_place(np.random.default_rng(4), np.int64(3)).shape == (3, 2)
    assert any_planet.random_place(np.random.default_rng(4), 0).shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rng": 0}, TypeError, "Generator"),
        ({"size": -1}, ValueError, "size"),
        ({"where": np.zeros(3, bool)}, ValueError, "where must have shape"),
        ({"where": "no cells"}, ValueError, "at least one cell"),
        ({"where": "ones"}, TypeError, "where must be boolean"),
    ],
)
def test_random_place_rejects_bad_arguments(any_planet, kwargs, error, match):
    n_cells = any_planet.space.n_cells
    where = {"no cells": np.zeros(n_cells, bool), "ones": np.ones(n_cells)}
    if isinstance(kwargs.get("where"), str):
        kwargs = kwargs | {"where": where[kwargs["where"]]}
    arguments = {"rng": np.random.default_rng(0)} | kwargs
    with pytest.raises(error, match=match):
        any_planet.random_place(**arguments)


def test_random_place_needs_an_integer_size(any_planet, not_an_integer):
    rng = np.random.default_rng(0)
    if not_an_integer is None:  # the default, which asks for one place
        assert any_planet.random_place(rng, not_an_integer).shape == (2,)
        return
    with pytest.raises(TypeError, match="size"):
        any_planet.random_place(rng, not_an_integer)


@pytest.mark.parametrize("on", ["sphere", "map"])
@pytest.mark.parametrize(("lon", "lat"), [(0.0, 90.5), (0.0, -91.0), (np.inf, 0.0)])
def test_cell_at_rejects_places_off_the_planet(make_caps_planet, on, lon, lat):
    world = make_caps_planet() if on == "sphere" else generate_planet(map_grid(), seed=0)
    with pytest.raises(ValueError, match=r"lon|lat"):
        world.cell_at(lon, lat)


def test_cell_at_needs_real_numbers(make_caps_planet, not_a_real):
    with pytest.raises(TypeError, match="longitude"):
        make_caps_planet().cell_at(not_a_real, 0.0)
    with pytest.raises(TypeError, match="latitude"):
        generate_planet(map_grid(), seed=0).cell_at(0.0, not_a_real)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"land": 1.5}, ValueError, "land"),
        ({"land": -0.1}, ValueError, "land"),
        ({"seed": -1}, ValueError, "seed"),
        ({"roughness": 0}, ValueError, "roughness"),
    ],
)
def test_generate_planet_rejects_values_out_of_range(make_icosphere, kwargs, error, match):
    with pytest.raises(error, match=match):
        generate_planet(make_icosphere(1), **kwargs)


def test_generate_planet_needs_an_integer_seed(make_icosphere, not_an_integer):
    with pytest.raises(TypeError, match="seed"):
        generate_planet(make_icosphere(1), seed=not_an_integer)


def test_generate_planet_needs_a_real_land_fraction(make_icosphere, not_a_real):
    with pytest.raises(TypeError, match="land"):
        generate_planet(make_icosphere(1), land=not_a_real)


@pytest.mark.parametrize(
    ("space", "error", "match"),
    [
        (LatticeSpace((4, 4, 4)), ValueError, "2-D lattice, got one with 3 axes"),
        (LatticeSpace((8,)), ValueError, "2-D lattice, got one with 1 axes"),
        (
            LatticeSpace((6, 4), tiling="triangle"),
            ValueError,
            "a planet's map needs a lattice of squares or hexagons, got triangles",
        ),
        ("square mesh", ValueError, "spherical mesh, got one with metric 'euclidean'"),
        ("text", TypeError, "spherical MeshSpace.*or a 2-D LatticeSpace for its map, got str"),
    ],
)
def test_planets_lie_on_spherical_meshes_and_2d_lattices(make_square_mesh, space, error, match):
    space = make_square_mesh() if space == "square mesh" else space
    with pytest.raises(error, match=match):
        generate_planet(space)
    with pytest.raises(error, match=match):
        Planet(space, np.zeros(4), np.zeros(4), np.zeros(4))


VALUES = ("elevation", "temperature", "moisture")


def test_planets_are_immutable_and_survive_pickling(any_planet):
    for name in VALUES:
        assert not getattr(any_planet, name).flags.writeable
    assert copy.copy(any_planet) is any_planet
    assert copy.deepcopy(any_planet) is any_planet
    restored = pickle.loads(pickle.dumps(any_planet))
    assert type(restored) is Planet
    np.testing.assert_array_equal(restored.space.centers, any_planet.space.centers)
    for name in VALUES:
        np.testing.assert_array_equal(getattr(restored, name), getattr(any_planet, name))
        assert not getattr(restored, name).flags.writeable
    np.testing.assert_array_equal(restored.landmasses, any_planet.landmasses)


@pytest.mark.parametrize(
    ("name", "value", "error", "match"),
    [
        ("elevation", 1.5, ValueError, r"elevation must lie in \[-1.0, 1\]"),
        ("elevation", -1.5, ValueError, "must lie in"),
        ("temperature", 1.5, ValueError, r"temperature must lie in \[0.0, 1\]"),
        ("temperature", -0.1, ValueError, "must lie in"),
        ("moisture", 1.5, ValueError, r"moisture must lie in \[0.0, 1\]"),
        ("moisture", -0.1, ValueError, "must lie in"),
        *((name, np.nan, ValueError, "NaN") for name in VALUES),
        *((name, "x", TypeError, "numbers") for name in VALUES),
    ],
)
def test_values_must_be_numbers_in_range(make_icosphere, name, value, error, match):
    values = dict.fromkeys(VALUES, np.zeros(12))
    values[name] = np.full(12, value)
    with pytest.raises(error, match=match):
        Planet(make_icosphere(1), **values)


@pytest.mark.parametrize("name", VALUES)
def test_values_need_one_entry_per_cell(make_icosphere, name):
    values = dict.fromkeys(VALUES, np.zeros(12))
    with pytest.raises(ValueError, match=rf"{name} must have shape \(12,\)"):
        Planet(make_icosphere(1), **(values | {name: np.zeros(13)}))
