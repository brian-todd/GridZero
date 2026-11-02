"""Random points in the cells of lattices and spherical meshes, spread evenly by area."""

import gc
import weakref

import numpy as np
import pytest
from scipy.stats import chisquare

from gridzero.core import LatticeSpace, State, icosphere, random_points

LATTICE_CASES = [
    (tiling, periodic)
    for tiling in ("hypercubic", "hex", "triangle")
    for periodic in ((True, True), (True, False), (False, True), (False, False))
]

SHAPES = {"hypercubic": (5, 4), "hex": (5, 4), "triangle": (6, 4)}
"""Small 2-D grids of each tiling. Triangles wrap only along even axes."""

SPACES = {
    "line": lambda: LatticeSpace((8,), periodic=False),
    "square": lambda: LatticeSpace((5, 4), periodic=False),
    "hex-wrapped": lambda: LatticeSpace((5, 4), tiling="hex", periodic=(True, False)),
    "triangle-wrapped": lambda: LatticeSpace((6, 4), tiling="triangle", periodic=(False, True)),
    "cube": lambda: LatticeSpace((4, 3, 3), periodic=(True, False, True)),
    "icosphere": lambda: icosphere(2),
}


@pytest.fixture(params=sorted(SPACES))
def space(request):
    return SPACES[request.param]()


def width(space):
    """How many coordinates a point of the space has."""
    return 3 if not isinstance(space, LatticeSpace) else space.dim


@pytest.mark.parametrize(("tiling", "periodic"), LATTICE_CASES)
def test_random_points_spread_evenly_over_the_cells_of_a_lattice(rng, tiling, periodic):
    # Cells along an edge that doesn't wrap are as likely as the others: hexagons and the
    # triangles at the ends of rows stick out.
    grid = LatticeSpace(SHAPES[tiling], tiling=tiling, periodic=periodic)
    counts = np.bincount(grid.locate(random_points(grid, rng, 20_000)), minlength=grid.n_cells)
    assert chisquare(counts).pvalue > 1e-3


@pytest.mark.parametrize("periodic", [True, (True, False, False), False])
def test_random_points_spread_evenly_over_the_cells_of_a_cube(rng, periodic):
    grid = LatticeSpace((3, 3, 4), periodic=periodic)
    counts = np.bincount(grid.locate(random_points(grid, rng, 20_000)), minlength=36)
    assert chisquare(counts).pvalue > 1e-3


@pytest.mark.parametrize(
    ("tiling", "area"),
    [("hypercubic", 1.0), ("hex", np.sqrt(3) / 2), ("triangle", 3 * np.sqrt(3) / 4)],
)
def test_random_points_spread_evenly_within_each_cell(rng, tiling, area):
    grid = LatticeSpace(SHAPES[tiling], tiling=tiling, periodic=False)
    points = random_points(grid, rng, 40_000)
    off_center = np.linalg.norm(points - grid.centers[grid.locate(points)], axis=1)
    for radius in (0.25, 0.45):  # disks inside every cell: a triangle's inradius is 1/2
        np.testing.assert_allclose(
            np.mean(off_center < radius), np.pi * radius**2 / area, atol=0.01
        )


@pytest.mark.parametrize(
    ("tiling", "sides", "radius"),
    [("hypercubic", 4, 2**-0.5), ("hex", 6, 3**-0.5), ("triangle", 3, 1.0)],
)
def test_random_points_reach_every_part_of_their_cells(rng, tiling, sides, radius):
    # Over a regular polygon with circumradius R, points spread evenly have their mean at the
    # center and a mean squared distance from it of R**2 * (1 + 2 cos(pi / sides)**2) / 6.
    # A box that cut off a triangle's far corner would miss 3% of it, which moves the mean by
    # 0.024 and the spread by 2.6%. Over 30 seeds, the widest misses were 0.0054 and 0.74%.
    grid = LatticeSpace(SHAPES[tiling], tiling=tiling, periodic=False)
    points = random_points(grid, rng, 80_000)
    cells = grid.locate(points)
    offsets = points - grid.centers[cells]
    pointing = grid.coords[cells].sum(axis=1) % 2 if tiling == "triangle" else np.zeros(len(cells))
    for way in np.unique(pointing):  # triangles pointing up, then down
        np.testing.assert_allclose(offsets[pointing == way].mean(axis=0), 0.0, atol=0.01)
    spread = radius**2 * (1 + 2 * np.cos(np.pi / sides) ** 2) / 6
    np.testing.assert_allclose(np.mean(np.sum(offsets**2, axis=1)), spread, rtol=0.015)


@pytest.mark.parametrize("seed", [0, 1])
def test_random_points_on_a_sphere_spread_over_its_cells_by_area(make_uneven_sphere, rng, seed):
    sphere = make_uneven_sphere(40, seed=seed)  # the largest cell has 4 times the smallest's area
    points = random_points(sphere, rng, 40_000)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, rtol=1e-14)
    counts = np.bincount(sphere.locate(points), minlength=40)
    expected = 40_000 * sphere.measures / sphere.measures.sum()
    assert chisquare(counts, expected).pvalue > 1e-3


def test_random_points_lie_in_the_cells_of_where_and_spread_over_them_by_area(space, rng):
    n_cells = space.n_cells
    where = np.random.default_rng(3).random(n_cells) < 0.4
    counts = np.bincount(
        space.locate(random_points(space, rng, 20_000, where=where)), None, n_cells
    )
    assert counts[~where].sum() == 0
    area = space.measures[where]
    assert chisquare(counts[where], 20_000 * area / area.sum()).pvalue > 1e-3
    one = np.arange(n_cells) == 7
    np.testing.assert_array_equal(space.locate(random_points(space, rng, 50, where=one)), 7)


@pytest.mark.parametrize(
    ("shape", "tiling", "periodic"),
    [
        *[
            (SHAPES[tiling], tiling, periodic)
            for tiling, periodic in LATTICE_CASES
            if periodic != (False,) * 2
        ],
        ((4, 3, 3), "hypercubic", (True, False, True)),
    ],
)
def test_random_points_lie_within_one_period_along_each_axis_that_wraps(
    rng, shape, tiling, periodic
):
    grid = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    points = random_points(grid, rng, 5_000)
    for axis in np.flatnonzero(grid.periodic):
        along = points[:, axis] / grid.basis[axis, axis]  # in steps, from the first cell
        assert ((along >= -0.5) & (along < grid.shape[axis] - 0.5)).all()
    # A cell beside the edge where an axis wraps gets the points of its whole area, in the
    # period: none of them is moved into another cell or lost.
    counts = np.bincount(grid.locate(points), minlength=grid.n_cells)
    assert chisquare(counts).pvalue > 1e-3


def test_random_points_repeat_with_the_generator_and_come_one_or_many(space):
    many = random_points(space, np.random.default_rng(4), 5)
    np.testing.assert_array_equal(random_points(space, np.random.default_rng(4), 5), many)
    assert not np.array_equal(random_points(space, np.random.default_rng(5), 5), many)
    assert many.shape == (5, width(space))
    assert many.dtype == np.float64
    one = random_points(space, np.random.default_rng(4))
    assert one.shape == (width(space),)
    assert random_points(space, np.random.default_rng(4), np.int64(3)).shape == (3, width(space))
    assert random_points(space, np.random.default_rng(4), 0).shape == (0, width(space))


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
def test_random_points_reject_bad_arguments(space, kwargs, error, match):
    where = {"no cells": np.zeros(space.n_cells, bool), "ones": np.ones(space.n_cells)}
    if isinstance(kwargs.get("where"), str):
        kwargs = kwargs | {"where": where[kwargs["where"]]}
    arguments = {"rng": np.random.default_rng(0)} | kwargs
    with pytest.raises(error, match=match):
        random_points(space, **arguments)


def test_random_points_need_an_integer_size(not_an_integer):
    grid, rng = LatticeSpace((4, 4)), np.random.default_rng(0)
    if not_an_integer is None:  # the default, which asks for one point
        assert random_points(grid, rng, not_an_integer).shape == (2,)
        return
    with pytest.raises(TypeError, match="size"):
        random_points(grid, rng, not_an_integer)


def test_random_points_need_a_lattice_or_a_spherical_mesh(make_square_mesh):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="got a mesh with metric 'euclidean'"):
        random_points(make_square_mesh(), rng)
    with pytest.raises(TypeError, match="LatticeSpace or a spherical MeshSpace, got State"):
        random_points(State(LatticeSpace((4, 4))), rng)  # ty: ignore[invalid-argument-type]


def test_random_points_keep_no_mesh_alive():
    sphere = icosphere(1)
    random_points(sphere, np.random.default_rng(0))  # makes and keeps the caps of its cells
    gone = weakref.ref(sphere)
    del sphere
    gc.collect()
    assert gone() is None
