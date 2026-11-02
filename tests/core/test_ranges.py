"""Distances between cells, and the cells within a range, on lattices and meshes."""

import itertools

import numpy as np
import pytest

from gridzero.core import LatticeSpace, Tiling, distance, icosphere, within

RADII = [0.0, 1.0, 1.5, 2.0, 2.7, 3.1]

WRAPPING: dict[str, tuple[tuple[int, ...], Tiling, bool | tuple[bool, ...]]] = {
    "hex-both": ((7, 3), "hex", True),
    "hex-long": ((30, 4), "hex", True),
    "hex-tall": ((4, 30), "hex", True),
    "hex-thin": ((3, 56), "hex", True),  # slanted periods that need reducing
    "hex-flat": ((56, 3), "hex", True),
    "hex-y": ((5, 9), "hex", (False, True)),
    "triangle": ((6, 8), "triangle", True),
    "line": ((5,), "hypercubic", True),
    "cube": ((3, 4, 5), "hypercubic", (True, False, True)),
}


def nearest_copies(space, a, b, turns=30):
    """Distances to the nearest copy, trying every copy up to ``turns`` periods away."""
    wrapped = [k for k in range(space.dim) if space.periodic[k]]
    periods = space.basis[:, wrapped] * np.array(space.shape)[wrapped]
    best = np.full(len(a), np.inf)
    for n in itertools.product(range(-turns, turns + 1), repeat=len(wrapped)):
        best = np.minimum(best, np.linalg.norm(b - a - periods @ np.array(n), axis=1))
    return best


def all_within(space, cell, radius):
    """The cells within ``radius``, by measuring the distance to every cell."""
    return np.flatnonzero(distance(space, cell, np.arange(space.n_cells)) <= radius * (1 + 1e-9))


@pytest.mark.parametrize("radius", RADII)
def test_within_is_the_cell_and_its_neighbors_in_the_radius_graph(any_space, radius):
    cells = (0, any_space.n_cells // 2, any_space.n_cells - 1)
    fresh = [within(any_space, cell, radius) for cell in cells]  # before any graph is cached
    try:
        graph = any_space.radius_graph(radius)
    except ValueError:  # too far for a short periodic axis: measure every cell instead
        graph = None
    for cell, before in zip(cells, fresh, strict=True):
        found = within(any_space, cell, radius)  # from the cached graph, if there is one
        assert found.dtype == before.dtype == np.int64
        if graph is None:
            expected = all_within(any_space, cell, radius)
        else:
            expected = np.union1d(graph.neighbors(cell), [cell])
        np.testing.assert_array_equal(before, expected)
        np.testing.assert_array_equal(found, expected)


def test_within_from_a_cached_graph_is_a_copy_the_caller_may_change():
    space = LatticeSpace((6, 6), tiling="hex", periodic=False)
    graph = space.radius_graph(1.0)
    found = within(space, 14, 1.0)
    found[:] = -1
    np.testing.assert_array_equal(within(space, 14, 1.0), np.union1d(graph.neighbors(14), [14]))


@pytest.mark.parametrize("lattice", WRAPPING)
@pytest.mark.parametrize("radius", [0.5, 1.0, 2.3, 4.0, 9.0, 50.0])
def test_within_wraps_around_short_axes_and_counts_each_cell_once(lattice, radius):
    shape, tiling, periodic = WRAPPING[lattice]
    space = LatticeSpace(shape, tiling, periodic)
    for cell in (0, space.n_cells - 1):
        np.testing.assert_array_equal(within(space, cell, radius), all_within(space, cell, radius))


def test_within_answers_each_radius_on_one_lattice_whatever_was_asked_before():
    space = LatticeSpace((20, 20), tiling="hex")
    for radius in (2.0, 5.5, 0.0, 3.1, 2.0, 5.5):
        np.testing.assert_array_equal(within(space, 7, radius), all_within(space, 7, radius))
    graph = space.radius_graph(3.1)
    np.testing.assert_array_equal(np.union1d(graph.neighbors(7), [7]), all_within(space, 7, 3.1))


def test_within_on_a_large_lattice_reaches_only_nearby():
    space = LatticeSpace((60, 60), tiling="triangle")
    cell = space.ravel([1, 1])  # near the edges, which wrap
    found = within(space, cell, 3.1)
    assert found.size == 25  # 24 neighbors, and the cell
    np.testing.assert_array_equal(found, all_within(space, cell, 3.1))


def test_within_a_radius_far_past_a_small_torus_is_every_cell():
    # The offsets within such a radius would outnumber the cells many times over.
    space = LatticeSpace((5, 6), tiling="hex")
    np.testing.assert_array_equal(within(space, 3, 1e9), np.arange(space.n_cells))


def test_within_on_a_fixed_lattice_stops_at_the_edge():
    space = LatticeSpace((5, 5), periodic=False)
    np.testing.assert_array_equal(space.unravel(within(space, 0, 1.0)), [[0, 0], [0, 1], [1, 0]])


def test_within_on_a_planar_mesh_is_by_straight_lines(make_square_mesh):
    mesh = make_square_mesh(4, 5)
    for radius in (0.0, 1.0, 1.5, 3.0):
        np.testing.assert_array_equal(within(mesh, 7, radius), all_within(mesh, 7, radius))


def test_distance_is_the_length_of_the_edge_between_neighbors(any_space):
    graph = any_space.adjacency("vertex")
    np.testing.assert_allclose(
        distance(any_space, graph.sources, graph.indices), graph.edge_data["distance"], rtol=1e-12
    )


def test_distance_is_the_same_both_ways_round(any_space):
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, any_space.n_cells, (2, 200))
    assert np.array_equal(distance(any_space, a, b), distance(any_space, b, a))
    np.testing.assert_array_equal(distance(any_space, a, a), 0.0)


def test_distance_broadcasts_ids():
    space = LatticeSpace((5, 5), periodic=False)
    found = distance(space, 0, [[1, 5], [6, 24]])
    np.testing.assert_allclose(found, [[1, 1], [2**0.5, 32**0.5]])
    assert distance(space, 0, 1).shape == ()


def test_distance_on_a_sphere_is_the_great_circle_arc():
    space = icosphere(3, radius=2.0)
    far = int(np.argmin(space.centers @ space.centers[0]))
    assert distance(space, 0, far) == pytest.approx(2 * np.pi, rel=1e-12)


@pytest.mark.parametrize("lattice", WRAPPING)
def test_lattice_distances_are_to_the_nearest_copy(lattice):
    shape, tiling, periodic = WRAPPING[lattice]
    space = LatticeSpace(shape, tiling, periodic)
    rng = np.random.default_rng(1)
    a, b = rng.uniform(-12, 12, (2, 2000, space.dim))
    np.testing.assert_allclose(space.distance(a, b), nearest_copies(space, a, b), atol=1e-12)


def test_lattice_distances_without_wrapping_are_straight_lines():
    space = LatticeSpace((5, 5), tiling="hex", periodic=False)
    a, b = np.array([0.0, 0.0]), np.array([[30.0, 40.0], [-3.0, 4.0]])
    np.testing.assert_array_equal(space.distance(a, b), [50.0, 5.0])


def test_lattice_distances_check_their_points():
    space = LatticeSpace((5, 5))
    with pytest.raises(ValueError, match="shape"):
        space.distance([0.0, 0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        space.distance(0.0, [1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        space.distance([np.nan, 0.0], [1.0, 1.0])
    with pytest.raises(TypeError):
        space.distance([True, False], [1.0, 1.0])


def test_bad_cells_raise(not_cell_ids, not_an_integer):
    space = LatticeSpace((5, 5))
    values, error = not_cell_ids
    with pytest.raises(error):
        distance(space, values, 0)
    with pytest.raises(IndexError):
        distance(space, 0, [3, 25])
    with pytest.raises(IndexError):
        distance(space, -1, 0)
    with pytest.raises(TypeError):
        within(space, not_an_integer, 1.0)
    with pytest.raises(IndexError):
        within(space, 25, 1.0)


def test_bad_radii_raise(not_a_real):
    space = LatticeSpace((5, 5))
    with pytest.raises(TypeError):
        within(space, 0, not_a_real)
    for bad in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="radius"):
            within(space, 0, bad)


def test_other_spaces_raise():
    with pytest.raises(TypeError, match="LatticeSpace or a MeshSpace"):
        distance(object(), 0, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="LatticeSpace or a MeshSpace"):
        within(object(), 0, 1.0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("lattice", WRAPPING)
def test_a_displacement_reaches_a_copy_of_its_end_and_is_as_long_as_the_distance(lattice):
    shape, tiling, periodic = WRAPPING[lattice]
    space = LatticeSpace(shape, tiling, periodic)
    rng = np.random.default_rng(2)
    a, b = rng.uniform(-12, 12, (2, 2000, space.dim))
    step = space.displacement(a, b)
    np.testing.assert_allclose(np.linalg.norm(step, axis=-1), space.distance(a, b), atol=1e-12)
    # a + step is b moved by whole periods: the same point of the lattice, which locate shows.
    np.testing.assert_array_equal(space.locate(a + step), space.locate(b))


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_the_displacement_between_neighbors_across_a_wrapped_edge_is_the_short_step(tiling):
    space = LatticeSpace((8, 6), tiling, periodic=True)
    graph = space.adjacency("vertex")
    centers = space.centers
    step = space.displacement(centers[graph.sources], centers[graph.indices])
    lengths = np.linalg.norm(step, axis=-1)
    assert lengths.max() < 2.5  # corner neighbors are at most a couple of units apart
    np.testing.assert_allclose(
        lengths, space.distance(centers[graph.sources], centers[graph.indices])
    )


def test_displacements_without_wrapping_are_differences():
    space = LatticeSpace((5, 5), tiling="hex", periodic=False)
    a, b = np.array([1.0, 2.0]), np.array([[30.0, 40.0], [-3.0, 4.0]])
    np.testing.assert_array_equal(space.displacement(a, b), b - a)


def test_displacements_check_their_points():
    space = LatticeSpace((5, 5))
    with pytest.raises(ValueError, match="shape"):
        space.displacement([0.0, 0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        space.displacement([np.nan, 0.0], [1.0, 1.0])
