"""Regions and borders of masks of cells, on lattices and spheres."""

from collections import deque

import numpy as np
import pytest

from gridzero.core import LatticeSpace, border, regions

LATTICES = {
    "square": ((7, 5), "hypercubic", False),
    "square-wrapped": ((7, 5), "hypercubic", True),
    "hex": ((6, 5), "hex", (True, False)),
    "triangle": ((10, 6), "triangle", (True, False)),
    "cube": ((6, 5, 5), "hypercubic", (True, False, False)),
}


def flood_fill(space, mask, kind):
    """The connected pieces of a mask, found breadth first: lists of cell ids, by lowest cell."""
    graph, seen, pieces = space.adjacency(kind), np.zeros(space.n_cells, bool), []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        seen[start], piece, queue = True, [start], deque([start])
        while queue:
            for other in graph.neighbors(queue.popleft()):
                if mask[other] and not seen[other]:
                    seen[other] = True
                    piece.append(other)
                    queue.append(other)
        pieces.append(piece)
    return pieces


def random_mask(space, seed, density=0.3):
    """Cells drawn at random: sparse enough to fall into several pieces."""
    return np.random.default_rng(seed).random(space.n_cells) < density


@pytest.mark.parametrize("lattice", LATTICES)
@pytest.mark.parametrize("kind", ["facet", "vertex"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_regions_on_a_lattice_are_its_connected_pieces_by_size_then_lowest_cell(
    make_lattice, lattice, kind, seed
):
    shape, tiling, periodic = LATTICES[lattice]
    space = make_lattice(shape, tiling, periodic)
    crowded = space.adjacency(kind).degree.max() > 8  # cubes' corners, or triangles'
    mask = random_mask(space, seed, 0.1 if crowded else 0.3)
    pieces = flood_fill(space, mask, kind)
    assert len(pieces) > 1
    expected = np.full(space.n_cells, -1)
    by_size = sorted(pieces, key=lambda piece: (-len(piece), min(piece)))  # cells of equal area
    for label, piece in enumerate(by_size):
        expected[piece] = label
    labels = regions(space, mask, kind)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("kind", ["facet", "vertex"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_regions_on_a_sphere_are_its_connected_pieces_largest_first(make_icosphere, kind, seed):
    space = make_icosphere(2)
    mask = random_mask(space, seed)
    labels = regions(space, mask, kind)
    assert labels.max() > 0
    pieces = {frozenset(piece) for piece in flood_fill(space, mask, kind)}
    assert {frozenset(np.flatnonzero(labels == k)) for k in range(labels.max() + 1)} == pieces
    np.testing.assert_array_equal(labels[~mask], -1)
    area = np.bincount(labels[mask], space.measures[mask])
    assert (np.diff(area) <= 0).all()


def test_regions_are_ranked_by_area_not_by_how_many_cells_they_have(make_uneven_sphere):
    space = make_uneven_sphere(40, seed=0)
    graph, area = space.adjacency(), space.measures
    big = int(np.argmax(area))
    by_big = [big, *graph.neighbors(big)]
    pairs = area[graph.sources] + area[graph.indices]  # two cells side by side
    pairs[np.isin(graph.sources, by_big) | np.isin(graph.indices, by_big)] = np.inf
    edge = int(np.argmin(pairs))
    assert pairs[edge] < area[big]  # the two cells have less area than the one
    cells = [big, graph.sources[edge], graph.indices[edge]]
    labels = regions(space, np.isin(np.arange(40), cells))
    np.testing.assert_array_equal(labels[cells], [0, 1, 1])


def test_regions_of_equal_area_are_numbered_by_their_lowest_cell():
    space = LatticeSpace((6, 3), periodic=False)
    view = np.full((6, 3), -1)
    view[0, :] = 1  # 3 cells
    view[2:4, :2] = 0  # 4 cells
    view[5, 0], view[5, 2] = 2, 3  # 1 cell each, apart: (5, 0) has the lower id
    expected = view[tuple(space.coords.T)]
    np.testing.assert_array_equal(regions(space, expected >= 0), expected)


def test_cells_that_meet_at_a_corner_touch_only_as_vertices():
    space = LatticeSpace((3, 3), periodic=False)
    diagonal = np.isin(np.arange(9), space.ravel([[0, 0], [1, 1]]))
    assert regions(space, diagonal).max() + 1 == 2
    assert regions(space, diagonal, "vertex").max() + 1 == 1
    plus = np.isin(np.arange(9), space.ravel([[1, 1], [0, 1], [2, 1], [1, 0], [1, 2]]))
    middle = space.ravel([1, 1])
    assert not border(space, plus)[middle]  # its sides all touch the plus
    assert border(space, plus, "vertex")[middle]  # its corners touch the cells outside


@pytest.mark.parametrize("space", ["icosphere", *LATTICES])
@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_the_border_is_the_cells_of_a_mask_that_touch_cells_outside_it(
    make_icosphere, make_lattice, space, kind
):
    space = make_icosphere(2) if space == "icosphere" else make_lattice(*LATTICES[space])
    graph = space.adjacency(kind)
    mask = random_mask(space, 0)
    expected = [
        mask[cell] and not mask[graph.neighbors(cell)].all() for cell in range(space.n_cells)
    ]
    found = border(space, mask, kind)
    assert found.dtype == np.bool_
    np.testing.assert_array_equal(found, expected)
    assert found.any()


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
@pytest.mark.parametrize("wraps", [True, False])
def test_regions_continue_across_the_edges_a_lattice_wraps_around(tiling, wraps):
    space = LatticeSpace((12, 8), tiling=tiling, periodic=(wraps, True))
    column = space.coords[:, 0]
    strips = (column < 3) | (column >= 9)  # beside the edge x = 0, on both sides
    assert regions(space, strips).max() + 1 == (1 if wraps else 2)
    # A strip along the edge borders the cells beyond it only where there are some.
    np.testing.assert_array_equal(border(space, column < 3)[column == 0], wraps)


def test_an_empty_mask_has_no_regions_and_a_full_one_has_no_border(make_icosphere, make_lattice):
    for space in (make_icosphere(2), make_lattice((5, 4), "hex", False)):
        nothing, everything = np.zeros(space.n_cells, bool), np.ones(space.n_cells, bool)
        np.testing.assert_array_equal(regions(space, nothing), -1)
        np.testing.assert_array_equal(border(space, nothing), False)
        np.testing.assert_array_equal(regions(space, everything), 0)  # one piece
        np.testing.assert_array_equal(border(space, everything), False)


@pytest.mark.parametrize("function", [regions, border])
def test_regions_and_borders_need_one_bool_per_cell_and_a_kind_of_touch(make_lattice, function):
    space = make_lattice()
    with pytest.raises(TypeError, match="mask must be boolean"):
        function(space, np.ones(space.n_cells))
    with pytest.raises(ValueError, match="mask must have shape"):
        function(space, np.ones(space.n_cells + 1, bool))
    with pytest.raises(ValueError, match="adjacency kind"):
        function(space, np.ones(space.n_cells, bool), "diagonal")
