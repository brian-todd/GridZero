import itertools
import tracemalloc

import numpy as np
import pytest

from gridzero.core import CellGraph, LatticeSpace, MeshSpace


def _edges_with_attributes(graph: CellGraph) -> dict[tuple[int, int], tuple[float, float]]:
    columns = (graph.edge_data["distance"].tolist(), graph.edge_data["boundary"].tolist())
    edges = zip(graph.sources.tolist(), graph.indices.tolist(), strict=True)
    return dict(zip(edges, zip(*columns, strict=True), strict=True))


def _reference_adjacency(shape, j) -> dict[tuple[int, int], tuple[float, float]]:
    """Adjacency of unit n-cubes from first principles, by brute force over all pairs.

    Two cells are j-adjacent when the affine span of their shared corners has dimension at
    least j. Shares no code with `LatticeSpace.offsets`.
    """
    dim = len(shape)
    corners = np.array(list(itertools.product((0, 1), repeat=dim)))
    cells = list(itertools.product(*map(range, shape)))  # row-major, like LatticeSpace ids
    corner_sets = [{tuple(v) for v in cell + corners} for cell in cells]
    edges = {}
    for a, b in itertools.permutations(range(len(cells)), 2):
        shared = np.array(sorted(corner_sets[a] & corner_sets[b]))
        if shared.size == 0:
            continue
        face_dim = np.linalg.matrix_rank(shared - shared[0])
        if face_dim >= j:
            distance = float(np.linalg.norm(np.subtract(cells[a], cells[b])))
            edges[(a, b)] = (distance, 1.0 if face_dim == dim - 1 else 0.0)
    return edges


def _lattice_as_mesh(space: LatticeSpace) -> MeshSpace:
    """Build a polygon mesh of a 2-D non-periodic lattice's cells, with exact shared vertices.

    Each corner gets an integer key in units of ``1 / scale`` lattice steps, so corners shared
    by neighboring cells get identical keys without any rounding. A square's corners sit at
    half-steps. A hexagon's corners are the centroids of three mutually adjacent cells. A
    triangle pointing up has its bottom corners a column to either side and half a row down,
    and its top corner half a row up. One pointing down is the same, upside down.
    """
    if space.tiling == "hex":
        ccw = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]])  # by angle
        corner_keys, scale = ccw + np.roll(ccw, -1, axis=0), 3
    elif space.tiling == "triangle":
        up = np.array([[-2, -1], [2, -1], [0, 1]])  # counterclockwise
        down = -up
        pointing_up = (space.coords.sum(axis=1) % 2 == 0)[:, None, None]
        corner_keys, scale = np.where(pointing_up, up, down), 2
    else:
        corner_keys, scale = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]), 2
    keys = scale * space.coords[:, None, :] + corner_keys
    unique, polygon_indices = np.unique(keys.reshape(-1, 2), axis=0, return_inverse=True)
    sides = corner_keys.shape[-2]
    return MeshSpace(
        centers=space.centers,
        measures=space.measures,
        vertices=unique @ space.basis.T / scale,
        polygon_indptr=np.arange(0, sides * space.n_cells + 1, sides),
        polygon_indices=polygon_indices.ravel(),
    )


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_the_integer_frame_of_a_tiling_holds_its_centers_and_corners_in_order(tiling):
    space = LatticeSpace((6, 4), tiling=tiling, periodic=False)
    frame = space._tiling.frame
    assert frame is not None
    assert np.linalg.det(frame.to_plane) > 0  # sides of lines stay the same
    types = space.coords.sum(axis=1) % 2 if tiling == "triangle" else np.zeros(24, int)
    centers = space.coords @ frame.scale.T + frame.center[types]
    np.testing.assert_allclose(centers @ frame.to_plane.T, space.centers, atol=1e-12)
    mesh = _lattice_as_mesh(space)
    for cell in range(space.n_cells):
        corners = (centers[cell] + frame.corners[types[cell]]) @ frame.to_plane.T
        x, y = corners.T
        area = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2
        assert area == pytest.approx(space.measures[cell])  # counterclockwise
        expected = mesh.polygon(cell)  # from some corner, either way round
        x, y = expected.T
        if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() < 0:
            expected = expected[::-1]
        start = np.argmin(np.linalg.norm(expected - corners[0], axis=1))
        np.testing.assert_allclose(corners, np.roll(expected, -start, axis=0), atol=1e-12)


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_the_metric_of_an_integer_frame_is_a_whole_multiple_of_the_planes(tiling):
    frame = LatticeSpace((6, 4), tiling=tiling)._tiling.frame
    assert frame is not None
    plane = frame.to_plane.T @ frame.to_plane
    ratio = frame.metric[0, 0] / plane[0, 0]
    assert ratio == pytest.approx(round(ratio))
    np.testing.assert_allclose(frame.metric, ratio * plane, atol=1e-12)


def test_only_2d_tilings_have_an_integer_frame():
    assert LatticeSpace((3, 3, 3))._tiling.frame is None
    assert LatticeSpace((5,))._tiling.frame is None


@pytest.mark.parametrize(
    ("shape", "tiling", "kind", "degree"),
    [
        ((5, 5), "hypercubic", "facet", 4),
        ((5, 5), "hypercubic", "vertex", 8),
        ((3, 4, 5), "hypercubic", "facet", 6),
        ((3, 4, 5), "hypercubic", 1, 18),
        ((3, 4, 5), "hypercubic", "vertex", 26),
        ((3, 3, 3, 3), "hypercubic", "facet", 8),
        ((3, 3, 3, 3), "hypercubic", "vertex", 80),
        ((4, 5), "hex", "facet", 6),
        ((4, 5), "hex", "vertex", 6),
        ((6, 4), "triangle", "facet", 3),
        ((6, 4), "triangle", "vertex", 12),
    ],
)
def test_periodic_lattices_have_known_coordination_numbers(
    make_lattice, shape, tiling, kind, degree
):
    graph = make_lattice(shape, tiling).adjacency(kind)
    np.testing.assert_array_equal(graph.degree, degree)


@pytest.mark.parametrize(
    ("shape", "j"),
    [(shape, j) for shape in [(3, 4), (3, 3, 3), (3, 3, 3, 3)] for j in range(len(shape))],
)
def test_hypercubic_adjacency_matches_shared_faces_of_the_cells(make_lattice, shape, j):
    graph = make_lattice(shape, periodic=False).adjacency(j)
    reference = _reference_adjacency(shape, j)
    edges = _edges_with_attributes(graph)
    assert edges.keys() == reference.keys()
    for edge, (distance, boundary) in reference.items():
        assert edges[edge] == pytest.approx((distance, boundary), rel=1e-12)


def _reference_radius_graph(space: LatticeSpace, radius: float) -> dict[tuple[int, int], float]:
    """Find pairs within ``radius`` by brute force over all pairs and their periodic images."""
    periods = (space.basis * np.array(space.shape)).T  # one row per axis
    difference = space.centers[None, :, :] - space.centers[:, None, :]  # (n, n, dim): j - i
    images = itertools.product(*[(-1, 0, 1) if wraps else (0,) for wraps in space.periodic])
    distance = np.full(difference.shape[:2], np.inf)
    for turns in images:
        image = np.linalg.norm(difference + np.array(turns) @ periods, axis=-1)
        distance = np.minimum(distance, image)
    within = (distance > 0) & (distance <= radius)
    return {
        (int(i), int(j)): float(distance[i, j]) for i, j in zip(*np.nonzero(within), strict=True)
    }


@pytest.mark.parametrize(
    ("shape", "tiling", "periodic", "radius"),
    [
        ((5, 5), "hypercubic", True, 2.3),
        ((4, 6), "hypercubic", False, 2.5),
        ((5, 4), "hypercubic", (True, False), 2.2),
        ((5, 5, 5), "hypercubic", True, 1.8),
        ((5, 5), "hex", True, 2.0),
        ((4, 6), "hex", False, 2.7),
        ((6, 4), "triangle", True, 2.2),
        ((8, 4), "triangle", (True, False), 2.7),
        ((5, 6), "triangle", False, 3.1),
    ],
)
def test_radius_graph_matches_brute_force_distances(shape, tiling, periodic, radius):
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    graph = space.radius_graph(radius)
    expected = _reference_radius_graph(space, radius)
    actual = dict(
        zip(
            zip(graph.sources.tolist(), graph.indices.tolist(), strict=True),
            graph.edge_data["distance"].tolist(),
            strict=True,
        )
    )
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(
        [actual[pair] for pair in expected], list(expected.values()), rtol=1e-12
    )


@pytest.mark.parametrize(("shrink", "count"), [(1e-12, 8), (1e-6, 4)])
def test_radius_graph_forgives_rounding_but_not_real_shortfalls(make_lattice, shrink, count):
    # Diagonal neighbors are sqrt(2) away: a radius short by rounding error still includes them.
    graph = make_lattice((5, 5)).radius_graph(2**0.5 * (1 - shrink))
    np.testing.assert_array_equal(graph.degree, np.full(25, count))


def test_radius_graph_rejects_radii_that_wrap_onto_themselves(make_lattice):
    space = make_lattice((4, 5))
    for _ in range(2):  # a refusal is not kept as a graph
        with pytest.raises(ValueError, match="periodic axis 0, which has length 4"):
            space.radius_graph(2.0)


def test_radius_graph_allows_long_radii_across_fixed_axes(make_lattice):
    graph = make_lattice((4, 5), periodic=False).radius_graph(10.0)
    np.testing.assert_array_equal(graph.degree, np.full(20, 19))


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_a_radius_beyond_a_fixed_grid_links_every_pair_without_searching_beyond_it(tiling):
    # The search box is clipped to the grid: unclipped, a radius of 1e6 would need a box of
    # (2e6 + 1)**2 offsets, far beyond memory.
    space = LatticeSpace((4, 5), tiling=tiling, periodic=False)
    np.testing.assert_array_equal(space.radius_graph(1e6).degree, np.full(20, 19))
    cube = LatticeSpace((4, 4, 4), periodic=False)
    np.testing.assert_array_equal(cube.radius_graph(1e6).degree, np.full(64, 63))


@pytest.mark.parametrize(
    "space",
    [
        LatticeSpace((3, 3, 3, 3)),
        LatticeSpace((5, 5), "hex"),
        LatticeSpace((6, 4), "triangle"),
        LatticeSpace((50, 3), periodic=(False, True)),
    ],
    ids=["tesseract", "hex", "triangle", "mixed"],
)
def test_a_radius_that_wraps_a_periodic_axis_fails_before_building_offsets(space):
    # Building first would need (2e6 + 1)**dim offsets: a MemoryError instead of this message.
    with pytest.raises(ValueError, match=r"periodic axes need length >= 2 \* reach \+ 1"):
        space.radius_graph(1e6)


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_lattice_adjacency_matches_a_mesh_of_its_cells(make_lattice, tiling, kind):
    space = make_lattice((4, 5), tiling, periodic=False)
    expected = _lattice_as_mesh(space).adjacency(kind)
    graph = space.adjacency(kind)
    np.testing.assert_array_equal(graph.indptr, expected.indptr)
    np.testing.assert_array_equal(graph.indices, expected.indices)
    for name in ("distance", "boundary"):
        np.testing.assert_allclose(graph.edge_data[name], expected.edge_data[name], rtol=1e-12)


def test_fixed_boundaries_drop_neighbors_beyond_the_edge(make_lattice):
    space = make_lattice((3, 4), periodic=False)
    facet = space.grid_view(space.adjacency("facet").degree)
    vertex = space.grid_view(space.adjacency("vertex").degree)
    np.testing.assert_array_equal(facet, [[2, 3, 3, 2], [3, 4, 4, 3], [2, 3, 3, 2]])
    np.testing.assert_array_equal(vertex, [[3, 5, 5, 3], [5, 8, 8, 5], [3, 5, 5, 3]])


def test_mixed_periodicity_wraps_only_periodic_axes(make_lattice):
    space = make_lattice((3, 4), periodic=(True, False))
    degree = space.grid_view(space.adjacency("facet").degree)
    np.testing.assert_array_equal(degree, [[3, 4, 4, 3]] * 3)


def test_periodic_neighbors_wrap_around(make_lattice):
    space = make_lattice((5, 5))
    corner = space.ravel([0, 0])
    expected = space.ravel([[4, 0], [1, 0], [0, 4], [0, 1]])
    np.testing.assert_array_equal(space.adjacency("facet").neighbors(corner), np.sort(expected))


def test_wrapped_edges_keep_their_true_length(make_lattice):
    graph = make_lattice((5, 5)).adjacency("vertex")
    distances = np.unique(np.round(graph.edge_data["distance"], 12))
    np.testing.assert_allclose(distances, [1.0, np.sqrt(2.0)])


def test_only_facet_neighbors_share_a_boundary(make_lattice):
    graph = make_lattice((4, 4)).adjacency("vertex")
    shares_facet = np.isclose(graph.edge_data["distance"], 1.0)
    np.testing.assert_array_equal(graph.edge_data["boundary"], np.where(shares_facet, 1.0, 0.0))


def test_hex_geometry_has_unit_spacing_and_regular_cells(make_lattice):
    space = make_lattice((4, 5), "hex")
    graph = space.adjacency()
    np.testing.assert_allclose(graph.edge_data["distance"], 1.0)
    np.testing.assert_allclose(graph.edge_data["boundary"], 1 / np.sqrt(3))
    np.testing.assert_allclose(space.measures, np.sqrt(3) / 2)
    np.testing.assert_allclose(space.centers[space.ravel([0, 1])], [0.5, np.sqrt(3) / 2])


def test_hex_perimeter_from_shared_boundaries_matches_a_regular_hexagon(make_lattice):
    space = make_lattice((4, 5), "hex")
    graph = space.adjacency()
    perimeter = np.bincount(graph.sources, weights=graph.edge_data["boundary"])
    np.testing.assert_allclose(perimeter, 6 / np.sqrt(3))


def test_triangle_geometry_has_unit_spacing_and_regular_cells(make_lattice):
    space = make_lattice((6, 4), "triangle")
    graph = space.adjacency()
    np.testing.assert_allclose(graph.edge_data["distance"], 1.0)
    np.testing.assert_allclose(graph.edge_data["boundary"], np.sqrt(3))  # each edge
    np.testing.assert_allclose(space.measures, 3 * np.sqrt(3) / 4)
    perimeter = np.bincount(graph.sources, weights=graph.edge_data["boundary"])
    np.testing.assert_allclose(perimeter, 3 * np.sqrt(3))
    # Up, down, then up again along row 0, and down, then up along column 0.
    expected = {(0, 0): (0, -0.25), (1, 0): (0.5, 0.25), (2, 0): (1, -0.25), (0, 1): (0, 1.75)}
    for coords, (columns, y) in expected.items():
        np.testing.assert_allclose(space.centers[space.ravel(coords)], [columns * 3**0.5, y])


def test_triangles_meet_their_neighbors_in_edges_and_corners(make_lattice):
    space = make_lattice((6, 4), "triangle")
    graph = space.adjacency("vertex")
    distance = np.round(graph.edge_data["distance"], 12)
    shares_edge = graph.edge_data["boundary"] > 0
    # Per triangle: 3 across its edges, 6 at its corners beside those (sqrt(3) away), and the 3
    # opposite its corners (2 away).
    for value, count, edge in [(1.0, 3, True), (np.round(3**0.5, 12), 6, False), (2.0, 3, False)]:
        at = distance == value
        np.testing.assert_array_equal(np.bincount(graph.sources[at], minlength=24), count)
        assert (shares_edge[at] == edge).all()


def test_triangles_pointing_down_have_the_opposite_offsets(make_lattice):
    space = make_lattice((6, 4), "triangle")
    up, down = space.ravel([2, 0]), space.ravel([2, 1])  # (i + j) even points up
    np.testing.assert_array_equal(space.offsets("facet", up), [[-1, 0], [1, 0], [0, -1]])
    for kind in ("facet", "vertex"):
        np.testing.assert_array_equal(space.offsets(kind, down), -space.offsets(kind, up))
        for cell in (up, down):
            expected = space.coords[space.adjacency(kind).neighbors(cell)] - space.coords[cell]
            found = {tuple(row) for row in space.offsets(kind, cell)}
            assert found == {tuple(row) for row in (expected + 2) % [6, 4] - 2}  # unwrapped
    assert len(space.offsets("vertex", up)) == 12


def test_offsets_need_a_cell_id(make_lattice, not_an_integer):
    space = make_lattice((6, 4), "triangle")
    with pytest.raises(IndexError, match="cell id 24"):
        space.offsets("facet", 24)
    if not_an_integer is not None:
        with pytest.raises(TypeError, match="cell id"):
            space.offsets("facet", not_an_integer)


@pytest.mark.parametrize("kind", ["facet", "vertex", 1])
def test_hypercubic_offsets_respect_the_shared_face_dimension(make_lattice, kind):
    space = make_lattice((3, 3, 3))
    offsets = space.offsets(kind)
    max_nonzero = {"facet": 1, "vertex": 3, 1: 2}[kind]
    assert np.count_nonzero(offsets, axis=1).max() == max_nonzero
    assert not np.any(np.all(offsets == 0, axis=1))


def test_ids_and_coordinates_round_trip(make_lattice):
    space = make_lattice((3, 4, 5))
    ids = np.arange(space.n_cells)
    np.testing.assert_array_equal(space.ravel(space.unravel(ids)), ids)
    np.testing.assert_array_equal(space.unravel(ids), space.coords)


@pytest.mark.parametrize("method", ["ravel", "unravel"])
def test_ravel_and_unravel_refuse_values_that_are_not_integers(make_lattice, not_cell_ids, method):
    values, error = not_cell_ids
    if method == "ravel":
        values = np.stack([values, values], axis=-1)
    with pytest.raises(error, match="coords" if method == "ravel" else "ids"):
        getattr(make_lattice((3, 3)), method)(values)


@pytest.mark.parametrize("ids", [9, -1, [0, 9]])
def test_unravel_rejects_ids_outside_the_grid_as_bad_indices(make_lattice, ids):
    with pytest.raises(IndexError, match=r"ids must hold cell ids in \[0, 9\)"):
        make_lattice((3, 3)).unravel(ids)


def test_ravel_rejects_coordinates_outside_the_grid(make_lattice):
    with pytest.raises(ValueError, match="invalid entry"):
        make_lattice((3, 3)).ravel([3, 0])


def test_grid_view_reshapes_without_copying(make_lattice, rng):
    space = make_lattice((3, 4))
    values = rng.random((2, space.n_cells))
    view = space.grid_view(values)
    assert view.shape == (2, 3, 4)
    assert np.shares_memory(view, values)
    assert view[1, 2, 3] == values[1, space.ravel([2, 3])]


def test_grid_view_rejects_a_wrong_cell_axis(make_lattice):
    with pytest.raises(ValueError, match="n_cells"):
        make_lattice((3, 4)).grid_view(np.zeros(11))


@pytest.mark.parametrize(
    ("shape", "tiling", "periodic"),
    [
        ((9, 7), "hypercubic", True),
        ((9, 7), "hypercubic", False),
        ((5, 4, 6), "hypercubic", (True, False, True)),
        ((8, 6), "triangle", True),
        ((7, 5), "triangle", False),
    ],
)
def test_graphs_built_in_blocks_equal_the_graph_built_at_once(shape, tiling, periodic):
    # Default blocks hold tens of thousands of cells, so only small blocks reach these
    # lattices' block boundaries. Triangles alternate, so odd blocks start on either kind.
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    offsets, _ = space._neighbors
    data = {"offset": np.arange(offsets[:, :, 0].size, dtype=np.float64).reshape(offsets.shape[:2])}
    whole = space._offset_graph(offsets, data, block=space.n_cells)
    for block in (1, 2, 5, 17):
        part = space._offset_graph(offsets, data, block=block)
        np.testing.assert_array_equal(part.indptr, whole.indptr)
        np.testing.assert_array_equal(part.indices, whole.indices)
        np.testing.assert_array_equal(part.edge_data["offset"], whole.edge_data["offset"])


@pytest.mark.slow
def test_building_a_large_adjacency_takes_little_memory_beyond_the_graph():
    # Sorting the whole edge list once took 5x the graph's size; rows built block by block
    # take about 1.7x here, the extra being a fixed budget per block.
    space = LatticeSpace((512, 512))
    tracemalloc.start()
    try:
        graph = space.adjacency("vertex")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    size = graph.indptr.nbytes + graph.indices.nbytes
    size += sum(values.nbytes for values in graph.edge_data.values())
    assert graph.n_edges == 8 * space.n_cells
    assert peak < 2 * size


def test_adjacency_is_built_once_per_kind(make_lattice):
    space = make_lattice((4, 4))
    assert space.adjacency("facet") is space.adjacency(1)
    assert space.adjacency("facet") is not space.adjacency("vertex")


def test_hex_facet_and_vertex_adjacency_are_one_graph(make_lattice):
    space = make_lattice((4, 5), "hex")
    assert space.adjacency("facet") is space.adjacency("vertex")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": ()}, "non-empty"),
        ({"shape": (3, 0)}, "positive"),
        ({"shape": (3, 3), "tiling": "penrose"}, "tiling"),
        ({"shape": (3, 3, 3), "tiling": "hex"}, "2-D"),
        ({"shape": (4,), "tiling": "triangle"}, "the triangle tiling is 2-D"),
        ({"shape": (3, 3), "periodic": (True,)}, "periodic has 1 entries, but the lattice has 2"),
        ({"shape": (2, 5)}, "length >= 3"),
        ({"shape": (7, 4), "tiling": "triangle"}, "axis 0 has length 7; .* divisible by 2"),
        ({"shape": (6, 5), "tiling": "triangle"}, "axis 1 has length 5; .* divisible by 2"),
        ({"shape": (4, 4), "tiling": "triangle"}, "axis 0 has length 4; .* length >= 6"),
        ({"shape": (6, 2), "tiling": "triangle"}, "axis 1 has length 2; .* length >= 4"),
    ],
    ids=[
        "empty",
        "zero-length",
        "unknown-tiling",
        "3d-hex",
        "1d-triangle",
        "periodic-mismatch",
        "short-periodic",
        "odd-triangle-row",
        "odd-triangle-column",
        "short-triangle-row",
        "short-triangle-column",
    ],
)
def test_invalid_lattices_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LatticeSpace(**kwargs)


@pytest.mark.parametrize(
    "shape", [5, "55", None, (3.9, 4), (4.0, 4), (True, 4)], ids=lambda shape: repr(shape)
)
def test_lattice_shapes_must_be_sequences_of_ints(shape):
    with pytest.raises(TypeError, match="shape"):
        LatticeSpace(shape, periodic=False)


@pytest.mark.parametrize(
    "periodic", ["no", 1, None, (1, 0), np.array([1, 0]), (True, "yes")], ids=repr
)
def test_periodic_must_be_bools(periodic):
    with pytest.raises(TypeError, match="periodic must be a bool or one bool per axis"):
        LatticeSpace((3, 3), periodic=periodic)


@pytest.mark.parametrize(
    ("periodic", "expected"),
    [
        (np.array(True), (True, True)),
        (np.False_, (False, False)),
        ([np.True_, False], (True, False)),
        (np.array([False, True]), (False, True)),
    ],
    ids=["0-d-array", "numpy-bool", "list", "array"],
)
def test_periodic_accepts_numpy_bools(periodic, expected):
    assert LatticeSpace((3, 3), periodic=periodic).periodic == expected


def test_short_axes_are_allowed_when_not_periodic():
    space = LatticeSpace((2, 1), periodic=False)
    np.testing.assert_array_equal(space.adjacency("facet").degree, [1, 1])
    triangles = LatticeSpace((3, 1), tiling="triangle", periodic=(False, False))
    np.testing.assert_array_equal(triangles.adjacency("facet").degree, [1, 2, 1])


@pytest.mark.parametrize("kind", ["edge", 2, -1])
def test_unknown_adjacency_kinds_are_rejected(make_lattice, kind):
    with pytest.raises(ValueError, match="adjacency kind"):
        make_lattice((3, 3)).adjacency(kind)


@pytest.mark.parametrize("kind", [True, 1.0, None])
def test_adjacency_kinds_must_be_strings_or_ints(make_lattice, kind):
    with pytest.raises(TypeError, match="adjacency kind"):
        make_lattice((3, 3)).adjacency(kind)


# Locating points


LOCATE_CASES = (
    [
        ((5, 4), "hypercubic", periodic)
        for periodic in [(True, True), (True, False), (False, True), (False, False)]
    ]
    + [
        ((5, 4), "hex", periodic)
        for periodic in [(True, True), (True, False), (False, True), (False, False)]
    ]
    + [
        ((6, 4), "triangle", periodic)
        for periodic in [(True, True), (True, False), (False, True), (False, False)]
    ]
    + [((4, 3, 3), "hypercubic", (True, False, True)), ((6,), "hypercubic", (False,))]
)


def _nearest_site(space: LatticeSpace, points: np.ndarray) -> np.ndarray:
    """The cell of the lattice site nearest each point, by brute force, or -1 beyond the grid.

    Searches every site within three steps of the grid, then wraps the nearest one's
    coordinates around periodic axes, and drops it if it lies beyond a fixed edge. A triangle's
    site is its center, 1/4 below or above ``basis @ coords``. Shares no code with
    `LatticeSpace.locate`.
    """
    ranges = [range(-3, length + 3) for length in space.shape]
    sites = np.array(list(itertools.product(*ranges)))
    centers = sites @ space.basis.T
    if space.tiling == "triangle":
        centers[:, 1] += np.where(sites.sum(axis=1) % 2 == 0, -0.25, 0.25)
    distance = np.linalg.norm(points[:, None, :] - centers[None], axis=-1)
    nearest = sites[np.argmin(distance, axis=1)]
    shape = np.array(space.shape)
    nearest = np.where(space.periodic, nearest % shape, nearest)
    inside = np.all((nearest >= 0) & (nearest < shape), axis=1)
    ids = np.ravel_multi_index(tuple(np.clip(nearest, 0, shape - 1).T), space.shape)
    return np.where(inside, ids, -1)


@pytest.mark.parametrize(("shape", "tiling", "periodic"), LOCATE_CASES)
def test_locate_finds_the_lattice_site_nearest_each_point(rng, shape, tiling, periodic):
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    coords = rng.uniform(-2, np.array(shape) + 1, size=(3000, len(shape)))  # around the grid
    points = coords @ space.basis.T
    expected = _nearest_site(space, points)
    assert (expected >= 0).any()
    assert (expected < 0).any() != all(periodic)  # points beyond every fixed edge
    np.testing.assert_array_equal(space.locate(points), expected)


@pytest.mark.parametrize(("shape", "tiling", "periodic"), LOCATE_CASES)
def test_each_center_locates_its_own_cell(shape, tiling, periodic):
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    found = space.locate(space.centers)
    assert found.dtype == np.int64
    np.testing.assert_array_equal(found, np.arange(space.n_cells))


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
@pytest.mark.parametrize("turns", [-3, 1, 10**9])
def test_locate_wraps_points_around_periodic_axes(rng, tiling, turns):
    space = LatticeSpace((6, 4), tiling=tiling, periodic=True)
    points = space.centers + rng.uniform(-0.3, 0.3, size=space.centers.shape)  # inside cells
    for period in (space.basis * np.array(space.shape)).T:
        np.testing.assert_array_equal(space.locate(points + turns * period), space.locate(points))


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_locate_finds_no_cell_beyond_a_fixed_edge_however_far(tiling):
    space = LatticeSpace((6, 4), tiling=tiling, periodic=(True, False))
    far = [[0.0, -1e9], [0.0, 1e300], [1e300, 1.0], [-1e300, -1e300]]
    found = space.locate(far)
    np.testing.assert_array_equal(found[[0, 1, 3]], -1)
    assert 0 <= found[2] < space.n_cells  # the wrapped axis has a cell everywhere


def test_locate_keeps_the_leading_shape_of_the_points(make_lattice):
    space = make_lattice((4, 3))
    points = space.centers.reshape(3, 4, 2)
    np.testing.assert_array_equal(space.locate(points), np.arange(12).reshape(3, 4))
    assert space.locate([1.0, 2.0]).shape == ()
    assert space.locate(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize(
    ("points", "error", "match"),
    [
        (np.zeros((4, 3)), ValueError, r"shape \(\.\.\., 2\)"),
        (1.0, ValueError, "shape"),
        ([[np.nan, 0.0]], ValueError, "NaN"),
        ([[np.inf, 0.0]], ValueError, "infinite"),
        ([["a", "b"]], TypeError, "numbers"),
        ([[1.7e308, -1.7e308]], ValueError, "too far"),  # its lattice coordinates overflow
    ],
)
def test_locate_rejects_bad_points(points, error, match):
    with pytest.raises(error, match=match):
        LatticeSpace((5, 4), tiling="hex").locate(points)


# The half turn

OPPOSITES = [
    ((5, 5), "hypercubic", True),
    ((4, 7), "hypercubic", False),
    ((7,), "hypercubic", False),
    ((3, 4, 5), "hypercubic", False),
    ((3, 4, 5), "hypercubic", True),
    ((6, 9), "hex", False),
    ((5, 8), "hex", True),
    ((6, 5), "triangle", False),
    ((7, 4), "triangle", (False, True)),
]


def edges(graph):
    rows = np.repeat(np.arange(graph.n_cells), np.diff(graph.indptr))
    return set(zip(rows.tolist(), graph.indices.tolist(), strict=True))


@pytest.mark.parametrize(("shape", "tiling", "periodic"), OPPOSITES)
def test_opposite_is_a_permutation_that_undoes_itself(shape, tiling, periodic):
    turn = LatticeSpace(shape, tiling, periodic).opposite()
    assert turn.dtype == np.int64
    np.testing.assert_array_equal(np.sort(turn), np.arange(turn.size))
    np.testing.assert_array_equal(turn[turn], np.arange(turn.size))


@pytest.mark.parametrize(("shape", "tiling", "periodic"), OPPOSITES)
def test_opposite_keeps_every_kind_of_adjacency(shape, tiling, periodic):
    space = LatticeSpace(shape, tiling, periodic)
    turn = space.opposite()
    for kind in ("facet", "vertex"):
        graph = edges(space.adjacency(kind))
        assert {(int(turn[a]), int(turn[b])) for a, b in graph} == graph


@pytest.mark.parametrize(("shape", "tiling", "periodic"), OPPOSITES)
def test_opposite_swaps_opposite_corners(shape, tiling, periodic):
    space = LatticeSpace(shape, tiling, periodic)
    turn = space.opposite()
    last = space.ravel(np.array(shape) - 1)
    assert turn[0] == last
    assert turn[last] == 0


@pytest.mark.parametrize(("shape", "periodic"), [((6, 5), False), ((7, 4), (False, True))])
def test_opposite_turns_triangles_over(shape, periodic):
    space = LatticeSpace(shape, "triangle", periodic)
    up = space.coords.sum(axis=1) % 2 == 0
    np.testing.assert_array_equal(up[space.opposite()], ~up)


@pytest.mark.parametrize(
    ("shape", "tiling"),
    [((4, 7), "hypercubic"), ((6, 9), "hex"), ((6, 5), "triangle"), ((3, 4, 5), "hypercubic")],
)
def test_opposite_keeps_distances_between_cells(shape, tiling):
    space = LatticeSpace(shape, tiling, periodic=False)
    turn = space.opposite()
    gaps = np.linalg.norm(space.centers[:, None] - space.centers[None], axis=-1)
    np.testing.assert_allclose(gaps[np.ix_(turn, turn)], gaps, atol=1e-12)


def test_opposite_cells_on_a_hex_board_are_half_a_turn_apart():
    space = LatticeSpace((6, 9), "hex", periodic=False)
    centers = space.centers - space.centers.mean(axis=0)
    np.testing.assert_allclose(centers[space.opposite()], -centers, atol=1e-12)


@pytest.mark.parametrize(("shape", "periodic"), [((6, 4), False), ((5, 5), False), ((6, 4), True)])
def test_triangles_with_an_even_q_plus_r_have_no_opposites(shape, periodic):
    with pytest.raises(ValueError, match="Q \\+ R odd"):
        LatticeSpace(shape, "triangle", periodic).opposite()


def test_opposite_is_computed_once_and_read_only(make_lattice):
    space = make_lattice((4, 6))
    assert space.opposite() is space.opposite()
    assert not space.opposite().flags.writeable


CORNERED = [
    ((4, 3), "hypercubic", 4, 0.5**0.5),
    ((4, 3), "hex", 6, 3**-0.5),
    ((6, 4), "triangle", 3, 1.0),
]


@pytest.mark.parametrize(("shape", "tiling", "k", "radius"), CORNERED)
@pytest.mark.parametrize("periodic", [False, True])
def test_corners_are_each_cells_polygon_around_its_center(shape, tiling, k, radius, periodic):
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    corners = space.corners()
    assert corners.shape == (space.n_cells, k, 2)
    assert corners.dtype == np.float64
    np.testing.assert_allclose(corners.mean(axis=1), space.centers, atol=1e-12)
    distances = np.linalg.norm(corners - space.centers[:, None, :], axis=2)
    np.testing.assert_allclose(distances, radius, rtol=1e-12)


@pytest.mark.parametrize(("shape", "tiling", "k", "radius"), CORNERED)
def test_corners_run_counterclockwise_around_the_cells_area(shape, tiling, k, radius):
    space = LatticeSpace(shape, tiling=tiling, periodic=False)
    x, y = space.corners()[..., 0], space.corners()[..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    np.testing.assert_allclose(area, space.measures, rtol=1e-12)


@pytest.mark.parametrize(("shape", "tiling", "k", "radius"), CORNERED)
def test_facet_neighbors_share_exactly_two_corners(shape, tiling, k, radius):
    space = LatticeSpace(shape, tiling=tiling, periodic=False)
    corners = np.round(space.corners(), 9)
    graph = space.adjacency("facet")
    for a in range(space.n_cells):
        for b in graph.indices[graph.indptr[a] : graph.indptr[a + 1]]:
            shared = {tuple(p) for p in corners[a]} & {tuple(p) for p in corners[b]}
            assert len(shared) == 2


def test_corners_are_cached_and_read_only():
    space = LatticeSpace((3, 3), tiling="hex", periodic=False)
    assert space.corners() is space.corners()
    assert not space.corners().flags.writeable


@pytest.mark.parametrize("shape", [(5,), (3, 3, 3)])
def test_corners_need_a_2d_lattice(shape):
    with pytest.raises(ValueError, match="2-D"):
        LatticeSpace(shape).corners()
