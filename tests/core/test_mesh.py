from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from gridzero.core import MeshSpace, icosphere


def _edge_set(graph) -> set[tuple[int, int]]:
    return set(zip(graph.sources.tolist(), graph.indices.tolist(), strict=True))


# --- hand-built planar mesh -------------------------------------------------------------


def test_square_mesh_adjacency_matches_the_grid(make_square_mesh):
    mesh = make_square_mesh(2, 2)  # cells: 0 1 / 2 3
    facet = mesh.adjacency("facet")
    vertex = mesh.adjacency("vertex")
    assert _edge_set(facet) == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)}
    assert _edge_set(vertex) == _edge_set(facet) | {(0, 3), (3, 0), (1, 2), (2, 1)}


def test_square_mesh_edge_geometry(make_square_mesh):
    vertex = make_square_mesh(2, 2).adjacency("vertex")
    diagonal = {(0, 3), (3, 0), (1, 2), (2, 1)}
    is_diagonal = np.array(
        [edge in diagonal for edge in zip(vertex.sources, vertex.indices, strict=True)]
    )
    np.testing.assert_allclose(vertex.edge_data["distance"], np.where(is_diagonal, np.sqrt(2), 1.0))
    np.testing.assert_allclose(vertex.edge_data["boundary"], np.where(is_diagonal, 0.0, 1.0))


def test_open_mesh_edges_have_no_neighbor(make_square_mesh):
    degree = make_square_mesh(3, 3).adjacency("facet").degree
    np.testing.assert_array_equal(degree.reshape(3, 3), [[2, 3, 2], [3, 4, 3], [2, 3, 2]])


def test_polygon_returns_vertices_in_cyclic_order(make_square_mesh):
    np.testing.assert_array_equal(
        make_square_mesh(2, 2).polygon(3), [[1, 1], [2, 1], [2, 2], [1, 2]]
    )


@pytest.mark.parametrize("cell", [-1, 4])
def test_polygon_rejects_ids_outside_the_mesh(make_square_mesh, cell):
    with pytest.raises(IndexError, match="out of range"):
        make_square_mesh(2, 2).polygon(cell)


@pytest.fixture
def split_border_mesh() -> MeshSpace:
    """Two unit squares side by side, with a vertex at the midpoint of their shared border.

    Vertex 4 = (1, 0.5) is a corner of both polygons, so the cells share two sides of length
    0.5 each.
    """
    return MeshSpace(
        centers=[[0.5, 0.5], [1.5, 0.5]],
        measures=[1.0, 1.0],
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1], [1, 0.5], [2, 0], [2, 1]],
        polygon_indptr=[0, 5, 10],
        polygon_indices=[0, 1, 4, 2, 3, 1, 5, 6, 2, 4],
    )


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_cells_sharing_several_sides_get_their_total_length(split_border_mesh, kind):
    graph = split_border_mesh.adjacency(kind)
    assert _edge_set(graph) == {(0, 1), (1, 0)}
    np.testing.assert_allclose(graph.edge_data["boundary"], [1.0, 1.0])
    np.testing.assert_allclose(graph.edge_data["distance"], [1.0, 1.0])


def test_an_edge_shared_by_three_cells_is_rejected():
    fan = MeshSpace(
        centers=np.zeros((3, 2)),
        measures=np.ones(3),
        vertices=[[0, 0], [1, 0], [0, 1], [0, -1], [1, 1]],
        polygon_indptr=[0, 3, 6, 9],
        polygon_indices=[0, 1, 2, 0, 1, 3, 0, 1, 4],
    )
    with pytest.raises(ValueError, match="more than two cells"):
        fan.adjacency("facet")


# Polygons of make_square_mesh(2, 2), whose vertices form a 3x3 grid with ids 0..8.
_SQUARE_POLYGONS = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"measures": np.ones(3)}, "one entry per cell"),
        ({"measures": [1.0, 0.0, 1.0, 1.0]}, "finite and positive"),
        ({"measures": [1.0, -1.0, 1.0, 1.0]}, "finite and positive"),
        ({"measures": [1.0, np.nan, 1.0, 1.0]}, "finite and positive"),
        ({"centers": [[0.5, 0.5], [np.inf, 0.5], [0.5, 1.5], [1.5, 1.5]]}, "finite"),
        ({"polygon_indptr": [0, 4, 8, 12, 15]}, "spanning"),
        ({"polygon_indices": _SQUARE_POLYGONS}, "1-D"),
        ({"polygon_indices": np.full(16, 99)}, "existing vertices"),
        ({"polygon_indices": [0, 1, 0, 3, *np.ravel(_SQUARE_POLYGONS[1:])]}, "same vertex"),
        ({"metric": "hyperbolic"}, "metric"),
        ({"metric": "spherical"}, "3-D"),
        (
            {
                "centers": np.zeros((0, 2)),
                "measures": [],
                "polygon_indptr": [0],
                "polygon_indices": [],
            },
            "at least one cell",
        ),
    ],
    ids=[
        "measures-count",
        "measures-zero",
        "measures-negative",
        "measures-nan",
        "centers-inf",
        "indptr",
        "indices-2d",
        "indices-range",
        "repeated-vertex",
        "metric",
        "spherical-2d",
        "empty",
    ],
)
def test_inconsistent_meshes_are_rejected(make_square_mesh, overrides, message):
    with pytest.raises(ValueError, match=message):
        MeshSpace(**(_mesh_fields(make_square_mesh(2, 2)) | overrides))


def _mesh_fields(mesh: MeshSpace) -> dict[str, Any]:
    return {
        "centers": mesh.centers,
        "measures": mesh.measures,
        "vertices": mesh.vertices,
        "polygon_indptr": mesh.polygon_indptr,
        "polygon_indices": mesh.polygon_indices,
        "metric": mesh.metric,
    }


@pytest.mark.parametrize("which", ["polygon_indptr", "polygon_indices"])
def test_polygon_arrays_must_hold_integers(make_square_mesh, not_cell_ids, which):
    ids, error = not_cell_ids
    with pytest.raises(error, match=which):
        MeshSpace(**(_mesh_fields(make_square_mesh(2, 2)) | {which: ids}))


def test_a_polygon_side_of_zero_length_is_rejected(make_square_mesh):
    # Vertex 9 sits where vertex 4 is: cell 0 would have a side of zero length.
    fields = _mesh_fields(make_square_mesh(2, 2))
    vertices = np.vstack([fields["vertices"], fields["vertices"][4]])
    indices = np.array(fields["polygon_indices"])
    indices[3] = 9  # cell 0 becomes 0 - 1 - 4 - 9: corners 4 and 9 coincide
    changed: dict[str, Any] = fields | {"vertices": vertices, "polygon_indices": indices}
    with pytest.raises(ValueError, match="consecutive vertices of a polygon are at the same point"):
        MeshSpace(**changed)


@pytest.mark.parametrize(
    "move",
    [lambda v: v * np.where(np.arange(len(v)) == 0, 1.01, 1.0)[:, None], lambda v: v + 1.0],
    ids=["one-vertex-off-the-sphere", "sphere-not-at-origin"],
)
def test_spherical_meshes_must_lie_on_one_origin_centered_sphere(make_icosphere, move):
    fields = _mesh_fields(make_icosphere(1))
    with pytest.raises(ValueError, match="one sphere"):
        MeshSpace(**(fields | {"vertices": move(fields["vertices"])}))


def test_a_planar_mesh_locates_the_cell_with_the_nearest_center(make_square_mesh, rng):
    mesh = make_square_mesh(3, 4)
    points = rng.uniform(-2, 6, size=(500, 2))  # also beyond the mesh's edges
    distance = np.linalg.norm(points[:, None] - mesh.centers[None], axis=-1)
    found = mesh.locate(points)
    assert found.dtype == np.int64
    np.testing.assert_array_equal(found, np.argmin(distance, axis=1))


def test_mesh_locate_keeps_the_leading_shape_of_the_points(make_square_mesh):
    mesh = make_square_mesh(2, 3)
    np.testing.assert_array_equal(
        mesh.locate(mesh.centers.reshape(2, 3, 2)), np.arange(6).reshape(2, 3)
    )
    assert mesh.locate([0.5, 0.5]).shape == ()
    assert mesh.locate(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize(
    ("points", "error", "match"),
    [
        (np.zeros((4, 3)), ValueError, r"shape \(\.\.\., 2\)"),
        ([[np.nan, 0.0]], ValueError, "NaN"),
        ([["a", "b"]], TypeError, "numbers"),
        ([[1e200, 0.0]], ValueError, "too far"),  # squared distances overflow
    ],
)
def test_planar_mesh_locate_rejects_bad_points(make_square_mesh, points, error, match):
    with pytest.raises(error, match=match):
        make_square_mesh().locate(points)


# --- icosphere --------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [1, 2, 3, 4])
def test_icosphere_has_10f2_plus_2_cells(make_icosphere, frequency):
    assert make_icosphere(frequency).n_cells == 10 * frequency**2 + 2


def test_frequency_one_icosphere_is_a_dodecahedron(make_icosphere):
    planet = make_icosphere(1)
    np.testing.assert_array_equal(planet.adjacency().degree, 5)
    assert len(planet.vertices) == 20


@pytest.mark.parametrize("frequency", [2, 3, 5])
def test_icosphere_has_exactly_twelve_pentagons(make_icosphere, frequency):
    planet = make_icosphere(frequency)
    degree = planet.adjacency().degree
    assert np.count_nonzero(degree == 5) == 12
    assert np.count_nonzero(degree == 6) == planet.n_cells - 12
    np.testing.assert_array_equal(np.diff(planet.polygon_indptr), degree)


@pytest.mark.parametrize("radius", [1.0, 6371.0])
def test_icosphere_cells_tile_the_sphere(make_icosphere, radius):
    planet = make_icosphere(4, radius)
    assert planet.measures.min() > 0
    assert planet.measures.sum() == pytest.approx(4 * np.pi * radius**2, rel=1e-9)


@pytest.mark.parametrize("frequency", [1, 2, 3])
def test_shared_boundaries_add_up_to_each_cell_perimeter(make_icosphere, frequency):
    # On a closed mesh every polygon side is shared with exactly one neighbor.
    planet = make_icosphere(frequency)
    graph = planet.adjacency("facet")
    shared = np.bincount(graph.sources, weights=graph.edge_data["boundary"])
    polygons = [planet.polygon(cell) for cell in range(planet.n_cells)]
    perimeter = [planet.distance(p, np.roll(p, -1, axis=0)).sum() for p in polygons]
    np.testing.assert_allclose(shared, perimeter, rtol=1e-12)


def test_icosphere_adjacency_matches_the_delaunay_triangulation(make_icosphere):
    planet = make_icosphere(5)
    triangles = ConvexHull(planet.centers).simplices
    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    delaunay = {(int(a), int(b)) for a, b in pairs} | {(int(b), int(a)) for a, b in pairs}
    assert _edge_set(planet.adjacency("facet")) == delaunay


def test_icosphere_vertex_adjacency_equals_facet_adjacency(make_icosphere):
    # Three cells meet at every Voronoi vertex, so touching at a corner implies sharing an edge.
    planet = make_icosphere(4)
    assert _edge_set(planet.adjacency("vertex")) == _edge_set(planet.adjacency("facet"))


def test_icosphere_lengths_scale_with_radius(make_icosphere):
    unit, earth = make_icosphere(3), make_icosphere(3, 6371.0)
    for name in ("distance", "boundary"):
        np.testing.assert_allclose(
            earth.adjacency().edge_data[name], 6371.0 * unit.adjacency().edge_data[name]
        )
    np.testing.assert_allclose(earth.measures, 6371.0**2 * unit.measures)


def test_icosphere_perimeters_are_consistent_with_areas(make_icosphere):
    # For near-regular hexagons, perimeter**2 / area is close to 8 * sqrt(3) ~ 13.86.
    planet = make_icosphere(6)
    graph = planet.adjacency()
    perimeter = np.bincount(graph.sources, weights=graph.edge_data["boundary"])
    ratio = perimeter**2 / planet.measures
    assert ratio.min() > 13.0
    assert ratio.max() < 15.5


def test_planar_radius_graph_reaches_diagonal_neighbors(make_square_mesh):
    graph = make_square_mesh(3, 3).radius_graph(1.5)
    np.testing.assert_array_equal(graph.degree, [3, 5, 3, 5, 8, 5, 3, 5, 3])
    np.testing.assert_allclose(np.unique(graph.edge_data["distance"]), [1.0, 2**0.5])


@pytest.mark.parametrize("radius", [0.6, 1.4, 3.2, 7.0])
def test_icosphere_radius_graph_matches_arc_lengths_between_all_pairs(make_icosphere, radius):
    space = make_icosphere(3, radius=2.0)
    unit = space.centers / 2.0
    arc = 2.0 * np.arccos(np.clip(unit @ unit.T, -1.0, 1.0))  # (n, n), independent of the code
    assert not np.any(np.isclose(arc, radius, rtol=1e-6))  # no pair sits on the boundary
    within = (arc <= radius) & ~np.eye(space.n_cells, dtype=bool)
    graph = space.radius_graph(radius)
    assert _edge_set(graph) == set(zip(*np.nonzero(within), strict=True))
    np.testing.assert_allclose(
        graph.edge_data["distance"], arc[graph.sources, graph.indices], rtol=1e-6
    )


def test_icosphere_is_deterministic():
    a, b = icosphere(3), icosphere(3)
    np.testing.assert_array_equal(a.centers, b.centers)
    np.testing.assert_array_equal(a.adjacency().indices, b.adjacency().indices)


@pytest.mark.parametrize("radius", [0.0, -1.0, np.inf])
def test_icosphere_rejects_a_radius_that_is_not_positive_and_finite(radius):
    with pytest.raises(ValueError, match="radius"):
        icosphere(2, radius)


@pytest.mark.parametrize("radius", [True, "1.0", None])
def test_icosphere_rejects_a_radius_that_is_not_a_number(radius):
    with pytest.raises(TypeError, match="radius must be a real number"):
        icosphere(2, radius)


@pytest.mark.parametrize("distance", [1e-300, 0.5, 1.0, 1e300])
def test_a_spherical_mesh_locates_the_cell_toward_each_point(make_icosphere, rng, distance):
    planet = make_icosphere(4, radius=3.0)  # neither the sphere's radius nor the point's matters
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    closeness = directions @ (planet.centers / 3.0).T  # larger is nearer
    np.testing.assert_array_equal(
        planet.locate(distance * directions), np.argmax(closeness, axis=1)
    )
    np.testing.assert_array_equal(
        planet.locate(distance * planet.centers), np.arange(planet.n_cells)
    )


@pytest.mark.parametrize(
    ("points", "match"),
    [
        (np.zeros((2, 3)), "center of the sphere"),
        (np.ones((2, 2)), "shape"),
        ([[np.inf, 0, 0]], "infinite"),
    ],
)
def test_spherical_mesh_locate_rejects_bad_points(make_icosphere, points, match):
    with pytest.raises(ValueError, match=match):
        make_icosphere(1).locate(points)
