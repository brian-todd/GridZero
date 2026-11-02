import pickle
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path

from gridzero.core import (
    Adjacency,
    AdjacencyKind,
    Ball,
    CellGraph,
    Explicit,
    Hops,
    LatticeSpace,
    Neighborhood,
    NeighborOperator,
    Space,
    icosphere,
    pull,
)


@dataclass(frozen=True)
class NearestCenters:
    """A hand-written custom neighborhood: the ``k`` cells with the closest centers.

    Distances are straight lines between centers, found by brute force, so this is only for
    tiny spaces. Ties go to the lower cell id. Weights are inverse distances.
    """

    k: int

    def compile(self, space: Space) -> NeighborOperator:
        centers = space.centers
        distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        idx = np.argsort(distance, axis=1, kind="stable")[:, : self.k]
        weight = 1.0 / np.take_along_axis(distance, idx, axis=1)
        return NeighborOperator.from_table(idx, weight)


def _edge_set(op: NeighborOperator) -> set[tuple[int, int]]:
    return set(zip(op.graph.sources.tolist(), op.graph.indices.tolist(), strict=True))


def _ring_sizes(op: NeighborOperator, reach: int) -> np.ndarray:
    """Count each cell's neighbors at each hop distance: shape (n_cells, reach)."""
    hops = op.query().weight  # 0 in unused slots
    return np.stack([np.sum(hops == k, axis=1) for k in range(1, reach + 1)], axis=1)


# The protocol.


@pytest.mark.parametrize(
    "hood",
    [Adjacency(), Hops(2), Ball(1.5), Explicit(CellGraph.from_edges([], [], 1)), NearestCenters(3)],
    ids=["adjacency", "hops", "ball", "explicit", "custom"],
)
def test_neighborhoods_satisfy_the_protocol(hood):
    assert isinstance(hood, Neighborhood)


def test_a_custom_neighborhood_plugs_in_like_a_built_in():
    # On the dodecahedron, the 5 nearest face centers are exactly the 5 adjacent faces.
    space = icosphere(1)
    custom = NearestCenters(5).compile(space)
    assert _edge_set(custom) == _edge_set(Adjacency().compile(space))
    np.testing.assert_allclose(pull(np.ones(12), custom, "mean"), np.ones(12), rtol=1e-12)


def test_neighborhoods_are_values_that_can_key_a_cache():
    assert Hops(2) == Hops(np.int64(2), "facet")
    assert {Hops(2): "a", Ball(1.5): "b"}[Hops(2)] == "a"
    assert pickle.loads(pickle.dumps(Ball(1.5, "distance"))) == Ball(1.5, "distance")


# Adjacency and Explicit.


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_adjacency_uses_the_space_graph_itself(any_space, kind):
    op = Adjacency(kind).compile(any_space)
    assert op.graph is any_space.adjacency(kind)


def test_adjacency_can_weight_by_shared_boundary(make_icosphere):
    space = make_icosphere(2)
    op = Adjacency(weight="boundary").compile(space)
    np.testing.assert_array_equal(op.weights, space.adjacency().edge_data["boundary"])


def test_explicit_neighborhood_uses_the_given_graph(make_lattice):
    graph = CellGraph.from_edges([0, 3], [24, 1], 25, edge_data={"flow": [2.0, 5.0]})
    op = Explicit(graph, weight="flow").compile(make_lattice())
    assert op.graph is graph
    np.testing.assert_array_equal(op.weights, [2.0, 5.0])


def test_explicit_neighborhood_must_fit_the_space(make_lattice):
    graph = CellGraph.from_edges([0], [1], 4)
    with pytest.raises(ValueError, match="4 cells but the space has 25"):
        Explicit(graph).compile(make_lattice())


@pytest.mark.parametrize(
    ("make", "wrong"),
    [
        (lambda weight: Adjacency(weight=weight), "hops"),
        (lambda weight: Hops(2, weight=weight), "distance"),
        (lambda weight: Ball(1.0, weight=weight), "boundary"),
        (lambda weight: Explicit(CellGraph.from_edges([0], [1], 2), weight=weight), "flow"),
    ],
)
def test_neighborhoods_reject_weights_they_do_not_provide_when_made(make, wrong):
    # Equal descriptions share one compiled operator, so a bad one must not get that far.
    with pytest.raises(ValueError, match=f"no edge attribute {wrong!r}"):
        make(wrong)
    with pytest.raises(TypeError, match="weight"):
        make(3)
    with pytest.raises(TypeError, match="weight"):
        make(b"distance")


# Hops.

HOP_RINGS: dict[str, tuple[Callable[[], Space], AdjacencyKind, list[int]]] = {
    "square-facet": (lambda: LatticeSpace((9, 9)), "facet", [4, 8, 12, 16]),
    "square-vertex": (lambda: LatticeSpace((9, 9)), "vertex", [8, 16, 24, 32]),
    "hex": (lambda: LatticeSpace((7, 7), tiling="hex"), "facet", [6, 12, 18]),
    "cube-facet": (lambda: LatticeSpace((9, 9, 9)), "facet", [6, 18, 38, 66]),
    "triangle-facet": (lambda: LatticeSpace((18, 10), tiling="triangle"), "facet", [3, 6, 9, 12]),
    "triangle-vertex": (
        lambda: LatticeSpace((24, 16), tiling="triangle"),
        "vertex",
        [12, 24, 36, 48],
    ),
}


@pytest.mark.parametrize("case", sorted(HOP_RINGS))
def test_hop_rings_on_periodic_lattices_have_the_known_sizes(case):
    make_space, kind, rings = HOP_RINGS[case]
    space = make_space()
    op = Hops(len(rings), kind, weight="hops").compile(space)
    expected = np.broadcast_to(rings, (space.n_cells, len(rings)))
    np.testing.assert_array_equal(_ring_sizes(op, len(rings)), expected)


def test_hop_rings_around_icosphere_pentagons_grow_by_five(make_icosphere):
    # Within f hops, a pentagon's neighborhood lies on the 5 icosahedron faces around it.
    space = make_icosphere(4)
    pentagons = np.flatnonzero(space.adjacency().degree == 5)
    op = Hops(4, weight="hops").compile(space)
    expected = np.broadcast_to([5, 10, 15, 20], (12, 4))
    np.testing.assert_array_equal(_ring_sizes(op, 4)[pentagons], expected)


@pytest.mark.parametrize("kind", ["facet", "vertex"])
@pytest.mark.parametrize("reach", [1, 2, 3])
def test_hops_match_breadth_first_search_distances(any_space, kind, reach):
    distance = shortest_path(any_space.adjacency(kind).to_scipy(), unweighted=True)
    op = Hops(reach, kind, weight="hops").compile(any_space)
    within = (distance > 0) & (distance <= reach)
    expected = set(zip(*np.nonzero(within), strict=True))
    assert _edge_set(op) == {(int(i), int(j)) for i, j in expected}
    np.testing.assert_array_equal(op.weights, distance[op.graph.sources, op.graph.indices])


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_one_hop_has_the_same_neighbors_as_adjacency(any_space, kind):
    hops = Hops(1, kind).compile(any_space)
    adjacency = any_space.adjacency(kind)
    np.testing.assert_array_equal(hops.graph.indptr, adjacency.indptr)
    np.testing.assert_array_equal(hops.graph.indices, adjacency.indices)


def test_hops_count_each_cell_once_on_a_small_torus():
    # On a 3x3 torus every other cell is within 2 facet hops, some of them along two routes.
    op = Hops(2, weight="hops").compile(LatticeSpace((3, 3)))
    np.testing.assert_array_equal(op.degree, np.full(9, 8))
    np.testing.assert_array_equal(_ring_sizes(op, 2), np.broadcast_to([4, 4], (9, 2)))


def test_hops_beyond_the_diameter_reach_every_cell(make_icosphere):
    space = make_icosphere(2)
    op = Hops(50).compile(space)
    np.testing.assert_array_equal(op.degree, np.full(space.n_cells, space.n_cells - 1))


@pytest.mark.parametrize("reach", [0, -1])
def test_hops_reject_a_reach_below_one(reach):
    with pytest.raises(ValueError, match="reach must be positive"):
        Hops(reach)


def test_hops_reject_a_reach_that_is_not_an_integer(not_an_integer):
    with pytest.raises(TypeError, match="reach must be an integer"):
        Hops(not_an_integer)


# Adjacency kinds.


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("edge", ValueError),
        (-1, ValueError),
        (1.0, TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
    ids=["unknown-name", "negative", "float", "bool", "none"],
)
@pytest.mark.parametrize("make", [Adjacency, lambda kind: Hops(1, kind)], ids=["adjacency", "hops"])
def test_neighborhood_kinds_are_checked_when_the_description_is_made(make, kind, error):
    # Checking only at compile would let Adjacency(1.0), which equals Adjacency(1), slip through
    # the simulation's cache of compiled neighborhoods whenever Adjacency(1) came first.
    with pytest.raises(error, match="adjacency kind"):
        make(kind)


def test_integer_kinds_are_stored_as_python_ints():
    assert type(Adjacency(np.int64(1)).kind) is int
    assert Adjacency(np.int64(1)) == Adjacency(1)
    assert type(Hops(2, np.int32(0)).kind) is int


def test_whether_an_integer_kind_fits_is_checked_against_the_space():
    hood = Adjacency(2)  # a face of dimension 2 exists between cubes, not between squares
    assert hood.compile(LatticeSpace((3, 3, 3))).n_edges > 0
    with pytest.raises(ValueError, match=r"an int in \[0, 2\)"):
        hood.compile(LatticeSpace((3, 3)))


# Ball.


@pytest.mark.parametrize(
    ("tiling", "radius", "count"),
    [
        ("hypercubic", 0.5, 0),
        ("hypercubic", 1.0, 4),
        ("hypercubic", 2**0.5, 8),
        ("hypercubic", 2.0, 12),
        ("hypercubic", 5**0.5, 20),
        ("hypercubic", 3.0, 28),
        ("hex", 1.0, 6),
        ("hex", 3**0.5, 12),
        ("hex", 2.0, 18),
        ("hex", 7**0.5, 30),
        ("hex", 3.0, 36),
        ("triangle", 1.0, 3),
        ("triangle", 3**0.5, 9),
        ("triangle", 2.0, 12),
        ("triangle", 7**0.5, 18),
        ("triangle", 3.0, 24),
    ],
)
def test_ball_sizes_count_the_lattice_points_in_a_disk(tiling, radius, count):
    shape = (10, 10) if tiling == "triangle" else (9, 9)  # triangles wrap along even axes
    op = Ball(radius).compile(LatticeSpace(shape, tiling=tiling))
    np.testing.assert_array_equal(op.degree, np.full(op.n_cells, count))


def test_ball_weights_are_center_distances(make_icosphere):
    space = make_icosphere(2)
    op = Ball(0.6, weight="distance").compile(space)
    centers = space.centers
    expected = space.distance(centers[op.graph.sources], centers[op.graph.indices])
    assert op.weights is not None
    np.testing.assert_allclose(op.weights, expected, rtol=1e-12)
    assert op.weights.max() <= 0.6


def test_ball_smaller_than_the_cell_spacing_has_no_neighbors(make_icosphere):
    op = Ball(0.01).compile(make_icosphere(2))
    assert op.n_edges == 0
    np.testing.assert_array_equal(pull(np.ones(42), op, "mean", fill=-1), np.full(42, -1.0))


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_ball_rejects_a_radius_that_is_not_finite_and_non_negative(radius):
    with pytest.raises(ValueError, match="radius"):
        Ball(radius)


@pytest.mark.parametrize("radius", [True, np.True_, "2", None], ids=repr)
def test_ball_rejects_a_radius_that_is_not_a_number(radius):
    with pytest.raises(TypeError, match="radius must be a real number"):
        Ball(radius)
