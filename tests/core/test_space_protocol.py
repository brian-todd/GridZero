"""Properties every `Space` implementation must satisfy, checked across all kinds of space."""

import copy
import pickle

import numpy as np
import pytest

from gridzero.core import Space
from gridzero.core.space import RadiusCache


def test_space_satisfies_the_protocol(any_space):
    assert isinstance(any_space, Space)


def test_geometry_arrays_have_one_entry_per_cell(any_space):
    assert any_space.centers.shape[0] == any_space.n_cells
    assert any_space.measures.shape == (any_space.n_cells,)
    assert np.all(any_space.measures > 0)


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_adjacency_is_symmetric_without_self_loops(any_space, kind):
    graph = any_space.adjacency(kind)
    assert graph.n_cells == any_space.n_cells
    assert graph.is_symmetric()
    assert not graph.has_self_loops()


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_edge_attributes_are_symmetric(any_space, kind):
    graph = any_space.adjacency(kind)
    reverse = graph.transpose()
    for name in ("distance", "boundary"):
        np.testing.assert_allclose(reverse.edge_data[name], graph.edge_data[name])


@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_transpose_is_the_exact_matrix_transpose(any_space, kind):
    graph = any_space.adjacency(kind)
    expected = graph.to_scipy("boundary").T.tocsr()
    expected.sort_indices()
    reverse = graph.transpose()
    np.testing.assert_array_equal(reverse.indptr, expected.indptr)
    np.testing.assert_array_equal(reverse.indices, expected.indices)
    np.testing.assert_array_equal(reverse.edge_data["boundary"], expected.data)


def test_vertex_adjacency_contains_facet_adjacency(any_space):
    facet, vertex = any_space.adjacency("facet"), any_space.adjacency("vertex")
    n = any_space.n_cells
    assert np.all(np.isin(facet.sources * n + facet.indices, vertex.sources * n + vertex.indices))


def test_facet_neighbors_have_positive_distance_and_boundary(any_space):
    graph = any_space.adjacency("facet")
    assert np.all(graph.edge_data["distance"] > 0)
    assert np.all(graph.edge_data["boundary"] > 0)


def test_space_arrays_are_read_only(any_space):
    for array in (any_space.centers, any_space.measures):
        assert not array.flags.writeable


def test_space_survives_a_pickle_round_trip(any_space):
    graphs = {kind: any_space.adjacency(kind) for kind in ("facet", "vertex")}  # fill caches
    restored = pickle.loads(pickle.dumps(any_space))
    assert restored.n_cells == any_space.n_cells
    np.testing.assert_array_equal(restored.centers, any_space.centers)
    np.testing.assert_array_equal(restored.measures, any_space.measures)
    for kind, graph in graphs.items():
        rebuilt = restored.adjacency(kind)
        np.testing.assert_array_equal(rebuilt.indptr, graph.indptr)
        np.testing.assert_array_equal(rebuilt.indices, graph.indices)
        np.testing.assert_array_equal(rebuilt.edge_data["distance"], graph.edge_data["distance"])
    for array in (restored.centers, restored.measures):
        assert not array.flags.writeable


def test_copying_a_space_returns_the_same_immutable_object(any_space):
    assert copy.copy(any_space) is any_space
    assert copy.deepcopy(any_space) is any_space


def _facet_spacing(space: Space) -> float:
    """Return the largest distance between facet neighbors."""
    return float(space.adjacency("facet").edge_data["distance"].max())


def _edge_keys(graph) -> np.ndarray:
    return graph.sources * graph.n_cells + graph.indices


def test_radius_graph_is_symmetric_and_stays_within_the_radius(any_space):
    radius = 1.5 * _facet_spacing(any_space)
    graph = any_space.radius_graph(radius)
    assert graph.n_cells == any_space.n_cells
    assert graph.is_symmetric()
    assert not graph.has_self_loops()
    distance = graph.edge_data["distance"]
    assert np.all((distance > 0) & (distance <= radius))
    np.testing.assert_array_equal(graph.transpose().edge_data["distance"], distance)


def test_radius_graph_contains_facet_neighbors_at_the_same_distance(any_space):
    facet = any_space.adjacency("facet")
    graph = any_space.radius_graph(_facet_spacing(any_space))
    slot = np.searchsorted(_edge_keys(graph), _edge_keys(facet))  # keys are sorted
    np.testing.assert_array_equal(_edge_keys(graph)[slot], _edge_keys(facet))
    np.testing.assert_allclose(
        graph.edge_data["distance"][slot], facet.edge_data["distance"], rtol=1e-12
    )


def test_radius_graphs_grow_with_the_radius(any_space):
    spacing = _facet_spacing(any_space)
    small = any_space.radius_graph(0.9 * spacing)
    large = any_space.radius_graph(1.5 * spacing)
    assert np.all(np.isin(_edge_keys(small), _edge_keys(large)))


def test_radius_graphs_are_kept_per_radius(any_space):
    spacing = _facet_spacing(any_space)
    graph = any_space.radius_graph(1.5 * spacing)
    assert any_space.radius_graph(np.float32(1.5) * np.float64(spacing)) is graph
    assert any_space.radius_graph(1.5 * spacing) is graph
    assert any_space.radius_graph(0.9 * spacing) is not graph
    assert any_space.radius_graph(0.9 * spacing).n_edges < graph.n_edges


def test_radius_zero_gives_no_neighbors(any_space):
    assert any_space.radius_graph(0.0).n_edges == 0


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_radius_graph_rejects_invalid_radii(any_space, radius):
    with pytest.raises(ValueError, match="radius"):
        any_space.radius_graph(radius)


def test_radius_graphs_are_kept_for_the_most_recently_used_radii(any_space):
    first = any_space.radius_graph(0.05)
    assert any_space.radius_graph(0.05) is first
    for k in range(1, RadiusCache.KEEP):  # fills the cache, 0.05 still among the radii kept
        any_space.radius_graph(0.05 + 0.01 * k)
    assert any_space.radius_graph(0.05) is first  # now the most recently used
    any_space.radius_graph(0.5)  # puts out 0.06, the least recently used, not 0.05
    assert any_space.radius_graph(0.05) is first
    for k in range(RadiusCache.KEEP):  # puts out every radius above
        any_space.radius_graph(1e-3 * (k + 1))
    rebuilt = any_space.radius_graph(0.05)
    assert rebuilt is not first
    np.testing.assert_array_equal(rebuilt.indptr, first.indptr)
    np.testing.assert_array_equal(rebuilt.indices, first.indices)


def test_a_radius_cache_builds_each_value_once_while_kept():
    built = []
    cache = RadiusCache[float]()
    for limit in [1.0, 2.0, 1.0, *range(3, 3 + RadiusCache.KEEP), 1.0]:
        cache.get(float(limit), lambda x: built.append(x) or x)
    assert built == [1.0, 2.0, *range(3, 3 + RadiusCache.KEEP), 1.0]  # 1.0 put out, then again
    assert cache.kept(2.0) is None
    assert cache.kept(1.0) == 1.0
