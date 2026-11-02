import copy
import pickle

import numpy as np
import pytest

from gridzero.core import CellGraph


@pytest.fixture
def path_graph() -> CellGraph:
    """Symmetric path 0 - 1 - 2 - 3; each edge is weighted by its smaller endpoint plus one."""
    src = [0, 1, 1, 2, 2, 3]
    dst = [1, 0, 2, 1, 3, 2]
    return CellGraph.from_edges(src, dst, 4, edge_data={"w": np.minimum(src, dst) + 1})


def test_from_edges_sorts_each_row_by_target():
    graph = CellGraph.from_edges([2, 0, 0, 2], [0, 2, 1, 1], 3)
    np.testing.assert_array_equal(graph.indptr, [0, 2, 2, 4])
    np.testing.assert_array_equal(graph.indices, [1, 2, 0, 1])


def test_from_edges_rejects_duplicate_edges():
    with pytest.raises(ValueError, match=r"duplicate edge \(0, 1\)"):
        CellGraph.from_edges([0, 0, 0], [1, 2, 1], 3, edge_data={"w": [10.0, 20.0, 99.0]})


def test_edge_data_stays_aligned_with_edges_after_sorting():
    graph = CellGraph.from_edges([1, 0, 1], [0, 1, 2], 3, edge_data={"w": [10.0, 1.0, 12.0]})
    weights = dict(
        zip(zip(graph.sources, graph.indices, strict=True), graph.edge_data["w"], strict=True)
    )
    assert weights == {(0, 1): 1.0, (1, 0): 10.0, (1, 2): 12.0}


@pytest.mark.parametrize(
    ("sources", "targets"),
    [([0, 3], [1, 0]), ([0, -1], [1, 0])],
    ids=["too-large", "negative"],
)
def test_from_edges_rejects_endpoints_outside_the_graph(sources, targets):
    with pytest.raises(ValueError, match="edge endpoints must lie in"):
        CellGraph.from_edges(sources, targets, 3)


def test_from_edges_rejects_misaligned_edge_data():
    with pytest.raises(ValueError, match="one value per edge"):
        CellGraph.from_edges([0, 1], [1, 0], 2, edge_data={"w": [1.0]})


def test_constructor_rejects_inconsistent_csr_arrays():
    with pytest.raises(ValueError, match="number of edges"):
        CellGraph(indptr=[0, 1, 3], indices=[1, 0])


@pytest.mark.parametrize(
    ("indptr", "indices"),
    [
        ([0, 2, 2, 2], [2, 1]),
        ([0, 2, 2, 2], [1, 1]),
        ([0, 1, 1, 3], [2, 1, 0]),
        ([0, 0, 0, 2], [0, 0]),
    ],
    ids=["unsorted-row", "repeated-edge", "unsorted-after-an-empty-row", "repeated-in-last-row"],
)
def test_constructor_rejects_rows_that_are_not_canonical(indptr, indices):
    with pytest.raises(ValueError, match="strictly increasing"):
        CellGraph(indptr=indptr, indices=indices)


@pytest.mark.parametrize("bad", ["indptr", "indices"])
def test_graphs_built_inside_the_package_check_dtypes_as_the_constructor_does(bad):
    # Builders hand fresh arrays to _owned, which keeps them without copying.
    arrays = {"indptr": np.array([0, 1, 1]), "indices": np.array([1])}
    arrays[bad] = arrays[bad].astype(np.float64)  # never truncated into ids
    with pytest.raises(TypeError, match=bad):
        CellGraph._owned(arrays["indptr"], arrays["indices"])


@pytest.mark.parametrize(
    "indptr", [[0, 1, 2, 2], [0, 1, 1, 2], [0, 0, 1, 2]], ids=["full-rows", "gap", "empty-first"]
)
def test_rows_may_restart_at_low_targets(indptr):
    # The canonical-row check compares edges within a row, never across a row boundary, also
    # when empty rows lie between.
    graph = CellGraph(indptr=indptr, indices=[2, 0])
    np.testing.assert_array_equal(graph.indices, [2, 0])


def test_degree_and_sources_describe_the_rows(path_graph):
    np.testing.assert_array_equal(path_graph.degree, [1, 2, 2, 1])
    np.testing.assert_array_equal(path_graph.sources, [0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(path_graph.neighbors(2), [1, 3])


@pytest.mark.parametrize("cell", [-1, -4, 4])
def test_neighbors_rejects_ids_outside_the_graph(path_graph, cell):
    with pytest.raises(IndexError, match="out of range"):
        path_graph.neighbors(cell)


def test_padded_table_fills_short_rows_with_the_fill_value(path_graph):
    np.testing.assert_array_equal(path_graph.padded(), [[1, -1], [0, 2], [1, 3], [2, -1]])
    np.testing.assert_array_equal(path_graph.padded(fill=9)[0], [1, 9])


@pytest.fixture
def directed_graph() -> CellGraph:
    """Weighted one-way edges 0->1, 0->2, 2->1, 2->0 and 3->1; cell 1 has no out-edges."""
    return CellGraph.from_edges(
        [0, 0, 2, 2, 3], [1, 2, 1, 0, 1], 4, edge_data={"w": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )


def test_transpose_reverses_edges_and_carries_attributes(directed_graph):
    reverse = directed_graph.transpose()
    edges = set(zip(reverse.sources, reverse.indices, reverse.edge_data["w"], strict=True))
    assert edges == {(1, 0, 1.0), (2, 0, 2.0), (1, 2, 3.0), (0, 2, 4.0), (1, 3, 5.0)}


def test_transpose_is_the_exact_matrix_transpose(directed_graph):
    expected = directed_graph.to_scipy("w").T.tocsr()
    expected.sort_indices()
    reverse = directed_graph.transpose()
    np.testing.assert_array_equal(reverse.indptr, expected.indptr)
    np.testing.assert_array_equal(reverse.indices, expected.indices)
    np.testing.assert_array_equal(reverse.edge_data["w"], expected.data)


def test_transpose_is_adjoint_for_pushing_values(directed_graph, rng):
    # Pull through the graph, push through its transpose: <A x, y> == <x, A^T y>.
    x, y = rng.random(4), rng.random(4)
    pull = directed_graph.to_scipy("w") @ x
    push = directed_graph.transpose().to_scipy("w") @ y
    assert np.dot(pull, y) == pytest.approx(np.dot(x, push), rel=1e-12)


def test_transpose_is_cached_and_transposing_twice_returns_the_graph(directed_graph):
    reverse = directed_graph.transpose()
    assert directed_graph.transpose() is reverse
    assert reverse.transpose() is directed_graph


def test_is_symmetric_distinguishes_one_way_edges(path_graph):
    assert path_graph.is_symmetric()
    assert not CellGraph.from_edges([0], [1], 2).is_symmetric()


def test_has_self_loops_detects_a_cell_listed_as_its_own_neighbor(path_graph):
    assert not path_graph.has_self_loops()
    assert CellGraph.from_edges([0, 1], [0, 0], 2).has_self_loops()


def test_to_scipy_matches_the_dense_weighted_adjacency(path_graph):
    expected = np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 3, 0]], dtype=float)
    np.testing.assert_array_equal(path_graph.to_scipy("w").toarray(), expected)
    np.testing.assert_array_equal(path_graph.to_scipy().toarray(), expected > 0)


def test_graph_arrays_are_read_only(path_graph):
    for array in (path_graph.indptr, path_graph.indices, path_graph.edge_data["w"]):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_graph_survives_a_pickle_round_trip(directed_graph):
    directed_graph.transpose()  # populate the cache, which must not break pickling
    restored = pickle.loads(pickle.dumps(directed_graph))
    np.testing.assert_array_equal(restored.indptr, directed_graph.indptr)
    np.testing.assert_array_equal(restored.indices, directed_graph.indices)
    np.testing.assert_array_equal(restored.edge_data["w"], directed_graph.edge_data["w"])
    for array in (restored.indptr, restored.indices, restored.edge_data["w"]):
        assert not array.flags.writeable


def test_copying_a_graph_returns_the_same_immutable_object(path_graph):
    assert copy.copy(path_graph) is path_graph
    assert copy.deepcopy(path_graph) is path_graph


def test_graph_does_not_alias_caller_arrays():
    indices = np.array([1, 0])
    graph = CellGraph(indptr=[0, 1, 2], indices=indices)
    indices[0] = 0
    np.testing.assert_array_equal(graph.indices, [1, 0])


GRAPH_BUILDERS = {
    "indptr": lambda ids: CellGraph(ids, []),
    "indices": lambda ids: CellGraph([0, 1, 2], ids),
    "sources": lambda ids: CellGraph.from_edges(ids, [1, 0], 2),
    "targets": lambda ids: CellGraph.from_edges([1, 0], ids, 2),
}


@pytest.mark.parametrize("which", list(GRAPH_BUILDERS))
def test_graphs_refuse_cell_ids_that_are_not_integers(not_cell_ids, which):
    ids, error = not_cell_ids
    with pytest.raises(error, match=which):
        GRAPH_BUILDERS[which](ids)


def test_from_edges_needs_an_integer_cell_count(not_an_integer):
    with pytest.raises(TypeError, match="n_cells must be an integer"):
        CellGraph.from_edges([], [], not_an_integer)


def test_neighbors_rejects_a_cell_that_is_not_an_integer(path_graph, not_an_integer):
    with pytest.raises(TypeError, match="cell id must be an integer"):
        path_graph.neighbors(not_an_integer)


def test_to_scipy_can_count_edges_exactly_in_integers(path_graph):
    matrix = path_graph.to_scipy(dtype=np.int64)
    assert matrix.dtype == np.int64
    np.testing.assert_array_equal(matrix.toarray(), path_graph.to_scipy().toarray())
    with pytest.raises(ValueError, match="floating dtype"):
        path_graph.to_scipy("w", dtype=np.int64)
