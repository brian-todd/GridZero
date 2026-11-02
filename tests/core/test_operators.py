import copy
import pickle
from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike

from gridzero.core import (
    Adjacency,
    Ball,
    CellGraph,
    LatticeSpace,
    NeighborOperator,
    Space,
    gather,
    pull,
    push,
)

pytestmark = pytest.mark.usefixtures("each_impl")  # every test on both paths

X = np.array([1.0, 10.0, 100.0, 1000.0])

INTEGER_DTYPES: list[type[np.generic]] = [np.bool_, np.int8, np.int16, np.int32, np.int64]
INTEGER_DTYPES += [np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float16, np.float32, np.float64]
INT64 = np.iinfo(np.int64)


def _limits(dtype: DTypeLike) -> tuple[Any, Any]:
    """Return the smallest and largest value of a dtype, as its scalars (bools for bool)."""
    if dtype is np.bool_:
        return False, True
    scalars = np.dtype(dtype)
    info = np.finfo(scalars) if scalars.kind == "f" else np.iinfo(scalars)
    return scalars.type(info.min), scalars.type(info.max)


@pytest.fixture
def small_operator() -> NeighborOperator:
    """Four cells, checkable by hand: 0 -> 1 (weight 2), 0 -> 2 (weight 3), 1 -> 2 (weight 1).

    Cells 2 and 3 have no neighbors, and nothing has cell 0 or 3 as a neighbor.
    """
    graph = CellGraph.from_edges([0, 0, 1], [1, 2, 2], 4, edge_data={"w": [2.0, 3.0, 1.0]})
    return NeighborOperator(graph, "w")


@pytest.fixture(params=[True, False], ids=["weighted", "unweighted"])
def random_operator(request: pytest.FixtureRequest, any_space: Space, rng) -> NeighborOperator:
    """An irregular, directed operator on each space: a random 70% of the vertex adjacency.

    Weighted operators get weights of both signs.
    """
    graph = any_space.adjacency("vertex")
    keep = rng.random(graph.n_edges) < 0.7
    directed = CellGraph.from_edges(
        graph.sources[keep],
        graph.indices[keep],
        graph.n_cells,
        edge_data={"w": rng.normal(size=int(keep.sum()))},
    )
    return NeighborOperator(directed, "w" if request.param else None)


def _dense(op: NeighborOperator) -> np.ndarray:
    """Return the operator as a dense matrix, ``W[i, j] = w_ij``, built straight from its edges."""
    matrix = np.zeros((op.n_cells, op.n_cells))
    matrix[op.graph.sources, op.graph.indices] = 1.0 if op.weights is None else op.weights
    return matrix


# Hand-checked results on the small operator.


def test_pull_sums_weighted_neighbor_values(small_operator):
    np.testing.assert_array_equal(pull(X, small_operator), [320.0, 100.0, 0.0, 0.0])


def test_push_sends_weighted_values_to_neighbors(small_operator):
    np.testing.assert_array_equal(push(X, small_operator), [0.0, 2.0, 13.0, 0.0])


def test_pull_mean_is_the_weighted_average_and_fills_cells_without_neighbors(small_operator):
    result = pull(X, small_operator, "mean", fill=np.nan)
    np.testing.assert_array_equal(result, [64.0, 100.0, np.nan, np.nan])


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [("max", [300.0, 100.0, -1.0, -1.0]), ("min", [20.0, 100.0, -1.0, -1.0])],
)
def test_pull_extremes_are_taken_over_weighted_contributions(small_operator, reduce, expected):
    np.testing.assert_array_equal(pull(X, small_operator, reduce, fill=-1), expected)


def test_gather_returns_raw_neighbor_values_with_fill_in_unused_slots(small_operator):
    expected = [[10.0, 100.0], [100.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
    np.testing.assert_array_equal(gather(X, small_operator, fill=-1), expected)


def test_gather_can_select_cells_and_keeps_leading_axes(small_operator):
    x = np.stack([X, -X])
    result = gather(x, small_operator, cells=[1, 0])
    np.testing.assert_array_equal(result, [[[100, 0], [10, 100]], [[-100, 0], [-10, -100]]])


def test_query_pads_rows_with_minus_one_and_zero_weight(small_operator):
    idx, weight = small_operator.query()
    np.testing.assert_array_equal(idx, [[1, 2], [2, -1], [-1, -1], [-1, -1]])
    np.testing.assert_array_equal(weight, [[2, 3], [1, 0], [0, 0], [0, 0]])


def test_query_selects_rows_in_the_order_given(small_operator):
    idx, weight = small_operator.query([1, 1, 0])
    np.testing.assert_array_equal(idx, [[2, -1], [2, -1], [1, 2]])
    np.testing.assert_array_equal(weight, [[1, 0], [1, 0], [2, 3]])


def test_query_of_an_unweighted_operator_has_no_weights(small_operator):
    assert NeighborOperator(small_operator.graph).query().weight is None


def test_query_tables_are_read_only(small_operator):
    idx, weight = small_operator.query()
    with pytest.raises(ValueError, match="read-only"):
        idx[0, 0] = 3
    with pytest.raises(ValueError, match="read-only"):
        weight[0, 0] = 3.0


@pytest.mark.parametrize("cells", [[-1], [4], [0, 5]])
def test_query_rejects_cell_ids_outside_the_operator(small_operator, cells):
    with pytest.raises(IndexError, match="cell ids"):
        small_operator.query(cells)


@pytest.mark.parametrize("cells", [[[0, 1]], 2])
def test_query_rejects_cells_that_are_not_a_list_of_ids(small_operator, cells):
    with pytest.raises(ValueError, match="1-D array"):
        small_operator.query(cells)


def test_query_refuses_cells_that_are_not_integers(small_operator, not_cell_ids):
    ids, error = not_cell_ids
    with pytest.raises(error, match="cells"):
        small_operator.query(ids)


def test_size_and_weight_sums_describe_each_row(small_operator):
    assert small_operator.size == 2
    np.testing.assert_array_equal(small_operator.weight_sums, [5.0, 1.0, 0.0, 0.0])
    unweighted = NeighborOperator(small_operator.graph)
    np.testing.assert_array_equal(unweighted.weight_sums, [2.0, 1.0, 0.0, 0.0])


# Reference results on every space.


def test_pull_matches_the_dense_matrix_product(random_operator, rng):
    x = rng.normal(size=(2, 3, random_operator.n_cells))
    expected = np.einsum("ij,...j->...i", _dense(random_operator), x)
    np.testing.assert_allclose(pull(x, random_operator), expected, rtol=1e-12, atol=1e-12)


def test_push_matches_the_transposed_dense_matrix_product(random_operator, rng):
    x = rng.normal(size=(2, random_operator.n_cells))
    expected = np.einsum("ij,...i->...j", _dense(random_operator), x)
    np.testing.assert_allclose(push(x, random_operator), expected, rtol=1e-12, atol=1e-12)


def test_push_is_the_adjoint_of_pull(random_operator, rng):
    x = rng.normal(size=(3, random_operator.n_cells))
    y = rng.normal(size=(3, random_operator.n_cells))
    lhs = np.sum(pull(x, random_operator) * y)
    rhs = np.sum(x * push(y, random_operator))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


@pytest.mark.parametrize("reduce", ["mean", "max", "min"])
def test_pull_reductions_match_a_dense_reference(random_operator, rng, reduce):
    x = rng.normal(size=(2, random_operator.n_cells))
    matrix = _dense(random_operator)
    is_edge = np.zeros(matrix.shape, dtype=bool)
    is_edge[random_operator.graph.sources, random_operator.graph.indices] = True
    contributions = matrix * x[:, None, :]  # (2, receiving cell, neighbor)
    has_neighbors = is_edge.any(axis=1)
    if reduce == "mean":
        total_weight = matrix.sum(axis=1)
        safe = np.where(total_weight != 0, total_weight, 1.0)
        expected = np.where(total_weight != 0, contributions.sum(axis=2) / safe, -7.0)
    else:
        extreme = np.max if reduce == "max" else np.min
        masked = np.where(is_edge, contributions, -np.inf if reduce == "max" else np.inf)
        expected = np.where(has_neighbors, extreme(masked, axis=2), -7.0)
    result = pull(x, random_operator, reduce, fill=-7)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_gather_times_weights_sums_to_pull(random_operator, rng):
    x = rng.normal(size=(2, random_operator.n_cells))
    idx, weight = random_operator.query()
    weight = (idx >= 0) if weight is None else weight
    total = np.sum(gather(x, random_operator) * weight, axis=-1)
    np.testing.assert_allclose(total, pull(x, random_operator), rtol=1e-12, atol=1e-12)


def test_from_table_rebuilds_the_operator_from_its_query(random_operator):
    idx, weight = random_operator.query()
    rebuilt = NeighborOperator.from_table(idx, weight)
    np.testing.assert_array_equal(rebuilt.graph.indptr, random_operator.graph.indptr)
    np.testing.assert_array_equal(rebuilt.graph.indices, random_operator.graph.indices)
    if random_operator.weights is None:
        assert rebuilt.weights is None
    else:
        np.testing.assert_array_equal(rebuilt.weights, random_operator.weights)


# Result dtypes.


def test_pull_counts_live_neighbors_of_a_blinker_exactly():
    space = LatticeSpace((5, 5))
    state = np.zeros((5, 5), dtype=bool)
    state[2, 1:4] = True
    counts = pull(state.ravel(), Adjacency("vertex").compile(space))
    expected = [
        [0, 0, 0, 0, 0],
        [1, 2, 3, 2, 1],
        [1, 1, 2, 1, 1],
        [1, 2, 3, 2, 1],
        [0, 0, 0, 0, 0],
    ]
    assert counts.dtype == np.int64
    np.testing.assert_array_equal(space.grid_view(counts), expected)


def test_pull_counts_more_than_255_neighbors_without_overflow():
    space = LatticeSpace((20, 20))
    op = Ball(9.5).compile(space)
    assert op.degree.min() > 255
    counts = pull(np.ones(space.n_cells, dtype=np.uint8), op)
    np.testing.assert_array_equal(counts, op.degree)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
def test_pull_keeps_the_floating_dtype_of_its_input(small_operator, dtype, reduce):
    assert pull(X.astype(dtype), small_operator, reduce).dtype == dtype


def test_push_keeps_the_floating_dtype_of_its_input(small_operator):
    assert push(X.astype(np.float32), small_operator).dtype == np.float32


@pytest.mark.parametrize(
    ("weighted", "reduce", "dtype"),
    [
        (False, "sum", np.int64),
        (False, "max", np.int64),
        (False, "mean", np.float64),
        (True, "sum", np.float64),
        (True, "min", np.float64),
    ],
)
def test_pull_of_integers_is_exact_unless_weights_or_averages_need_floats(
    small_operator, weighted, reduce, dtype
):
    op = small_operator if weighted else NeighborOperator(small_operator.graph)
    assert pull(X.astype(np.int32), op, reduce).dtype == dtype


def test_gather_keeps_the_dtype_of_its_input(small_operator):
    state = np.array([True, False, True, True])
    result = gather(state, small_operator)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, [[False, True], [True, False], [False] * 2, [False] * 2])


# Invalid input.


@pytest.mark.parametrize("x", [np.zeros(3), np.zeros((4, 2)), np.float64(1.0)])
def test_operations_reject_fields_without_one_value_per_cell(small_operator, x):
    for operation in (pull, push, gather):
        with pytest.raises(ValueError, match="n_cells"):
            operation(x, small_operator)


def test_pull_rejects_unknown_reductions(small_operator):
    with pytest.raises(ValueError, match="reduce"):
        pull(X, small_operator, "median")  # ty: ignore[invalid-argument-type]


def test_pull_rejects_values_that_are_not_real_numbers(small_operator):
    with pytest.raises(TypeError, match="dtype"):
        pull(X.astype(np.complex128), small_operator)


@pytest.mark.parametrize(
    ("x", "fill"), [(X.astype(np.int64), np.nan), (X.astype(np.int64), 0.5), (X, "a")]
)
def test_pull_rejects_fill_values_the_result_cannot_hold(small_operator, x, fill):
    with pytest.raises(ValueError, match="fill"):
        pull(x, NeighborOperator(small_operator.graph), "max", fill=fill)


@pytest.mark.parametrize(
    ("dtype", "fill"),
    [
        (np.bool_, 2),
        (np.uint8, -1),
        (np.int8, 128),
        (np.uint64, -1),
        (np.int64, 0.5),
        (np.int64, np.nan),
        (np.float16, 1e5),
        (np.float32, 1e40),
        (np.float32, -1e40),
        (np.float64, "1.5"),
    ],
    ids=repr,
)
def test_gather_rejects_fill_values_the_input_dtype_cannot_hold(small_operator, dtype, fill):
    with pytest.raises(ValueError, match="fill"):
        gather(np.ones(4, dtype=dtype), small_operator, fill=fill)


@pytest.mark.parametrize(("dtype", "fill"), [(np.float16, 1e5), (np.float32, 1e40)])
def test_pull_rejects_float_fills_that_would_overflow(small_operator, dtype, fill):
    with pytest.raises(ValueError, match="fill"):
        pull(np.ones(4, dtype=dtype), small_operator, "max", fill=fill)


def test_float_fills_may_round_and_may_be_infinite(small_operator):
    # Cell 3 has no neighbors, so it gets the fill.
    result = pull(np.ones(4, dtype=np.float32), small_operator, "max", fill=0.1)
    assert result.dtype == np.float32
    assert result[3] == np.float32(0.1)
    assert gather(np.ones(4, dtype=np.float16), small_operator, fill=-np.inf)[3, 0] == -np.inf


# Results at the limits of each dtype: exact, or an error, never a wrong value.


@pytest.mark.parametrize(
    ("dtype", "reduce"),
    [(dtype, reduce) for dtype in INTEGER_DTYPES for reduce in ("sum", "push", "max", "min")]
    + [(dtype, reduce) for dtype in FLOAT_DTYPES for reduce in ("max", "min")],
    ids=lambda value: getattr(value, "__name__", value),
)
@pytest.mark.parametrize("extreme", ["smallest", "largest"])
def test_unweighted_results_at_the_limits_of_each_dtype_are_exact_or_raise(dtype, reduce, extreme):
    # Every cell holds the extreme value and has 4 neighbors, so each sum is 4 times it.
    op = Adjacency().compile(LatticeSpace((3, 3)))
    value = _limits(dtype)[extreme == "largest"]
    x = np.full(9, value, dtype=dtype)
    if reduce in ("sum", "push"):
        expected = 4 * int(value)
        compute = partial(pull if reduce == "sum" else push, x, op)
    else:
        expected = value if np.dtype(dtype).kind == "f" else int(value)
        compute = partial(pull, x, op, reduce)
    if np.dtype(dtype).kind != "f" and not INT64.min <= expected <= INT64.max:
        with pytest.raises(ValueError, match="int64 cannot hold"):
            compute()
        return
    result = compute()
    assert result.dtype == (dtype if np.dtype(dtype).kind == "f" else np.int64)
    assert result.tolist() == [expected] * 9


@pytest.mark.parametrize(("dtype", "value"), [(np.float16, 40000.0), (np.float32, 1e38)])
@pytest.mark.parametrize("reduce", ["sum", "push", "weighted max"])
def test_float_results_too_large_for_their_narrow_dtype_raise(dtype, value, reduce):
    # Each cell has 4 neighbors: a sum of 4 values, or one value at weight 4, fits float64 only.
    space = LatticeSpace((3, 3))
    x = np.full(9, value, dtype=dtype)
    if reduce == "weighted max":
        graph = space.adjacency()
        weighted = CellGraph(graph.indptr, graph.indices, {"w": np.full(graph.n_edges, 4.0)})
        compute = partial(pull, x, NeighborOperator(weighted, "w"), "max")
    else:
        compute = partial(pull if reduce == "sum" else push, x, Adjacency().compile(space))
    with pytest.raises(ValueError, match=f"does not fit in {np.dtype(dtype).name}"):
        compute()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: d.__name__)
def test_infinite_values_and_float64_overflow_give_inf(dtype):
    op = Adjacency().compile(LatticeSpace((3, 3)))
    x = np.ones(9, dtype=dtype)
    x[4] = np.inf  # the center, every edge cell's neighbor
    assert pull(x, op)[1] == np.inf
    huge = np.full(9, np.finfo(np.float64).max)
    with np.errstate(over="ignore"):  # NumPy warns on the reference path, Numba doesn't
        assert (pull(huge, op) == np.inf).all()


def test_64_bit_integers_well_inside_the_range_pull_without_complaint():
    op = Adjacency("vertex").compile(LatticeSpace((3, 3)))
    x = np.array([2**58, -(2**58), 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    np.testing.assert_array_equal(pull(x, op), x.sum() - x)  # every other cell is a neighbor
    largest = np.array([INT64.max, 5, 7, 0, 1, 2, 3, 4, 6], dtype=np.uint64)  # it fits int64
    np.testing.assert_array_equal(pull(largest, op, "max"), [7] + [INT64.max] * 8)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES + FLOAT_DTYPES, ids=lambda d: d.__name__)
def test_gather_keeps_the_limits_of_each_dtype(dtype):
    op = Adjacency().compile(LatticeSpace((3, 3)))
    x = np.resize(np.array(_limits(dtype), dtype=dtype), 9)
    out = gather(x, op)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, x[op.query().idx])


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_extremes_match_a_loop_over_cells_when_every_cell_has_neighbors(any_space, rng, reduce):
    # The adjacency of every space gives each cell a neighbor: the case without any fill.
    op = NeighborOperator(any_space.adjacency("vertex"), "distance")
    x = rng.normal(size=(2, op.n_cells))
    weights = op.weights
    assert weights is not None
    extreme = max if reduce == "max" else min
    expected = np.zeros_like(x)
    for cell in range(op.n_cells):
        start, stop = op.graph.indptr[cell], op.graph.indptr[cell + 1]
        for row in range(2):
            contributions = x[row, op.graph.indices[start:stop]] * weights[start:stop]
            expected[row, cell] = extreme(contributions)
    np.testing.assert_array_equal(pull(x, op, reduce), expected)


# Construction.


def test_from_table_ignores_weights_in_unused_slots():
    op = NeighborOperator.from_table([[1, -1], [-1, 0]], weight=[[2.0, np.nan], [99.0, 3.0]])
    np.testing.assert_array_equal(op.graph.indices, [1, 0])
    np.testing.assert_array_equal(op.weights, [2.0, 3.0])


@pytest.mark.parametrize(
    ("idx", "weight", "match"),
    [
        ([[1, 1], [0, -1]], None, "duplicate edge"),
        ([[-2, 1], [0, -1]], None, "cell ids"),
        ([[2, -1], [0, -1]], None, "cell ids"),
        ([1, 0], None, "2-D table"),
        ([[1], [0]], [1.0, 2.0], "weight has shape"),
    ],
)
def test_from_table_rejects_inconsistent_tables(idx, weight, match):
    with pytest.raises(ValueError, match=match):
        NeighborOperator.from_table(idx, weight)


def test_from_table_refuses_entries_that_are_not_integers(not_cell_ids):
    # A uint64 entry of 2**64 - 1 would otherwise wrap to -1 and pass for an unused slot.
    ids, error = not_cell_ids
    with pytest.raises(error, match="idx"):
        NeighborOperator.from_table(ids.reshape(-1, 1))


def test_operator_rejects_unknown_weight_attributes(small_operator):
    with pytest.raises(ValueError, match="no edge attribute 'area'"):
        NeighborOperator(small_operator.graph, "area")


def test_operator_rejects_weights_that_are_not_finite():
    graph = CellGraph.from_edges([0], [1], 2, edge_data={"w": [np.inf]})
    with pytest.raises(ValueError, match="finite"):
        NeighborOperator(graph, "w")


def test_operator_needs_a_cell_graph():
    with pytest.raises(TypeError, match="CellGraph"):
        NeighborOperator(np.eye(3))  # ty: ignore[invalid-argument-type]


def test_transpose_is_cached_and_transposing_twice_gives_the_operator_back(small_operator):
    transpose = small_operator.transpose()
    assert small_operator.transpose() is transpose
    assert transpose.transpose() is small_operator
    assert transpose.weight == small_operator.weight


def test_operator_without_edges_pulls_zero_sums_and_fills_averages():
    op = NeighborOperator(CellGraph.from_edges([], [], 3))
    assert op.size == 0
    np.testing.assert_array_equal(pull(np.ones(3), op), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(pull(np.ones(3), op, "mean", fill=-1), [-1.0, -1.0, -1.0])
    assert gather(np.ones(3), op).shape == (3, 0)


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
def test_operations_on_no_cells_keep_their_components(reduce):
    op = NeighborOperator(CellGraph([0], []))
    for dtype in (np.float64, np.int64):
        x = np.zeros((2, 3, 0), dtype=dtype)
        assert pull(x, op, reduce).shape == (2, 3, 0)
        assert push(x, op).shape == (2, 3, 0)
        assert gather(x, op).shape == (2, 3, 0, 0)


def test_operator_survives_a_pickle_round_trip(small_operator):
    small_operator.query()  # fill the caches, which are rebuilt rather than pickled
    restored = pickle.loads(pickle.dumps(small_operator))
    assert restored.weight == small_operator.weight
    np.testing.assert_array_equal(restored.weights, small_operator.weights)
    np.testing.assert_array_equal(pull(X, restored), pull(X, small_operator))
    assert not restored.query().idx.flags.writeable


def test_copying_an_operator_returns_the_same_immutable_object(small_operator):
    assert copy.copy(small_operator) is small_operator
    assert copy.deepcopy(small_operator) is small_operator


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_a_zero_maximum_or_minimum_is_positive_zero(make_icosphere, reduce):
    # Whatever zeros the neighbors hold, in any order: NumPy's float64 maximum keeps the later
    # of two equal values, and Numba's max the earlier.
    op = Adjacency(weight="boundary").compile(make_icosphere(1))
    for x in (np.where(np.arange(op.n_cells) % 2, -0.0, 0.0), np.full(op.n_cells, -0.0)):
        y = pull(x, op, reduce)
        np.testing.assert_array_equal(y, 0.0)
        assert not np.signbit(y).any()
