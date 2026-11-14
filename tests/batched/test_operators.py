"""`batched`'s operators against `core`'s: integers bit for bit, floats to float32's precision."""

import jax
import numpy as np
import pytest

from gridzero import batched
from gridzero.core import (
    Adjacency,
    Ball,
    Hops,
    LatticeSpace,
    NeighborOperator,
    gather,
    icosphere,
    pull,
    push,
)

REDUCTIONS = ("sum", "mean", "max", "min")


def directed(n_cells, size, seed, weighted=False):
    """An operator whose edges go one way, with rows of every length, some empty."""
    rng = np.random.default_rng(seed)
    idx = np.full((n_cells, size), -1)
    for cell in range(n_cells):
        degree = int(rng.integers(0, size + 1))
        idx[cell, :degree] = np.sort(rng.choice(n_cells, size=degree, replace=False))
    weight = rng.normal(size=idx.shape) * (idx >= 0) if weighted else None
    return NeighborOperator.from_table(idx, weight)


OPERATORS = {
    "squares": lambda: Adjacency().compile(LatticeSpace((5, 4), periodic=False)),
    "hexes, two hops": lambda: Hops(2).compile(LatticeSpace((6, 5), tiling="hex")),
    "planet": lambda: Adjacency().compile(icosphere(2)),
    "planet, by distance": lambda: Ball(0.7, weight="distance").compile(icosphere(2)),
    "one way": lambda: directed(23, 5, seed=0),
    "one way, weighted": lambda: directed(23, 5, seed=1, weighted=True),
}


@pytest.fixture(params=list(OPERATORS))
def op(request):
    return OPERATORS[request.param]()


def test_the_tables_are_the_operators_with_n_cells_in_unused_slots(op):
    tables = batched.operator(op)
    for found, of in ((tables.idx, op), (tables.back, op.transpose())):
        idx = of.query().idx
        assert found.dtype == np.int32
        np.testing.assert_array_equal(found, np.where(idx >= 0, idx, op.n_cells))
    if op.weight is None:
        assert tables.weight is None
        assert tables.back_weight is None
    else:
        assert tables.weight is not None
        assert tables.back_weight is not None
        assert tables.weight.dtype == np.float32
        np.testing.assert_allclose(tables.weight, op.query().weight, rtol=1e-6)
        np.testing.assert_allclose(tables.back_weight, op.transpose().query().weight, rtol=1e-6)


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_pull_of_integers_and_booleans_is_cores(op, reduce, rng):
    tables = batched.operator(op)
    for x in (rng.integers(-50, 50, size=(3, op.n_cells)), rng.random(op.n_cells) < 0.4):
        want = pull(x, op, reduce, fill=-7)
        found = batched.pull(x, tables, reduce, fill=-7)
        assert found.shape == want.shape
        if op.weight is None and reduce != "mean":
            assert found.dtype == np.int32
            np.testing.assert_array_equal(found, want)  # exact
        else:
            assert found.dtype == np.float32
            np.testing.assert_allclose(found, want, rtol=2e-5, atol=1e-4)


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_pull_of_floats_is_cores_to_float32(op, reduce, rng):
    x = rng.normal(size=(2, 3, op.n_cells)).astype(np.float32)
    found = batched.pull(x, batched.operator(op), reduce, fill=2.5)
    assert found.dtype == np.float32
    np.testing.assert_allclose(found, pull(x, op, reduce, fill=2.5), rtol=2e-5, atol=1e-5)


def test_push_is_cores_and_the_transpose_of_pull(op, rng):
    tables = batched.operator(op)
    counts = rng.integers(0, 9, size=(2, op.n_cells))
    if op.weight is None:
        np.testing.assert_array_equal(batched.push(counts, tables), push(counts, op))
    x = rng.normal(size=op.n_cells).astype(np.float32)
    y = rng.normal(size=op.n_cells).astype(np.float32)
    np.testing.assert_allclose(batched.push(x, tables), push(x, op), rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.dot(batched.pull(x, tables), y), np.dot(x, batched.push(y, tables)), rtol=1e-4
    )


def test_push_conserves_what_an_operator_of_shares_sends(rng):
    op = directed(30, 4, seed=3)
    amounts = rng.random(30).astype(np.float32)
    degree = np.maximum(op.degree, 1)
    sent = np.where(op.degree > 0, amounts / degree, 0).astype(np.float32)
    received = batched.push(sent, batched.operator(op))
    assert float(received.sum()) == pytest.approx(float(amounts[op.degree > 0].sum()), rel=1e-5)


def test_gather_is_cores_for_every_dtype(op, rng):
    tables = batched.operator(op)
    for x, fill in (
        (rng.integers(-9, 9, size=(2, op.n_cells)).astype(np.int8), -3),
        (rng.random(op.n_cells) < 0.5, False),
        (rng.normal(size=op.n_cells).astype(np.float32), np.nan),
    ):
        found = batched.gather(x, tables, fill=fill)
        assert found.dtype == x.dtype
        np.testing.assert_array_equal(found, gather(x, op, fill=fill))


def test_a_batch_is_each_state_alone(op, rng):
    tables = batched.operator(op)
    x = rng.integers(0, 5, size=(4, op.n_cells))
    together = jax.jit(jax.vmap(lambda row: batched.pull(row, tables, "max")))(x)
    for row, found in zip(x, together, strict=True):
        np.testing.assert_array_equal(found, batched.pull(row, tables, "max"))


def test_cells_without_neighbors_get_the_fill_except_in_sums():
    op = NeighborOperator.from_table(np.array([[1, 2], [-1, -1], [0, -1]]))
    tables = batched.operator(op)
    x = np.array([4, 5, 6])
    assert batched.pull(x, tables, "sum", fill=9).tolist() == [11, 0, 4]
    assert batched.pull(x, tables, "max", fill=9).tolist() == [6, 9, 4]
    assert batched.pull(x, tables, "min", fill=9).tolist() == [5, 9, 4]
    assert batched.pull(x, tables, "mean", fill=9).tolist() == [5.5, 9.0, 4.0]


def test_a_zero_extreme_is_positive_zero():
    op = NeighborOperator.from_table(np.array([[1], [0]]))
    found = batched.pull(np.array([-0.0, -0.0], np.float32), batched.operator(op), "max")
    assert not np.signbit(np.asarray(found)).any()


def test_bad_arguments_raise():
    op = Adjacency().compile(LatticeSpace((3, 3)))
    tables = batched.operator(op)
    with pytest.raises(TypeError, match="NeighborOperator"):
        batched.operator(op.graph)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="reduce"):
        batched.pull(np.zeros(9), tables, "median")  # ty: ignore[invalid-argument-type]
    for call in (batched.pull, batched.push, batched.gather):
        with pytest.raises(ValueError, match="shape"):
            call(np.zeros(8), tables)
