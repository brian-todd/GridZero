"""Compiled neighborhoods and the operations on them: pull, push, and gather.

A `NeighborOperator` lists, for every cell ``i``, its neighbors ``j`` and a weight ``w_ij``.
`pull` aggregates neighbor values into each cell, `push` sends each cell's value out to its
neighbors (the exact transpose of pull), and `gather` returns the raw neighbor values in a
padded table.

Each operation has a NumPy/SciPy reference implementation and a Numba fast path with
bit-identical results. `gridzero.core.implementation` chooses between them.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from gridzero.core import _kernels
from gridzero.core._arrays import as_rows, check_cells, int_array, read_only
from gridzero.core.graph import CellGraph
from gridzero.core.implementations import fast

type Reduction = Literal["sum", "mean", "max", "min"]
"""How `pull` combines the contributions ``w_ij * x[..., j]`` of a cell's neighbors."""

_REDUCTIONS = ("sum", "mean", "max", "min")

_INT64 = np.iinfo(np.int64)


class Neighbors(NamedTuple):
    """The neighbors of a batch of cells, as padded tables.

    Attributes
    ----------
    idx : ndarray of int64, shape (n_query, size)
        Neighbor cell ids of each queried cell, in ascending order. Unused slots are -1.
    weight : ndarray of float64, shape (n_query, size), or None
        The matching weights, 0 in unused slots. None if the operator is unweighted.
    """

    idx: NDArray[np.int64]
    weight: NDArray[np.float64] | None


@dataclass(frozen=True, eq=False)
class NeighborOperator:
    """A compiled neighborhood: each cell's neighbors and their weights, in CSR form.

    Row ``i`` of ``graph`` lists the neighbors ``j`` of cell ``i``, and the edge ``i -> j``
    carries the weight ``w_ij``. `pull` computes ``y_i = sum_j w_ij x_j``, and `push` is its
    transpose, ``y_j = sum_i w_ij x_i``. Slots in padded tables follow CSR order, so slot ``k``
    holds a cell's ``k``-th smallest neighbor id, not a fixed direction.

    The operator is immutable, so copying it returns the same object. Its transpose and
    padded tables are computed on first use and cached.

    Parameters
    ----------
    graph : CellGraph
        Neighbor structure, with any per-edge attributes.
    weight : str, optional
        Name of the edge attribute in ``graph.edge_data`` to use as weights, such as
        ``"boundary"``. If omitted, every weight is 1. Weights must be finite.

    Raises
    ------
    TypeError
        If ``graph`` is not a `CellGraph`.
    ValueError
        If ``weight`` names a missing attribute, or a weight is not finite.
    """

    graph: CellGraph
    weight: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, CellGraph):
            raise TypeError(f"graph must be a CellGraph, got {type(self.graph).__name__}")
        if self.weight is not None:
            if self.weight not in self.graph.edge_data:
                names = sorted(self.graph.edge_data)
                raise ValueError(f"the graph has no edge attribute {self.weight!r}; it has {names}")
            if not np.all(np.isfinite(self.graph.edge_data[self.weight])):
                raise ValueError(f"weights from edge attribute {self.weight!r} must be finite")

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor. Cached tables and matrices are recomputed on demand.
        return type(self), (self.graph, self.weight)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    @classmethod
    def from_table(cls, idx: ArrayLike, weight: ArrayLike | None = None) -> Self:
        """Build an operator from padded neighbor tables, the form a custom query returns.

        Parameters
        ----------
        idx : array_like of int, shape (n_cells, m)
            Row ``i`` lists the neighbors of cell ``i`` in any order. -1 marks an unused slot.
        weight : array_like of float, shape (n_cells, m), optional
            Weight of each neighbor. Values in unused slots are ignored. If omitted, the
            operator is unweighted.

        Returns
        -------
        NeighborOperator
            Operator over ``n_cells`` cells. Weights, if given, are stored as the edge
            attribute ``"weight"``.

        Raises
        ------
        TypeError
            If ``idx`` does not hold integers.
        ValueError
            If the tables have the wrong shape, an entry is not a cell id or -1, or a row lists
            the same neighbor twice.
        """
        idx = int_array(idx, "idx")
        if idx.ndim != 2:
            raise ValueError(f"idx must be a 2-D table of shape (n_cells, m), got {idx.shape}")
        n_cells = idx.shape[0]
        if np.any((idx < -1) | (idx >= n_cells)):
            raise ValueError(f"idx entries must be cell ids in [0, {n_cells}) or -1")
        valid = idx >= 0
        rows = np.broadcast_to(np.arange(n_cells, dtype=np.int64)[:, None], idx.shape)
        edge_data = None
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float64)
            if weight.shape != idx.shape:
                raise ValueError(f"weight has shape {weight.shape}, expected {idx.shape}")
            edge_data = {"weight": weight[valid]}
        graph = CellGraph.from_edges(rows[valid], idx[valid], n_cells, edge_data=edge_data)
        return cls(graph, None if weight is None else "weight")

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.graph.n_cells

    @property
    def n_edges(self) -> int:
        """Number of (cell, neighbor) pairs."""
        return self.graph.n_edges

    @property
    def degree(self) -> NDArray[np.int64]:
        """Number of neighbors of each cell: int64, shape (n_cells,)."""
        return self.graph.degree

    @cached_property
    def size(self) -> int:
        """Largest number of neighbors of any cell: the width of padded tables."""
        return int(self.degree.max()) if self.n_cells else 0

    @property
    def weights(self) -> NDArray[np.float64] | None:
        """Weight of each edge: float64, shape (n_edges,), aligned with ``graph.indices``.

        None if the operator is unweighted.
        """
        return None if self.weight is None else self.graph.edge_data[self.weight]

    @cached_property
    def weight_sums(self) -> NDArray[np.float64]:
        """Total weight of each cell's neighbors: float64, shape (n_cells,).

        Equal to ``degree`` for an unweighted operator.
        """
        weights = self.weights
        if weights is None:
            return read_only(self.degree.astype(np.float64))
        sums = np.bincount(self.graph.sources, weights=weights, minlength=self.n_cells)
        return read_only(sums)

    def transpose(self) -> Self:
        """Return the operator with every edge reversed, keeping its weight.

        The transpose is computed once and cached, and transposing it again returns this
        operator. `push` through an operator is `pull` through its transpose.

        Returns
        -------
        NeighborOperator
            Operator whose row ``j`` lists every cell ``i`` that has ``j`` as a neighbor.
        """
        return self._transpose

    @cached_property
    def _transpose(self) -> Self:
        """The transpose, cached both ways: the transpose's own transpose is this operator."""
        reverse = type(self)(self.graph.transpose(), self.weight)
        reverse.__dict__["_transpose"] = self
        return reverse

    def query(self, cells: ArrayLike | None = None) -> Neighbors:
        """Return the neighbors of some cells as padded tables.

        Parameters
        ----------
        cells : array_like of int, shape (n_query,), optional
            Cell ids to look up, in ``[0, n_cells)``. Defaults to every cell, in id order.

        Returns
        -------
        Neighbors
            Tables of width `size`: neighbor ids with -1 in unused slots, and weights (or None
            if the operator is unweighted). Tables for all cells are cached and read-only.

        Raises
        ------
        TypeError
            If ``cells`` does not hold integers.
        ValueError
            If ``cells`` is not 1-D.
        IndexError
            If a cell id is out of range. Negative ids are not counted from the end.
        """
        if cells is None:
            return Neighbors(self._idx_table, self._weight_table)
        cells = check_cells(cells, self.n_cells)
        weight = None if self._weight_table is None else self._weight_table[cells]
        return Neighbors(self._idx_table[cells], weight)

    @cached_property
    def _idx_table(self) -> NDArray[np.int64]:
        """Neighbor ids of every cell, shape (n_cells, size), -1 in unused slots."""
        return read_only(self.graph.padded(fill=-1))

    @cached_property
    def _weight_table(self) -> NDArray[np.float64] | None:
        """Weights of `_idx_table`'s slots, float64 of the same shape, 0 in unused slots.

        None if the operator is unweighted.
        """
        if self.weight is None:
            return None
        table = np.zeros(self._idx_table.shape)
        table[self._idx_table >= 0] = self.weights  # row-major order matches CSR order
        return read_only(table)

    @cached_property
    def _stay_slots(self) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64]]:
        """Padded tables with a first slot for the cell itself, the layout of action spaces.

        Returns the cell in each slot, int64 of shape (n_cells, size + 1), with -1 in unused
        slots; whether each slot is used, bool of the same shape; and the flat position of each
        edge's slot in such a table, int64 of shape (n_edges,), in CSR order.
        """
        n_cells, n_slots = self.n_cells, self.size + 1
        targets = np.empty((n_cells, n_slots), dtype=np.int64)
        targets[:, 0] = np.arange(n_cells)
        targets[:, 1:] = self._idx_table
        sources = self.graph.sources
        slots = np.arange(self.n_edges) - self.graph.indptr[sources] + 1  # position in row, + 1
        return read_only(targets), read_only(targets >= 0), read_only(sources * n_slots + slots)

    @cached_property
    def _csr(self) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.float64] | None]:
        """The CSR arrays as the Numba kernels take them.

        Returns indptr and indices viewed as uint64 (they are never negative), of shapes
        (n_cells + 1,) and (n_edges,), and the weights, float64 of shape (n_edges,), or None if
        the operator is unweighted.
        """
        graph = self.graph
        return graph.indptr.view(np.uint64), graph.indices.view(np.uint64), self.weights

    @cached_property
    def _float_matrix(self) -> sp.csr_array:
        """The operator as a float64 SciPy matrix of its weights (ones if unweighted)."""
        return self.graph.to_scipy(self.weight)

    @cached_property
    def _int_matrix(self) -> sp.csr_array:
        """The operator's structure as an int64 SciPy matrix of ones, for exact sums."""
        return self.graph.to_scipy(dtype=np.int64)

    @cached_property
    def _nonnegative(self) -> bool:
        """Whether no weight is negative, as moving amounts in proportion to weights needs."""
        weights = self.weights
        return weights is None or bool(weights.min(initial=0.0) >= 0)

    @cached_property
    def _source_incidence(self) -> sp.csr_array:
        """Row ``i`` has a 1 for each edge leaving cell ``i``: shape (n_cells, n_edges)."""
        n_edges = self.n_edges
        return sp.csr_array(
            (np.ones(n_edges), np.arange(n_edges), self.graph.indptr.copy()),
            shape=(self.n_cells, n_edges),
        )

    @cached_property
    def _target_incidence(self) -> sp.csr_array:
        """Row ``j`` has a 1 for each edge entering cell ``j``: shape (n_cells, n_edges)."""
        entering = np.argsort(self.graph.indices, kind="stable")  # edge ids grouped by target
        return sp.csr_array(
            (np.ones(self.n_edges), entering, self.transpose().graph.indptr.copy()),
            shape=(self.n_cells, self.n_edges),
        )

    @cached_property
    def _segments(self) -> tuple[NDArray[np.int64], NDArray[np.bool_] | None]:
        """Where each non-empty row starts in ``graph.indices``, and which rows are non-empty.

        Returns the starts, int64 of shape (n_nonempty,), and the mask, bool of shape
        (n_cells,), which is None when every cell has a neighbor, so reductions can skip it.
        """
        has_neighbors = self.degree > 0
        starts = read_only(self.graph.indptr[:-1][has_neighbors])
        if self.n_cells and has_neighbors.all():
            return starts, None
        return starts, read_only(has_neighbors)


def pull(
    x: ArrayLike,
    op: NeighborOperator,
    reduce: Reduction = "sum",
    *,
    fill: float = 0,
) -> NDArray:
    """Aggregate neighbor values into each cell.

    Each neighbor ``j`` of cell ``i`` contributes ``w_ij * x[..., j]``. The contributions are
    then summed, averaged by weight, or reduced to their maximum or minimum.

    Parameters
    ----------
    x : array_like of bool, int, or float, shape (..., n_cells)
        Per-cell values. Leading axes (components, batches) are carried through.
    op : NeighborOperator
        Compiled neighborhood.
    reduce : {"sum", "mean", "max", "min"}, default "sum"
        ``"sum"`` gives ``y_i = sum_j w_ij x_j``. ``"mean"`` divides that by ``sum_j w_ij``,
        a weighted average. ``"max"`` and ``"min"`` give the extreme contribution, +0.0 if
        it is a zero of either sign.
    fill : float, default 0
        Result for a cell with no neighbors, or with zero total weight for ``"mean"``. A sum
        over no neighbors is always 0. It must fit the result dtype: exactly for integers, and
        without overflowing for floats, which may round. NaN therefore needs a floating result.

    Returns
    -------
    ndarray, shape (..., n_cells)
        Floating ``x`` keeps its dtype: sums are accumulated in float64 and rounded once.
        Boolean and integer ``x`` give exact int64 results on an unweighted operator, and
        float64 results on a weighted operator or for ``"mean"``. Infinite values in ``x``
        give infinite results, and so does float64 arithmetic that overflows, as in NumPy.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not ``n_cells`` long, ``reduce`` is unknown, ``fill``
        does not fit the result dtype, ``x`` holds 64-bit integers whose exact result might
        not fit in int64, or ``x`` is float16 or float32 and a result, finite in float64, is
        too large for its dtype. Convert such ``x`` to float64.
    TypeError
        If ``x`` is not boolean, integer, or floating.
    """
    x = _check_field(x, op.n_cells)
    if reduce not in _REDUCTIONS:
        raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
    # Counts and integer sums stay exact in int64. Everything else accumulates in float64.
    exact = x.dtype.kind in "biu" and op.weight is None and reduce != "mean"
    dtype = x.dtype if x.dtype.kind == "f" else np.dtype(np.int64 if exact else np.float64)
    if exact and x.dtype.itemsize == 8:  # only 64-bit integers can overflow int64
        _check_int64(x, op.size if reduce == "sum" else 1)
    if reduce in ("sum", "mean"):
        if fast():
            total = _row_sums(op, x, exact)
        else:
            total = _matvec(op._int_matrix if exact else op._float_matrix, x, exact)
        if reduce == "sum":
            return _narrow(total, dtype)
        out = np.full(total.shape, _fill_value(fill, dtype), dtype=np.float64)
        np.divide(total, op.weight_sums, out=out, where=op.weight_sums != 0)
        return _narrow(out, dtype)
    fill_value = _fill_value(fill, dtype)
    if fast():
        return _row_extremes(op, x, reduce == "max", fill_value, dtype)
    values = np.take(x, op.graph.indices, axis=-1)  # ~2x faster than x[..., indices]
    if op.weight is not None:
        values = values * op.weights
    reducer = np.maximum if reduce == "max" else np.minimum
    starts, has_neighbors = op._segments  # each segment ends where the next one starts
    if not starts.size:
        return np.full(x.shape, fill_value)
    extremes = reducer.reduceat(values, starts, axis=-1)
    if extremes.dtype.kind == "f":
        extremes += 0.0  # a zero result is +0.0, whichever zeros its row held
    if has_neighbors is None:
        return _narrow(extremes, dtype)
    out = np.full(x.shape, fill_value)
    out[..., has_neighbors] = _narrow(extremes, dtype)
    return out


def push(x: ArrayLike, op: NeighborOperator) -> NDArray:
    """Send each cell's value out to its neighbors: the transpose of `pull`.

    Cell ``j`` receives ``y_j = sum_i w_ij x_i`` over every cell ``i`` that has ``j`` as a
    neighbor, so ``dot(pull(x, op), y) == dot(x, push(y, op))``. Use it to move conserved
    quantities as fluxes. It is computed as a pull through the cached transpose.

    Parameters
    ----------
    x : array_like of bool, int, or float, shape (..., n_cells)
        Per-cell values to send. Leading axes are carried through.
    op : NeighborOperator
        Compiled neighborhood.

    Returns
    -------
    ndarray, shape (..., n_cells)
        Values received by each cell, with the same dtype rules as a `pull` sum.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not ``n_cells`` long, or a result does not fit, as for
        a `pull` sum.
    TypeError
        If ``x`` is not boolean, integer, or floating.
    """
    return pull(x, op.transpose())


def gather(
    x: ArrayLike,
    op: NeighborOperator,
    cells: ArrayLike | None = None,
    *,
    fill: float = 0,
) -> NDArray:
    """Return the raw values of each cell's neighbors, for custom logic.

    Parameters
    ----------
    x : array_like, shape (..., n_cells)
        Per-cell values. Leading axes are carried through.
    op : NeighborOperator
        Compiled neighborhood.
    cells : array_like of int, shape (n_query,), optional
        Cells whose neighbors to gather. Defaults to every cell, in id order.
    fill : float, default 0
        Value for unused slots. It must fit the dtype of ``x``: ``False`` or 0 for booleans, an
        integer in range for integers, and a float that does not overflow (it may round) for
        floats.

    Returns
    -------
    ndarray, same dtype as ``x``, shape (..., n_query, size)
        ``out[..., b, k] = x[..., op.query(cells).idx[b, k]]``, and ``fill`` where that index
        is -1. Weights are not applied; they are in ``op.query(cells).weight``.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not ``n_cells`` long, ``cells`` is not 1-D, or ``fill``
        does not fit the dtype of ``x``.
    TypeError
        If ``cells`` does not hold integers.
    IndexError
        If a cell id is out of range.
    """
    x = _cells_last(x, op.n_cells)
    idx = op.query(cells).idx
    fill_value = _fill_value(fill, x.dtype)
    if fast() and x.dtype.kind in "biufc" and x.dtype.itemsize in (1, 2, 4, 8):
        # The kernel copies bits, so any such dtype goes through as unsigned integers.
        bits = np.dtype(f"u{x.dtype.itemsize}")
        flat = as_rows(x)
        out = np.empty((flat.shape[0], *idx.shape), dtype=x.dtype)
        fill_bits = np.array([fill_value], dtype=x.dtype).view(bits)[0]  # a scalar of type bits
        _kernels.gather_table(idx, flat.view(bits), out.view(bits), fill_bits)
        return out.reshape(*x.shape[:-1], *idx.shape)
    # take, then overwrite unused slots (whose -1 picked the last cell): ~8x faster than
    # assigning through a boolean mask.
    out = np.take(x, idx, axis=-1)
    np.copyto(out, fill_value, where=idx < 0)
    return out


def _cells_last(x: ArrayLike, n_cells: int) -> NDArray:
    """Return ``x`` as an array of shape ``(..., n_cells)``, of any dtype.

    Raises
    ------
    ValueError
        If ``x`` is a scalar, or its last axis is not ``n_cells`` long.
    """
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] != n_cells:
        raise ValueError(f"x must have shape (..., n_cells) with n_cells={n_cells}")
    return x


def _check_field(x: ArrayLike, n_cells: int) -> NDArray:
    """Return ``x`` as a boolean, integer or floating array of shape ``(..., n_cells)``.

    Raises
    ------
    ValueError
        If ``x`` is a scalar, or its last axis is not ``n_cells`` long.
    TypeError
        If ``x`` is of another dtype.
    """
    x = _cells_last(x, n_cells)
    if x.dtype.kind not in "biuf":
        raise TypeError(f"x must be boolean, integer, or floating, got dtype {x.dtype}")
    return x


def _check_int64(x: NDArray, terms: int) -> None:
    """Raise unless int64 holds every sum of up to ``terms`` values of 64-bit integer ``x``."""
    if x.size == 0:
        return
    low, high = int(x.min()), int(x.max())
    if low * terms < _INT64.min or high * terms > _INT64.max:
        what = "these values" if terms == 1 else f"sums of up to {terms} of them"
        raise ValueError(
            f"x holds integers from {low} to {high}, and int64 cannot hold {what}; "
            "convert x to float64"
        )


def _fill_value(fill: float, dtype: np.dtype) -> np.generic:
    """Return ``fill`` as a ``dtype`` scalar: exact for integers, finite for floats."""
    value = None
    if not isinstance(fill, str | bytes):  # NumPy would parse "1.5" as a number
        try:
            with np.errstate(over="raise"):
                value = dtype.type(fill)
        except FloatingPointError, OverflowError, TypeError, ValueError:
            pass
    if value is None or (dtype.kind != "f" and value != fill):
        raise ValueError(f"fill {fill!r} does not fit in {dtype}")
    return value


def _narrow(result: NDArray, dtype: np.dtype) -> NDArray:
    """Return ``result``, computed in float64 or int64, as ``dtype``, the result's dtype.

    A float16 or float32 result whose float64 value is finite but too large for ``dtype``
    raises ValueError instead of becoming inf. Infinite values stay infinite.
    """
    if result.dtype == dtype:
        return result
    try:
        with np.errstate(over="raise"):
            return result.astype(dtype)
    except FloatingPointError:
        raise ValueError(
            f"a result does not fit in {dtype}: its largest magnitude is "
            f"{np.nanmax(np.abs(result), initial=0):.6g}; convert x to float64"
        ) from None


def _sum_by_source(values: NDArray[np.float64], op: NeighborOperator) -> NDArray[np.float64]:
    """Add up per-edge values over the edges leaving each cell.

    ``values`` is float64 of shape (..., n_edges), aligned with ``op.graph.indices``. Returns
    float64 of shape (..., n_cells).
    """
    return _sum_rows(op._source_incidence, values)


def _sum_by_target(values: NDArray[np.float64], op: NeighborOperator) -> NDArray[np.float64]:
    """Add up per-edge values over the edges entering each cell.

    Shapes as for `_sum_by_source`.
    """
    return _sum_rows(op._target_incidence, values)


def _sum_rows(incidence: sp.csr_array, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``incidence @ values`` over the last axis: shape ``(..., n_rows)``.

    Parameters
    ----------
    incidence : scipy.sparse.csr_array, shape (n_rows, n_edges)
        Incidence matrix of the edges.
    values : ndarray of float64, shape (..., n_edges)
        A value per edge.

    Returns
    -------
    ndarray of float64, shape (..., n_rows)
        The sum over each row's edges.
    """
    # One matrix-vector product per leading index: about 3x faster than one product with a
    # (n_edges, k) matrix, which SciPy first copies.
    if values.ndim == 1:
        return incidence @ values
    n_rows: int = incidence.shape[0]
    out = np.empty((*values.shape[:-1], n_rows))
    for index in np.ndindex(values.shape[:-1]):
        out[index] = incidence @ values[index]
    return out


def _row_sums(op: NeighborOperator, x: NDArray, exact: bool) -> NDArray:
    """Fast path of `_matvec` with the operator's matrix: float64, or int64 if ``exact``.

    ``x`` has shape ``(..., n_cells)``, and the result the same shape.
    """
    dtype = np.dtype(np.int64 if exact else np.float64)
    flat = as_rows(x, dtype)
    out = np.empty_like(flat)
    indptr, indices, weights = op._csr  # weights are None whenever the sum is exact
    _kernels.row_sums(indptr, indices, weights, flat, out, dtype.type(0))
    return out.reshape(x.shape)


def _row_extremes(
    op: NeighborOperator, x: NDArray, largest: bool, fill: np.generic, dtype: np.dtype
) -> NDArray:
    """Fast path of the ``"max"`` and ``"min"`` reductions, returning ``dtype``.

    The kernel works in float64, or in int64 for integers on an unweighted operator. Both hold
    every value of ``x`` exactly (`pull` has refused uint64 values beyond int64), and a weighted
    product is formed in float64, as NumPy forms it. ``x`` has shape ``(..., n_cells)``, and the
    result the same shape.
    """
    floating = x.dtype.kind == "f" or op.weight is not None
    work = np.dtype(np.float64 if floating else np.int64)
    flat = as_rows(x, work)
    out = np.empty_like(flat)
    indptr, indices, weights = op._csr
    _kernels.row_extremes(
        indptr, indices, weights, flat, out, work.type(fill), work.type(0), largest
    )
    return _narrow(out.reshape(x.shape), dtype)


def _matvec(matrix: sp.csr_array, x: NDArray, exact: bool) -> NDArray:
    """Return ``matrix @ x`` over the last axis of ``x``: float64, or int64 if ``exact``.

    ``matrix`` is square, (n_cells, n_cells), and ``x`` and the result are (..., n_cells).
    """
    x = x.astype(np.int64 if exact else np.float64, copy=False)
    if x.ndim == 1:
        return matrix @ x
    flat = x.reshape(math.prod(x.shape[:-1]), x.shape[-1])  # -1 fails with no cells
    return np.ascontiguousarray((matrix @ flat.T).T).reshape(x.shape)
