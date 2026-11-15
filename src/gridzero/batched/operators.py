"""Neighbor operators as JAX arrays: `core.pull`, `core.push` and `core.gather` for one state.

An `Operator` holds a `core.NeighborOperator`'s padded tables, so every neighborhood on every
space works here unchanged: compile it in `core`, then convert it once with `operator`.
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from gridzero.core import NeighborOperator

type Reduction = Literal["sum", "mean", "max", "min"]
"""How `pull` combines the contributions of a cell's neighbors."""

_REDUCTIONS = ("sum", "mean", "max", "min")


class Operator(NamedTuple):
    """A compiled neighborhood as padded tables of JAX arrays (see `operator`).

    Slot ``k`` of a row holds the cell's ``k``-th smallest neighbor id, as in
    `core.NeighborOperator.query`. Unused slots hold ``n_cells``, one past the last cell, so
    that values padded with one more entry can be gathered without a mask.
    """

    idx: jax.Array
    """Neighbor ids of every cell: int32, shape (n_cells, size)."""
    weight: jax.Array | None
    """The weights of those slots, 0 in unused ones: float32, shape (n_cells, size). None if
    the operator is unweighted."""
    back: jax.Array
    """The transpose's neighbor ids, the cells each cell is a neighbor of: int32, shape
    (n_cells, back_size)."""
    back_weight: jax.Array | None
    """The weights of the transpose's slots: float32, shape (n_cells, back_size), or None."""
    sources: jax.Array
    """The cell each edge leaves, in the order of `core.CellGraph`'s edges (by source, then by
    target): int32, shape (n_edges,). What a rate per edge is aligned with."""
    targets: jax.Array
    """The cell each edge enters, the neighbor: int32, shape (n_edges,)."""


def operator(op: NeighborOperator) -> Operator:
    """Return a compiled neighborhood's tables as JAX arrays.

    Parameters
    ----------
    op : NeighborOperator
        A neighborhood compiled on a space, such as ``Adjacency().compile(space)``.

    Returns
    -------
    Operator
        The tables of ``op`` and of its transpose.

    Raises
    ------
    TypeError
        If ``op`` is not a `NeighborOperator`.
    """
    if not isinstance(op, NeighborOperator):
        raise TypeError(f"op must be a NeighborOperator, got {type(op).__name__}")
    idx, weight = _tables(op)
    back, back_weight = _tables(op.transpose())
    sources = jnp.asarray(op.graph.sources.astype(np.int32))
    targets = jnp.asarray(op.graph.indices.astype(np.int32))
    return Operator(idx, weight, back, back_weight, sources, targets)


def _tables(op: NeighborOperator) -> tuple[jax.Array, jax.Array | None]:
    """Return an operator's padded ids, ``n_cells`` in unused slots, and weights or None."""
    found = op.query()
    idx = np.where(found.idx >= 0, found.idx, op.n_cells).astype(np.int32)
    weight = None if found.weight is None else jnp.asarray(found.weight, dtype=jnp.float32)
    return jnp.asarray(idx), weight


def pull(x: ArrayLike, op: Operator, reduce: Reduction = "sum", *, fill: float = 0) -> jax.Array:
    """Aggregate neighbor values into each cell, as `core.pull` does.

    Parameters
    ----------
    x : Array of bool, int or float, shape (..., n_cells)
        Per-cell values. Leading axes (components) are carried through.
    op : Operator
        The neighborhood's tables.
    reduce : {"sum", "mean", "max", "min"}, default "sum"
        ``"sum"`` gives ``y_i = sum_j w_ij x_j``, ``"mean"`` divides that by ``sum_j w_ij``,
        and ``"max"`` and ``"min"`` give the extreme contribution.
    fill : float, default 0
        The result for a cell with no neighbors, or with zero total weight for ``"mean"``. A
        sum over no neighbors is always 0.

    Returns
    -------
    Array, shape (..., n_cells)
        Floating ``x`` keeps its dtype. Boolean and integer ``x`` give exact int32 results on
        an unweighted operator, while they fit, and float32 results on a weighted one or for
        ``"mean"``. Floats equal `core.pull`'s to float32's precision.

    Raises
    ------
    ValueError
        If ``reduce`` is unknown, or the last axis of ``x`` is not ``n_cells`` long.
    """
    return _pull(jnp.asarray(x), op.idx, op.weight, reduce, fill)


def push(x: ArrayLike, op: Operator) -> jax.Array:
    """Send each cell's value out to its neighbors, as `core.push` does: `pull`'s transpose.

    Parameters
    ----------
    x : Array of bool, int or float, shape (..., n_cells)
        Per-cell values to send. Leading axes are carried through.
    op : Operator
        The neighborhood's tables.

    Returns
    -------
    Array, shape (..., n_cells)
        ``y_j = sum_i w_ij x_i``, with the dtypes of a `pull` sum.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not ``n_cells`` long.
    """
    return _pull(jnp.asarray(x), op.back, op.back_weight, "sum", 0)


def gather(x: ArrayLike, op: Operator, *, fill: float = 0) -> jax.Array:
    """Return the raw values of each cell's neighbors, as `core.gather` does.

    Parameters
    ----------
    x : Array, shape (..., n_cells)
        Per-cell values. Leading axes are carried through.
    op : Operator
        The neighborhood's tables.
    fill : float, default 0
        The value in unused slots.

    Returns
    -------
    Array, same dtype as ``x``, shape (..., n_cells, size)
        ``out[..., i, k] = x[..., op.idx[i, k]]``, and ``fill`` in unused slots. Weights are
        not applied.

    Raises
    ------
    ValueError
        If the last axis of ``x`` is not ``n_cells`` long.
    """
    x = jnp.asarray(x)
    _check_cells(x, op.idx)
    return _padded(x, jnp.asarray(fill, dtype=x.dtype))[..., op.idx]


def _pull(
    x: jax.Array, idx: jax.Array, weight: jax.Array | None, reduce: str, fill: float
) -> jax.Array:
    """Reduce the values at ``idx`` (and their weights) per row: the work of `pull` and `push`."""
    if reduce not in _REDUCTIONS:
        raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
    _check_cells(x, idx)
    floating = jnp.issubdtype(x.dtype, jnp.floating)
    exact = not floating and weight is None and reduce != "mean"
    dtype = x.dtype if floating else (jnp.int32 if exact else jnp.float32)
    x = x.astype(dtype)
    used = idx < x.shape[-1]
    if reduce in ("sum", "mean"):
        values = _padded(x, jnp.zeros((), dtype))[..., idx]
        total = (values if weight is None else values * weight.astype(dtype)).sum(axis=-1)
        if reduce == "sum":
            return total
        sums = used.sum(axis=-1).astype(dtype) if weight is None else weight.sum(axis=-1)
        return jnp.where(sums != 0, total / jnp.where(sums != 0, sums, 1), fill).astype(dtype)
    if floating or not exact:
        worst = jnp.asarray(-jnp.inf if reduce == "max" else jnp.inf, dtype)
    else:
        info = jnp.iinfo(jnp.int32)
        worst = jnp.asarray(info.min if reduce == "max" else info.max, dtype)
    values = _padded(x, jnp.zeros((), dtype))[..., idx]
    if weight is not None:
        values = values * weight.astype(dtype)
    values = jnp.where(used, values, worst)
    extreme = values.max(axis=-1) if reduce == "max" else values.min(axis=-1)
    if not exact:
        extreme = extreme + 0.0  # a zero result is +0.0, whichever zeros its row held
    return jnp.where(used.any(axis=-1), extreme, jnp.asarray(fill, dtype))


def _padded(x: jax.Array, value: jax.Array) -> jax.Array:
    """Return ``x`` with one more entry along its last axis, which unused slots index."""
    pad = jnp.broadcast_to(value, (*x.shape[:-1], 1))
    return jnp.concatenate([x, pad], axis=-1)


def _check_cells(x: jax.Array, idx: jax.Array) -> None:
    """Raise unless the last axis of ``x`` is one entry per cell of the tables."""
    if x.ndim == 0 or x.shape[-1] != idx.shape[0]:
        raise ValueError(f"x must have shape (..., {idx.shape[0]}), got {x.shape}")
