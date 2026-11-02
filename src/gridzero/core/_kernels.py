"""Numba kernels behind the fast paths of the operations on operators, intents and actions.

Every kernel does the reference's arithmetic in the reference's order, so its results are
bit-identical to the NumPy/SciPy path, and the tests compare them with ``assert_array_equal``:

- A sum over a CSR row starts from 0 and adds the row's terms in CSR order, as SciPy's
  ``csr_matvec`` does.
- A sum over the edges entering a cell adds them in the order of their sources, as the product
  with the operator's target incidence matrix does.

Index arrays are uint64 views of the int64 CSR arrays. Numba checks every signed index for
wraparound, which made a CSR sum 1.5x slower than SciPy; with unsigned indices it is faster.
Index arithmetic must not mix uint64 and int64, which Numba promotes to float64.

Equal results include the sign of a zero. Numba's ``max(a, b)`` keeps ``a`` when the two are
equal, so ``max(-0.0, 0.0)`` is -0.0 where ``np.maximum`` gives 0.0: clips are written as
selects, ``w if w > 0.0 else 0.0``. NaN payloads may differ: which NaN a sum keeps depends on
the order of its operands, which compilers are free to swap.

Kernels that divide use ``error_model="numpy"``, which skips Numba's check of every divisor
for zero (to raise ZeroDivisionError). None of them divides by 0.

Kernels write into outputs the caller allocates. Arrays of values are C-contiguous, with
components first: ``(k, n_cells)`` or ``(k, n_edges)``. Everything runs on one thread: at
working sizes threads cost more than they save, and one thread keeps every sum in a fixed order.
"""

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

type Index = NDArray[np.uint64]
"""CSR offsets, or cell ids, viewed as uint64."""
type Floats = NDArray[np.float64]
type Sums = NDArray[np.float64] | NDArray[np.int64]
"""Values summed or compared exactly: float64, or int64 for integers."""
type Zero = np.float64 | np.int64
"""The 0 of a sum's type, ``Sums``."""

# ---------------------------------------------------------------------------------------------
# Sums over CSR rows: pull and push.
#
# Components are handled in blocks of 4, then one block of 3, 2 or 1. Each edge's index and
# weight are then read once per block instead of once per component, and the independent sums
# of a block keep the processor busy. At 2 to 8 components this was 1.6x to 2x faster than one
# component at a time, and 2.3x to 3x faster than SciPy, which copies a transposed input.


@njit(cache=True)
def _row_sums_4(
    indptr: Index, indices: Index, weights: Floats | None, x: Sums, out: Sums, c: int, zero: Zero
) -> None:
    """`row_sums` for the block of components ``c`` to ``c + 3``.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_rows + 1,)
        CSR offsets: the edges of row ``i`` are ``indptr[i]:indptr[i + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge reads from.
    weights : ndarray of float64, shape (n_edges,), or None
        Weight of each edge. None means 1.
    x : ndarray of float64 or int64, shape (k, n_cells)
        Values summed.
    out : ndarray, the dtype of ``x``, shape (k, n_rows)
        Written for rows ``c`` to ``c + 3``.
    c : int
        First component of the block.
    zero : float64 or int64
        Start of each sum, of the accumulator type.
    """
    x0, x1, x2, x3 = x[c], x[c + 1], x[c + 2], x[c + 3]
    o0, o1, o2, o3 = out[c], out[c + 1], out[c + 2], out[c + 3]
    for i in range(indptr.size - 1):
        s0 = s1 = s2 = s3 = zero
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if weights is None:
                s0 += x0[j]
                s1 += x1[j]
                s2 += x2[j]
                s3 += x3[j]
            else:
                w = weights[e]
                s0 += w * x0[j]
                s1 += w * x1[j]
                s2 += w * x2[j]
                s3 += w * x3[j]
        o0[i], o1[i], o2[i], o3[i] = s0, s1, s2, s3


@njit(cache=True)
def _row_sums_3(
    indptr: Index, indices: Index, weights: Floats | None, x: Sums, out: Sums, c: int, zero: Zero
) -> None:
    """`row_sums` for the block of components ``c`` to ``c + 2``, as `_row_sums_4`."""
    x0, x1, x2 = x[c], x[c + 1], x[c + 2]
    o0, o1, o2 = out[c], out[c + 1], out[c + 2]
    for i in range(indptr.size - 1):
        s0 = s1 = s2 = zero
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if weights is None:
                s0 += x0[j]
                s1 += x1[j]
                s2 += x2[j]
            else:
                w = weights[e]
                s0 += w * x0[j]
                s1 += w * x1[j]
                s2 += w * x2[j]
        o0[i], o1[i], o2[i] = s0, s1, s2


@njit(cache=True)
def _row_sums_2(
    indptr: Index, indices: Index, weights: Floats | None, x: Sums, out: Sums, c: int, zero: Zero
) -> None:
    """`row_sums` for the block of components ``c`` and ``c + 1``, as `_row_sums_4`."""
    x0, x1 = x[c], x[c + 1]
    o0, o1 = out[c], out[c + 1]
    for i in range(indptr.size - 1):
        s0 = s1 = zero
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            if weights is None:
                s0 += x0[j]
                s1 += x1[j]
            else:
                w = weights[e]
                s0 += w * x0[j]
                s1 += w * x1[j]
        o0[i], o1[i] = s0, s1


@njit(cache=True)
def _row_sums_1(
    indptr: Index, indices: Index, weights: Floats | None, x: Sums, out: Sums, c: int, zero: Zero
) -> None:
    """`row_sums` for component ``c`` alone, as `_row_sums_4`."""
    x0, o0 = x[c], out[c]
    for i in range(indptr.size - 1):
        s0 = zero
        for e in range(indptr[i], indptr[i + 1]):
            if weights is None:
                s0 += x0[indices[e]]
            else:
                s0 += weights[e] * x0[indices[e]]
        o0[i] = s0


_BLOCKS = {4: _row_sums_4, 3: _row_sums_3, 2: _row_sums_2, 1: _row_sums_1}


def row_sums(
    indptr: Index, indices: Index, weights: Floats | None, x: Sums, out: Sums, zero: Zero
) -> None:
    """``out[c, i] = sum_e weights[e] * x[c, indices[e]]`` over the edges ``e`` of row ``i``.

    ``weights`` None means 1 (no multiply). ``zero`` sets the accumulator type: 0.0 for float64
    ``x``, or an int64 0 for exact integer sums.

    This driver is plain Python, calling one kernel per block. Called from another jitted
    function, the same kernel ran 35% slower at one component.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_rows + 1,)
        CSR offsets: the edges of row ``i`` are ``indptr[i]:indptr[i + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge reads from.
    weights : ndarray of float64, shape (n_edges,), or None
        Weight of each edge. None means 1.
    x : ndarray of float64 or int64, shape (k, n_cells)
        Values summed.
    out : ndarray, the dtype of ``x``, shape (k, n_rows)
        Written with the sums.
    zero : float64 or int64
        Start of each sum, of the accumulator type.
    """
    k = x.shape[0]
    for c in range(0, k, 4):
        _BLOCKS[min(4, k - c)](indptr, indices, weights, x, out, c, zero)


# ---------------------------------------------------------------------------------------------
# Extremes over CSR rows.


@njit(cache=True)
def row_extremes(
    indptr: Index,
    indices: Index,
    weights: Floats | None,
    x: Sums,
    out: Sums,
    fill: Zero,
    zero: Zero,
    largest: bool,
) -> None:
    """``out[c, i]`` = the largest (or smallest) ``weights[e] * x[c, indices[e]]`` of row ``i``.

    NaN wins, as in ``np.maximum``. Rows without edges get ``fill``. ``weights`` None means 1.
    Adding ``zero`` (0.0, or 0 for int64) makes a zero result +0.0, as in the reference.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_rows + 1,)
        CSR offsets: the edges of row ``i`` are ``indptr[i]:indptr[i + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge reads from.
    weights : ndarray of float64, shape (n_edges,), or None
        Weight of each edge.
    x : ndarray of float64 or int64, shape (k, n_cells)
        Values compared.
    out : ndarray, the dtype of ``x``, shape (k, n_rows)
        Written with the extremes.
    fill : float64 or int64
        Result for a row without edges.
    zero : float64 or int64
        0 of the type of ``x``, added to every extreme.
    largest : bool
        Whether to take the largest contribution, or the smallest.
    """
    for c in range(x.shape[0]):
        xc, oc = x[c], out[c]
        for i in range(indptr.size - 1):
            start, stop = indptr[i], indptr[i + 1]
            if start == stop:
                oc[i] = fill
                continue
            best = xc[indices[start]] if weights is None else weights[start] * xc[indices[start]]
            nan = best  # holds a NaN once one is seen
            seen_nan = best != best
            for e in range(start + np.uint64(1), stop):
                value = xc[indices[e]] if weights is None else weights[e] * xc[indices[e]]
                if value != value:
                    seen_nan = True
                    nan = value
                best = max(best, value) if largest else min(best, value)
            oc[i] = (nan if seen_nan else best) + zero


# ---------------------------------------------------------------------------------------------
# Padded tables.


@njit(cache=True)
def gather_table(
    idx: NDArray[np.int64],
    x: NDArray[np.unsignedinteger[Any]],
    out: NDArray[np.unsignedinteger[Any]],
    fill: np.unsignedinteger[Any],
) -> None:
    """``out[c, b, s] = x[c, idx[b, s]]``, or ``fill`` where ``idx`` is -1.

    Values are copied bit for bit, so callers pass any dtype viewed as unsigned integers of the
    same size.

    Parameters
    ----------
    idx : ndarray of int64, shape (n_query, size)
        Padded neighbor table: a cell id per slot, or -1 for an unused slot.
    x : ndarray of uint8, uint16, uint32 or uint64, shape (k, n_cells)
        Values gathered.
    out : ndarray, the dtype of ``x``, shape (k, n_query, size)
        Written with the gathered values.
    fill : unsigned integer, of the dtype of ``x``
        Value of unused slots.
    """
    for c in range(x.shape[0]):
        xc, oc = x[c], out[c]
        for b in range(idx.shape[0]):
            for s in range(idx.shape[1]):
                j = idx[b, s]
                oc[b, s] = xc[j] if j >= 0 else fill


# ---------------------------------------------------------------------------------------------
# Flows.


@njit(cache=True)
def flow_rate_sums(
    indptr: Index,
    indices: Index,
    rate: Floats,
    domain: Floats | None,
    total: Floats,
    kept: Floats | None,
) -> None:
    """Add up each cell's outflow rates, clipped at 0, and those toward cells in the domain.

    ``total[c, i] = sum_e max(rate[c, e], 0)`` over the edges leaving ``i``. ``kept`` gets
    the same sum with each term times ``domain[indices[e]]`` (1.0 or 0.0), unless ``domain``
    and ``kept`` are None.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets: the edges leaving cell ``i`` are ``indptr[i]:indptr[i + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge leads to.
    rate : ndarray of float64, shape (k, n_edges)
        Fraction of each cell's amount sent along each edge.
    domain : ndarray of float64, shape (n_cells,), or None
        1.0 for a cell in the domain, 0.0 for one outside. None if there is no domain.
    total : ndarray of float64, shape (k, n_cells)
        Written with each cell's total outflow rate.
    kept : ndarray of float64, shape (k, n_cells), or None
        Written with each cell's outflow rate toward the domain. None exactly when ``domain``
        is.
    """
    for c in range(rate.shape[0]):
        r, t = rate[c], total[c]
        for i in range(indptr.size - 1):
            s = 0.0
            s_in = 0.0
            for e in range(indptr[i], indptr[i + 1]):
                term = r[e] if r[e] > 0.0 else 0.0  # not max(r[e], 0.0), which keeps -0.0
                s += term
                if domain is not None:
                    s_in += term * domain[indices[e]]
            t[i] = s
            if kept is not None:  # with domain
                kept[c, i] = s_in


@njit(cache=True)
def flow_send(
    indptr: Index, indices: Index, rate: Floats, domain: Floats | None, source: Floats, out: Floats
) -> None:
    """Deliver ``source[c, i] * max(rate[c', e], 0)`` along each edge ``e = i -> j`` into ``j``.

    ``c'`` is ``c``, or 0 when ``rate`` has one row shared by every component. Terms toward a
    cell outside ``domain`` (1.0 or 0.0 per cell; None for every cell) are multiplied by 0.
    Each cell adds what it receives in the order of the sending cells, starting from 0.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets: the edges leaving cell ``i`` are ``indptr[i]:indptr[i + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge leads to.
    rate : ndarray of float64, shape (k, n_edges) or (1, n_edges)
        Fraction of each cell's amount sent along each edge.
    domain : ndarray of float64, shape (n_cells,), or None
        1.0 for a cell in the domain, 0.0 for one outside. None if there is no domain.
    source : ndarray of float64, shape (k, n_cells)
        Amount each cell sends from, before the rates.
    out : ndarray of float64, shape (k, n_cells)
        Written with what each cell receives.
    """
    out[:] = 0.0
    shared = rate.shape[0] == 1
    for c in range(source.shape[0]):
        r = rate[0] if shared else rate[c]
        sc, oc = source[c], out[c]
        for i in range(indptr.size - 1):
            amount = sc[i]
            for e in range(indptr[i], indptr[i + 1]):
                j = indices[e]
                term = r[e] if r[e] > 0.0 else 0.0
                if domain is not None:
                    term = term * domain[j]
                oc[j] += amount * term


@njit(cache=True, error_model="numpy")
def ration(x: Floats, demand: Floats, granted: Floats, remaining: Floats) -> bool:
    """Share amounts among takers that ask for ``demand`` out of ``x``, elementwise.

    ``remaining`` is ``x - demand``, or 0 where the demand exceeds the amount.

    Parameters
    ----------
    x : ndarray of float64, shape (n,)
        Amounts available.
    demand : ndarray of float64, shape (n,)
        What the takers of each amount ask for, together.
    granted : ndarray of float64, shape (n,)
        Written, only if some element was short, with the fraction ``x / demand`` that every
        taker gets where it was short, and 1 elsewhere.
    remaining : ndarray of float64, shape (n,)
        Written with what remains of each amount, exactly non-negative.

    Returns
    -------
    bool
        Whether any element was short.
    """
    short = False
    for k in range(x.size):
        over = demand[k] > x[k]
        short |= over
        remaining[k] = 0.0 if over else x[k] - demand[k]
    if short:
        for k in range(x.size):
            granted[k] = x[k] / demand[k] if demand[k] > x[k] else 1.0
    return short


# ---------------------------------------------------------------------------------------------
# Action spaces: rows of slots, slot 0 for staying and slot s >= 1 for edge indptr[i] + s - 1.
#
# The two allocation kernels repeat the same normalization. As a shared helper it ran 1.6x
# slower, called or inlined, and taking row views (``weights[c, i]``) ran 3x slower: Numba
# counts references to every view.


@njit(cache=True, error_model="numpy")
def fractions(weights: Floats, open_slots: Floats, out: Floats) -> None:
    """Normalize slot weights ``(k, n_cells, n_slots)``, as `NeighborAllocation.fractions`.

    Negative weights and closed slots (``open_slots`` 0.0) count as 0. A row whose total
    overflows is first divided by its largest weight, and a row of zeros stays in slot 0.

    Parameters
    ----------
    weights : ndarray of float64, shape (k, n_cells, n_slots)
        Proposed weight of each slot.
    open_slots : ndarray of float64, shape (n_cells, n_slots)
        1.0 for an open slot, 0.0 for a closed one.
    out : ndarray of float64, shape (k, n_cells, n_slots)
        Written with the fractions, which add up to 1 over each row's slots.
    """
    n_slots = weights.shape[2]
    for c in range(weights.shape[0]):
        for i in range(weights.shape[1]):
            total = 0.0
            for s in range(n_slots):
                w = weights[c, i, s]
                value = (w if w > 0.0 else 0.0) * open_slots[i, s]
                out[c, i, s] = value
                total = value if s == 0 else total + value
            if np.isinf(total):
                largest = out[c, i, 0]
                for s in range(1, n_slots):
                    largest = max(largest, out[c, i, s])
                for s in range(n_slots):
                    value = out[c, i, s] / largest
                    out[c, i, s] = value
                    total = value if s == 0 else total + value
            if total == 0:
                out[c, i, 0] += 1.0
            else:
                for s in range(n_slots):
                    out[c, i, s] = out[c, i, s] / total


@njit(cache=True, error_model="numpy")
def allocation_rates(indptr: Index, weights: Floats, open_slots: Floats, rates: Floats) -> None:
    """Write the fraction moving along each edge, ``rates[c, e]``, as `fractions` gives it.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets: slot ``s >= 1`` of cell ``i`` is edge ``indptr[i] + s - 1``.
    weights : ndarray of float64, shape (k, n_cells, n_slots)
        Proposed weight of each slot.
    open_slots : ndarray of float64, shape (n_cells, n_slots)
        1.0 for an open slot, 0.0 for a closed one.
    rates : ndarray of float64, shape (k, n_edges)
        Written with the fraction of each cell's amount moving along each edge. Slot 0, which
        stays, has no edge.
    """
    n_slots = weights.shape[2]
    row = np.empty(n_slots)
    for c in range(weights.shape[0]):
        for i in range(weights.shape[1]):
            total = 0.0
            for s in range(n_slots):
                w = weights[c, i, s]
                value = (w if w > 0.0 else 0.0) * open_slots[i, s]
                row[s] = value
                total = value if s == 0 else total + value
            if np.isinf(total):
                largest = row[0]
                for s in range(1, n_slots):
                    largest = max(largest, row[s])
                for s in range(n_slots):
                    value = row[s] / largest
                    row[s] = value
                    total = value if s == 0 else total + value
            if total == 0:
                total = 1.0  # nothing proposed: all stays, and the zeros divide as they are
            slot = 1
            for e in range(indptr[i], indptr[i + 1]):
                rates[c, e] = row[slot] / total
                slot += 1


@njit(cache=True)
def choice_rates(
    indptr: Index,
    valid: NDArray[np.bool_],
    slots: NDArray[np.int64],
    rate: Floats,
    rates: Floats,
) -> None:
    """Set ``rates[c, e] = rate[c, i]`` on the edge each cell ``i`` chose; 0 elsewhere.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets: slot ``s >= 1`` of cell ``i`` is edge ``indptr[i] + s - 1``.
    valid : ndarray of bool, shape (n_cells, n_slots)
        Whether each slot holds a neighbor.
    slots : ndarray of int64, shape (k, n_cells)
        Slot each cell chose. Slot 0, and an invalid slot, choose no edge.
    rate : ndarray of float64, shape (k, n_cells)
        Fraction of each cell's amount that moves to the chosen neighbor.
    rates : ndarray of float64, shape (k, n_edges)
        Written with the fraction moving along each edge.
    """
    rates[:] = 0.0
    for c in range(slots.shape[0]):
        for i in range(slots.shape[1]):
            slot = slots[c, i]
            if slot > 0 and valid[i, slot]:
                rates[c, indptr[i] + np.uint64(slot - 1)] = rate[c, i]


@njit(cache=True)
def slot_mask(
    indptr: Index, indices: Index, inside: NDArray[np.bool_], out: NDArray[np.bool_]
) -> None:
    """Write which slots of a neighbor action space are valid in a domain, as `mask` gives it.

    Slot 0 (stay) is always valid, slot ``s >= 1`` of cell ``i`` is valid if its neighbor, along
    edge ``indptr[i] + s - 1``, is inside, and slots past the cell's degree are not.

    Parameters
    ----------
    indptr, indices : ndarray of uint64, shapes (n_cells + 1,) and (n_edges,)
        CSR structure of the operator.
    inside : ndarray of bool, shape (n_cells,)
        The domain.
    out : ndarray of bool, shape (n_cells, n_slots)
        Written with the mask.
    """
    n_slots = out.shape[1]
    for i in range(out.shape[0]):
        out[i, 0] = True
        s = 1
        for e in range(indptr[i], indptr[i + 1]):
            out[i, s] = inside[indices[e]]
            s += 1
        for rest in range(s, n_slots):
            out[i, rest] = False


@njit(cache=True)
def best_slots(
    scores: Sums | NDArray[np.uint64], valid: NDArray[np.bool_], out: NDArray[np.int64]
) -> None:
    """``out[c, i]`` = the valid slot with the highest ``scores[c, i, s]``, the lowest on ties.

    Slot 0 is always valid, and ``scores`` hold no NaN. The maximum comes first, then the slot
    that holds it, both without branches on the scores: branching on each comparison ran 1.8x
    slower on random scores, mispredicting.

    Parameters
    ----------
    scores : ndarray of float64, int64 or uint64, shape (k, n_cells, n_slots)
        Score of each slot, without NaN.
    valid : ndarray of bool, shape (n_cells, n_slots)
        Whether each slot holds a neighbor.
    out : ndarray of int64, shape (k, n_cells)
        Written with the slot each cell chose.
    """
    n_slots = scores.shape[2]
    for c in range(scores.shape[0]):
        for i in range(scores.shape[1]):
            best = scores[c, i, 0]
            for s in range(1, n_slots):
                best = max(best, scores[c, i, s]) if valid[i, s] else best
            slot = 0
            for s in range(n_slots - 1, -1, -1):
                slot = s if valid[i, s] & (scores[c, i, s] == best) else slot  # not `and`
            out[c, i] = slot


# ---------------------------------------------------------------------------------------------
# Shortest paths: `paths.steps` and `paths.path`.
#
# The distance of a cell is the smallest rounded sum ``dist[u] + cost[v]`` over the cells
# ``u`` it can be entered from, which doesn't depend on the order cells are settled in. So
# these searches give SciPy's distances bit for bit, whatever order their heaps break ties in.
# A search that stops early, at a limit or at a goal, has settled every cell nearer than where
# it stopped.


@njit(cache=True)
def bfs_steps(
    indptr: Index,
    indices: Index,
    sources: Index,
    blocked: NDArray[np.bool_],
    limit: float,
    goal: np.uint64,
    dist: Floats,
    queue: Index,
) -> None:
    """Write the number of steps from the nearest source to each cell into ``dist``.

    Cells farther than ``limit`` steps, or unreachable, get inf. With ``goal`` below
    ``dist.size``, stops once the goal is reached: cells nearer than it are then all final.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets of the adjacency: the edges leaving ``u`` are ``indptr[u]:indptr[u + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge enters.
    sources : ndarray of uint64, shape (n_sources,)
        Cells the search starts from, at 0 steps. Repeats are allowed.
    blocked : ndarray of bool, shape (n_cells,) or (0,)
        Cells no move may enter, which a search may still start from; empty for none.
    limit : float
        Largest number of steps searched, inf for no limit.
    goal : uint64
        Cell to stop at, or ``n_cells`` or more to search everywhere.
    dist : ndarray of float64, shape (n_cells,)
        Written with the number of steps to each cell.
    queue : ndarray of uint64, shape (n_cells,)
        Workspace for the cells waiting to be expanded.
    """
    # Walls start at -inf, which the search never enters (it enters cells still at inf), so
    # they cost nothing per edge. Those that aren't sources go back to inf at the end.
    dist[:] = np.inf
    for c in range(blocked.size):
        if blocked[c]:
            dist[c] = -np.inf
    tail = 0
    for k in range(sources.size):
        s = sources[k]
        if dist[s] != 0.0:
            dist[s] = 0.0
            queue[tail] = s
            tail += 1
    head = 0
    while head < tail:
        u = queue[head]
        head += 1
        if u == goal:
            break
        du = dist[u] + 1.0
        if du > limit:
            continue
        for e in range(indptr[u], indptr[u + np.uint64(1)]):
            v = indices[e]
            if dist[v] == np.inf:
                dist[v] = du
                queue[tail] = v
                tail += 1
    for c in range(blocked.size):
        if dist[c] == -np.inf:
            dist[c] = np.inf


@njit(cache=True)
def dijkstra_steps(
    indptr: Index,
    indices: Index,
    cost: Floats,
    sources: Index,
    limit: float,
    goal: np.uint64,
    dist: Floats,
    heap_d: Floats,
    heap_v: Index,
) -> None:
    """Write the cheapest cost from the nearest source to each cell into ``dist``.

    Entering cell ``v`` costs ``cost[v]``, positive or inf. Cells costing more than ``limit``,
    or unreachable, get inf. With ``goal`` below ``dist.size``, stops once the goal is settled:
    every cell nearer than it is then final, which is all a path traced back from it reads. A
    binary heap with stale entries: a cell is pushed again whenever its distance falls, and
    the entries left behind are skipped.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets of the adjacency: the edges leaving ``u`` are ``indptr[u]:indptr[u + 1]``.
    indices : ndarray of uint64, shape (n_edges,)
        Cell each edge enters.
    cost : ndarray of float64, shape (n_cells,)
        Cost of entering each cell, positive or inf.
    sources : ndarray of uint64, shape (n_sources,)
        Cells the search starts from, at cost 0. Repeats are allowed.
    limit : float
        Largest cost searched, inf for no limit.
    goal : uint64
        Cell to stop at, or ``n_cells`` or more to search everywhere.
    dist : ndarray of float64, shape (n_cells,)
        Written with the cheapest cost of each cell.
    heap_d : ndarray of float64, shape (n_sources + n_edges,)
        Workspace for the distances in the heap: one push per source, and at most one per
        edge.
    heap_v : ndarray of uint64, shape (n_sources + n_edges,)
        Workspace for the cells in the heap.

    """
    dist[:] = np.inf
    n = 0
    for k in range(sources.size):
        s = sources[k]
        if dist[s] != 0.0:
            dist[s] = 0.0
            heap_d[n] = 0.0
            heap_v[n] = s
            n += 1  # all zeros: still a heap
    while n > 0:
        d = heap_d[0]
        u = heap_v[0]
        n -= 1
        last_d = heap_d[n]
        last_v = heap_v[n]
        i = 0
        while True:  # sift the last entry down from the top
            c = 2 * i + 1
            if c >= n:
                break
            if c + 1 < n and heap_d[c + 1] < heap_d[c]:
                c += 1
            if heap_d[c] < last_d:
                heap_d[i] = heap_d[c]
                heap_v[i] = heap_v[c]
                i = c
            else:
                break
        heap_d[i] = last_d
        heap_v[i] = last_v
        if d > dist[u]:
            continue  # stale
        if u == goal:
            return
        for e in range(indptr[u], indptr[u + np.uint64(1)]):
            v = indices[e]
            nd = d + cost[v]
            if nd < dist[v] and nd <= limit:
                dist[v] = nd
                i = n
                n += 1
                while i > 0:  # sift up
                    p = (i - 1) // 2
                    if heap_d[p] > nd:
                        heap_d[i] = heap_d[p]
                        heap_v[i] = heap_v[p]
                        i = p
                    else:
                        break
                heap_d[i] = nd
                heap_v[i] = v


@njit(cache=True)
def walk_back(
    indptr: Index,
    indices: Index,
    cost: Floats | None,
    dist: Floats,
    start: np.uint64,
    goal: np.uint64,
    out: Index,
) -> int:
    """Write the cells of a cheapest path from ``start`` to ``goal`` into ``out``, goal first.

    From each cell ``v``, steps back to the lowest neighbor ``u`` with ``dist[u] < dist[v]``
    and ``dist[u] + cost[v] == dist[v]``.

    Parameters
    ----------
    indptr : ndarray of uint64, shape (n_cells + 1,)
        CSR offsets of a symmetric adjacency: the neighbors of ``v`` are
        ``indices[indptr[v]:indptr[v + 1]]``.
    indices : ndarray of uint64, shape (n_edges,)
        Neighbors of each cell, in id order.
    cost : ndarray of float64, shape (n_cells,), or None
        Cost of entering each cell. None means 1.
    dist : ndarray of float64, shape (n_cells,)
        Cheapest cost of each cell from ``start``, final at least for the cells nearer than
        ``goal``, which is finite.
    start, goal : uint64
        Ends of the path.
    out : ndarray of uint64, shape (n_cells,)
        Written with the path's cells, goal first.

    Returns
    -------
    int
        Number of cells written, or 0 if some cell has no such neighbor, where rounding hid a
        cost.
    """
    v = goal
    out[0] = v
    count = 1
    while v != start:
        dv = dist[v]
        step = 1.0 if cost is None else cost[v]
        found = False
        u = v
        for e in range(indptr[v], indptr[v + np.uint64(1)]):
            u = indices[e]
            if dist[u] < dv and dist[u] + step == dv:
                found = True
                break
        if not found:
            return 0
        v = u
        out[count] = v
        count += 1
    return count


# ---------------------------------------------------------------------------------------------
# Lines of sight: `sight.visible` and `sight.sight`.


@njit(cache=True)
def clear_lines(
    scale: NDArray[np.int64],
    center: NDArray[np.int64],
    corners: NDArray[np.int64],
    type_steps: NDArray[np.int64],
    shape: NDArray[np.int64],
    periodic: NDArray[np.bool_],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    lows: NDArray[np.int64],
    highs: NDArray[np.int64],
    walls: NDArray[np.bool_],
    out: NDArray[np.bool_],
    contacts: NDArray[np.int64],
) -> None:
    """Write whether the line from each of ``starts`` to ``ends`` is clear into ``out``.

    As `sight._clear_reference` decides from `sight._trace`, in the same integer frame, but
    looking only at blocked cells in a band around the line, row by row, and stopping at the
    first that blocks.

    Parameters
    ----------
    scale : ndarray of int64, shape (2, 2)
        Maps lattice coordinates into the tiling's integer frame, as in `lattice._Frame`.
    center : ndarray of int64, shape (n_types, 2)
        Offset of the center of a cell of each type, in the frame.
    corners : ndarray of int64, shape (n_types, n_corners, 2)
        Corners of a cell of each type around its center, counterclockwise, in the frame.
    type_steps : ndarray of int64, shape (2,)
        Type of the cell at coordinates ``c``: ``c @ type_steps % n_types``.
    shape : ndarray of int64, shape (2,)
        Number of cells along each axis.
    periodic : ndarray of bool, shape (2,)
        Whether each axis wraps.
    starts, ends : ndarray of int64, shape (n, 2)
        Lattice coordinates of the two ends of each line.
    lows, highs : ndarray of int64, shape (n,)
        Ids of the two cells of each line, which never block it.
    walls : ndarray of bool, shape (n_cells,)
        Whether each cell blocks sight.
    out : ndarray of bool, shape (n,)
        Written with whether each line is clear.
    contacts : ndarray of int64, shape (4, size)
        Workspace for the stretches of a line that blocked cells touch, at least one per cell
        searched: their starts and ends from the left, then from the right.
    """
    for p in range(starts.shape[0]):
        out[p] = lows[p] == highs[p] or _line_clear(
            scale,
            center,
            corners,
            type_steps,
            shape,
            periodic,
            starts[p, 0],
            starts[p, 1],
            ends[p, 0],
            ends[p, 1],
            lows[p],
            highs[p],
            walls,
            contacts,
        )


@njit(cache=True)
def sight_offsets(
    scale: NDArray[np.int64],
    center: NDArray[np.int64],
    corners: NDArray[np.int64],
    type_steps: NDArray[np.int64],
    shape: NDArray[np.int64],
    periodic: NDArray[np.bool_],
    viewers: NDArray[np.int64],
    offsets: NDArray[np.int64],
    walls: NDArray[np.bool_],
    out: NDArray[np.bool_],
    contacts: NDArray[np.int64],
) -> None:
    """Write which cells each viewer sees, among those at ``offsets`` from it, into ``out``.

    As `clear_lines` decides for the lines from each viewer to the cell at each offset, traced
    from the lower id, as `sight._ends` places them. Each offset must be the step to the
    nearest copy of the cell it reaches, and no two offsets may reach one cell.

    Parameters
    ----------
    scale, center, corners, type_steps, shape, periodic
        The tiling's frame and the grid, as for `clear_lines`.
    viewers : ndarray of int64, shape (k,)
        Cell ids of the viewers.
    offsets : ndarray of int64, shape (n_types, m, 2)
        Steps in lattice coordinates from a cell of each type to the cells it may see.
    walls : ndarray of bool, shape (n_cells,)
        Whether each cell blocks sight.
    out : ndarray of bool, shape (k, n_cells)
        Written at each viewer's row, for the cells at its offsets. Other entries are left.
    contacts : ndarray of int64, shape (4, size)
        Workspace, as for `clear_lines`, big enough for the longest offset's line.
    """
    n_types = center.shape[0]
    for v in range(viewers.shape[0]):
        cell = viewers[v]
        a0 = cell // shape[1]
        a1 = cell % shape[1]
        ta = (a0 * type_steps[0] + a1 * type_steps[1]) % n_types
        for k in range(offsets.shape[1]):
            o0 = offsets[ta, k, 0]
            o1 = offsets[ta, k, 1]
            b0 = a0 + o0
            b1 = a1 + o1
            if periodic[0]:
                b0 %= shape[0]
            if periodic[1]:
                b1 %= shape[1]
            if b0 < 0 or b0 >= shape[0] or b1 < 0 or b1 >= shape[1]:
                continue
            target = b0 * shape[1] + b1
            if target > cell:  # from the viewer, to the copy of the target at the offset
                clear = _line_clear(
                    scale, center, corners, type_steps, shape, periodic,
                    a0, a1, a0 + o0, a1 + o1, cell, target, walls, contacts,
                )  # fmt: skip
            else:  # from the target, to the copy of the viewer back along the offset
                clear = _line_clear(
                    scale, center, corners, type_steps, shape, periodic,
                    b0, b1, b0 - o0, b1 - o1, target, cell, walls, contacts,
                )  # fmt: skip
            out[v, target] = clear


@njit(cache=True)
def _line_clear(
    scale: NDArray[np.int64],
    center: NDArray[np.int64],
    corners: NDArray[np.int64],
    type_steps: NDArray[np.int64],
    shape: NDArray[np.int64],
    periodic: NDArray[np.bool_],
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    low: int,
    high: int,
    walls: NDArray[np.bool_],
    contacts: NDArray[np.int64],
) -> bool:
    """Return whether the line from lattice coordinates ``(a0, a1)`` to ``(b0, b1)`` is clear.

    The line joins cells ``low`` and ``high``, which never block it. See `clear_lines`.
    """
    n_types = center.shape[0]
    n_corners = corners.shape[1]
    margin = 2
    ta = (a0 * type_steps[0] + a1 * type_steps[1]) % n_types
    tb = (b0 * type_steps[0] + b1 * type_steps[1]) % n_types
    p0 = scale[0, 0] * a0 + scale[0, 1] * a1 + center[ta, 0]
    p1 = scale[1, 0] * a0 + scale[1, 1] * a1 + center[ta, 1]
    d0 = scale[0, 0] * b0 + scale[0, 1] * b1 + center[tb, 0] - p0
    d1 = scale[1, 0] * b0 + scale[1, 1] * b1 + center[tb, 1] - p1
    dd = d0 * d0 + d1 * d1
    n_left = 0
    n_right = 0
    for j in range(min(a1, b1) - margin, max(a1, b1) + margin + 1):
        # The columns where the line, in lattice coordinates, comes within `margin` rows.
        if a1 == b1:
            x_lo, x_hi = float(min(a0, b0)), float(max(a0, b0))
        else:
            t0 = (j - margin - a1) / (b1 - a1)
            t1 = (j + margin - a1) / (b1 - a1)
            if t0 > t1:
                t0, t1 = t1, t0
            t0 = max(t0, 0.0)
            t1 = min(t1, 1.0)
            xa = a0 + t0 * (b0 - a0)
            xb = a0 + t1 * (b0 - a0)
            x_lo, x_hi = min(xa, xb), max(xa, xb)
        for i in range(int(np.floor(x_lo)) - margin, int(np.ceil(x_hi)) + margin + 1):
            if (i == a0 and j == a1) or (i == b0 and j == b1):
                continue
            c0, c1 = i, j
            if periodic[0]:
                c0 %= shape[0]
            if periodic[1]:
                c1 %= shape[1]
            if c0 < 0 or c0 >= shape[0] or c1 < 0 or c1 >= shape[1]:
                continue
            cell = c0 * shape[1] + c1
            if not walls[cell] or cell in (low, high):
                continue
            t = (i * type_steps[0] + j * type_steps[1]) % n_types
            u0 = scale[0, 0] * i + scale[0, 1] * j + center[t, 0]
            u1 = scale[1, 0] * i + scale[1, 1] * j + center[t, 1]
            left = False
            right = False
            inside = False
            lo = np.iinfo(np.int64).max
            hi = np.iinfo(np.int64).min
            for k in range(n_corners):
                w0 = u0 + corners[t, k, 0] - p0
                w1 = u1 + corners[t, k, 1] - p1
                s = d0 * w1 - d1 * w0
                if s > 0:
                    left = True
                elif s < 0:
                    right = True
                else:
                    along = w0 * d0 + w1 * d1
                    lo = min(lo, along)
                    hi = max(hi, along)
                    inside |= (along > 0) & (along < dd)
                # The edge to the next corner, if its ends lie on either side of the line.
                n = (k + 1) % n_corners
                e0 = corners[t, n, 0] - corners[t, k, 0]
                e1 = corners[t, n, 1] - corners[t, k, 1]
                v0 = w0 + e0
                v1 = w1 + e1
                s_next = d0 * v1 - d1 * v0
                if (s > 0 and s_next < 0) or (s < 0 and s_next > 0):
                    num = w0 * e1 - w1 * e0
                    den = d0 * e1 - d1 * e0
                    if den < 0:
                        num = -num
                        den = -den
                    inside |= (num > 0) & (num < den)
            if not inside:
                continue
            if left and right:
                return False
            if left:
                contacts[0, n_left] = lo
                contacts[1, n_left] = hi
                n_left += 1
            else:
                contacts[2, n_right] = lo
                contacts[3, n_right] = hi
                n_right += 1
    for x in range(n_left):
        for y in range(n_right):
            if max(contacts[0, x], contacts[2, y]) <= min(contacts[1, x], contacts[3, y]):
                return False
    return True
