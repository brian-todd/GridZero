"""Moving through cells: how far every cell is, what a budget reaches, and the cheapest path.

Moves go between adjacent cells, as ``space.adjacency(kind)`` says, so these work on any space,
except `hops`, which counts moves between any cells of a lattice without searching.
Entering a cell costs ``cost[cell]``, 1 by default, and inf where no move may enter, such as a
wall, which ``blocked`` also marks. Costs add up exactly as floats do: keep them integers or
binary fractions (0.5, 0.25) when sums must come out even, because ``0.1 + 0.2 > 0.3``.
"""

import math
import numbers
from typing import SupportsIndex

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import dijkstra

from gridzero.core import _kernels
from gridzero.core._arrays import (
    cell_mask,
    cell_selection,
    check_cell,
    check_ids,
    read_only,
)
from gridzero.core.graph import CellGraph
from gridzero.core.implementations import fast
from gridzero.core.lattice import LatticeSpace
from gridzero.core.space import AdjacencyKind, Space


def steps(
    space: Space,
    sources: ArrayLike,
    *,
    kind: AdjacencyKind = "facet",
    cost: ArrayLike | None = None,
    blocked: ArrayLike | None = None,
    limit: float = math.inf,
    separate: bool = False,
) -> NDArray[np.float64]:
    """Return what it costs to reach each cell from the nearest source.

    Without costs, that is the number of moves. Use it for distances that walls bend, or for
    the nearest cell of some kind: the sources are then the cells of that kind.

    Parameters
    ----------
    space : Space
        The cells.
    sources : int, array_like of int, shape (k,), or array_like of bool, shape (n_cells,)
        A cell id, cell ids, or a mask of cells. They cost 0 to reach, whatever their cost.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells a move may go between, as for `Space.adjacency`. On squares, ``"vertex"``
        allows diagonal moves, at the same cost.
    cost : array_like of float, shape (n_cells,), optional
        What entering each cell costs: positive, or inf where no move may enter. By default,
        every move costs 1.
    blocked : array_like of bool, shape (n_cells,), optional
        Cells no move may enter, such as walls: the same as a cost of inf there, whatever
        ``cost`` says.
    limit : float, default inf
        The most a cell may cost. Cells beyond it count as unreachable, and the search stops
        there, which is far faster for small limits.
    separate : bool, default False
        Whether to give each source a row of its own instead of the nearest source's cost.

    Returns
    -------
    ndarray of float64, shape (n_cells,), or (k, n_cells) if ``separate``
        The cost of the cheapest way from a source to each cell: a sum of costs in the order
        they are paid, 0 at a source, and inf where no source reaches within ``limit``.

    Raises
    ------
    TypeError
        If ``sources`` holds neither integers nor booleans, ``cost`` doesn't hold numbers,
        ``blocked`` is not boolean, ``limit`` is not a real number, or ``kind`` is neither a
        string nor an integer.
    ValueError
        If ``sources``, ``cost`` or ``blocked`` has the wrong shape, a cost is NaN or not
        positive, ``limit`` is NaN or negative, or ``kind`` is unknown or out of range.
    IndexError
        If a source is not a cell id.
    """
    graph = space.adjacency(kind)
    cells = cell_selection(sources, space.n_cells, "sources")
    costs, walls = _costs(cost, blocked, space.n_cells)
    limit = _limit(limit, "limit")
    if separate:
        out = np.empty((cells.size, space.n_cells))
        for row, cell in zip(out, cells, strict=True):
            row[:] = _steps(graph, cell[None], costs, walls, limit)
        return out
    return _steps(graph, cells, costs, walls, limit)


def reach(
    space: Space,
    sources: ArrayLike,
    budget: float,
    *,
    kind: AdjacencyKind = "facet",
    cost: ArrayLike | None = None,
    blocked: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Return the cells some source reaches for at most ``budget``: where a piece may move.

    Parameters
    ----------
    space : Space
        The cells.
    sources : int, array_like of int, shape (k,), or array_like of bool, shape (n_cells,)
        A cell id, cell ids, or a mask of cells. They are reached for free.
    budget : float
        The most a move may cost, at least 0: the number of moves without costs.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells a move may go between, as for `Space.adjacency`.
    cost : array_like of float, shape (n_cells,), optional
        What entering each cell costs: positive, or inf where no move may enter. By default,
        every move costs 1.
    blocked : array_like of bool, shape (n_cells,), optional
        Cells no move may enter, such as walls: the same as a cost of inf there, whatever
        ``cost`` says.

    Returns
    -------
    ndarray of bool, shape (n_cells,)
        Whether a source reaches each cell, for at most ``budget``: where `steps` is finite
        and at most ``budget``. Sources are included.

    Raises
    ------
    TypeError, ValueError, IndexError
        As for `steps`, with ``budget`` checked as its ``limit``.
    """
    graph = space.adjacency(kind)
    cells = cell_selection(sources, space.n_cells, "sources")
    costs, walls = _costs(cost, blocked, space.n_cells)
    return np.isfinite(_steps(graph, cells, costs, walls, _limit(budget, "budget")))


def path(
    space: Space,
    start: SupportsIndex,
    goal: SupportsIndex,
    *,
    kind: AdjacencyKind = "facet",
    cost: ArrayLike | None = None,
    blocked: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Return the cells of a cheapest path from ``start`` to ``goal``.

    Of the paths that cost as little, it is the one found by stepping back from the goal, each
    time to the neighbor with the lowest id that a cheapest path may come from. On squares that
    gives paths shaped like an L, or a staircase with diagonal moves, rather than zigzags.

    Parameters
    ----------
    space : Space
        The cells.
    start, goal : int
        Cell ids.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells a move may go between, as for `Space.adjacency`.
    cost : array_like of float, shape (n_cells,), optional
        What entering each cell costs: positive, or inf where no move may enter. By default,
        every move costs 1.
    blocked : array_like of bool, shape (n_cells,), optional
        Cells no move may enter, such as walls: the same as a cost of inf there, whatever
        ``cost`` says.

    Returns
    -------
    ndarray of int64, shape (n_moves + 1,)
        The cells from ``start`` to ``goal``, both included, each adjacent to the next, or no
        cells if ``goal`` can't be reached. Adding up the cost of each cell after the first, in
        order, gives ``steps(space, start, ...)[goal]`` exactly.

    Raises
    ------
    TypeError
        If ``start`` or ``goal`` is not an integer, ``cost`` doesn't hold numbers, ``blocked``
        is not boolean, or ``kind`` is neither a string nor an integer.
    ValueError
        If ``cost`` or ``blocked`` has the wrong shape, or a cost is NaN or not positive,
        ``kind`` is unknown or out of range, or some cost along the way is so small beside the
        cost of getting there (below about 1e-16 of it) that adding it changes nothing.
    IndexError
        If ``start`` or ``goal`` is not a cell id.
    """
    graph = space.adjacency(kind)
    start = check_cell(start, space.n_cells)
    goal = check_cell(goal, space.n_cells)
    costs, walls = _costs(cost, blocked, space.n_cells)
    indptr, indices = graph.indptr.view(np.uint64), graph.indices.view(np.uint64)
    out = np.empty(space.n_cells, dtype=np.uint64)
    if fast():
        dist = np.empty(space.n_cells)
        source = np.array([start], dtype=np.uint64)
        _search(graph, source, costs, walls, math.inf, goal, dist)
        if dist[goal] == math.inf:
            return np.empty(0, dtype=np.int64)
        # Each cell after the start was entered, so it costs 1 if only walls were given.
        count = _kernels.walk_back(
            indptr, indices, costs, dist, np.uint64(start), np.uint64(goal), out
        )
    else:
        merged = _merged(costs, walls)
        dist = _reference_steps(graph, np.array([start]), merged, math.inf)
        if dist[goal] == math.inf:
            return np.empty(0, dtype=np.int64)
        count = _walk_back(graph, merged, dist, start, goal, out)
    if count == 0:
        raise ValueError(
            "a cost along the path is too small beside the cost of getting there: adding it "
            "changes nothing, so the path can't be traced back"
        )
    return out[:count][::-1].astype(np.int64)


def hops(
    space: LatticeSpace, a: ArrayLike, b: ArrayLike, *, kind: AdjacencyKind = "facet"
) -> NDArray[np.int64]:
    """Return how many moves between adjacent cells it takes from each cell to another.

    That is ``steps(space, a, kind=kind)[b]`` for every pair at once, without searching: the
    distance `Hops` counts. Squares and cubes have a closed form. On hexes and triangles, the
    moves to every displacement from a cell of each type are counted once per lattice and kind,
    and looked up after that, so they are exact on any torus, however thin.

    Parameters
    ----------
    space : LatticeSpace
        The cells.
    a, b : int or array_like of int, shape (...)
        Cell ids. They broadcast against each other.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells a move may go between, as for `Space.adjacency`. On squares, ``"vertex"``
        allows diagonal moves.

    Returns
    -------
    ndarray of int64, shape (...)
        The fewest moves from ``a`` to ``b``: 0 from a cell to itself, and the same both ways
        round.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, ``a`` or ``b`` does not hold integers, or
        ``kind`` is neither a string nor an integer.
    ValueError
        If ``kind`` is unknown or out of range.
    IndexError
        If ``a`` or ``b`` holds an id that is not a cell's.
    """
    if not isinstance(space, LatticeSpace):
        raise TypeError(f"hops needs a LatticeSpace, got {type(space).__name__}")
    a, b = check_ids(a, space.n_cells, "a"), check_ids(b, space.n_cells, "b")
    j = space._face_dim(kind)
    coords = space.coords
    moved = coords[b] - coords[a]  # broadcasts a against b
    if space.tiling == "hypercubic":
        lengths, wraps = np.array(space.shape, dtype=np.int64), np.array(space.periodic)
        # A move changes up to dim - j coordinates by 1 each.
        moved = np.where(wraps, moved % lengths, np.abs(moved))
        moved = np.where(wraps, np.minimum(moved, lengths - moved), moved)
        most = space.dim - j
        return np.maximum(moved.max(axis=-1, initial=0), -(-moved.sum(axis=-1) // most))
    table, base = _hop_table(space, j)
    types = space._types[a]
    at = base[types] + moved
    if any(space.periodic):
        lengths, wraps = np.array(space.shape, dtype=np.int64), np.array(space.periodic)
        at = np.where(wraps, at % lengths, at)
    return table[types, at[..., 0], at[..., 1]]  # hexes and triangles are 2-D


def _hop_table(space: LatticeSpace, j: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the moves from a cell of each type to every displacement, and where it lies.

    The moves are counted on a lattice of displacements: the same as ``space`` along the axes
    that wrap, and twice as long along the others, so that every displacement on ``space``
    lies on it. Returns read-only int64 arrays of shapes (n_types, *shape) and (n_types, dim):
    the moves from the cell of each type at ``base[type]`` to each cell, which a cell of that
    type at ``c`` needs to reach ``c + displacement``.
    """
    if j in space._hops:
        return space._hops[j]
    lengths = np.array(space.shape, dtype=np.int64)
    wraps = np.array(space.periodic)
    shape = np.where(wraps, lengths, 2 * lengths)
    room = LatticeSpace(tuple(shape.tolist()), space.tiling, space.periodic)
    tiling = space._tiling
    base = np.tile(np.where(wraps, 0, lengths - 1), (tiling.n_types, 1))
    if tiling.n_types > 1:  # move along axis 0, which changes the type by 1 a step
        base[:, 0] += (np.arange(tiling.n_types) - base @ tiling.type_steps) % tiling.n_types
    sources = room.ravel(base)
    moves = steps(room, sources, kind=j, separate=True).astype(np.int64)
    found = read_only(moves.reshape(tiling.n_types, *room.shape)), read_only(base)
    space._hops[j] = found
    return found


def _steps(
    graph: CellGraph,
    cells: NDArray[np.int64],
    costs: NDArray[np.float64] | None,
    walls: NDArray[np.bool_] | None,
    limit: float,
) -> NDArray[np.float64]:
    """Return the cheapest cost from the nearest of ``cells`` to each cell, on either path.

    ``cells`` are checked int64 cell ids, shape (n_sources,), and ``costs`` and ``walls`` the
    checked costs of entering each cell, as `_costs` returns them. Returns float64 of shape
    (n_cells,), inf beyond ``limit``.
    """
    if fast():
        dist = np.empty(graph.n_cells)
        _search(graph, cells.view(np.uint64), costs, walls, limit, graph.n_cells, dist)
        return dist
    return _reference_steps(graph, cells, _merged(costs, walls), limit)


def _search(
    graph: CellGraph,
    cells: NDArray[np.uint64],
    costs: NDArray[np.float64] | None,
    walls: NDArray[np.bool_] | None,
    limit: float,
    goal: int,
    dist: NDArray[np.float64],
) -> None:
    """Run the search kernel from ``cells`` into ``dist``, stopping at ``goal`` if a cell.

    ``cells`` are cell ids viewed as uint64, shape (n_sources,). ``costs`` and ``walls`` are
    as `_costs` returns them, and ``dist`` float64 of shape (n_cells,), as in
    `_kernels.dijkstra_steps`. Unit costs, with or without walls, search breadth first.
    """
    indptr, indices = graph.indptr.view(np.uint64), graph.indices.view(np.uint64)
    if costs is None:
        queue = np.empty(graph.n_cells, dtype=np.uint64)
        blocked = _NO_WALLS if walls is None else np.ascontiguousarray(walls)
        _kernels.bfs_steps(indptr, indices, cells, blocked, limit, np.uint64(goal), dist, queue)
    else:
        size = cells.size + graph.n_edges  # a push per source, and at most one per edge
        heap_d, heap_v = np.empty(size), np.empty(size, dtype=np.uint64)
        _kernels.dijkstra_steps(
            indptr, indices, costs, cells, limit, np.uint64(goal), dist, heap_d, heap_v
        )


def _reference_steps(
    graph: CellGraph,
    cells: NDArray[np.int64],
    costs: NDArray[np.float64] | None,
    limit: float,
) -> NDArray[np.float64]:
    """Return the cheapest cost from the nearest of ``cells`` to each cell, with SciPy.

    Arguments and result as for `_steps`.
    """
    if cells.size == 0:
        return np.full(graph.n_cells, math.inf)
    if costs is None:
        matrix, unweighted = graph.to_scipy(), True
    else:
        shape = (graph.n_cells, graph.n_cells)
        matrix = sp.csr_array((costs[graph.indices], graph.indices, graph.indptr), shape)
        unweighted = False
    return dijkstra(matrix, indices=cells, unweighted=unweighted, limit=limit, min_only=True)


def _walk_back(
    graph: CellGraph,
    costs: NDArray[np.float64] | None,
    dist: NDArray[np.float64],
    start: int,
    goal: int,
    out: NDArray[np.uint64],
) -> int:
    """Write a cheapest path into ``out``, goal first, as `_kernels.walk_back` does.

    Arguments and result as for the kernel, with ``start`` and ``goal`` as ints.
    """
    v, count = goal, 1
    out[0] = v
    while v != start:
        before = graph.neighbors(v)
        step = 1.0 if costs is None else costs[v]
        tight = (dist[before] < dist[v]) & (dist[before] + step == dist[v])
        if not tight.any():
            return 0
        v = int(before[np.argmax(tight)])  # the lowest: rows are sorted
        out[count] = v
        count += 1
    return count


_NO_WALLS = np.zeros(0, dtype=bool)
"""`_kernels.bfs_steps`' ``blocked`` when no cell is blocked."""


def _costs(
    cost: ArrayLike | None, blocked: ArrayLike | None, n_cells: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None]:
    """Return the costs of entering cells as float64, after checking them, or None for 1.

    Returns the costs, C-contiguous float64 of shape (n_cells,), positive or inf: inf where
    ``blocked``. Or, without ``cost``, None for costs of 1, and ``blocked``'s mask, bool of
    shape (n_cells,), or None: unit costs search breadth first on the fast path, around walls
    too. Merge the two with `_merged` for the reference path.

    Raises
    ------
    TypeError
        If ``cost`` does not hold numbers, or ``blocked`` is not boolean.
    ValueError
        If either does not have shape (n_cells,), or a cost is 0, negative or NaN.
    """
    costs = _checked_costs(cost, n_cells)
    if blocked is None:
        return costs, None
    walls = cell_mask(blocked, n_cells, "blocked")
    if costs is None:
        return None, walls
    return np.where(walls, np.inf, costs), None


def _merged(
    costs: NDArray[np.float64] | None, walls: NDArray[np.bool_] | None
) -> NDArray[np.float64] | None:
    """Return `_costs`' two results as one: float64 of shape (n_cells,), inf at walls, or None."""
    if walls is None:
        return costs
    return np.where(walls, np.inf, 1.0)


def _checked_costs(cost: ArrayLike | None, n_cells: int) -> NDArray[np.float64] | None:
    """Return ``cost`` as C-contiguous float64 of shape (n_cells,) after checking it, or None."""
    if cost is None:
        return None
    array = np.asarray(cost)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"cost must hold numbers, got dtype {array.dtype}")
    if array.shape != (n_cells,):
        raise ValueError(f"cost must have shape ({n_cells},), got {array.shape}")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not (array > 0).all():  # NaN fails too
        raise ValueError("cost must be positive, or inf where no move may enter, never NaN")
    return array


def _limit(value: object, name: str) -> float:
    """Return a limit on costs as a float: a real number, at least 0, inf allowed."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not number >= 0:  # NaN fails too
        raise ValueError(f"{name} must be at least 0, got {value!r}")
    return number
