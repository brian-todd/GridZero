"""Lines between cells on a 2-D lattice: the cells a line crosses, and who sees whom.

A line runs straight between two cells' centers, to the nearest copy of the second across the
edges that wrap. It is traced exactly, in integer coordinates in which every center and corner
of the tiling is a lattice point, so whether it passes through a corner or along an edge is
settled without rounding.

A line **crosses** the cells whose inside it passes through. Where it passes exactly through a
corner or runs along an edge, it only **touches** the cells beside it there. Sight follows one
rule: a line is blocked by a blocked cell it crosses, or where blocked cells touch it from both
sides at one point. So nothing sees between two blocked squares that meet at a corner, but a
line may graze a single blocked cell. The two ends never block: a wall itself can be seen.
Lines are traced from the lower cell id, so sight is symmetric.
"""

import itertools
from dataclasses import dataclass
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import _kernels
from gridzero.core._arrays import cell_mask, cell_selection, check_cell, read_only
from gridzero.core.implementations import fast
from gridzero.core.lattice import LatticeSpace, _Frame
from gridzero.core.space import RADIUS_RTOL, check_radius

_MARGIN = 2
"""Lattice steps around a line's ends, along each axis, that may hold cells it touches.

A cell spans at most one step either way from its lattice coordinates (triangles, along x),
and a triangle's center lies 1/6 of a row from them.
"""


def line(
    space: LatticeSpace, a: SupportsIndex, b: SupportsIndex, *, touching: bool = False
) -> NDArray[np.int64]:
    """Return the cells a line from the center of ``a`` to the center of ``b`` passes through.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice: squares, hexagons or triangles.
    a, b : int
        Cell ids: where the line starts and ends.
    touching : bool, default False
        Whether to include the cells the line only touches, through a corner or along an
        edge. Without them, a line running along an edge leaves a gap there: on hexes, a line
        toward a corner of a hexagon runs along the edges between its neighbors.

    Returns
    -------
    ndarray of int64, shape (n,)
        Cell ids in order along the line, from ``a`` to ``b``, both included, each once. Cells
        touched at one point lie in order of their ids.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, or ``a`` or ``b`` is not an integer.
    ValueError
        If the lattice is not 2-D.
    IndexError
        If ``a`` or ``b`` is not a cell id.
    """
    _check_lattice(space)
    a, b = check_cell(a, space.n_cells), check_cell(b, space.n_cells)
    if a == b:
        return np.array([a], dtype=np.int64)
    low, high = min(a, b), max(a, b)
    start, end = _ends(space, np.array([low]), np.array([high]))
    trace = _trace(space, start[0], end[0])
    keep = trace.crossed | (touching & trace.touched)
    order = np.lexsort((trace.ids[keep], trace.keys[keep]))
    cells = trace.ids[keep][order]
    _, first = np.unique(cells, return_index=True)  # a short axis that wraps: once each
    cells = cells[np.sort(first)]
    return cells if a == low else cells[::-1]


def visible(space: LatticeSpace, a: SupportsIndex, b: SupportsIndex, blocked: ArrayLike) -> bool:
    """Return whether a line between the centers of two cells is clear, both ways round.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice: squares, hexagons or triangles.
    a, b : int
        Cell ids.
    blocked : array_like of bool, shape (n_cells,)
        The cells that block sight, such as walls.

    Returns
    -------
    bool
        False if a blocked cell other than ``a`` and ``b`` crosses the line, or blocked cells
        touch it from both sides at one point (see the module's notes).

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, ``a`` or ``b`` is not an integer, or
        ``blocked`` is not boolean.
    ValueError
        If the lattice is not 2-D, or ``blocked`` has the wrong shape.
    IndexError
        If ``a`` or ``b`` is not a cell id.
    """
    _check_lattice(space)
    a, b = check_cell(a, space.n_cells), check_cell(b, space.n_cells)
    walls = cell_mask(blocked, space.n_cells, "blocked")
    return bool(_clear(space, np.array([a]), np.array([b]), walls)[0])


def sight(
    space: LatticeSpace,
    viewers: ArrayLike,
    blocked: ArrayLike,
    radius: float | None = None,
    *,
    separate: bool = False,
) -> NDArray[np.bool_]:
    """Return the cells some viewer sees: those whose line from that viewer is clear.

    The same as `visible` for each cell, so a viewer sees a cell exactly when that cell's
    viewer would see it back. Many viewers cost one call: their lines are traced together.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice: squares, hexagons or triangles.
    viewers : int, array_like of int, shape (k,), or array_like of bool, shape (n_cells,)
        A cell id, cell ids, or a mask of cells, from which to look.
    blocked : array_like of bool, shape (n_cells,)
        The cells that block sight, such as walls. They can be seen themselves.
    radius : float, optional
        How far each viewer sees, between centers, as `within` measures. By default, the whole
        lattice.
    separate : bool, default False
        Whether to give each viewer a row of its own instead of what any of them sees.

    Returns
    -------
    ndarray of bool, shape (n_cells,), or (k, n_cells) if ``separate``
        Whether a viewer sees each cell. Each sees its own.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, ``viewers`` holds neither integers nor booleans,
        ``blocked`` is not boolean, or ``radius`` is not a real number.
    ValueError
        If the lattice is not 2-D, ``viewers`` or ``blocked`` has the wrong shape, or
        ``radius`` is negative, infinite or NaN.
    IndexError
        If a viewer is not a cell id.
    """
    _check_lattice(space)
    cells = cell_selection(viewers, space.n_cells, "viewers")
    walls = cell_mask(blocked, space.n_cells, "blocked")
    n = space.n_cells
    seen = np.zeros((cells.size, n), dtype=np.bool_)
    seen[np.arange(cells.size), cells] = True
    limit = None if radius is None else check_radius(radius) * (1 + RADIUS_RTOL)
    offsets = None if limit is None or not fast() else _sight_offsets(space, limit)
    if offsets is not None:
        frame, tiling = _frame(space), space._tiling
        rows = np.abs(offsets).max(axis=(0, 1), initial=0)
        size = int((rows[0] + 2 * _MARGIN + 3) * (rows[1] + 2 * _MARGIN + 1))
        _kernels.sight_offsets(
            frame.scale,
            frame.center,
            frame.corners,
            tiling.type_steps,
            np.array(space.shape, dtype=np.int64),
            np.array(space.periodic),
            cells,
            offsets,
            walls,
            seen,
            np.empty((4, size), dtype=np.int64),
        )
        return seen if separate else seen.any(axis=0)
    if limit is None:
        rows = np.repeat(np.arange(cells.size, dtype=np.int64), n)
        targets = np.tile(np.arange(n, dtype=np.int64), cells.size)
    else:
        rows, targets = space._pairs_within(cells, limit)
    others = targets != cells[rows]
    rows, targets = rows[others], targets[others]
    seen[rows, targets] = _clear(space, cells[rows], targets, walls)
    return seen if separate else seen.any(axis=0)


class SightLines(NamedTuple):
    """What may block the line from a cell to each cell within a radius, by lattice offset.

    Row ``i`` of every array is about the line from a cell at coordinates ``c`` to the cell at
    ``c + offsets[i]``. That line is clear unless a cell of ``crossed[i]`` is blocked, or both
    cells of a pair of ``touched[i]`` are. Cells beyond the lattice's edge block nothing.
    """

    offsets: NDArray[np.int64]
    """Lattice offsets to the cells within the radius, shape (m, 2): the cell itself first, then
    by distance between centers, nearest first, and by offset among those equally far. So the
    cells within any smaller radius come first."""
    crossed: NDArray[np.int64]
    """Rows of `offsets` whose cells the line crosses, its two ends aside: shape (m, s), in
    ascending order, -1 in unused slots."""
    touched: NDArray[np.int64]
    """Pairs of rows of `offsets` whose cells touch the line from both sides at one point:
    shape (m, p, 2), the cell on the line's left first, -1 in unused slots."""


def sight_lines(space: LatticeSpace, radius: float, cell: SupportsIndex = 0) -> SightLines:
    """Return what may block the line from a cell to each cell within ``radius``, as tables.

    Lines between cells of a lattice depend only on the offset between them, so sight within a
    radius is a table: `sight` from a cell, read from the tables, is ``True`` at the cell of
    each offset whose row finds nothing blocking (see `SightLines`). Use it where lines are
    looked up rather than traced, such as rules that run for many boards at once.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice: squares, hexagons or triangles.
    radius : float
        How far to look, between centers, as `within` measures.
    cell : int, default 0
        A cell id. On squares and hexagons every cell has the same tables. On triangles, those
        pointing up have one set and those pointing down another.

    Returns
    -------
    SightLines
        The offsets to the cells within ``radius``, and for each the cells that block its
        line. Offsets that leave the lattice from every cell are left out.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, ``radius`` is not a real number, or ``cell`` is
        not an integer.
    ValueError
        If the lattice is not 2-D, ``radius`` is negative, infinite or NaN, or it reaches ``r``
        cells along an axis that wraps and is shorter than ``2 * r + 1``: two offsets would
        then reach one cell.
    IndexError
        If ``cell`` is not a cell id.
    """
    _check_lattice(space)
    tiling, frame = space._tiling, _frame(space)
    kind = int(space.coords[check_cell(cell, space.n_cells)] @ tiling.type_steps % tiling.n_types)
    limit = check_radius(radius) * (1 + RADIUS_RTOL)
    reach = space._offset_reach(limit)
    space._check_wrap(radius, np.where(space.periodic, reach, 0))
    near = space._offsets_within(limit)[0][kind]
    # A room around one cell of the same type, with every cell a line within the radius meets.
    base = reach + _MARGIN + 1
    base[0] += (kind - base @ tiling.type_steps) % tiling.n_types
    room = LatticeSpace(tuple((2 * base + 1).tolist()), space.tiling, False)
    here, _ = _frame_centers(room, base)
    there, _ = _frame_centers(room, base + near)
    step = there - here
    squared = np.einsum("...i,ij,...j->...", step, frame.metric, step)
    near = near[np.lexsort((near[:, 1], near[:, 0], squared))]
    offsets = np.concatenate([np.zeros((1, 2), dtype=np.int64), near])
    row = {(int(dq), int(dr)): i for i, (dq, dr) in enumerate(offsets)}
    start = int(room.ravel(base))
    crossed: list[list[int]] = [[]]
    touched: list[list[tuple[int, int]]] = [[]]
    for offset in near:
        end = int(room.ravel(base + offset))
        low, high = min(start, end), max(start, end)
        trace = _trace(room, room.coords[low], room.coords[high])
        between = (trace.ids != start) & (trace.ids != end)
        # A cell the table lacks shares no lattice with the viewer, so it never blocks: the
        # cells a line meets are no further from its start than its end is.
        rows = np.array(
            [row.get((int(dq), int(dr)), -1) for dq, dr in room.coords[trace.ids] - base],
            dtype=np.int64,
        )
        between &= rows >= 0
        crossed.append(sorted(rows[between & trace.crossed].tolist()))
        left = (between & trace.touched & (trace.side > 0)).nonzero()[0]
        right = (between & trace.touched & (trace.side < 0)).nonzero()[0]
        touched.append(
            sorted(
                (int(rows[a]), int(rows[b]))
                for a, b in itertools.product(left.tolist(), right.tolist())
                if max(trace.low[a], trace.low[b]) <= min(trace.high[a], trace.high[b])
            )
        )
    alone = np.full((len(offsets), max(map(len, crossed))), -1, dtype=np.int64)
    pairs = np.full((len(offsets), max(map(len, touched)), 2), -1, dtype=np.int64)
    for i, (cells, both) in enumerate(zip(crossed, touched, strict=True)):
        alone[i, : len(cells)] = cells
        if both:
            pairs[i, : len(both)] = both
    return SightLines(read_only(offsets), read_only(alone), read_only(pairs))


def _sight_offsets(space: LatticeSpace, limit: float) -> NDArray[np.int64] | None:
    """Return the offsets to the cells within ``limit``, if each is a line's nearest copy.

    Offsets are int64 of shape (n_types, m, 2), as `LatticeSpace._offsets_within` gives them.
    They are None where `_pairs_within` compares every center instead, or where an axis that
    wraps is shorter than ``2 * reach + 1``: two offsets could then reach one cell, and one
    of them would not be the step `_ends` takes.
    """
    reach = space._offset_reach(limit)
    if np.prod(2 * reach + 1, dtype=np.float64) > space.n_cells:
        return None
    if any(
        w and length < 2 * r + 1
        for w, length, r in zip(space.periodic, space.shape, reach, strict=True)
    ):
        return None
    offsets, _ = space._offsets_within(limit)
    return offsets


@dataclass(frozen=True, eq=False)
class _Trace:
    """The cells around a line, as `_trace` finds them.

    Attributes, one entry per cell near the line, in no particular order: ``ids`` (int64),
    ``crossed`` and ``touched`` (bool: the line passes through the inside, or touches it from
    ``side`` +1, the left, or -1, the right), ``low`` and ``high`` (int64: where along the line
    the cell's contact begins and ends, in units of ``1 / |D|**2`` of the line), and ``keys``
    (float64: the middle of its stretch of line, from 0 at the start to 1 at the end).
    """

    ids: NDArray[np.int64]
    crossed: NDArray[np.bool_]
    touched: NDArray[np.bool_]
    side: NDArray[np.int64]
    low: NDArray[np.int64]
    high: NDArray[np.int64]
    keys: NDArray[np.float64]


def _trace(space: LatticeSpace, start: NDArray[np.int64], end: NDArray[np.int64]) -> _Trace:
    """Find the cells a line crosses and touches, from lattice coordinates ``start`` to ``end``.

    ``end`` may lie beyond the grid, on a copy across an edge that wraps. Every cell in a box
    around the two is classified by the sides of the line its corners lie on, with integers.
    ``start`` and ``end`` are int64 of shape (2,).
    """
    frame = _frame(space)
    p, _ = _frame_centers(space, start)
    q, _ = _frame_centers(space, end)
    d = q - p
    dd = int(d @ d)
    low, high = np.minimum(start, end) - _MARGIN, np.maximum(start, end) + _MARGIN
    coords = np.indices((high - low + 1).tolist(), dtype=np.int64).reshape(2, -1).T + low
    centers, types = _frame_centers(space, coords)
    corners = centers[:, None, :] + frame.corners[types]  # (m, k, 2)
    w = corners - p
    side = np.sign(d[0] * w[..., 1] - d[1] * w[..., 0])
    along = w @ d  # where each corner projects onto the line, in units of 1 / |d|**2
    on_line = side == 0
    left, right = (side > 0).any(axis=1), (side < 0).any(axis=1)
    touches = on_line.any(axis=1)
    crossed, touched = left & right, touches & ~(left & right)
    # Where the line enters and leaves each crossed cell: through corners on it, or across an
    # edge whose ends lie on either side, at t = cross(w, e) / cross(d, e).
    e = np.roll(corners, -1, axis=1) - corners
    across = side * np.roll(side, -1, axis=1) < 0
    num = w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]
    den = d[0] * e[..., 1] - d[1] * e[..., 0]
    num, den = np.where(den < 0, -num, num), np.abs(den)
    # Cells other than the ends' meet the line apart from its ends, all inside it or all out.
    # One point where each meets the line tells which.
    inside = (on_line & (along > 0) & (along < dd)) | (across & (num > 0) & (num < den))
    inside = inside.any(axis=1)
    meets = on_line | across
    t = np.where(on_line, along / dd, num / np.where(across, den, 1))
    first = np.where(meets, t, np.inf).min(axis=1)
    last = np.where(meets, t, -np.inf).max(axis=1)
    keys = np.where(meets.any(axis=1), first, 0.0) / 2 + np.where(meets.any(axis=1), last, 0.0) / 2
    is_start, is_end = (coords == start).all(axis=1), (coords == end).all(axis=1)
    keys = np.where(is_start, -np.inf, np.where(is_end, np.inf, keys))
    inside |= is_start | is_end
    ids = space._wrapped_ids(coords)
    real = inside & (ids >= 0)
    big = np.iinfo(np.int64)
    return _Trace(
        ids=ids[real],
        crossed=crossed[real],
        touched=touched[real],
        side=np.where(left, 1, -1)[real],
        low=np.where(on_line, along, big.max).min(axis=1)[real],
        high=np.where(on_line, along, big.min).max(axis=1)[real],
        keys=keys[real],
    )


def _clear(
    space: LatticeSpace,
    a: NDArray[np.int64],
    b: NDArray[np.int64],
    walls: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return whether the line between each pair of cells is clear, on either path.

    ``a`` and ``b`` are cell ids, int64 of shape (n,), and ``walls`` marks blocked cells, bool
    of shape (n_cells,). Returns bool of shape (n,).
    """
    low, high = np.minimum(a, b), np.maximum(a, b)
    start, end = _ends(space, low, high)
    if fast():
        frame, tiling = _frame(space), space._tiling
        rows = np.abs(end - start)
        size = int(((rows[:, 0] + 2 * _MARGIN + 3) * (rows[:, 1] + 2 * _MARGIN + 1)).max(initial=0))
        out = np.empty(a.size, dtype=np.bool_)
        _kernels.clear_lines(
            frame.scale,
            frame.center,
            frame.corners,
            tiling.type_steps,
            np.array(space.shape, dtype=np.int64),
            np.array(space.periodic),
            start,
            end,
            low,
            high,
            walls,
            out,
            np.empty((4, max(size, 1)), dtype=np.int64),
        )
        return out
    out = np.empty(a.size, dtype=np.bool_)
    for k in range(a.size):
        out[k] = low[k] == high[k] or _clear_reference(
            _trace(space, start[k], end[k]), walls, low[k], high[k]
        )
    return out


def _clear_reference(trace: _Trace, walls: NDArray[np.bool_], a: int, b: int) -> bool:
    """Return whether no blocked cell blocks a traced line between cells ``a`` and ``b``.

    ``walls`` is bool of shape (n_cells,). The line's own cells never block it.
    """
    blocking = walls[trace.ids] & (trace.ids != a) & (trace.ids != b)
    if (blocking & trace.crossed).any():
        return False
    left = blocking & trace.touched & (trace.side > 0)
    right = blocking & trace.touched & (trace.side < 0)
    overlap = np.maximum.outer(trace.low[left], trace.low[right]) <= np.minimum.outer(
        trace.high[left], trace.high[right]
    )
    return not overlap.any()


def _ends(
    space: LatticeSpace, a: NDArray[np.int64], b: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the lattice coordinates of cells ``a``, and of the copies of ``b`` nearest them.

    Shapes (n, 2). The copies lie whole periods away along the axes that wrap. Lengths are
    compared exactly, in the integer frame, among the copies around the one `_nearest_copy`
    finds. Of copies equally near, the one with the lowest turns along axis 0, then axis 1,
    is taken, so the choice never depends on what else is asked along with it.
    """
    start, end = space.coords[a], space.coords[b]
    wrapped = np.flatnonzero(space.periodic)
    if not wrapped.size:
        return start, end
    lengths = np.array(space.shape, dtype=np.int64)[wrapped]
    step = space.centers[b] - space.centers[a]
    turns = (space._nearest_copy(step) - step) @ space._inverse_basis.T
    turns = np.rint(turns[:, wrapped] / lengths).astype(np.int64)
    around = np.array(list(itertools.product((-1, 0, 1), repeat=wrapped.size)), dtype=np.int64)
    turns = turns[:, None, :] + around @ space._period_turns.T  # (n, 3**m, m)
    copies = np.repeat(end[:, None, :], len(around), axis=1)
    copies[..., wrapped] += turns * lengths
    centers, _ = _frame_centers(space, copies)
    first, _ = _frame_centers(space, start)
    d = centers - first[:, None, :]
    squared = np.einsum("...i,ij,...j->...", d, _frame(space).metric, d)
    # The lowest turns: read as digits of a number, in a base wider than any two apart.
    base = 2 * int(np.abs(turns).max(initial=0)) + 1
    rank = turns[..., 0] * base + turns[..., 1] if wrapped.size == 2 else turns[..., 0]
    rank = np.where(squared == squared.min(axis=1, keepdims=True), rank, np.iinfo(np.int64).max)
    best = np.argmin(rank, axis=1)
    return start, copies[np.arange(len(copies)), best]


def _frame_centers(
    space: LatticeSpace, coords: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the centers of cells at lattice coordinates in the integer frame, and types.

    ``coords`` is int64 of shape (..., 2). Returns int64 centers of the same shape, and int64
    types of shape (...).
    """
    tiling, frame = space._tiling, _frame(space)
    types = coords @ tiling.type_steps % tiling.n_types
    return coords @ frame.scale.T + frame.center[types], types


def _check_lattice(space: object) -> None:
    """Raise unless ``space`` is a 2-D lattice."""
    if not isinstance(space, LatticeSpace):
        raise TypeError(f"lines need a 2-D LatticeSpace, got {type(space).__name__}")
    _frame(space)


def _frame(space: LatticeSpace) -> _Frame:
    """Return the integer frame of a 2-D lattice, or raise if the lattice has none."""
    frame = space._tiling.frame
    if frame is None:
        raise ValueError(f"lines need a 2-D lattice, got one with {space.dim} axes")
    return frame
