"""Queries about cells as JAX arrays: moves on any space, and sight and paths around a cell.

Two kinds. `steps` and `reach` search a whole space through an `Operator`, one dilation per
move. The others look a few cells around one cell of a lattice, through a `Patch`: tables by
lattice offset, which `core` fills once (`core.sight_lines`, `core.snap`, `core.in_front`) and
these functions only gather. A piece's moves, targets and lines of fire then cost the same on
any board size.
"""

from dataclasses import dataclass
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from gridzero.batched.operators import Operator
from gridzero.core import LatticeSpace, check_int, in_front, sight_lines, snap
from gridzero.core.space import RADIUS_RTOL, check_radius

FAR: int = 1 << 30
"""The steps, or the cost, to a cell that isn't reached: what stands for `core`'s inf."""


def steps(
    op: Operator,
    sources: ArrayLike,
    *,
    blocked: ArrayLike | None = None,
    limit: SupportsIndex | None = None,
) -> jax.Array:
    """Return how many moves it takes to reach each cell from the nearest source.

    `core.steps` for moves that cost 1, between the cells ``op`` joins: with
    ``operator(Adjacency().compile(space))``, between adjacent cells.

    Parameters
    ----------
    op : Operator
        Which cells a move may go between: from a cell to its neighbors.
    sources : Array of bool, shape (n_cells,)
        The cells to start from. They take 0 moves, blocked or not.
    blocked : Array of bool, shape (n_cells,), optional
        Cells no move may enter, such as walls.
    limit : int, optional
        The most moves to count, at least 0. Cells beyond it count as not reached. By
        default, every cell that can be reached is.

    Returns
    -------
    Array of int32, shape (n_cells,)
        The fewest moves from a source to each cell, and `FAR` where none reaches it.

    Raises
    ------
    TypeError
        If ``limit`` is not an integer.
    ValueError
        If ``limit`` is negative.
    """
    n = op.idx.shape[0]
    most = n if limit is None else check_int(limit, "limit", minimum=0)
    sources = jnp.asarray(sources, dtype=jnp.bool_)
    shut = jnp.zeros(n, jnp.bool_) if blocked is None else jnp.asarray(blocked, jnp.bool_)

    def going(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        moves, _, front = carry
        return front.any() & (moves < most)

    def move(
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        moves, found, front = carry
        entered = jnp.append(front, False)[op.back].any(axis=-1) & (found >= FAR) & ~shut
        return moves + 1, jnp.where(entered, moves + 1, found), entered

    start = (jnp.int32(0), jnp.where(sources, 0, FAR).astype(jnp.int32), sources)
    return jax.lax.while_loop(going, move, start)[1]


def reach(
    op: Operator,
    sources: ArrayLike,
    budget: SupportsIndex | None = None,
    *,
    blocked: ArrayLike | None = None,
) -> jax.Array:
    """Return the cells some source reaches in at most ``budget`` moves.

    `core.reach` for moves that cost 1. Without a budget it is the flood from the sources:
    every cell joined to one, as `core.regions` groups them.

    Parameters
    ----------
    op : Operator
        Which cells a move may go between: from a cell to its neighbors.
    sources : Array of bool, shape (n_cells,)
        The cells to start from. They are reached, blocked or not.
    budget : int, optional
        The most moves, at least 0. By default, as many as it takes.
    blocked : Array of bool, shape (n_cells,), optional
        Cells no move may enter, such as walls.

    Returns
    -------
    Array of bool, shape (n_cells,)
        Whether a source reaches each cell.

    Raises
    ------
    TypeError
        If ``budget`` is not an integer.
    ValueError
        If ``budget`` is negative.
    """
    limit = None if budget is None else check_int(budget, "budget", minimum=0)
    return steps(op, sources, blocked=blocked, limit=limit) < FAR


@dataclass(frozen=True, eq=False)
class Patch:
    """The cells around each cell of a lattice, by offset, as tables (see `patch`).

    A slot is an offset between cells. Slot 0 is the cell itself, and the others follow by the
    distance between centers, nearest first, as `core.sight_lines` orders them: so the cells
    within a smaller radius are the first slots (`within`). ``size`` stands for "no slot" in
    every table of slots, and ``n_cells`` for "off the lattice" in `at`.
    """

    offsets: NDArray[np.int64]
    """The lattice offset of each slot: shape (size, 2), read-only."""
    distance: NDArray[np.float64]
    """How far each slot's center is: shape (size,), read-only."""
    moves: int
    """The most moves that surely stay inside the patch: every cell that many moves from the
    cell has a slot."""
    at: jax.Array
    """The cell in each slot of each cell: int32, shape (n_cells + 1, size). The last row, for
    a cell off the lattice, is off the lattice everywhere."""
    beside: jax.Array
    """The slots adjacent to each slot, in order of `core.LatticeSpace.directions`: int32,
    shape (size, n_directions)."""
    crossed: jax.Array
    """The slots whose cells block the line to each slot when blocked: int32, shape (size, s)."""
    touched: jax.Array
    """Pairs of slots whose cells block the line to each slot when both are blocked: int32,
    shape (size, p, 2)."""
    turned: jax.Array
    """The direction nearest each slot's offset, per direction faced before (`core.snap`):
    int32, shape (size, n_directions)."""
    front: jax.Array
    """Whether each slot's offset points into the half a piece faces (`core.in_front`), per
    facing: bool, shape (n_directions, size)."""
    step: jax.Array
    """`offsets` as a JAX array: int32, shape (size, 2)."""
    rows: jax.Array
    """The slot of each offset from ``(-span, -span)`` to ``(span, span)``: int32, shape
    (2 * span + 1, 2 * span + 1)."""
    span: int
    """The largest offset of a slot along either axis."""

    @property
    def size(self) -> int:
        """The number of slots, the cell's own included."""
        return len(self.offsets)

    @property
    def n_cells(self) -> int:
        """The number of cells of the lattice."""
        return self.at.shape[0] - 1

    def within(self, radius: float) -> int:
        """Return how many slots lie within a radius: the first that many.

        Parameters
        ----------
        radius : float
            A distance between centers, as `core.within` measures, at most the patch's own.

        Returns
        -------
        int
            The count, the cell's own slot included.

        Raises
        ------
        TypeError
            If ``radius`` is not a real number.
        ValueError
            If ``radius`` is negative, infinite or NaN.
        """
        return int(np.count_nonzero(self.distance <= check_radius(radius) * (1 + RADIUS_RTOL)))


def patch(space: LatticeSpace, radius: float) -> Patch:
    """Return the tables of the cells within ``radius`` of each cell of a lattice.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of squares or hexes, whose cells are all alike.
    radius : float
        How far the patch reaches, between centers, as `core.within` measures.

    Returns
    -------
    Patch
        One slot per offset within ``radius``.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, or ``radius`` is not a real number.
    ValueError
        If the lattice is not of squares or hexes in 2-D, ``radius`` is negative, infinite or
        NaN, or it reaches ``r`` cells along an axis that wraps and is shorter than
        ``2 * r + 1``.
    """
    lines = sight_lines(space, radius)  # checks the space and the radius
    directions = space.directions()  # raises on triangles
    offsets = lines.offsets
    size = len(offsets)
    shape, wraps = np.array(space.shape), np.array(space.periodic)
    coords = space.coords[:, None, :] + offsets
    coords = np.where(wraps, coords % shape, coords)
    inside = ((coords >= 0) & (coords < shape)).all(axis=-1)
    at = np.full((space.n_cells + 1, size), space.n_cells, dtype=np.int32)
    at[:-1] = np.where(inside, coords @ np.array([space.shape[1], 1]), space.n_cells)
    span = int(np.abs(offsets).max())
    rows = np.full((2 * span + 1, 2 * span + 1), size, dtype=np.int32)
    rows[offsets[:, 0] + span, offsets[:, 1] + span] = np.arange(size)
    beyond = offsets[:, None, :] + directions  # (size, n_directions, 2)
    known = (np.abs(beyond) <= span).all(axis=-1)
    near = np.clip(beyond, -span, span) + span
    beside = np.where(known, rows[near[..., 0], near[..., 1]], size)
    # A neighbor without a slot ends the patch there, unless no cell has such a neighbor.
    exists = (wraps | (np.abs(beyond) < shape)).all(axis=-1)
    hop = np.full(size + 1, size, dtype=np.int64)
    hop[0] = 0
    for move in range(1, size):
        entered = (hop[beside] == move - 1).any(axis=1) & (hop[:size] == size)
        hop[:size][entered] = move
    open_ = ((beside == size) & exists).any(axis=1)
    moves = int(hop[:size][open_].min()) if open_.any() else space.n_cells
    turned = np.array(
        [[snap(space, offset, d) for d in range(len(directions))] for offset in offsets]
    )
    front = np.stack([in_front(space, d, offsets) for d in range(len(directions))])
    distance = np.linalg.norm(offsets @ space.basis.T, axis=-1)
    distance.setflags(write=False)
    return Patch(
        offsets=offsets,
        distance=distance,
        moves=moves,
        at=jnp.asarray(at),
        beside=jnp.asarray(beside, dtype=jnp.int32),
        crossed=jnp.asarray(np.where(lines.crossed >= 0, lines.crossed, size), dtype=jnp.int32),
        touched=jnp.asarray(np.where(lines.touched >= 0, lines.touched, size), dtype=jnp.int32),
        turned=jnp.asarray(turned, dtype=jnp.int32),
        front=jnp.asarray(front),
        step=jnp.asarray(offsets, dtype=jnp.int32),
        rows=jnp.asarray(rows),
        span=span,
    )


def slot(patch: Patch, offset: ArrayLike) -> jax.Array:
    """Return the slot of a patch at each lattice offset.

    Parameters
    ----------
    patch : Patch
        The tables of a lattice.
    offset : Array of int, shape (..., 2)
        Offsets between cells: the coordinates of one less those of the other.

    Returns
    -------
    Array of int32, shape (...)
        The slot, and ``patch.size`` for an offset beyond the patch.
    """
    offset = jnp.asarray(offset, dtype=jnp.int32)
    span = patch.span
    known = (jnp.abs(offset) <= span).all(axis=-1)
    at = jnp.clip(offset, -span, span) + span
    return jnp.where(known, patch.rows[at[..., 0], at[..., 1]], patch.size)


def clear(patch: Patch, blocked: ArrayLike, cell: ArrayLike) -> jax.Array:
    """Return which of the cells around a cell it sees: those with a clear line to it.

    One viewer's `core.sight`, by slot.

    Parameters
    ----------
    patch : Patch
        The tables of a lattice.
    blocked : Array of bool, shape (n_cells,)
        The cells that block sight, such as walls. They can be seen themselves.
    cell : Array of int, shape ()
        The viewer's cell id, or ``n_cells`` for none.

    Returns
    -------
    Array of bool, shape (size,)
        Whether the line to each slot's cell is clear. False where the slot is off the
        lattice. A cell sees itself.
    """
    at = patch.at[cell]
    there = jnp.append(jnp.append(jnp.asarray(blocked, jnp.bool_), False)[at], False)
    alone = there[patch.crossed].any(axis=-1)
    both = (there[patch.touched[..., 0]] & there[patch.touched[..., 1]]).any(axis=-1)
    return ~alone & ~both & (at < patch.n_cells)


def sight(
    patch: Patch, blocked: ArrayLike, viewers: ArrayLike, count: ArrayLike | None = None
) -> jax.Array:
    """Return the cells some viewer sees, as `core.sight` does within a radius.

    Parameters
    ----------
    patch : Patch
        The tables of a lattice.
    blocked : Array of bool, shape (n_cells,)
        The cells that block sight, such as walls. They can be seen themselves.
    viewers : Array of int, shape (k,)
        The cells to look from. A negative entry is no viewer.
    count : Array of int, shape (k,) or (), optional
        How many slots each viewer looks at, nearest first: ``patch.within(radius)`` for a
        viewer that sees as far as ``radius``. By default, the whole patch.

    Returns
    -------
    Array of bool, shape (n_cells,)
        Whether a viewer sees each cell. Each sees its own.
    """
    n = patch.n_cells
    viewers = jnp.asarray(viewers)
    cells = jnp.where(viewers >= 0, viewers, n)
    seen = jax.vmap(lambda cell: clear(patch, blocked, cell))(cells)
    if count is not None:
        limit = jnp.broadcast_to(jnp.asarray(count), cells.shape)
        seen = seen & (jnp.arange(patch.size) < limit[:, None])
    return jnp.zeros(n + 1, jnp.bool_).at[patch.at[cells]].max(seen)[:n]


def steps_around(
    patch: Patch,
    cell: ArrayLike,
    blocked: ArrayLike,
    moves: SupportsIndex,
    cost: ArrayLike | None = None,
) -> jax.Array:
    """Return what it costs to reach each cell around a cell in at most ``moves`` moves.

    `core.steps` from one cell, by slot, wherever the cheapest way takes at most ``moves``
    moves: always, when every move costs 1.

    Parameters
    ----------
    patch : Patch
        The tables of a lattice.
    cell : Array of int, shape ()
        The cell to start from.
    blocked : Array of bool, shape (n_cells,)
        Cells no move may enter, such as walls and other pieces.
    moves : int
        The most moves, from 0 to ``patch.moves``.
    cost : Array of int, shape (n_cells,), optional
        What entering each cell costs, positive, with sums well below `FAR`. By default,
        every move costs 1.

    Returns
    -------
    Array of int32, shape (size,)
        The cost of the cheapest way to each slot's cell in at most ``moves`` moves: 0 for
        the cell itself, and `FAR` where there is none.

    Raises
    ------
    TypeError
        If ``moves`` is not an integer.
    ValueError
        If ``moves`` is negative or more than ``patch.moves``.
    """
    return _steps_around(patch, cell, blocked, _check_moves(patch, moves), cost)[0]


def path_around(
    patch: Patch,
    cell: ArrayLike,
    goal: ArrayLike,
    blocked: ArrayLike,
    moves: SupportsIndex,
    cost: ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the cheapest path from a cell to one of the cells around it, as slots.

    `core.path`, wherever the cheapest way takes at most ``moves`` moves: of the paths that
    cost as little, the one found by stepping back from the goal, each time to the neighbor
    with the lowest cell id that a cheapest path may come from.

    Parameters
    ----------
    patch : Patch
        The tables of a lattice.
    cell : Array of int, shape ()
        The cell to start from.
    goal : Array of int, shape ()
        The slot to reach.
    blocked : Array of bool, shape (n_cells,)
        Cells no move may enter, such as walls and other pieces.
    moves : int
        The most moves, from 0 to ``patch.moves``.
    cost : Array of int, shape (n_cells,), optional
        What entering each cell costs, positive, with sums well below `FAR`. By default,
        every move costs 1.

    Returns
    -------
    route : Array of int32, shape (moves + 1,)
        The slot after each move: slot 0 first, then the path's, then the goal's for the
        moves it doesn't need. All 0 if the goal isn't reached.
    length : Array of int32, shape ()
        The path's moves, or -1 if the goal isn't reached in ``moves`` moves.

    Raises
    ------
    TypeError
        If ``moves`` is not an integer.
    ValueError
        If ``moves`` is negative or more than ``patch.moves``.
    """
    most = _check_moves(patch, moves)
    found, entry, cells = _steps_around(patch, cell, blocked, most, cost)
    padded = jnp.append(found, FAR)
    ids = jnp.append(cells, patch.n_cells)
    here = jnp.asarray(goal, dtype=jnp.int32)
    back = []
    for _ in range(most):
        back.append(here)
        before = patch.beside[here]
        tight = (padded[before] < found[here]) & (padded[before] + entry[here] == found[here])
        lowest = before[jnp.argmin(jnp.where(tight, ids[before], patch.n_cells + 1))]
        here = jnp.where(tight.any(), lowest, here)
    reached = found[goal] < FAR
    if not back:
        return jnp.zeros(1, jnp.int32), jnp.where(reached, 0, -1).astype(jnp.int32)
    trail = jnp.stack(back)  # the goal first
    length = (trail != 0).sum().astype(jnp.int32)
    after = trail[jnp.clip(length - jnp.arange(1, most + 1), 0, most - 1)]
    route = jnp.concatenate([jnp.zeros(1, jnp.int32), after])
    return jnp.where(reached, route, 0), jnp.where(reached, length, -1)


def _steps_around(
    patch: Patch, cell: ArrayLike, blocked: ArrayLike, moves: int, cost: ArrayLike | None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the cost to each slot, what entering each slot costs, and each slot's cell.

    The work of `steps_around`: three int32 arrays of shape (size,).
    """
    cells = patch.at[cell]
    shut = jnp.append(jnp.asarray(blocked, jnp.bool_), True)[cells]
    if cost is None:
        entry = jnp.ones(patch.size, jnp.int32)
    else:
        entry = jnp.append(jnp.asarray(cost, jnp.int32), 1)[cells]
    found = jnp.where(jnp.arange(patch.size) == 0, 0, FAR).astype(jnp.int32)
    for _ in range(moves):
        near = jnp.append(found, FAR)[patch.beside].min(axis=-1)
        through = jnp.where(shut | (near >= FAR), FAR, near + entry)
        found = jnp.minimum(found, through)
    return found, entry, cells


def _check_moves(patch: Patch, moves: SupportsIndex) -> int:
    """Return ``moves`` as an int, checked to stay inside the patch."""
    most = check_int(moves, "moves", minimum=0)
    if most > patch.moves:
        raise ValueError(
            f"moves must be at most {patch.moves}, as far as this patch surely reaches, got {most}"
        )
    return most
