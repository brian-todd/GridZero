"""Boards for Tactics: a hex rhombus with walls, rubble and a point, symmetric for both sides.

The rhombus of an axial hex lattice is symmetric under a half turn about its center
(`core.LatticeSpace.opposite`), which swaps the two corners of its short diagonal. Each team's
home is one of those corners, so a board made symmetric under that half turn favors neither
side. The point is the one tile that half turn leaves in place, so boards have odd lengths.
"""

import functools
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import LatticeSpace, check_int, check_real
from gridzero.game import highest, joined, pockets, symmetric, zones
from gridzero.worlds import noise

from .state import TacticsState

WAVENUMBERS = (2 * np.pi / 12, 2 * np.pi / 4)
"""Wavenumbers of the noise obstacles come from: wavelengths of 4 to 12 tiles, so cover is spread
in pieces of a few tiles over the whole board, rather than in a few large masses."""
_ATTEMPTS = 100
"""Boards drawn before giving up on one whose homes and point are joined by open tiles."""


_AROUND = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
"""The axial offsets of a hex's neighbors, in order around it."""


@functools.cache
def rings(shape: tuple[int, int]) -> NDArray[np.int64]:
    """Return each tile's neighbors in order around it, shared by every board of that shape.

    Parameters
    ----------
    shape : tuple of int
        ``(Q, R)``: the lengths of the two axial axes.

    Returns
    -------
    ndarray of int64, shape (n_cells, 6)
        The cell ids around each tile, ``n_cells`` where the board ends. Read-only.
    """
    q, r = shape
    table = np.empty((q * r, 6), dtype=np.int64)
    at_q, at_r = np.divmod(np.arange(q * r), r)
    for slot, (dq, dr) in enumerate(_AROUND):
        to_q, to_r = at_q + dq, at_r + dr
        inside = (to_q >= 0) & (to_q < q) & (to_r >= 0) & (to_r < r)
        table[:, slot] = np.where(inside, to_q * r + to_r, q * r)
    table.setflags(write=False)
    return table


@functools.cache
def board_space(shape: tuple[int, int]) -> LatticeSpace:
    """Return the hex lattice of a board, shared by every board of that shape.

    Parameters
    ----------
    shape : tuple of int
        ``(Q, R)``: the lengths of the two axial axes, each at least 2.

    Returns
    -------
    LatticeSpace
        A hex lattice that doesn't wrap. Spaces compare by identity, so sharing one lets
        positions of different games share their caches.
    """
    return LatticeSpace(shape, tiling="hex", periodic=False)


def center_cell(space: LatticeSpace) -> int:
    """Return the cell the half turn leaves in place, where the point is.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of odd shape ``(Q, R)``.

    Returns
    -------
    int
        The cell ``((Q - 1) / 2, (R - 1) / 2)``.

    Raises
    ------
    ValueError
        If an axis has an even length: the half turn then leaves no cell in place.
    """
    q, r = space.shape[0], space.shape[1]
    if q % 2 == 0 or r % 2 == 0:
        raise ValueError(f"a board needs odd lengths to have a center, got {space.shape}")
    return int(space.ravel([(q - 1) // 2, (r - 1) // 2]))


def home_corners(space: LatticeSpace) -> tuple[int, int]:
    """Return the cells at the corners of the short diagonal: team 0's home, then team 1's.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D hex lattice of shape ``(Q, R)``.

    Returns
    -------
    tuple of int
        The cells ``(Q - 1, 0)`` and ``(0, R - 1)``.
    """
    q, r = space.shape[0], space.shape[1]
    return int(space.ravel([q - 1, 0])), int(space.ravel([0, r - 1]))


def homes(space: LatticeSpace, depth: SupportsIndex) -> NDArray[np.int8]:
    """Return the team that deploys on each tile: those fewer than ``depth`` steps from its corner.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D hex lattice.
    depth : int
        Rows of each home, counted from its corner, at least 1. A home ``depth`` rows deep
        holds ``depth**2`` tiles when the board is large enough.

    Returns
    -------
    ndarray of int8, shape (n_cells,)
        0 or 1 on the homes, -1 elsewhere.

    Raises
    ------
    ValueError
        If ``depth`` is below 1, or the homes would touch or overlap.
    TypeError
        If ``depth`` is not an integer.
    """
    depth = check_int(depth, "depth", minimum=1)
    try:
        return zones(space, home_corners(space), depth - 1)
    except ValueError as error:
        message = f"homes {depth} rows deep overlap on a board of shape {space.shape}"
        raise ValueError(message) from error


def make_board(
    shape: tuple[int, int],
    *,
    seed: SupportsIndex,
    walls: float = 0.08,
    rubble: float = 0.05,
    home_depth: SupportsIndex = 5,
) -> TacticsState:
    """Make a board with no units: walls, rubble and homes, the same from both sides.

    Obstacles come from smooth noise (`worlds.noise`, see `WAVENUMBERS`), made symmetric under
    the half turn that swaps the homes (`game.symmetric`). Walls take the highest tiles and
    rubble the next ones, sparing the homes (`game.highest`) and the point at the center with the
    tiles around it. Open tiles cut off from the homes become walls (`game.pockets`), so every
    open tile can be reached. A board whose homes and point are cut off from each other
    (`game.joined`) is drawn again.

    Parameters
    ----------
    shape : tuple of int
        ``(Q, R)``, each odd and at least 3, so that the board has a center (`center_cell`).
    seed : int
        Seed of the noise. The same seed gives the same board.
    walls : float, default 0.08
        The share of the tiles outside the homes made walls, in [0, 1), before cut-off tiles
        are added.
    rubble : float, default 0.05
        The share of the tiles outside the homes made rubble, with ``walls + rubble < 1``.
    home_depth : int, default 5
        Rows of each home (see `homes`).

    Returns
    -------
    TacticsState
        A state with the board's static fields set and no units, at step 0.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range, an axis has an even length, or no board joining the
        homes and the point is found.
    """
    space = board_space((check_int(shape[0], "shape", 2), check_int(shape[1], "shape", 2)))
    seed = check_int(seed, "seed", minimum=0)
    walls = check_real(walls, "walls", 0.0, 1.0)
    rubble = check_real(rubble, "rubble", 0.0, 1.0)
    if walls + rubble >= 1:
        raise ValueError(f"walls + rubble must be below 1, got {walls} + {rubble}")
    home = homes(space, home_depth)
    opposite = space.opposite()
    outside = home < 0
    rng = np.random.default_rng(seed)
    corners = home_corners(space)
    center = center_cell(space)
    around = space.radius_graph(1.0)
    clear = np.append(around.indices[around.indptr[center] : around.indptr[center + 1]], center)
    point = np.zeros(space.n_cells, dtype=np.bool_)
    point[center] = True
    for _ in range(_ATTEMPTS):
        height = symmetric(noise(space.centers, rng, scales=WAVENUMBERS), opposite)
        wall = highest(height, walls, outside)
        rough = highest(height, walls + rubble, outside) & ~wall
        wall[clear] = rough[clear] = False
        open_ = ~(wall | rough)
        if joined(space, open_, [*corners, center]):
            wall |= pockets(space, open_, corners[0])
            return TacticsState(
                space,
                {
                    TacticsState.wall: wall,
                    TacticsState.rubble: rough,
                    TacticsState.home: home,
                    TacticsState.point: point,
                },
            )
    raise ValueError(f"no board of {walls} walls and {rubble} rubble joins homes and point")
