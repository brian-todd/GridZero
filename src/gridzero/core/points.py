"""Random points in the cells of a space, spread evenly by area."""

import weakref
from collections.abc import Callable
from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import cell_mask, check_int, read_only
from gridzero.core.lattice import LatticeSpace
from gridzero.core.mesh import MeshSpace

_CAPS: weakref.WeakKeyDictionary[MeshSpace, tuple[NDArray[np.float64], NDArray[np.float64]]]
_CAPS = weakref.WeakKeyDictionary()
"""The caps of each spherical mesh drawn in, made once: meshes are immutable."""


def random_points(
    space: LatticeSpace | MeshSpace,
    rng: np.random.Generator,
    size: SupportsIndex | None = None,
    *,
    where: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return points drawn at random, evenly over the cells of a space or some of them.

    Each cell gets points in proportion to its area (its volume, in 3-D and up), and every
    point of a cell is as likely as every other. The points are in the coordinates `locate`
    takes, so ``space.locate(points)`` gives their cells:

    - On a lattice, they are points of the space of ``space.centers``. Along each axis that
      wraps, they lie in one period, from half a step before the first cell: the part of a 2-D
      lattice that `render.flat_map` draws.
    - On a spherical mesh, they are points of the unit sphere.

    Parameters
    ----------
    space : LatticeSpace or MeshSpace
        A lattice, or a spherical mesh whose cells are the Voronoi cells of their centers, as
        on `icosphere`.
    rng : numpy.random.Generator
        Source of the points.
    size : int, optional
        How many points to draw, at least 0. By default, one.
    where : array_like of bool, shape (n_cells,), optional
        The cells the points must lie in. By default, every cell.

    Returns
    -------
    ndarray of float64, shape (dim,) or (size, dim)
        One point, or ``size`` of them, where ``dim`` is the number of axes of a lattice, and
        3 on a sphere.

    Raises
    ------
    TypeError
        If ``space`` is neither a `LatticeSpace` nor a `MeshSpace`, ``rng`` is not a
        `numpy.random.Generator`, ``size`` is not an integer, or ``where`` is not boolean.
    ValueError
        If the mesh is not spherical, ``size`` is negative, or ``where`` has the wrong shape or
        holds no cell.
    """
    if isinstance(space, LatticeSpace):
        low, high = space._tiling.low, space._tiling.high
        areas = np.prod(high - low, axis=1)  # of a box around a cell of each type
        box_areas = np.full(space.n_cells, areas[0]) if len(areas) == 1 else areas[space._types]

        def in_boxes(rng: np.random.Generator, cells: NDArray[np.int64]) -> NDArray[np.float64]:
            return _in_lattice_boxes(space, low, high, rng, cells)

        width = space.dim
    elif isinstance(space, MeshSpace):
        if space.metric != "spherical":
            raise ValueError(
                f"random points need a lattice or a spherical mesh, got a mesh with metric "
                f"{space.metric!r}"
            )
        centers, cos_radius = _caps(space)
        box_areas = 2 * np.pi * (1 - cos_radius)

        def in_boxes(rng: np.random.Generator, cells: NDArray[np.int64]) -> NDArray[np.float64]:
            return _in_caps(rng, centers[cells], cos_radius[cells])

        width = 3
    else:
        raise TypeError(
            f"random points need a LatticeSpace or a spherical MeshSpace, got "
            f"{type(space).__name__}"
        )
    return _draw_in_cells(rng, size, where, box_areas, in_boxes, space.locate, width)


def _draw_in_cells(
    rng: np.random.Generator,
    size: SupportsIndex | None,
    where: ArrayLike | None,
    box_areas: NDArray[np.float64],
    in_boxes: Callable[[np.random.Generator, NDArray[np.int64]], NDArray[np.float64]],
    cells_of: Callable[[NDArray[np.float64]], NDArray[np.int64]],
    width: int,
) -> NDArray[np.float64]:
    """Return points drawn evenly over the area of the cells of ``where``.

    ``box_areas`` is the area of a box around each cell that contains it, ``in_boxes(rng,
    cells)`` draws a point evenly over the box of each cell, shape (len(cells), width), and
    ``cells_of(points)`` finds the cell at each point. Checks the arguments `random_points`
    takes, and returns one point, shape (width,), or ``size`` of them, shape (size, width).
    Also shared with `worlds`, whose planets draw places in their own coordinates.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    count = 1 if size is None else check_int(size, "size", minimum=0)
    n_cells = box_areas.size
    allowed = np.ones(n_cells, bool) if where is None else cell_mask(where, n_cells, "where")
    if not allowed.any():
        raise ValueError("where must hold at least one cell")
    # Pick a cell by the area of a box around it, draw a point evenly in the box, and keep it
    # if it lies in the cell. Otherwise pick again, cell and all: then every point of an
    # allowed cell is as likely as every other, whatever the boxes' sizes.
    boxes = box_areas * allowed
    chance = boxes / boxes.sum()
    batches, found, drawn, rate = [np.empty((0, width))], 0, 0, 0.5
    while found < count:
        batch = min(int(1.5 * (count - found) / rate) + 16, 1 << 20)
        cells = rng.choice(n_cells, batch, p=chance)
        points = in_boxes(rng, cells)
        kept = cells_of(points) == cells
        batches.append(points[kept])
        found, drawn = found + int(kept.sum()), drawn + batch
        rate = (found + 1) / drawn
    points = np.concatenate(batches)[:count]
    return points[0] if size is None else points


def _cell_reach(space: LatticeSpace) -> NDArray[np.float64]:
    """Return how far a lattice's cells reach from their centers along each axis: (dim,).

    Half a step either way on squares and cubes. A hexagon spans one step across its flat
    sides, along x, and ``2 / sqrt(3)`` between its corners, along y. A triangle reaches
    ``sqrt(3) / 2`` along x, and 1 along y, toward its far corner.
    """
    low, high = space._tiling.low, space._tiling.high
    return np.maximum(-low, high).max(axis=0)


def _in_lattice_boxes(
    space: LatticeSpace,
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    rng: np.random.Generator,
    cells: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return a point drawn evenly over the box around each cell: shape (len(cells), dim).

    ``low`` and ``high`` are the corners of the box around a cell of each type, from its
    center: shape (n_types, dim).
    """
    if len(low) == 1:
        offsets = rng.uniform(low[0], high[0], (len(cells), space.dim))
    else:
        types = space._types[cells]
        offsets = rng.uniform(low[types], high[types])
    points = space.centers[cells] + offsets
    # Move points by whole periods along the axes that wrap, into one period: coordinate k
    # from half a step before the first cell. The basis is upper triangular, so a period of
    # axis k moves coordinates k and below, and going from the last axis to the first keeps
    # each coordinate where it was put.
    for axis in reversed(range(space.dim)):
        if space.periodic[axis]:
            length = space.shape[axis]
            turns = np.floor((points[:, axis] / space.basis[axis, axis] + 0.5) / length)
            points -= np.outer(turns * length, space.basis[:, axis])
    return points


def _caps(space: MeshSpace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a cap around each cell of a spherical mesh that holds the cell.

    The cells' centers on the unit sphere, shape (n_cells, 3), and the cosines of the caps'
    angular radii, to each cell's farthest corner, shape (n_cells,). Made once per mesh.
    """
    if space in _CAPS:
        return _CAPS[space]
    centers = space.centers / np.linalg.norm(space.centers, axis=1, keepdims=True)
    corners = space.vertices / np.linalg.norm(space.vertices, axis=1, keepdims=True)
    owner = np.repeat(np.arange(space.n_cells), np.diff(space.polygon_indptr))
    cosines = np.einsum("ij,ij->i", centers[owner], corners[space.polygon_indices])
    farthest = np.minimum.reduceat(cosines, space.polygon_indptr[:-1])
    cos_radius = np.clip(farthest - 1e-9, -1.0, 1.0)  # a little wider, for rounding
    _CAPS[space] = read_only(centers), read_only(cos_radius)
    return _CAPS[space]


def _in_caps(
    rng: np.random.Generator, center: NDArray[np.float64], cos_radius: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a point of the unit sphere drawn evenly over each cap: shape (len(center), 3)."""
    height = rng.uniform(cos_radius, 1.0)  # even by area on a sphere
    turn = rng.uniform(0.0, 2 * np.pi, len(center))
    axis = np.where(np.abs(center[:, [2]]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    east = np.cross(axis, center)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(center, east)
    side = np.sqrt(1 - height**2)[:, None]
    return height[:, None] * center + side * (
        np.cos(turn)[:, None] * east + np.sin(turn)[:, None] * north
    )
