"""Straight-line questions about cells: how far apart they are, and which lie within a range.

Distances run between cells' centers, straight across the plane of a lattice (to the nearest
copy along the axes that wrap) and along great circles on a sphere. They are the distances
`Ball` and `Space.radius_graph` use.
"""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import check_cell, check_ids
from gridzero.core.lattice import LatticeSpace
from gridzero.core.mesh import MeshSpace
from gridzero.core.space import RADIUS_RTOL, check_radius


def distance(space: LatticeSpace | MeshSpace, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the distances between the centers of cells.

    Parameters
    ----------
    space : LatticeSpace or MeshSpace
        The cells.
    a, b : int or array_like of int, shape (...)
        Cell ids. They broadcast against each other.

    Returns
    -------
    ndarray of float64, shape (...)
        Straight-line distances on a lattice, to the nearest copy across the edges that wrap,
        and great-circle arcs on a spherical mesh. The same both ways round.

    Raises
    ------
    TypeError
        If ``space`` is neither a `LatticeSpace` nor a `MeshSpace`, or ``a`` or ``b`` does not
        hold integers.
    IndexError
        If ``a`` or ``b`` holds an id that is not a cell's.
    """
    _check_space(space, "distance")
    a, b = check_ids(a, space.n_cells, "a"), check_ids(b, space.n_cells, "b")
    a, b = np.broadcast_arrays(a, b)
    low, high = np.minimum(a, b), np.maximum(a, b)  # measured one way round, so symmetric
    return space.distance(space.centers[low], space.centers[high])


def within(
    space: LatticeSpace | MeshSpace, cell: SupportsIndex, radius: float
) -> NDArray[np.int64]:
    """Return the cells whose centers lie within ``radius`` of a cell's, such as an attack's range.

    It is ``cell`` itself and its neighbors in ``Ball(radius)``, found without building the
    neighborhood of every cell: from offsets on a lattice, and a tree of centers on a mesh. On
    a lattice it needs no periodic axis to be long enough, unlike `Ball`. Once the space holds
    its `radius_graph` for ``radius``, it reads the cell's row there instead: 1.8 us against
    12-14 us on a lattice and 30 us on an icosphere (2026-10-28).

    Parameters
    ----------
    space : LatticeSpace or MeshSpace
        The cells.
    cell : int
        A cell id.
    radius : float
        Largest distance between centers, finite and at least 0, measured as by `distance`.
        Distances within a relative `RADIUS_RTOL` of it count as within it.

    Returns
    -------
    ndarray of int64, shape (n,)
        Cell ids in ascending order, ``cell`` among them.

    Raises
    ------
    TypeError
        If ``space`` is neither a `LatticeSpace` nor a `MeshSpace`, ``cell`` is not an integer,
        or ``radius`` is not a real number.
    ValueError
        If ``radius`` is negative, infinite or NaN.
    IndexError
        If ``cell`` is not a cell id.
    """
    _check_space(space, "within")
    cell = check_cell(cell, space.n_cells)
    limit = check_radius(radius) * (1 + RADIUS_RTOL)
    graph = space._radius_graphs.kept(limit)  # keyed by the same limit, its tolerance included
    if graph is not None:
        row = graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]]
        at = int(row.searchsorted(cell))
        found = np.empty(row.size + 1, dtype=np.int64)  # 4x faster than np.insert
        found[:at], found[at], found[at + 1 :] = row[:at], cell, row[at:]
        return found
    return space._within(cell, limit)


def _check_space(space: object, name: str) -> None:
    """Raise TypeError unless ``space`` is a lattice or a mesh."""
    if not isinstance(space, LatticeSpace | MeshSpace):
        raise TypeError(f"{name} needs a LatticeSpace or a MeshSpace, got {type(space).__name__}")
