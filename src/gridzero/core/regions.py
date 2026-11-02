"""Masks of cells on any space: their connected regions, and their borders."""

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import connected_components

from gridzero.core._arrays import cell_mask
from gridzero.core.space import AdjacencyKind, Space


def regions(space: Space, mask: ArrayLike, kind: AdjacencyKind = "facet") -> NDArray[np.int64]:
    """Return the connected region of each cell of a mask.

    A region is a set of cells of the mask connected through cells of the mask that touch, as
    ``space.adjacency(kind)`` says, so on a lattice that wraps around, a region continues
    across the edge. Regions are numbered from 0 by area (``space.measures``), largest first,
    and regions of equal area by their lowest cell id.

    Parameters
    ----------
    space : Space
        The cells.
    mask : array_like of bool, shape (n_cells,)
        The cells to divide into regions, such as land.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells touch: those that share a side, or also those that share only a corner,
        as for `Space.adjacency`. On squares, ``"vertex"`` joins cells diagonally.

    Returns
    -------
    ndarray of int64, shape (n_cells,)
        The region of each cell of the mask, and -1 for the other cells.

    Raises
    ------
    TypeError
        If ``mask`` is not boolean, or ``kind`` is neither a string nor an integer.
    ValueError
        If ``mask`` does not have shape (n_cells,), or ``kind`` is unknown or out of range.
    """
    n_cells = space.n_cells
    inside = cell_mask(mask, n_cells, "mask")
    graph = space.adjacency(kind)
    linked = inside[graph.sources] & inside[graph.indices]
    sources, targets = graph.sources[linked], graph.indices[linked]
    links = sp.csr_matrix((np.ones(sources.size), (sources, targets)), (n_cells, n_cells))
    _, component = connected_components(links, directed=False)
    _, component = np.unique(component[inside], return_inverse=True)  # numbered 0, 1, ...
    area = np.bincount(component, space.measures[inside])
    rank = np.empty(area.size, dtype=np.int64)
    rank[np.argsort(-area, kind="stable")] = np.arange(area.size)
    labels = np.full(n_cells, -1, dtype=np.int64)
    labels[inside] = rank[component]
    return labels


def border(space: Space, mask: ArrayLike, kind: AdjacencyKind = "facet") -> NDArray[np.bool_]:
    """Return the cells of a mask that touch a cell outside it.

    The edge of a lattice that doesn't wrap around is not a side: no cell lies beyond it.

    Parameters
    ----------
    space : Space
        The cells.
    mask : array_like of bool, shape (n_cells,)
        The cells whose border to find, such as land, whose border is its coast.
    kind : {"facet", "vertex"} or int, default "facet"
        Which cells touch, as for `Space.adjacency`.

    Returns
    -------
    ndarray of bool, shape (n_cells,)

    Raises
    ------
    TypeError
        If ``mask`` is not boolean, or ``kind`` is neither a string nor an integer.
    ValueError
        If ``mask`` does not have shape (n_cells,), or ``kind`` is unknown or out of range.
    """
    n_cells = space.n_cells
    inside = cell_mask(mask, n_cells, "mask")
    graph = space.adjacency(kind)
    outside = np.bincount(graph.sources, ~inside[graph.indices], minlength=n_cells)
    return inside & (outside > 0)
