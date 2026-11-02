"""Cell adjacency graphs in compressed sparse row (CSR) form."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Self, SupportsIndex

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, DTypeLike, NDArray

from gridzero.core._arrays import (
    check_cell,
    check_int,
    frozen_copy,
    int_array,
    read_only,
)


@dataclass(frozen=True, eq=False, init=False)
class CellGraph:
    """Directed edges between cells, stored in CSR form, with optional per-edge attributes.

    The neighbors of cell ``i`` are ``indices[indptr[i]:indptr[i + 1]]``. Spatial adjacency
    graphs are symmetric, so every edge ``i -> j`` has a matching ``j -> i``.

    Rows are canonical: the targets in each row are strictly increasing, so no edge appears
    twice. Use `from_edges` to build a graph from edges in any order. All arrays are read-only
    copies, and the graph is immutable, so copying it returns the same object.

    Parameters
    ----------
    indptr : array_like of int, shape (n_cells + 1,)
        Row offsets. Starts at 0, never decreases, and ends at ``n_edges``.
    indices : array_like of int, shape (n_edges,)
        Target cell of each edge, in ``[0, n_cells)``, strictly increasing within each row.
    edge_data : mapping of str to array_like of float, each shape (n_edges,), optional
        Per-edge attributes aligned with ``indices``, such as ``"distance"`` (between cell
        centers) or ``"boundary"`` (measure of the shared face).

    Raises
    ------
    TypeError
        If ``indptr`` or ``indices`` does not hold integers.
    ValueError
        If the arrays are inconsistent with each other, or a row is unsorted or repeats an
        edge.
    """

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    edge_data: Mapping[str, NDArray[np.float64]]

    def __init__(
        self,
        indptr: ArrayLike,
        indices: ArrayLike,
        edge_data: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        data = {name: frozen_copy(values, np.float64) for name, values in (edge_data or {}).items()}
        self._set(
            frozen_copy(int_array(indptr, "indptr"), np.int64),
            frozen_copy(int_array(indices, "indices"), np.int64),
            data,
        )

    @classmethod
    def _owned(
        cls,
        indptr: NDArray[np.int64],
        indices: NDArray[np.int64],
        edge_data: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> Self:
        """Build a graph around new arrays that nothing else holds, without copying them.

        For builders in this package: the arrays are checked as by the constructor, and made
        read-only in place. Copying them would double the memory of a large graph.
        """
        graph = object.__new__(cls)
        data = {name: _freeze(values, np.float64) for name, values in (edge_data or {}).items()}
        indptr = _freeze(int_array(indptr, "indptr"), np.int64)
        indices = _freeze(int_array(indices, "indices"), np.int64)
        graph._set(indptr, indices, data)
        return graph

    def _set(
        self,
        indptr: NDArray[np.int64],
        indices: NDArray[np.int64],
        data: dict[str, NDArray[np.float64]],
    ) -> None:
        """Check read-only arrays and store them as the graph's fields."""
        if indptr.ndim != 1 or indptr.size < 1 or indptr[0] != 0:
            raise ValueError("indptr must be a 1-D array starting at 0")
        if indices.ndim != 1 or indptr[-1] != indices.size:
            raise ValueError("indptr[-1] must equal the number of edges in indices")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        n_cells = indptr.size - 1
        if indices.size and (indices.min() < 0 or indices.max() >= n_cells):
            raise ValueError(f"edge targets must lie in [0, {n_cells})")
        for name, array in data.items():
            if array.shape != indices.shape:
                raise ValueError(
                    f"edge_data[{name!r}] has shape {array.shape}, expected {indices.shape}"
                )
        # Targets must increase from each edge to the next, except where a new row begins.
        # One byte per edge: comparing sources * n_cells + indices took three int64 arrays.
        increasing = indices[1:] > indices[:-1]
        starts = indptr[1:-1]
        increasing[starts[(starts > 0) & (starts < indices.size)] - 1] = True
        if not increasing.all():
            raise ValueError(
                "the targets in each row must be strictly increasing (sorted, with no edge "
                "repeated); use CellGraph.from_edges to build a graph from unordered edges"
            )
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "edge_data", MappingProxyType(data))

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor so unpickled arrays are validated and read-only again.
        return type(self), (self.indptr, self.indices, dict(self.edge_data))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    @classmethod
    def from_edges(
        cls,
        sources: ArrayLike,
        targets: ArrayLike,
        n_cells: SupportsIndex,
        edge_data: Mapping[str, ArrayLike] | None = None,
    ) -> Self:
        """Build a graph from an edge list in any order.

        Each row is sorted by target. Duplicate ``(source, target)`` pairs are rejected rather
        than merged, because no single rule (sum, first, mean) is right for every attribute:
        shared boundary lengths add up, but a distance between centers does not. Combine
        duplicates before calling.

        Parameters
        ----------
        sources, targets : array_like of int, shape (n,)
            Edge endpoints, in ``[0, n_cells)``.
        n_cells : int
            Number of cells (rows) in the graph.
        edge_data : mapping of str to array_like of float, each shape (n,), optional
            Per-edge attributes aligned with ``sources``.

        Returns
        -------
        CellGraph
            The graph with canonical rows.

        Raises
        ------
        TypeError
            If the endpoints or ``n_cells`` are not integers.
        ValueError
            If an endpoint is out of range, the array lengths differ, or an edge appears more
            than once.
        """
        src = int_array(sources, "sources").ravel()
        dst = int_array(targets, "targets").ravel()
        if src.shape != dst.shape:
            raise ValueError("sources and targets must have the same length")
        n_cells = check_int(n_cells, "n_cells", minimum=0)
        if src.size and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n_cells):
            raise ValueError(f"edge endpoints must lie in [0, {n_cells})")
        data = {
            name: np.asarray(v, dtype=np.float64).ravel() for name, v in (edge_data or {}).items()
        }
        for name, values in data.items():
            if values.shape != src.shape:
                raise ValueError(f"edge_data[{name!r}] must have one value per edge")

        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        repeated = (src[1:] == src[:-1]) & (dst[1:] == dst[:-1])
        if np.any(repeated):
            k = int(np.argmax(repeated))
            raise ValueError(
                f"duplicate edge ({src[k]}, {dst[k]}); combine duplicate edges before building "
                "the graph"
            )
        data = {name: values[order] for name, values in data.items()}

        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n_cells), out=indptr[1:])
        return cls._owned(indptr, dst, data)

    @property
    def n_cells(self) -> int:
        """Number of cells (rows)."""
        return self.indptr.size - 1

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return self.indices.size

    @cached_property
    def degree(self) -> NDArray[np.int64]:
        """Out-degree of each cell: int64, shape (n_cells,)."""
        return read_only(np.diff(self.indptr))

    @cached_property
    def sources(self) -> NDArray[np.int64]:
        """Source cell of each edge: int64, shape (n_edges,), aligned with ``indices``."""
        return read_only(np.repeat(np.arange(self.n_cells, dtype=np.int64), self.degree))

    def neighbors(self, cell: SupportsIndex) -> NDArray[np.int64]:
        """Return the neighbors of one cell.

        Parameters
        ----------
        cell : int
            Cell id, in ``[0, n_cells)``.

        Returns
        -------
        ndarray of int64, shape (degree[cell],)
            Read-only view of the cell's row, in ascending order.

        Raises
        ------
        TypeError
            If ``cell`` is not an integer.
        IndexError
            If ``cell`` is out of range. Negative ids are not counted from the end.
        """
        cell = check_cell(cell, self.n_cells)
        return self.indices[self.indptr[cell] : self.indptr[cell + 1]]

    def padded(self, fill: int = -1) -> NDArray[np.int64]:
        """Return the neighbor lists as a rectangular table.

        Parameters
        ----------
        fill : int, default -1
            Value for unused slots in rows shorter than the maximum degree.

        Returns
        -------
        ndarray of int64, shape (n_cells, max_degree)
            Row ``i`` holds the neighbors of cell ``i`` in CSR order, followed by ``fill``.
        """
        width = int(self.degree.max()) if self.n_cells else 0
        table = np.full((self.n_cells, width), fill, dtype=np.int64)
        slots = np.arange(self.n_edges) - self.indptr[self.sources]
        table[self.sources, slots] = self.indices
        return table

    def transpose(self) -> Self:
        """Return the graph with every edge reversed, carrying its attributes along.

        The transpose is computed once and cached, and transposing it again returns this
        graph. Pushing values along edges is a pull through the transpose, so this is cheap
        to call every step.

        Returns
        -------
        CellGraph
            Graph with an edge ``j -> i`` for every edge ``i -> j``.
        """
        return self._transpose

    @cached_property
    def _transpose(self) -> Self:
        """The transpose, cached both ways: the transpose's own transpose is this graph."""
        # Counting sort by target. Edges are ordered by (source, target), so a stable sort on
        # target leaves each new row's sources strictly increasing: the result is canonical.
        order = np.argsort(self.indices, kind="stable")
        indptr = np.zeros(self.n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.indices, minlength=self.n_cells), out=indptr[1:])
        reverse = type(self)._owned(
            indptr,
            self.sources[order],
            {name: values[order] for name, values in self.edge_data.items()},
        )
        reverse.__dict__["_transpose"] = self
        return reverse

    def is_symmetric(self) -> bool:
        """Return whether every edge ``i -> j`` has a matching ``j -> i``."""
        n = self.n_cells
        forward = np.sort(self.sources * n + self.indices)
        backward = np.sort(self.indices * n + self.sources)
        return bool(np.array_equal(forward, backward))

    def has_self_loops(self) -> bool:
        """Return whether any cell is its own neighbor."""
        return bool(np.any(self.sources == self.indices))

    def to_scipy(self, weights: str | None = None, dtype: DTypeLike = np.float64) -> sp.csr_array:
        """Return the graph as a SciPy sparse matrix.

        Parameters
        ----------
        weights : str, optional
            Name of the edge attribute to use as matrix values. If omitted, every edge has
            value 1.
        dtype : dtype_like, default float64
            Type of the matrix values. An integer type gives exact counts, so it needs
            ``weights`` to be omitted.

        Returns
        -------
        scipy.sparse.csr_array of ``dtype``, shape (n_cells, n_cells)
            Entry ``(i, j)`` is the weight of edge ``i -> j``.

        Raises
        ------
        ValueError
            If ``weights`` is given with a ``dtype`` that is not floating.
        """
        dtype = np.dtype(dtype)
        if weights is None:
            data = np.ones(self.n_edges, dtype=dtype)
        elif dtype.kind == "f":
            data = self.edge_data[weights].astype(dtype)
        else:
            raise ValueError(f"weights are floats, so they need a floating dtype, not {dtype}")
        return sp.csr_array(
            (data, self.indices.copy(), self.indptr.copy()),
            shape=(self.n_cells, self.n_cells),
        )


def _freeze(array: NDArray, dtype: type) -> NDArray:
    """Return a new ``array`` read-only, converted to C-contiguous ``dtype`` only if need be."""
    return read_only(np.ascontiguousarray(array, dtype=dtype))
