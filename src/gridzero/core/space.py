"""The `Space` protocol: a finite set of cells with ids, geometry, and adjacency."""

import math
import numbers
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from gridzero.core.graph import CellGraph

RADIUS_RTOL = 1e-9
"""Relative slack in `Space.radius_graph`.

Centers count as within ``radius`` when their distance is at most ``radius * (1 + RADIUS_RTOL)``,
so a radius such as ``sqrt(3)`` includes cells exactly that far apart despite rounding.
"""

type AdjacencyKind = int | np.integer[Any] | Literal["facet", "vertex"]
"""How cells must touch to count as adjacent.

``"facet"`` means they share a face of dimension ``dim - 1``: an edge between polygons, a square
between cubes. On hypercubic grids this gives the von Neumann neighborhood. ``"vertex"`` means
they share any vertex, which gives the Moore neighborhood on hypercubic grids. An integer ``j``
means they share a face of dimension at least ``j``.
"""


@runtime_checkable
class Space(Protocol):
    """A finite set of cells, identified by flat ids ``0 .. n_cells - 1``.

    Every space provides cell geometry (a representative center and a measure such as area or
    volume) and adjacency graphs. Neighborhoods, fields, and update rules are built on top of
    this interface. They never depend on a concrete space type.
    """

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        ...

    @property
    def dim(self) -> int:
        """Intrinsic dimension of the cells: 2 for polygons, including polygons on a sphere."""
        ...

    @property
    def centers(self) -> NDArray[np.float64]:
        """Representative point of each cell: float64, shape (n_cells, embed_dim)."""
        ...

    @property
    def measures(self) -> NDArray[np.float64]:
        """Size of each cell (length, area, or volume): float64, shape (n_cells,), all > 0."""
        ...

    def adjacency(self, kind: AdjacencyKind = "facet") -> CellGraph:
        """Return the symmetric graph of cells that touch in the given way.

        Parameters
        ----------
        kind : {"facet", "vertex"} or int, default "facet"
            How cells must touch to be adjacent (see `AdjacencyKind`).

        Returns
        -------
        CellGraph
            Symmetric graph without self-loops. Edge attributes include ``"distance"``
            (between centers) and ``"boundary"`` (measure of the shared face, or 0 when cells
            only meet in a lower-dimensional face).
        """
        ...

    def radius_graph(self, radius: float) -> CellGraph:
        """Return the symmetric graph of cells whose centers lie within ``radius`` of each other.

        Parameters
        ----------
        radius : float
            Largest center-to-center distance, finite and non-negative. Distances are measured
            as in `adjacency`: across periodic boundaries, and along great circles on a sphere.
            Distances within a relative `RADIUS_RTOL` of ``radius`` count as within it.

        Returns
        -------
        CellGraph
            Symmetric graph without self-loops, with edge attribute ``"distance"``.
        """
        ...


class RadiusCache[V]:
    """What a space computed for its most recently used radii, at most `KEEP` of them.

    Queries ask for the same few radii again and again, so spaces cache what depends only on
    a radius. A sweep over many radii would keep them all; this keeps the last used.
    """

    KEEP = 8
    """How many radii are kept."""

    def __init__(self) -> None:
        self._values: dict[float, V] = {}  # least recently used first

    def get(self, limit: float, make: Callable[[float], V]) -> V:
        """Return the value for ``limit``, made by ``make(limit)`` if it isn't kept.

        Parameters
        ----------
        limit : float
            A checked radius, its tolerance included.
        make : callable
            Builds the value of a radius.

        Returns
        -------
        V
            The kept value, or the new one, now kept as the most recently used.
        """
        value = self.kept(limit)
        if value is None:
            value = make(limit)
            if len(self._values) >= self.KEEP:
                del self._values[next(iter(self._values))]
            self._values[limit] = value
        return value

    def kept(self, limit: float) -> V | None:
        """Return the value for ``limit`` if it is kept, as the most recently used, else None.

        Parameters
        ----------
        limit : float
            A checked radius, its tolerance included.

        Returns
        -------
        V or None
            The kept value, or None.
        """
        if limit not in self._values:
            return None
        self._values[limit] = value = self._values.pop(limit)
        return value


def check_radius(radius: float) -> float:
    """Return ``radius`` as a float after checking that it is finite and non-negative.

    Parameters
    ----------
    radius : float
        Radius to check: a real number, such as a Python or NumPy integer or float.

    Returns
    -------
    float
        ``radius`` as a Python float.

    Raises
    ------
    TypeError
        If ``radius`` is not a real number, or is a bool.
    ValueError
        If ``radius`` is negative, infinite, or NaN.
    """
    if isinstance(radius, bool | np.bool_) or not isinstance(radius, numbers.Real):
        raise TypeError(f"radius must be a real number, got {radius!r}")
    value = float(radius)
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"radius must be finite and non-negative, got {radius!r}")
    return value


def check_kind(kind: object) -> int | Literal["facet", "vertex"]:
    """Return an adjacency kind after checking it as far as possible without a space.

    ``"facet"`` means ``dim - 1``, so it cannot be turned into an int here, and whether an int
    is below ``dim`` depends on the space: `min_face_dim` checks that.

    Parameters
    ----------
    kind : object
        Kind to check: ``"facet"``, ``"vertex"``, or the smallest dimension of a shared face,
        a Python or NumPy integer.

    Returns
    -------
    int or {"facet", "vertex"}
        ``kind``, an integer as a Python int.

    Raises
    ------
    TypeError
        If ``kind`` is neither a string nor an integer (bools are rejected).
    ValueError
        If ``kind`` is an unknown string or a negative integer.
    """
    if isinstance(kind, str):
        if kind not in ("facet", "vertex"):
            raise ValueError(f"adjacency kind must be 'facet', 'vertex', or an int, got {kind!r}")
        return kind
    if isinstance(kind, bool | np.bool_) or not isinstance(kind, int | np.integer):
        raise TypeError(f"adjacency kind must be 'facet', 'vertex', or an int, got {kind!r}")
    if kind < 0:
        raise ValueError(f"adjacency kind must be a non-negative int, got {kind!r}")
    return int(kind)


def min_face_dim(kind: AdjacencyKind, dim: int) -> int:
    """Resolve an adjacency kind to the minimum dimension of the face cells must share.

    Parameters
    ----------
    kind : {"facet", "vertex"} or int
        Adjacency kind.
    dim : int
        Intrinsic dimension of the cells.

    Returns
    -------
    int
        A value in ``[0, dim - 1]``.

    Raises
    ------
    TypeError
        If ``kind`` is neither a string nor an integer.
    ValueError
        If ``kind`` is not recognized or is out of range.
    """
    kind = check_kind(kind)
    if kind == "facet":
        return dim - 1
    if kind == "vertex":
        return 0
    if kind >= dim:
        raise ValueError(
            f"adjacency kind must be 'facet', 'vertex', or an int in [0, {dim}), got {kind!r}"
        )
    return kind
