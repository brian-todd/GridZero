"""Regular lattice spaces: n-dimensional hypercubic grids, and hexagonal and triangular grids."""

import functools
import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Self, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import (
    check_cell,
    check_ids,
    check_int,
    finite_floats,
    int_array,
    read_only,
)
from gridzero.core.graph import CellGraph
from gridzero.core.space import (
    RADIUS_RTOL,
    AdjacencyKind,
    RadiusCache,
    check_radius,
    min_face_dim,
)

type Tiling = Literal["hypercubic", "hex", "triangle"]

_SQRT3 = 3.0**0.5

_BLOCK_ENTRIES = 1 << 20
"""Coordinates per block of cells when building offset graphs: 8 MB of int64 per array."""

# Axial-coordinate offsets to the six neighbors of a hexagon.
_HEX_OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]], dtype=np.int64)

# Offsets from a triangle that points up to the 12 cells that touch it: the 3 that share an
# edge (beside it, and below it), then the 9 that share only a corner (2 more in its row, 4
# more along its bottom edge in the row below, and 3 at its top corner in the row above). A
# triangle that points down is turned half around, so its offsets are the opposites.
_TRIANGLE_OFFSETS = np.array(
    [
        *([-1, 0], [1, 0], [0, -1]),
        *([-2, 0], [2, 0]),
        *([-2, -1], [-1, -1], [1, -1], [2, -1]),
        *([-1, 1], [0, 1], [1, 1]),
    ],
    dtype=np.int64,
)
_TRIANGLE_FACES = np.array([1] * 3 + [0] * 9, dtype=np.int64)


@dataclass(frozen=True, eq=False)
class _Frame:
    """Integer coordinates for a 2-D tiling, in which every center and corner is exact.

    A cell at lattice coordinates ``c`` of type ``t`` has its center at
    ``scale @ c + center[t]`` and its corners at that plus ``corners[t]``, counterclockwise.
    The frame maps to the plane of `LatticeSpace.centers` by ``to_plane``, whose determinant is
    positive, so the side of a line a point lies on is the same in both. Lines of sight are
    traced in it with integers, which settles exactly whether a line passes through a corner.
    ``metric`` is a whole multiple of ``to_plane.T @ to_plane``, so ``v @ metric @ v`` compares
    the lengths of steps ``v`` in the frame exactly.

    Attributes
    ----------
    scale : ndarray of int64, shape (2, 2)
    center : ndarray of int64, shape (n_types, 2)
    corners : ndarray of int64, shape (n_types, n_corners, 2)
    to_plane : ndarray of float64, shape (2, 2)
    metric : ndarray of int64, shape (2, 2)
    """

    scale: NDArray[np.int64]
    center: NDArray[np.int64]
    corners: NDArray[np.int64]
    to_plane: NDArray[np.float64]
    metric: NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class _Tiling:
    """What sets one tiling apart, in one dimension: the facts `LatticeSpace` reads.

    Cells come in types, which lie differently in the lattice: one type on squares, cubes and
    hexagons, and two on triangles, which point up (type 0) or down (type 1). A cell's type is
    ``coords @ type_steps % n_types``. Arrays of shape (n_types, ...) hold one row per type.

    Attributes
    ----------
    basis : ndarray of float64, shape (dim, dim)
        Lattice vectors as columns, upper triangular.
    shift : ndarray of float64, shape (n_types, dim)
        From ``basis @ coords`` to the center of a cell of each type.
    type_steps : ndarray of int64, shape (dim,)
        How a step along each axis changes a cell's type.
    low, high : ndarray of float64, shape (n_types, dim)
        Corners of a box around a cell of each type, from its center, that holds the cell.
    facet_measure : float
        Measure of the face that facet neighbors share.
    wrap_minimum : tuple of int
        Shortest length of each axis that wraps around, so that each neighbor is reached
        through only one offset.
    wrap_multiple : tuple of int
        Each axis that wraps around needs a length divisible by this, so that cells of each
        type meet their kind across the edge.
    round : callable
        Maps lattice coordinates, shape (..., dim), to those of the cell holding each point.
    neighbors : callable
        Returns the offsets to the cells that touch a cell of each type, int64, shape
        (n_types, n_neighbors, dim), and the dimension of the face each shares with it, int64,
        shape (n_types, n_neighbors).
    frame : _Frame or None
        Integer coordinates for lines of sight, on 2-D tilings only.
    """

    basis: NDArray[np.float64]
    shift: NDArray[np.float64]
    type_steps: NDArray[np.int64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    facet_measure: float
    wrap_minimum: tuple[int, ...]
    wrap_multiple: tuple[int, ...]
    round: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    neighbors: Callable[[], tuple[NDArray[np.int64], NDArray[np.int64]]]
    frame: _Frame | None = None

    @property
    def n_types(self) -> int:
        """Number of types of cell."""
        return len(self.shift)


def _cubes(dim: int) -> _Tiling:
    """Describe unit squares, cubes, or n-cubes."""
    half = read_only(np.full((1, dim), 0.5))
    return _Tiling(
        basis=read_only(np.eye(dim)),
        shift=read_only(np.zeros((1, dim))),
        type_steps=read_only(np.zeros(dim, dtype=np.int64)),
        low=read_only(-half),
        high=half,
        facet_measure=1.0,
        wrap_minimum=(3,) * dim,
        wrap_multiple=(1,) * dim,
        round=_round_each,
        neighbors=functools.partial(_cube_neighbors, dim),
        # Twice the coordinates: centers even, corners odd.
        frame=_frame(
            [[2, 0], [0, 2]],
            [[0, 0]],
            [[[1, -1], [1, 1], [-1, 1], [-1, -1]]],
            np.eye(2) / 2,
            [[1, 0], [0, 1]],  # 4 to_plane.T @ to_plane
        )
        if dim == 2
        else None,
    )


def _cube_neighbors(dim: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return every offset in ``{-1, 0, 1}**dim`` but 0, and the dimension of the shared face.

    The cell at offset ``d`` shares a face of dimension ``dim - count_nonzero(d)``. Shapes are
    those of `_Tiling.neighbors`, with one type: (1, 3**dim - 1, dim) and (1, 3**dim - 1).
    """
    cube = np.array(list(itertools.product((-1, 0, 1), repeat=dim)), dtype=np.int64)
    nonzero = np.count_nonzero(cube, axis=1)
    return cube[None, nonzero > 0], (dim - nonzero[nonzero > 0])[None]


def _hexagons(dim: int) -> _Tiling:
    """Describe hexagons in axial coordinates, 1 apart across their sides."""
    # A hexagon spans 1 across its sides, along x, and 2 / sqrt(3) between its corners, along y.
    reach = read_only(np.array([[0.5, 3**-0.5]]))
    return _Tiling(
        basis=read_only(np.array([[1.0, 0.5], [0.0, _SQRT3 / 2]])),
        shift=read_only(np.zeros((1, dim))),
        type_steps=read_only(np.zeros(dim, dtype=np.int64)),
        low=read_only(-reach),
        high=reach,
        facet_measure=1.0 / _SQRT3,
        wrap_minimum=(3, 3),
        wrap_multiple=(1, 1),
        round=_hex_round,
        neighbors=_hex_neighbors,
        # Three times the axial coordinates: corners lie a third of the way to the neighbors.
        frame=_frame(
            [[3, 0], [0, 3]],
            [[0, 0]],
            [[[1, -2], [2, -1], [1, 1], [-1, 2], [-2, 1], [-1, -1]]],
            np.array([[1.0, 0.5], [0.0, _SQRT3 / 2]]) / 3,
            [[4, 2], [2, 4]],  # 36 to_plane.T @ to_plane
        ),
    )


def _hex_neighbors() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the offsets to the six hexagons around a hexagon, which all share an edge.

    Shapes are those of `_Tiling.neighbors`, with one type: (1, 6, 2) and (1, 6).
    """
    return _HEX_OFFSETS[None], np.ones((1, len(_HEX_OFFSETS)), dtype=np.int64)


def _triangles(dim: int) -> _Tiling:
    """Describe equilateral triangles in rows, with edges sqrt(3) long, centers 1 apart."""
    step = _SQRT3 / 2  # from one column to the next: half an edge
    return _Tiling(
        basis=read_only(np.array([[step, 0.0], [0.0, 1.5]])),
        # Centers lie a third of the way up a triangle pointing up, and two thirds of the way
        # up one pointing down: 1/4 below and above the middle of the row.
        shift=read_only(np.array([[0.0, -0.25], [0.0, 0.25]])),
        type_steps=read_only(np.ones(dim, dtype=np.int64)),
        low=read_only(np.array([[-step, -0.5], [-step, -1.0]])),
        high=read_only(np.array([[step, 1.0], [step, 0.5]])),
        facet_measure=_SQRT3,
        wrap_minimum=(6, 4),  # neighbors span 5 columns and 3 rows
        wrap_multiple=(2, 2),
        round=_triangle_round,
        neighbors=_triangle_neighbors,
        # Columns, and quarters of a unit along y: rows 6 apart, centers 1 below or above the
        # middle of their row, and corners 3 above or below it.
        frame=_frame(
            [[1, 0], [0, 6]],
            [[0, -1], [0, 1]],
            [[[1, -2], [0, 4], [-1, -2]], [[0, -4], [1, 2], [-1, 2]]],
            np.diag([step, 0.25]),
            [[12, 0], [0, 1]],  # 16 to_plane.T @ to_plane
        ),
    )


def _triangle_neighbors() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the offsets to the 12 triangles around one pointing up, then pointing down.

    Shapes are those of `_Tiling.neighbors`, with two types: (2, 12, 2) and (2, 12).
    """
    return np.stack([_TRIANGLE_OFFSETS, -_TRIANGLE_OFFSETS]), np.stack([_TRIANGLE_FACES] * 2)


def _frame(
    scale: object, center: object, corners: object, to_plane: object, metric: object
) -> _Frame:
    """Return the integer frame of a 2-D tiling, from nested lists."""
    return _Frame(
        scale=read_only(np.array(scale, dtype=np.int64)),
        center=read_only(np.array(center, dtype=np.int64)),
        corners=read_only(np.array(corners, dtype=np.int64)),
        to_plane=read_only(np.array(to_plane, dtype=np.float64)),
        metric=read_only(np.array(metric, dtype=np.int64)),
    )


_TILINGS: dict[str, Callable[[int], _Tiling]] = {
    "hypercubic": _cubes,
    "hex": _hexagons,
    "triangle": _triangles,
}


def _parse_shape(shape: object) -> tuple[int, ...]:
    """Return ``shape`` as a non-empty tuple of positive ints, rejecting floats and bools."""
    message = f"shape must be a non-empty sequence of positive ints, got {shape!r}"
    if isinstance(shape, str | bytes) or not isinstance(shape, Iterable):
        raise TypeError(message)
    entries = tuple(shape)
    if not entries:
        raise ValueError(message)
    return tuple(check_int(length, "each shape entry", minimum=1) for length in entries)


def _parse_periodic(periodic: object, dim: int) -> tuple[bool, ...]:
    """Return ``periodic`` as one bool per axis, from a single bool or a sequence of them."""
    flags = np.asarray(periodic)
    if flags.dtype != np.bool_ or flags.ndim > 1:
        raise TypeError(f"periodic must be a bool or one bool per axis, got {periodic!r}")
    if flags.ndim == 0:
        return (bool(flags),) * dim
    if flags.size != dim:
        raise ValueError(f"periodic has {flags.size} entries, but the lattice has {dim} axes")
    return tuple(bool(wraps) for wraps in flags)


@dataclass(frozen=True, eq=False, init=False)
class LatticeSpace:
    """Cells of a regular lattice, stored row-major over a grid ``shape``.

    Cell ids are row-major (C-order) flat indices into ``shape``, so a field with shape
    ``(..., n_cells)`` can be viewed as ``(..., *shape)`` without copying (see `grid_view`).
    Neighboring centers are 1 apart. Squares, cubes and hexagons are centered at
    ``basis @ coords``. Triangles are centered 1/4 below that point when they point up, and
    1/4 above it when they point down. The space is immutable, so copying it returns the same
    object.

    Parameters
    ----------
    shape : tuple of int
        Grid extent along each axis. Its length is the dimension of the lattice.
    tiling : {"hypercubic", "hex", "triangle"}, default "hypercubic"
        ``"hypercubic"`` gives unit squares, cubes, or n-cubes. ``"hex"`` (2-D only) gives
        hexagons in axial coordinates, with lattice vectors ``(1, 0)`` and ``(1/2, sqrt(3)/2)``,
        so the grid is a rhombus. ``"triangle"`` (2-D only) gives equilateral triangles in
        rows, with edges ``sqrt(3)`` long, so the grid is a rectangle. Cell ``(i, j)`` lies in
        column ``i`` of row ``j`` and points up when ``i + j`` is even. Columns are
        ``sqrt(3)/2`` apart and rows 3/2.
    periodic : bool or tuple of bool, default True
        Whether each axis wraps around. A single bool applies to every axis.

    Raises
    ------
    TypeError
        If the shape is not a sequence of ints, or ``periodic`` is not a bool or a sequence of
        bools.
    ValueError
        If the shape is empty or has an entry below 1, the tiling is unknown, a hex or triangle
        lattice is not 2-D, ``periodic`` has the wrong length, or a periodic axis is too short:
        shorter than 3, or on triangles odd or shorter than 6 along axis 0 and 4 along axis 1.
        On an axis that short, one neighbor would be reached through two different offsets, and
        on an odd one, triangles would meet others pointing the same way across the edge.
    """

    shape: tuple[int, ...]
    tiling: Tiling
    periodic: tuple[bool, ...]
    _graphs: dict[int, CellGraph] = field(repr=False)
    _near: RadiusCache[tuple[NDArray[np.int64], NDArray[np.float64]]] = field(repr=False)
    _reaches: RadiusCache[NDArray[np.int64]] = field(repr=False)
    _hops: dict[int, tuple[NDArray[np.int64], NDArray[np.int64]]] = field(repr=False)
    _radius_graphs: RadiusCache[CellGraph] = field(repr=False)
    _tiling: _Tiling = field(repr=False)

    def __init__(
        self,
        shape: Sequence[int],
        tiling: Tiling = "hypercubic",
        periodic: bool | Sequence[bool] = True,
    ) -> None:
        shape = _parse_shape(shape)
        if tiling not in tuple(_TILINGS):
            raise ValueError(f"tiling must be 'hypercubic', 'hex' or 'triangle', got {tiling!r}")
        if tiling != "hypercubic" and len(shape) != 2:
            raise ValueError(f"the {tiling} tiling is 2-D; shape must have length 2")
        record = _TILINGS[tiling](len(shape))
        periodic = _parse_periodic(periodic, len(shape))
        for axis, (length, wraps) in enumerate(zip(shape, periodic, strict=True)):
            if wraps and length % record.wrap_multiple[axis]:
                raise ValueError(
                    f"periodic axis {axis} has length {length}; periodic axes of a "
                    f"{tiling} lattice need a length divisible by "
                    f"{record.wrap_multiple[axis]}, so that the pattern of cells continues "
                    "across the edge"
                )
            if wraps and length < record.wrap_minimum[axis]:
                raise ValueError(
                    f"periodic axis {axis} has length {length}; periodic axes need length >= "
                    f"{record.wrap_minimum[axis]} so that each neighbor is reached through "
                    "only one offset"
                )
        set_ = object.__setattr__
        set_(self, "shape", shape)
        set_(self, "tiling", tiling)
        set_(self, "periodic", periodic)
        set_(self, "_graphs", {})
        set_(self, "_near", RadiusCache())
        set_(self, "_reaches", RadiusCache())
        set_(self, "_hops", {})
        set_(self, "_radius_graphs", RadiusCache())
        set_(self, "_tiling", record)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor: cached arrays and graphs are recomputed on demand.
        return type(self), (self.shape, self.tiling, self.periodic)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    @property
    def dim(self) -> int:
        """Dimension of the lattice (number of axes)."""
        return len(self.shape)

    @cached_property
    def n_cells(self) -> int:
        """Number of cells, ``prod(shape)``."""
        return math.prod(self.shape)

    @property
    def basis(self) -> NDArray[np.float64]:
        """Lattice vectors as columns: float64, shape (dim, dim), upper triangular.

        One step along axis ``k`` moves a cell's center by ``basis[:, k]``, or to the center of
        a cell of another type on triangles. A period along a wrapped axis is
        ``shape[k] * basis[:, k]``.
        """
        return self._tiling.basis

    @cached_property
    def coords(self) -> NDArray[np.int64]:
        """Integer lattice coordinates of every cell: int64, shape (n_cells, dim)."""
        grids = np.indices(self.shape, dtype=np.int64).reshape(self.dim, -1)
        return read_only(np.ascontiguousarray(grids.T))

    @cached_property
    def centers(self) -> NDArray[np.float64]:
        """Cell centers in physical coordinates: float64, shape (n_cells, dim)."""
        centers = self.coords @ self.basis.T
        if self._tiling.n_types > 1:
            centers += self._tiling.shift[self._types]
        return read_only(centers)

    @cached_property
    def _inverse_basis(self) -> NDArray[np.float64]:
        """The inverse of `basis`: float64, shape (dim, dim). Maps steps to lattice coordinates."""
        return read_only(np.linalg.inv(self.basis))

    @cached_property
    def measures(self) -> NDArray[np.float64]:
        """Cell volumes: float64, shape (n_cells,). Every cell has volume ``|det(basis)|``."""
        return read_only(np.full(self.n_cells, abs(np.linalg.det(self.basis))))

    @property
    def facet_measure(self) -> float:
        """Measure of the face shared by facet neighbors.

        1 for n-cubes, 1/sqrt(3) for hexagons, and sqrt(3) for triangles.
        """
        return self._tiling.facet_measure

    @cached_property
    def _types(self) -> NDArray[np.int64]:
        """The type of each cell, as `_Tiling` defines them: int64, shape (n_cells,)."""
        tiling = self._tiling
        return read_only(self.coords @ tiling.type_steps % tiling.n_types)

    @cached_property
    def _neighbors(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The offsets to the cells touching a cell of each type, and their faces' dimensions.

        Read-only int64 arrays of shapes (n_types, n_neighbors, dim) and (n_types, n_neighbors).
        """
        offsets, faces = self._tiling.neighbors()
        return read_only(offsets), read_only(faces)

    def unravel(self, ids: ArrayLike) -> NDArray[np.int64]:
        """Convert flat cell ids to lattice coordinates.

        Parameters
        ----------
        ids : array_like of int, shape (...)
            Cell ids.

        Returns
        -------
        ndarray of int64, shape (..., dim)
            Lattice coordinates.

        Raises
        ------
        TypeError
            If ``ids`` are not integers.
        IndexError
            If an id is outside ``[0, n_cells)``.
        """
        return np.stack(np.unravel_index(check_ids(ids, self.n_cells, "ids"), self.shape), axis=-1)

    def ravel(self, coords: ArrayLike) -> NDArray[np.int64]:
        """Convert lattice coordinates to flat cell ids.

        Parameters
        ----------
        coords : array_like of int, shape (..., dim)
            Lattice coordinates, inside the grid.

        Returns
        -------
        ndarray of int64, shape (...)
            Cell ids.

        Raises
        ------
        TypeError
            If ``coords`` are not integers.
        ValueError
            If a coordinate lies outside the grid.
        """
        coords = int_array(coords, "coords")
        ids = np.ravel_multi_index(tuple(np.moveaxis(coords, -1, 0)), self.shape)
        return np.asarray(ids, dtype=np.int64)

    def opposite(self) -> NDArray[np.int64]:
        """Return the cell opposite each cell, through the lattice's center.

        This is the point reflection through the center, the half turn in 2-D, which swaps
        opposite corners: fair boards for two sides, and one side's view of the board as the
        other's. Lattice coordinates map as ``c -> shape - 1 - c`` on every axis. Squares and
        cubes in any dimension and the rhombus of hexes have it, and so do triangles when
        ``Q + R`` is odd, which turns each triangle pointing up into one pointing down. Wrapped
        axes keep it.

        Returns
        -------
        ndarray of int64, shape (n_cells,)
            ``opposite[c]`` is the cell opposite ``c``: a permutation that is its own inverse.
            Read-only, computed once per space.

        Raises
        ------
        ValueError
            If the lattice has no such symmetry: triangles with ``Q + R`` even.
        """
        return self._opposite

    @cached_property
    def _opposite(self) -> NDArray[np.int64]:
        """The permutation `opposite` returns, checked against the cells' centers."""
        coords = np.asarray(self.shape, dtype=np.int64) - 1 - self.coords
        turn = self.ravel(coords)
        centers = self.centers
        middle = (centers.min(axis=0) + centers.max(axis=0)) / 2
        if not np.allclose(centers[turn], 2 * middle - centers, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"a {self.tiling} lattice of shape {self.shape} has no opposite cells"
                + (": triangles need Q + R odd" if self.tiling == "triangle" else "")
            )
        return read_only(turn)

    def corners(self) -> NDArray[np.float64]:
        """Return each cell's polygon: its corners in the plane of `centers`, counterclockwise.

        For drawing a 2-D lattice cell by cell (`render.cells`): squares have 4 corners,
        hexagons 6 and triangles 3. Cells keep their place in the plane where an axis wraps,
        as their centers do.

        Returns
        -------
        ndarray of float64, shape (n_cells, n_corners, 2)
            Read-only, computed once per space.

        Raises
        ------
        ValueError
            If the lattice is not 2-D.
        """
        return self._corners

    @cached_property
    def _corners(self) -> NDArray[np.float64]:
        """The polygons `corners` returns, from the tiling's integer frame (see `_Frame`)."""
        frame = self._tiling.frame
        if frame is None or self.dim != 2:
            raise ValueError(f"corners need a 2-D lattice, got {self.dim}-D")
        types = self._types
        at = (self.coords @ frame.scale.T + frame.center[types])[:, None, :] + frame.corners[types]
        return read_only(at @ frame.to_plane.T)

    def directions(self) -> NDArray[np.int64]:
        """Return the directions a piece may face: the facet offsets, counterclockwise.

        Direction 0 points along the first lattice vector (``+x`` in the plane of `centers`),
        and the others follow it counterclockwise: 4 on squares, 6 on hexes. Direction
        ``d + n // 2`` (mod ``n``) is the opposite of ``d``, as `opposite` turns it. All have the
        same length, so the one nearest a vector in angle has the largest dot product with it,
        which `core.snap` computes exactly in integers.

        Returns
        -------
        ndarray of int64, shape (n_directions, 2)
            Lattice offsets. Read-only, computed once per space.

        Raises
        ------
        ValueError
            If the lattice is not of squares or hexes in 2-D: triangles' facets point
            differently for the two kinds of cell.
        """
        return self._directions[0]

    @cached_property
    def _directions(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """`directions`, and the integer metric of lattice offsets (see `core.directions`).

        The metric is ``scale.T @ metric @ scale`` of the tiling's frame divided by the
        greatest common divisor of its entries: a positive multiple of ``basis.T @ basis``.
        """
        frame = self._tiling.frame
        if self.dim != 2 or frame is None or self.tiling == "triangle":
            raise ValueError(
                f"directions need squares or hexes in 2-D, got a {self.dim}-D {self.tiling} lattice"
            )
        offsets = self.offsets("facet")
        steps = offsets @ self.basis.T
        offsets = offsets[np.argsort(np.arctan2(steps[:, 1], steps[:, 0]) % (2 * np.pi))]
        metric = frame.scale.T @ frame.metric @ frame.scale
        return read_only(offsets), read_only(metric // np.gcd.reduce(metric.ravel()))

    def grid_view[T: np.generic](self, values: NDArray[T]) -> NDArray[T]:
        """View per-cell values on the grid.

        Parameters
        ----------
        values : ndarray, shape (..., n_cells)
            Per-cell values, cells last.

        Returns
        -------
        ndarray, shape (..., *shape)
            The same data viewed on the grid. No copy is made when ``values`` is contiguous.

        Raises
        ------
        ValueError
            If the last axis does not have length ``n_cells``.
        """
        if values.shape[-1:] != (self.n_cells,):
            raise ValueError(f"last axis must have length n_cells={self.n_cells}")
        return values.reshape(*values.shape[:-1], *self.shape)

    def locate(self, points: ArrayLike) -> NDArray[np.int64]:
        """Return the cell containing each point, or -1 for a point beyond the grid.

        The cells (squares, cubes, hexagons, triangles) are the Voronoi cells of their centers,
        so this is also the cell whose center is nearest. Points wrap around periodic axes. A
        point on the border between cells goes to one of them.

        Parameters
        ----------
        points : array_like of float, shape (..., dim)
            Points in the coordinates of `centers`.

        Returns
        -------
        ndarray of int64, shape (...)
            Cell ids, or -1 where a point lies beyond the edge of a non-periodic axis.

        Raises
        ------
        TypeError
            If ``points`` does not hold numbers.
        ValueError
            If ``points`` does not end in an axis of length ``dim``, holds NaN or infinite
            values, or lies so far out (about 1e308) that its lattice coordinates overflow.
        """
        points = finite_floats(points, "points")
        if points.ndim == 0 or points.shape[-1] != self.dim:
            raise ValueError(f"points must have shape (..., {self.dim}), got {points.shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            coords = points @ self._inverse_basis.T  # fractional lattice coordinates
        if not np.isfinite(coords).all():
            raise ValueError("points lie too far out to locate")
        shape = np.array(self.shape, dtype=np.float64)
        periodic = np.array(self.periodic)
        # Wrap and bound the coordinates before rounding, so that huge ones round exactly.
        # Moving a point by whole periods, or further beyond an edge, changes no answer.
        coords = np.where(periodic, np.mod(coords, shape), np.clip(coords, -2.0, shape + 1))
        return self._wrapped_ids(self._tiling.round(coords).astype(np.int64))

    def distance(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return the straight-line distances between points, across the edges that wrap.

        Along an axis that wraps around, a point has a copy every period, and the distance is
        to the nearest copy, as `radius_graph` measures between centers.

        Parameters
        ----------
        a, b : array_like of float, shape (..., dim)
            Points in the coordinates of `centers`. Leading axes broadcast.

        Returns
        -------
        ndarray of float64, shape (...)
            Distances, the same both ways round: ``distance(a, b) == distance(b, a)``.

        Raises
        ------
        TypeError
            If ``a`` or ``b`` does not hold numbers.
        ValueError
            If ``a`` or ``b`` does not end in an axis of length ``dim``, or holds NaN or
            infinite values.
        """
        return np.linalg.norm(self.displacement(a, b), axis=-1)

    def displacement(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return the shortest vectors from points to points, across the edges that wrap.

        Along an axis that wraps around, a point has a copy every period, and the vector goes
        to the nearest copy, so its length is `distance`. Between the centers of two neighbors
        across a wrapped edge, it is the short step between them, not the long way round: the
        direction from a cell to its neighbor, as wind or a gradient needs.

        Parameters
        ----------
        a, b : array_like of float, shape (..., dim)
            Points in the coordinates of `centers`. Leading axes broadcast.

        Returns
        -------
        ndarray of float64, shape (..., dim)
            The nearest copy of ``b`` minus ``a``. Where two copies are equally near (exactly
            half a period apart along an axis), it goes to one of them.

        Raises
        ------
        TypeError
            If ``a`` or ``b`` does not hold numbers.
        ValueError
            If ``a`` or ``b`` does not end in an axis of length ``dim``, or holds NaN or
            infinite values.
        """
        a, b = finite_floats(a, "a"), finite_floats(b, "b")
        for name, points in (("a", a), ("b", b)):
            if points.ndim == 0 or points.shape[-1] != self.dim:
                raise ValueError(f"{name} must have shape (..., {self.dim}), got {points.shape}")
        return self._nearest_copy(b - a)

    def _nearest_copy(self, steps: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the shortest step, shape (..., dim), to a copy of where each step leads.

        Copies lie whole periods apart along the axes that wrap. Rounding the step's
        coordinates in a reduced basis of the periods finds the nearest copy along orthogonal
        periods; slanted ones (hexes that wrap both ways) also try the copies around it.
        """
        periods, orthogonal = self._periods
        if not periods.shape[1]:
            return steps
        turns = np.rint(steps @ self._period_inverse.T)  # half-even, so symmetric
        nearest = steps - turns @ periods.T
        if orthogonal:
            return nearest
        around = np.array(list(itertools.product((-1, 0, 1), repeat=periods.shape[1])))
        copies = nearest[..., None, :] - around @ periods.T  # (..., 3**m, dim)
        lengths = np.einsum("...i,...i->...", copies, copies)
        best = np.argmin(lengths, axis=-1)[..., None, None]
        return np.take_along_axis(copies, best, axis=-2)[..., 0, :]

    @cached_property
    def _periods(self) -> tuple[NDArray[np.float64], bool]:
        """A reduced basis of the periods along the axes that wrap, and whether it's orthogonal.

        The periods are columns, shape (dim, n_wrapped). Two slanted periods are reduced as
        Lagrange and Gauss did, to the shortest pair spanning the same copies, so that the
        nearest copy lies among those around the rounded one.
        """
        wrapped = [k for k in range(self.dim) if self.periodic[k]]
        periods = self.basis[:, wrapped] * np.array(self.shape)[wrapped]
        if len(wrapped) == 2:
            u, v = periods.T
            if u @ u > v @ v:
                u, v = v, u
            while True:
                v = v - np.rint((u @ v) / (u @ u)) * u
                if v @ v >= u @ u:
                    break
                u, v = v, u
            periods = np.stack([u, v], axis=1)
        gram = periods.T @ periods
        slant = np.abs(gram - np.diag(np.diag(gram)))
        orthogonal = bool(np.all(slant <= 1e-12 * gram.max(initial=0.0)))
        return read_only(periods), orthogonal

    @cached_property
    def _period_turns(self) -> NDArray[np.int64]:
        """The reduced periods of `_periods` in whole turns along each axis that wraps.

        Columns, int64 of shape (n_wrapped, n_wrapped): ``_periods`` is
        ``basis[:, wrapped] * shape[wrapped] @ _period_turns``.
        """
        wrapped = [k for k in range(self.dim) if self.periodic[k]]
        periods, _ = self._periods
        axes = self.basis[:, wrapped] * np.array(self.shape)[wrapped]
        return read_only(np.rint(np.linalg.pinv(axes) @ periods).astype(np.int64))

    @cached_property
    def _period_inverse(self) -> NDArray[np.float64]:
        """The pseudo-inverse of the reduced periods: float64, shape (n_wrapped, dim)."""
        periods, _ = self._periods
        return read_only(np.linalg.pinv(periods))

    def _within(self, cell: int, limit: float) -> NDArray[np.int64]:
        """Return the cells whose centers lie within ``limit`` of ``cell``'s, sorted, itself too.

        Returns int64 of shape (n_near,). See `_pairs_within`, which this does for one cell
        with less set-up.
        """
        if np.prod(2 * self._offset_reach(limit) + 1, dtype=np.float64) > self.n_cells:
            _, ids = self._pairs_within(np.array([cell], dtype=np.int64), limit)
            return ids
        offsets, _ = self._offsets_within(limit)
        ids = self._wrapped_ids(self.coords[cell] + offsets[self._types[cell]])
        return np.unique(np.append(ids[ids >= 0], cell))  # a short wrapped axis: once each

    def _pairs_within(
        self, cells: NDArray[np.int64], limit: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the cells whose centers lie within ``limit`` of each of ``cells``'s.

        ``cells`` is int64 of shape (k,). Returns two int64 arrays of shape (n_pairs,): the
        position in ``cells`` of each pair's first cell, and the cell near it, sorted by both,
        each cell among its own. From the offsets within ``limit``, while there are fewer than
        cells. On an axis that wraps, two offsets may lead to one cell, which then appears once.
        """
        reach = self._offset_reach(limit)
        if np.prod(2 * reach + 1, dtype=np.float64) > self.n_cells:
            centers = self.centers
            near = self.distance(centers[cells][:, None, :], centers) <= limit
            rows, ids = np.nonzero(near)
            return rows.astype(np.int64), ids.astype(np.int64)
        offsets, _ = self._offsets_within(limit)
        ids = self._wrapped_ids(self.coords[cells][:, None, :] + offsets[self._types[cells]])
        ids = np.concatenate([ids, cells[:, None]], axis=1)
        rows = np.broadcast_to(np.arange(cells.size, dtype=np.int64)[:, None], ids.shape)
        keys = np.unique(rows[ids >= 0] * self.n_cells + ids[ids >= 0])
        return keys // self.n_cells, keys % self.n_cells

    def offsets(self, kind: AdjacencyKind = "facet", cell: SupportsIndex = 0) -> NDArray[np.int64]:
        """Return the coordinate offsets from a cell to its adjacent cells.

        Parameters
        ----------
        kind : {"facet", "vertex"} or int, default "facet"
            Adjacency kind (see `AdjacencyKind`).
        cell : int, default 0
            A cell id. On squares, cubes and hexagons every cell has the same offsets. On
            triangles, those pointing up have one set and those pointing down the opposites.

        Returns
        -------
        ndarray of int64, shape (n_offsets, dim)
            One row per neighbor direction. On hypercubic lattices a cell at offset ``d``
            shares a face of dimension ``dim - count_nonzero(d)``, so ``"facet"`` keeps offsets
            with one nonzero entry and ``"vertex"`` keeps all of ``{-1, 0, 1}**dim`` except 0.

        Raises
        ------
        TypeError
            If ``cell`` is not an integer.
        IndexError
            If ``cell`` is not a cell id.
        """
        j = min_face_dim(kind, self.dim)
        tiling = self._tiling
        cell_type = self.unravel(check_cell(cell, self.n_cells)) @ tiling.type_steps
        offsets, faces = self._neighbors
        cell_type %= tiling.n_types
        return offsets[cell_type][faces[cell_type] >= j]

    def adjacency(self, kind: AdjacencyKind = "facet") -> CellGraph:
        """Return the symmetric graph of cells that touch in the given way.

        The graph is built once per kind and cached. On the hex tiling every kind gives the
        same graph object, because hexagons meet only along edges.

        Parameters
        ----------
        kind : {"facet", "vertex"} or int, default "facet"
            Adjacency kind (see `AdjacencyKind`).

        Returns
        -------
        CellGraph
            Edge attributes are ``"distance"`` (between centers, measured along the offset, so
            periodic wrap-around edges have their true length) and ``"boundary"`` (the shared
            facet's measure, or 0 for cells that meet only at a lower-dimensional face).
        """
        j = self._face_dim(kind)
        if j not in self._graphs:
            self._graphs[j] = self._build_adjacency(j)
        return self._graphs[j]

    def _face_dim(self, kind: AdjacencyKind) -> int:
        """Return the smallest dimension of a face that ``kind`` neighbors share.

        Kinds that keep the same neighbors give the same dimension, so they share a graph.
        """
        j = min_face_dim(kind, self.dim)
        _, faces = self._neighbors
        return int(faces[faces >= j].min())

    def radius_graph(self, radius: float) -> CellGraph:
        """Return the symmetric graph of cells whose centers lie within ``radius`` of each other.

        Parameters
        ----------
        radius : float
            Largest center-to-center distance, finite and non-negative. Distances are measured
            along the lattice offset, so they wrap across periodic axes. Distances within a
            relative `RADIUS_RTOL` of ``radius`` count as within it.

        Returns
        -------
        CellGraph
            Symmetric graph without self-loops, with edge attribute ``"distance"``. Cached for
            the last few radii (`RadiusCache`): asking again returns the same graph.

        Raises
        ------
        ValueError
            If ``radius`` is invalid, or reaches ``r`` cells along a periodic axis shorter than
            ``2 * r + 1``. On such an axis, one neighbor would be reached through two offsets.
        """
        limit = check_radius(radius) * (1 + RADIUS_RTOL)
        return self._radius_graphs.get(limit, lambda limit: self._radius_graph(limit, radius))

    def _radius_graph(self, limit: float, radius: float) -> CellGraph:
        """Build `radius_graph` for a checked ``limit``; ``radius`` names it in errors."""
        periodic = np.array(self.periodic)
        # The offset of k steps along axis k alone is within `limit` for k up to this bound, so
        # the radius surely reaches that far. Check periodic axes before building any offsets.
        surely = np.floor(max(limit - self._spread, 0.0) / np.linalg.norm(self.basis, axis=0))
        self._check_wrap(radius, np.where(periodic, surely.astype(np.int64), 0))
        offsets, distance = self._offsets_within(limit)
        if offsets.size:
            self._check_wrap(radius, np.where(periodic, np.abs(offsets).max(axis=(0, 1)), 0))
        return self._offset_graph(offsets, {"distance": distance})

    @cached_property
    def _spread(self) -> float:
        """How much further apart, or closer, two centers lie than their lattice points.

        0 on lattices with one type of cell.
        """
        shift = self._tiling.shift
        return float(np.linalg.norm(shift[:, None] - shift[None], axis=-1).max())

    def _offset_reach(self, limit: float) -> NDArray[np.int64]:
        """Return how many steps along each axis a center within ``limit`` may lie: (dim,).

        A center within ``limit`` is at most ``|row k of inv(basis)| * (limit + spread)`` steps
        along axis k. Along a non-periodic axis, offsets of ``length`` steps or more never land
        in the grid. Read-only, and cached for the last few limits.
        """
        return self._reaches.get(limit, self._build_offset_reach)

    def _build_offset_reach(self, limit: float) -> NDArray[np.int64]:
        """Compute `_offset_reach`, uncached."""
        steps = np.linalg.norm(self._inverse_basis, axis=1) * (limit + self._spread)
        reach = np.floor(steps).astype(np.int64)
        lengths = np.array(self.shape, dtype=np.int64)
        return read_only(np.where(self.periodic, reach, np.minimum(reach, lengths - 1)))

    def _offsets_within(self, limit: float) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return the offsets from a cell of each type to the other centers within ``limit``.

        Offsets, int64 of shape (n_types, n_offsets, dim), and the distance along each,
        float64 of shape (n_types, n_offsets). Every type has as many: on triangles, those of a
        cell pointing down are those of one pointing up, turned half around. Both are read-only
        and cached for the last few limits, since queries ask for the same few radii again and
        again.
        """
        return self._near.get(limit, self._build_offsets_within)

    def _build_offsets_within(self, limit: float) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Compute `_offsets_within`, uncached."""
        reach = self._offset_reach(limit)
        box = np.indices((2 * reach + 1).tolist(), dtype=np.int64).reshape(self.dim, -1).T - reach
        boxes = np.broadcast_to(box, (self._tiling.n_types, *box.shape))  # from each type
        distance = np.linalg.norm(self._center_steps(boxes), axis=-1)
        inside = (distance > 0) & (distance <= limit)
        return read_only(_keep_in_rows(boxes, inside)), read_only(_keep_in_rows(distance, inside))

    def _check_wrap(self, radius: float, reached: NDArray[np.int64]) -> None:
        """Raise if ``radius`` reaches ``reached[k]`` cells along a periodic axis too short.

        ``reached`` is int64 of shape (dim,), one count per axis.
        """
        for axis, (length, r) in enumerate(zip(self.shape, reached, strict=True)):
            if length < 2 * r + 1:
                raise ValueError(
                    f"radius {radius} reaches {r} cells along periodic axis {axis}, which has "
                    f"length {length}; periodic axes need length >= 2 * reach + 1 so that each "
                    "neighbor is reached through only one offset"
                )

    def _center_steps(self, offsets: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return the step from a cell's center to the center of the cell at each offset.

        ``offsets`` has shape (n_types, n_offsets, dim), with the offsets from a cell of each
        type, and so does the result.
        """
        tiling = self._tiling
        steps = offsets @ tiling.basis.T
        if tiling.n_types > 1:
            types = np.arange(tiling.n_types)[:, None]
            reached = (types + offsets @ tiling.type_steps) % tiling.n_types
            steps += tiling.shift[reached] - tiling.shift[types]
        return steps

    def _build_adjacency(self, j: int) -> CellGraph:
        """Link the cells that share a face of dimension at least ``j``."""
        offsets, faces = self._neighbors
        keep = faces >= j
        offsets = _keep_in_rows(offsets, keep)
        faces = _keep_in_rows(faces, keep)
        distance = np.linalg.norm(self._center_steps(offsets), axis=-1)
        boundary = np.where(faces == self.dim - 1, self.facet_measure, 0.0)
        return self._offset_graph(offsets, {"distance": distance, "boundary": boundary})

    def _offset_graph(
        self,
        offsets: NDArray[np.int64],
        offset_data: dict[str, NDArray[np.float64]],
        block: int | None = None,
    ) -> CellGraph:
        """Link every cell to the cell at each offset, dropping offsets that leave the grid.

        ``offsets`` has shape (n_types, n_offsets, dim): the offsets from a cell of each type
        (see `_types`). ``offset_data`` holds one value per offset, shape (n_types, n_offsets),
        copied to every edge along that offset. Rows are assembled directly in CSR form, a
        block of cells at a time: sorting each row's few targets replaces a sort of the whole
        edge list, and nothing larger than the graph itself is ever allocated. At 1024x1024
        with vertex adjacency (8.4M edges) the peak memory fell from 1.26 GB to about 0.35 GB.
        ``block``, the number of cells per block, is for tests: by default a block holds about
        a million coordinates.
        """
        n_cells = self.n_cells
        n_offsets = offsets.shape[1]
        if block is None:
            block = max(1024, _BLOCK_ENTRIES // max(1, n_offsets * self.dim))
        starts = range(0, n_cells, block)
        indptr = np.zeros(n_cells + 1, dtype=np.int64)  # first each row's degree, then sums
        if all(self.periodic):  # every offset stays on the grid
            indptr[1:] = n_offsets
        else:
            for start in starts:
                stop = min(start + block, n_cells)
                targets, _ = self._targets(start, stop, offsets)
                indptr[start + 1 : stop + 1] = np.count_nonzero(targets < n_cells, axis=1)
        np.cumsum(indptr, out=indptr)
        indices = np.empty(indptr[-1], dtype=np.int64)
        data = {name: np.empty(indptr[-1]) for name in offset_data}
        for start in starts:
            stop = min(start + block, n_cells)
            targets, types = self._targets(start, stop, offsets)
            order = np.argsort(targets, axis=1, kind="stable")
            targets = np.take_along_axis(targets, order, axis=1)
            inside = targets < n_cells
            edges = slice(indptr[start], indptr[stop])
            indices[edges] = targets[inside]  # row by row, each in ascending order
            for name, values in offset_data.items():
                along = values[0][order] if types is None else values[types[:, None], order]
                data[name][edges] = along[inside]
        return CellGraph._owned(indptr, indices, data)

    def _targets(
        self, start: int, stop: int, offsets: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:
        """Return the cell at each offset from cells ``start`` to ``stop``, or ``n_cells``.

        Shape (stop - start, n_offsets), with the offsets of each cell's type. Offsets that
        leave a non-periodic axis give ``n_cells``, past every cell id, so they sort last. Also
        returns the cells' types, shape (stop - start,), or None on lattices with one type.
        """
        coords = np.stack(np.unravel_index(np.arange(start, stop), self.shape), axis=-1)
        if len(offsets) == 1:
            types, moved = None, coords[:, None, :] + offsets[0]
        else:
            types = coords @ self._tiling.type_steps % self._tiling.n_types
            moved = coords[:, None, :] + offsets[types]
        return self._wrapped_ids(moved, missing=self.n_cells), types

    @cached_property
    def _strides(self) -> NDArray[np.int64]:
        """How far a step along each axis moves a cell id, row-major: int64, shape (dim,)."""
        return read_only(np.cumprod((1, *self.shape[:0:-1]), dtype=np.int64)[::-1])

    def _wrapped_ids(self, coords: NDArray[np.int64], missing: int = -1) -> NDArray[np.int64]:
        """Return the ids of the cells at integer lattice coordinates, shape (..., dim).

        Coordinates wrap around periodic axes. Beyond the edge of a non-periodic axis, the id
        is ``missing``. Shape (...).
        """
        shape = np.array(self.shape, dtype=np.int64)
        coords = np.where(self.periodic, coords % shape, coords)
        inside = np.all((coords >= 0) & (coords < shape), axis=-1)
        return np.where(inside, coords @ self._strides, missing)


def _keep_in_rows[D: np.generic](values: NDArray[D], keep: NDArray[np.bool_]) -> NDArray[D]:
    """Return ``values[t][keep[t]]`` for each row ``t`` (one per type of cell), stacked.

    Every row keeps as many entries, so the result has shape
    (n_rows, n_kept, *values.shape[2:]).
    """
    return np.stack([values[t][keep[t]] for t in range(len(keep))])


def _round_each(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Round lattice coordinates, shape (..., dim), to those of the nearest n-cube's center."""
    return np.floor(coords + 0.5)


def _hex_round(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Round axial coordinates, shape (..., 2), to those of the nearest hexagon's center.

    This is cube rounding: round ``q``, ``r`` and ``s = -q - r`` each, then recompute the one
    that moved most from the other two, so that they add up to 0 again.
    """
    q, r = coords[..., 0], coords[..., 1]
    s = -q - r
    rq, rr, rs = np.floor(q + 0.5), np.floor(r + 0.5), np.floor(s + 0.5)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    return np.stack([np.where(fix_q, -rr - rs, rq), np.where(fix_r, -rq - rs, rr)], axis=-1)


def _triangle_round(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Round lattice coordinates, shape (..., 2), to those of the triangle holding each point.

    Row ``j`` holds the points between ``j - 1/2`` and ``j + 1/2`` steps along axis 1. At
    height ``h`` in it, from 0 at its bottom to 1 at its top, the triangle at column ``i``
    pointing up spans ``|x - i| <= 1 - h``, and those pointing down fill the rest. Of the two
    columns on either side of a point, one holds a triangle pointing up.
    """
    x, y = coords[..., 0], coords[..., 1]
    row = np.floor(y + 0.5)
    height = y + 0.5 - row
    left = np.floor(x)
    up = np.where(np.mod(left + row, 2) == 0, left, left + 1)  # the column pointing up
    down = 2 * left + 1 - up
    return np.stack([np.where(np.abs(x - up) + height <= 1, up, down), row], axis=-1)
