"""Mesh spaces: polygonal cells with explicit geometry, including the icosphere planet grid."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Self, SupportsIndex

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree, SphericalVoronoi

from gridzero.core._arrays import (
    check_cell,
    finite_floats,
    frozen_copy,
    int_array,
)
from gridzero.core.geometry import great_circle_angle, subdivided_icosahedron
from gridzero.core.graph import CellGraph
from gridzero.core.space import (
    RADIUS_RTOL,
    AdjacencyKind,
    RadiusCache,
    check_radius,
    min_face_dim,
)

type Metric = Literal["euclidean", "spherical"]

# Relative spread allowed in the distance from the origin for points of a spherical mesh.
_SPHERE_RTOL = 1e-9

# Relative slack when searching a k-d tree for candidate pairs, before the exact distance test.
_SEARCH_SLACK = 1e-6


@dataclass(frozen=True, eq=False, init=False)
class MeshSpace:
    """Polygonal cells on a 2-D surface (a plane or a sphere), defined by explicit geometry.

    Cell ``i`` is the polygon whose vertices are
    ``vertices[polygon_indices[polygon_indptr[i]:polygon_indptr[i + 1]]]``, listed in cyclic
    order. Adjacency is derived from shared geometry: cells sharing a polygon edge are
    facet-adjacent, and cells sharing any vertex are vertex-adjacent. This assumes an
    edge-to-edge tiling, where two cells meet along whole edges of each or not at all. Two
    cells may share several consecutive edges, for example when a vertex lies part-way along
    their border. The space is immutable, so copying it returns the same object.

    Parameters
    ----------
    centers : array_like of float, shape (n_cells, embed_dim)
        Representative point of each cell, for example the Voronoi site.
    measures : array_like of float, shape (n_cells,)
        Area of each cell. Must be finite and positive.
    vertices : array_like of float, shape (n_vertices, embed_dim)
        Polygon vertex coordinates.
    polygon_indptr : array_like of int, shape (n_cells + 1,)
        Offsets into ``polygon_indices`` for each cell.
    polygon_indices : array_like of int, shape (n_polygon_vertices,)
        Vertex ids of each cell's polygon, in cyclic order, with no vertex repeated.
    metric : {"euclidean", "spherical"}, default "euclidean"
        How distances and boundary lengths are measured: straight lines, or great-circle arcs
        on the sphere the points lie on. A spherical mesh needs 3-D coordinates, all on one
        sphere centered at the origin.

    Raises
    ------
    TypeError
        If ``polygon_indptr`` or ``polygon_indices`` does not hold integers.
    ValueError
        If the mesh is empty, the arrays are inconsistent with each other, a measure or
        coordinate is invalid, a polygon has fewer than 3 vertices or lists one twice, or two
        consecutive vertices of a polygon are at the same point. Other degenerate shapes, such
        as polygons with collinear vertices, are not detected.
    """

    centers: NDArray[np.float64]
    measures: NDArray[np.float64]
    vertices: NDArray[np.float64]
    polygon_indptr: NDArray[np.int64]
    polygon_indices: NDArray[np.int64]
    metric: Metric
    _graphs: dict[int, CellGraph] = field(repr=False)
    _radius_graphs: RadiusCache[CellGraph] = field(repr=False)

    def __init__(
        self,
        centers: ArrayLike,
        measures: ArrayLike,
        vertices: ArrayLike,
        polygon_indptr: ArrayLike,
        polygon_indices: ArrayLike,
        metric: Metric = "euclidean",
    ) -> None:
        centers = frozen_copy(centers, np.float64)
        measures = frozen_copy(measures, np.float64)
        vertices = frozen_copy(vertices, np.float64)
        indptr = frozen_copy(int_array(polygon_indptr, "polygon_indptr"), np.int64)
        indices = frozen_copy(int_array(polygon_indices, "polygon_indices"), np.int64)
        if centers.ndim != 2 or vertices.ndim != 2 or vertices.shape[1] != centers.shape[1]:
            raise ValueError("centers and vertices must be 2-D with the same embedding dimension")
        n_cells = centers.shape[0]
        if n_cells == 0:
            raise ValueError("a mesh needs at least one cell")
        if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(vertices))):
            raise ValueError("centers and vertices must be finite")
        if measures.shape != (n_cells,):
            raise ValueError("measures must have one entry per cell")
        if not np.all(np.isfinite(measures) & (measures > 0)):
            raise ValueError("measures must be finite and positive")
        if indices.ndim != 1:
            raise ValueError("polygon_indices must be a 1-D array")
        if indptr.shape != (n_cells + 1,) or indptr[0] != 0 or indptr[-1] != indices.size:
            raise ValueError(
                "polygon_indptr must have n_cells + 1 offsets spanning polygon_indices"
            )
        sides = np.diff(indptr)
        if np.any(sides < 3):
            raise ValueError("every polygon needs at least 3 vertices")
        if indices.min() < 0 or indices.max() >= len(vertices):
            raise ValueError("polygon_indices must refer to existing vertices")
        # A repeated vertex would give a polygon a side shared with itself (a self-loop).
        keys = np.sort(np.repeat(np.arange(n_cells), sides) * len(vertices) + indices)
        if np.any(keys[1:] == keys[:-1]):
            raise ValueError("a polygon lists the same vertex more than once")
        # Distinct vertices at one point would make a side of zero length: a zero boundary.
        following = indices[_next_corner(indptr)]
        if np.any(np.all(vertices[indices] == vertices[following], axis=1)):
            raise ValueError("two consecutive vertices of a polygon are at the same point")
        if metric not in ("euclidean", "spherical"):
            raise ValueError(f"metric must be 'euclidean' or 'spherical', got {metric!r}")
        if metric == "spherical":
            if centers.shape[1] != 3:
                raise ValueError("a spherical mesh needs 3-D coordinates")
            radius = np.linalg.norm(np.concatenate([centers, vertices]), axis=1)
            if not radius.min() > 0 or np.ptp(radius) > _SPHERE_RTOL * radius.max():
                raise ValueError(
                    "the centers and vertices of a spherical mesh must lie on one sphere "
                    "centered at the origin"
                )
        for name, value in (
            ("centers", centers),
            ("measures", measures),
            ("vertices", vertices),
            ("polygon_indptr", indptr),
            ("polygon_indices", indices),
            ("metric", metric),
            ("_graphs", {}),
            ("_radius_graphs", RadiusCache()),
        ):
            object.__setattr__(self, name, value)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor so unpickled arrays are validated and read-only again.
        # Cached graphs are dropped and rebuilt on demand.
        fields = (self.centers, self.measures, self.vertices, self.polygon_indptr)
        return type(self), (*fields, self.polygon_indices, self.metric)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.centers.shape[0]

    @property
    def dim(self) -> int:
        """Intrinsic dimension of the cells. Mesh spaces currently hold 2-D polygons."""
        return 2

    def polygon(self, cell: SupportsIndex) -> NDArray[np.float64]:
        """Return the vertex coordinates of one cell's polygon.

        Parameters
        ----------
        cell : int
            Cell id, in ``[0, n_cells)``.

        Returns
        -------
        ndarray of float64, shape (n_sides, embed_dim)
            Vertices in cyclic order.

        Raises
        ------
        TypeError
            If ``cell`` is not an integer.
        IndexError
            If ``cell`` is out of range. Negative ids are not counted from the end.
        """
        cell = check_cell(cell, self.n_cells)
        return self.vertices[
            self.polygon_indices[self.polygon_indptr[cell] : self.polygon_indptr[cell + 1]]
        ]

    def distance(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Return distances between points under this space's metric.

        Parameters
        ----------
        a, b : array_like of float, shape (..., embed_dim)
            Points. Leading axes broadcast. For the spherical metric, the points must lie on
            the sphere, and the radius is taken from ``|a|``.

        Returns
        -------
        ndarray of float64, shape (...)
            Straight-line distances, or great-circle arc lengths.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if self.metric == "spherical":
            return np.linalg.norm(a, axis=-1) * great_circle_angle(a, b)
        return np.linalg.norm(a - b, axis=-1)

    def locate(self, points: ArrayLike) -> NDArray[np.int64]:
        """Return the cell whose center is nearest each point.

        On a Voronoi mesh, such as `icosphere`, that is the cell containing the point. Every
        point gets a cell, also one beyond the edge of a planar mesh.

        Parameters
        ----------
        points : array_like of float, shape (..., embed_dim)
            Points in the coordinates of `centers`. On a spherical mesh a point stands for its
            direction from the center of the sphere, so its distance from the center does not
            matter.

        Returns
        -------
        ndarray of int64, shape (...)
            Cell ids.

        Raises
        ------
        TypeError
            If ``points`` does not hold numbers.
        ValueError
            If ``points`` does not end in an axis of length ``embed_dim``, or holds NaN or
            infinite values. On a spherical mesh, also if a point is the center of the sphere.
            On a planar mesh, also if a point lies so far out (about 1e154) that its squared
            distances overflow.
        """
        points = finite_floats(points, "points")
        width = self.centers.shape[1]
        if points.ndim == 0 or points.shape[-1] != width:
            raise ValueError(f"points must have shape (..., {width}), got {points.shape}")
        if self.metric == "spherical":
            # The tree's centers lie on the unit sphere, so the nearest is the same all along a
            # point's ray. Scale each point near the sphere, where its distances neither
            # overflow nor all round to 1.
            scale = np.abs(points).max(axis=-1, keepdims=True, initial=0.0)
            if not (scale > 0).all():
                raise ValueError("points must not be the center of the sphere")
            points = points / scale
        distance, cells = self._tree.query(points)
        if not np.isfinite(distance).all():
            raise ValueError("points lie too far out to locate")
        return np.asarray(cells, dtype=np.int64)  # a single point gives a Python int

    @cached_property
    def _tree(self) -> KDTree:
        """k-d tree of the centers, on the unit sphere for a spherical mesh."""
        centers = self.centers
        if self.metric == "spherical":
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        return KDTree(centers)

    def adjacency(self, kind: AdjacencyKind = "facet") -> CellGraph:
        """Return the symmetric graph of cells that touch in the given way.

        The graph is built once per kind and cached.

        Parameters
        ----------
        kind : {"facet", "vertex"} or int, default "facet"
            ``"facet"`` (or 1) means sharing a polygon edge. ``"vertex"`` (or 0) means sharing
            at least one vertex.

        Returns
        -------
        CellGraph
            Edge attributes are ``"distance"`` (between centers) and ``"boundary"`` (total
            length of the shared edges, or 0 for cells that meet only at a vertex).

        Raises
        ------
        ValueError
            If a polygon edge is shared by more than two cells (the mesh is not a surface).
        """
        j = min_face_dim(kind, self.dim)
        if j not in self._graphs:
            self._graphs[j] = self._facet_graph() if j == 1 else self._vertex_graph()
        return self._graphs[j]

    def radius_graph(self, radius: float) -> CellGraph:
        """Return the symmetric graph of cells whose centers lie within ``radius`` of each other.

        Parameters
        ----------
        radius : float
            Largest center-to-center distance, finite and non-negative: a straight line, or a
            great-circle arc on a spherical mesh. Distances within a relative `RADIUS_RTOL` of
            ``radius`` count as within it.

        Returns
        -------
        CellGraph
            Symmetric graph without self-loops, with edge attribute ``"distance"``. Cached for
            the last few radii (`RadiusCache`): asking again returns the same graph.
        """
        limit = check_radius(radius) * (1 + RADIUS_RTOL)
        return self._radius_graphs.get(limit, self._radius_graph)

    def _radius_graph(self, limit: float) -> CellGraph:
        """Build `radius_graph` for a checked ``limit``, its tolerance included."""
        reach = limit
        if self.metric == "spherical":
            # Search by chord length, which grows with arc length up to half way round.
            sphere = float(np.linalg.norm(self.centers[0]))
            reach = 2 * sphere * np.sin(min(limit / sphere, np.pi) / 2)
        tree = KDTree(self.centers)
        pairs = tree.query_pairs(reach * (1 + _SEARCH_SLACK), output_type="ndarray")
        a, b = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        distance = self.distance(self.centers[a], self.centers[b])
        inside = distance <= limit
        return _symmetric_graph(a[inside], b[inside], self.n_cells, {"distance": distance[inside]})

    def _within(self, cell: int, limit: float) -> NDArray[np.int64]:
        """Return the cells whose centers lie within ``limit`` of ``cell``'s, sorted, itself too.

        Distances are measured as `radius_graph` measures them, from the lower cell id, so the
        two agree. Returns int64 of shape (n_near,).
        """
        center = self.centers[cell]
        reach = limit
        if self.metric == "spherical":
            # The tree is on the unit sphere: search by chord length there.
            sphere = float(np.linalg.norm(self.centers[0]))
            reach = 2 * np.sin(min(limit / sphere, np.pi) / 2)
            center = center / np.linalg.norm(center)
        found = np.asarray(self._tree.query_ball_point(center, reach * (1 + _SEARCH_SLACK)))
        found = found.astype(np.int64)
        low, high = np.minimum(found, cell), np.maximum(found, cell)
        near = self.distance(self.centers[low], self.centers[high]) <= limit
        return np.unique(np.append(found[near], cell))

    def _polygon_edges(self) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Return ``(cell, lo, hi)`` for every polygon side, with vertex ids sorted ``lo <= hi``."""
        cell = np.repeat(np.arange(self.n_cells), np.diff(self.polygon_indptr))
        v0 = self.polygon_indices
        v1 = self.polygon_indices[_next_corner(self.polygon_indptr)]
        return cell, np.minimum(v0, v1), np.maximum(v0, v1)

    def _facet_graph(self) -> CellGraph:
        """Build the facet adjacency: cells that share a polygon side, as `adjacency` documents.

        Raises
        ------
        ValueError
            If a polygon side is shared by more than two cells.
        """
        cell, lo, hi = self._polygon_edges()
        order = np.lexsort((cell, hi, lo))
        cell, lo, hi = cell[order], lo[order], hi[order]
        same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
        if np.any(same[1:] & same[:-1]):
            raise ValueError("a polygon edge is shared by more than two cells")
        a, b = cell[:-1][same], cell[1:][same]  # a < b: each side's cells are sorted by id
        side = self.distance(self.vertices[lo[:-1][same]], self.vertices[hi[:-1][same]])
        # Cells may share several sides. Their boundary is the total length of those sides.
        _, first, pair = np.unique(a * self.n_cells + b, return_index=True, return_inverse=True)
        boundary = np.bincount(pair, weights=side, minlength=first.size)
        a, b = a[first], b[first]
        distance = self.distance(self.centers[a], self.centers[b])
        return _symmetric_graph(a, b, self.n_cells, {"distance": distance, "boundary": boundary})

    def _vertex_graph(self) -> CellGraph:
        """Build the vertex adjacency: cells that share a polygon corner.

        Edges get ``"distance"`` and ``"boundary"`` as in `adjacency`. A pair that shares only
        corners has a boundary of 0.
        """
        sides = np.diff(self.polygon_indptr)
        incidence = sp.csr_array(
            (
                np.ones(self.polygon_indices.size),
                (np.repeat(np.arange(self.n_cells), sides), self.polygon_indices),
            ),
            shape=(self.n_cells, len(self.vertices)),
        )
        shared = sp.coo_array(incidence @ incidence.T)
        off_diagonal = shared.row != shared.col
        a = shared.row[off_diagonal].astype(np.int64)
        b = shared.col[off_diagonal].astype(np.int64)

        facets = self.adjacency("facet")
        n = self.n_cells
        keys = a * n + b
        boundary = np.zeros(keys.size)
        if facets.n_edges:
            facet_keys = facets.sources * n + facets.indices  # sorted, because rows are sorted
            slot = np.minimum(np.searchsorted(facet_keys, keys), facets.n_edges - 1)
            is_facet = facet_keys[slot] == keys
            boundary[is_facet] = facets.edge_data["boundary"][slot[is_facet]]
        return CellGraph.from_edges(
            a,
            b,
            n,
            edge_data={
                "distance": self.distance(self.centers[a], self.centers[b]),
                "boundary": boundary,
            },
        )


def _next_corner(indptr: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return, for each entry of ``polygon_indices``, the position of the next corner.

    Corners are cyclic, so the last corner of each polygon is followed by its first.
    ``indptr`` is `MeshSpace.polygon_indptr`, int64 of shape (n_cells + 1,). Returns int64 of
    shape (indptr[-1],), the length of ``polygon_indices``.
    """
    sides = np.diff(indptr)
    start = np.repeat(indptr[:-1], sides)
    position = np.arange(indptr[-1])
    return start + (position - start + 1) % np.repeat(sides, sides)


def _symmetric_graph(
    a: NDArray[np.int64], b: NDArray[np.int64], n_cells: int, data: dict[str, NDArray[np.float64]]
) -> CellGraph:
    """Link each pair ``a[k], b[k]`` both ways, giving both edges the pair's attributes.

    ``a`` and ``b`` are cell ids, int64 of shape (n_pairs,), with no pair twice in either
    order. Each array of ``data`` is float64 of shape (n_pairs,).
    """
    return CellGraph.from_edges(
        np.concatenate([a, b]),
        np.concatenate([b, a]),
        n_cells,
        edge_data={name: np.concatenate([values, values]) for name, values in data.items()},
    )


def check_spherical(space: object) -> MeshSpace:
    """Return ``space`` after checking that it is a spherical mesh, such as an icosphere.

    Parameters
    ----------
    space : object
        Space to check.

    Returns
    -------
    MeshSpace
        ``space`` itself.

    Raises
    ------
    TypeError
        If ``space`` is not a `MeshSpace`.
    ValueError
        If its metric is not ``"spherical"``.
    """
    if not isinstance(space, MeshSpace):
        raise TypeError(
            f"need a spherical MeshSpace such as icosphere(32), got {type(space).__name__}"
        )
    if space.metric != "spherical":
        raise ValueError(f"need a spherical MeshSpace, got one with metric {space.metric!r}")
    return space


def icosphere(frequency: SupportsIndex, radius: float = 1.0) -> MeshSpace:
    """Build a planet grid: the spherical Voronoi cells of a subdivided icosahedron.

    The result is a Goldberg-polyhedron-like tiling of ``10 * frequency**2 + 2`` cells: exactly
    12 pentagons, with all other cells hexagons. Naive subdivision makes the largest cell
    about twice the area of the smallest.

    Parameters
    ----------
    frequency : int
        Subdivisions per icosahedron edge. ``1`` gives 12 pentagons (a dodecahedron), ``16``
        gives 2,562 cells, and ``32`` gives 10,242 cells.
    radius : float, default 1.0
        Sphere radius. Distances scale with ``radius`` and areas with ``radius**2``.

    Returns
    -------
    MeshSpace
        Spherical-metric mesh. Cell centers are the Voronoi sites.

    Raises
    ------
    TypeError
        If ``frequency`` is not an integer or ``radius`` is not a real number.
    ValueError
        If ``frequency`` is less than 1 or ``radius`` is not positive and finite.
    """
    if check_radius(radius) == 0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    sites = subdivided_icosahedron(frequency)
    voronoi = SphericalVoronoi(sites, radius=1.0, center=np.zeros(3))
    voronoi.sort_vertices_of_regions()
    regions = [np.asarray(region, dtype=np.int64) for region in voronoi.regions]
    indptr = np.zeros(len(regions) + 1, dtype=np.int64)
    np.cumsum([len(region) for region in regions], out=indptr[1:])
    return MeshSpace(
        centers=sites * radius,
        measures=voronoi.calculate_areas() * radius**2,
        vertices=voronoi.vertices * radius,
        polygon_indptr=indptr,
        polygon_indices=np.concatenate(regions),
        metric="spherical",
    )
