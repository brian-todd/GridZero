"""Geometric helpers: icosahedral spheres, longitude and latitude, and great-circle angles."""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import check_int

_GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0

# Faces of the icosahedron returned by `icosahedron`, as indices into its vertex array.
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)  # fmt: skip


def icosahedron() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return the vertices and faces of an icosahedron inscribed in the unit sphere.

    Returns
    -------
    vertices : ndarray of float64, shape (12, 3)
        Unit vectors to the vertices.
    faces : ndarray of int64, shape (20, 3)
        Vertex indices of each triangular face.
    """
    t = _GOLDEN_RATIO
    vertices = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, _ICOSAHEDRON_FACES.copy()


def subdivided_icosahedron(frequency: SupportsIndex) -> NDArray[np.float64]:
    """Return the vertices of a geodesic subdivision of the icosahedron.

    Each icosahedron edge is split into ``frequency`` segments, each face into ``frequency**2``
    triangles, and every point is projected radially onto the unit sphere. The points are
    built combinatorially as the original vertices, then edge interiors, then face interiors,
    so no point is generated twice and no tolerance-based deduplication is needed.

    Parameters
    ----------
    frequency : int
        Number of segments per icosahedron edge. Must be at least 1.

    Returns
    -------
    ndarray of float64, shape (10 * frequency**2 + 2, 3)
        Unit vectors, in a deterministic order.

    Raises
    ------
    TypeError
        If ``frequency`` is not an integer.
    ValueError
        If ``frequency`` is less than 1.
    """
    f = check_int(frequency, "frequency", minimum=1)
    vertices, faces = icosahedron()
    parts = [vertices]

    if f > 1:
        edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
        t = (np.arange(1, f) / f)[None, :, None]  # (1, f-1, 1)
        a, b = vertices[edges[:, 0]][:, None, :], vertices[edges[:, 1]][:, None, :]
        parts.append(((1 - t) * a + t * b).reshape(-1, 3))

    if f > 2:
        i, j = np.meshgrid(np.arange(1, f), np.arange(1, f), indexing="ij")
        interior = i + j <= f - 1
        u = (i[interior] / f)[None, :, None]  # weight of a face's second vertex
        v = (j[interior] / f)[None, :, None]  # weight of a face's third vertex
        a, b, c = (vertices[faces[:, k]][:, None, :] for k in range(3))
        parts.append(((1 - u - v) * a + u * b + v * c).reshape(-1, 3))

    points = np.concatenate(parts)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def sphere_points(lon: ArrayLike, lat: ArrayLike) -> NDArray[np.float64]:
    """Return the points of the unit sphere at given longitudes and latitudes.

    Longitude turns from the x axis toward the y axis, and latitude rises toward the z axis.

    Parameters
    ----------
    lon, lat : array_like of float, shape (...)
        Longitudes and latitudes in degrees. They broadcast against each other.

    Returns
    -------
    ndarray of float64, shape (..., 3)
        Unit vectors.
    """
    lon, lat = np.radians(lon), np.radians(lat)
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def lon_lat(points: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the longitudes and latitudes of points, the inverse of `sphere_points`.

    Parameters
    ----------
    points : array_like of float, shape (..., 3)
        Points other than the origin. They need not be unit length.

    Returns
    -------
    lon : ndarray of float64, shape (...)
        Longitudes in degrees, in ``[-180, 180]``.
    lat : ndarray of float64, shape (...)
        Latitudes in degrees, in ``[-90, 90]``.
    """
    x, y, z = np.moveaxis(np.asarray(points, dtype=np.float64), -1, 0)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.hypot(x, y)))


def great_circle_angle(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the angle between direction vectors, in radians.

    Uses ``atan2(|a x b|, a . b)``, which stays accurate for both tiny and near-antipodal
    angles, unlike ``arccos`` of the dot product.

    Parameters
    ----------
    a, b : array_like of float, shape (..., 3)
        Direction vectors. They need not be unit length. Leading axes broadcast.

    Returns
    -------
    ndarray of float64, shape (...)
        Angles in ``[0, pi]``. Multiply by the sphere radius to get arc lengths.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.einsum("...i,...i->...", a, b)
    return np.arctan2(cross, dot)
