"""Terrain colors, shared by planets and the terrain of lattices."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# As RGB.
_DEEP = np.array([0.04, 0.11, 0.33])
_SHALLOW = np.array([0.22, 0.47, 0.72])
_SEA_ICE = np.array([0.80, 0.87, 0.93])
_SAND = np.array([0.84, 0.76, 0.52])
_GRASS = np.array([0.47, 0.65, 0.31])
_FOREST = np.array([0.16, 0.39, 0.18])
_HILLS = np.array([0.66, 0.62, 0.40])
_ROCK = np.array([0.52, 0.47, 0.42])
_SNOW = np.array([0.95, 0.95, 0.97])


def _sea_colors(elevation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the colors of the sea at each elevation, as RGB: shape (n_cells, 3)."""
    return _mix(_SHALLOW, _DEEP, np.sqrt(np.clip(-elevation[:, None], 0.0, 1.0)))


def _mix(a: ArrayLike, b: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
    """Return ``a`` blended toward ``b`` by ``t``, clipped to [0, 1].

    ``a``, ``b`` and ``t`` broadcast together: RGB colors of shape (3,) or (n, 3), and
    fractions of shape (n, 1). Returns float64 of the broadcast shape, (n, 3).
    """
    t = np.clip(t, 0.0, 1.0)
    return (1 - t) * np.asarray(a) + t * np.asarray(b)
