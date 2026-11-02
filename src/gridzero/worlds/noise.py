"""Random smooth fields for generating worlds, and the relief of land and sea made from them.

`noise` sums plane waves, so its value at a point depends only on the point and the seed. The
same seed then gives the same features on a coarse and a fine grid, and fields on a lattice
that wraps around can repeat over its periods.
"""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import check_int, check_real, finite_floats

_CHUNK = 4096
"""Points per block when summing waves, so a block holds ``4096 * waves`` values at most."""


def noise(
    points: ArrayLike,
    rng: np.random.Generator,
    *,
    scales: tuple[float, float] = (1.0, 64.0),
    roughness: float = 0.5,
    waves: SupportsIndex = 512,
    periods: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return a random smooth field at points of a sphere or a plane: a sum of plane waves.

    Wave ``i`` is ``a_i * cos(dot(k_i, p) + phase_i)``, with a wavevector ``k_i`` in a random
    direction and a random phase. Its wavelength is about ``2 * pi / |k_i|``: along the unit
    sphere, ``|k_i|`` waves go around it. The wavenumbers ``|k_i|`` are spread evenly over the
    octaves from ``scales[0]`` to ``scales[1]``, and each octave's amplitude is ``roughness``
    times that of the octave below.

    The field is a function of position: the value at a point depends only on the point and
    the draws from ``rng``, never on the other points. Evaluated at the cells of a coarse and
    a fine mesh, it shows the same features.

    Parameters
    ----------
    points : array_like of float, shape (..., d)
        Where to evaluate the field, with ``d >= 1``: usually points of the unit sphere
        (``d = 3``) or of a plane (``d = 2``).
    rng : numpy.random.Generator
        Source of the random directions and phases.
    scales : (float, float), default (1.0, 64.0)
        The lowest and highest wavenumber, with ``0 < scales[0] <= scales[1]``.
    roughness : float, default 0.5
        Amplitude ratio between successive octaves, positive. Larger values give more small
        features, such as ragged coasts.
    waves : int, default 512
        Number of waves, at least 1. More waves give a more even, less streaky field.
    periods : array_like of float, shape (n_periods, d), optional
        Shifts under which the field repeats, at most ``d`` of them and linearly independent,
        such as the sides of a grid that wraps around. Each wavevector then moves to a nearby
        one that fits a whole number of waves into every period. Waves much longer than a
        period move the most, and some become constant.

    Returns
    -------
    ndarray of float64, shape (...)
        Field values. At every point, their mean over random draws is 0 and their variance is
        1.

    Raises
    ------
    TypeError
        If ``rng`` is not a `numpy.random.Generator`, ``waves`` is not an integer, a scale or
        ``roughness`` is not a real number, or ``points`` or ``periods`` does not hold
        numbers.
    ValueError
        If ``points`` has no last axis or an empty one, ``points`` or ``periods`` holds NaN or
        infinite values, the scales are out of order or not positive, ``roughness`` is not
        positive, ``waves`` is less than 1, or ``periods`` has the wrong shape or is not
        linearly independent.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    points = finite_floats(points, "points")
    if points.ndim == 0 or points.shape[-1] == 0:
        raise ValueError(f"points must have shape (..., d) with d >= 1, got {points.shape}")
    dim = points.shape[-1]
    low, high = (check_real(scale, "each scale", 0.0) for scale in scales)
    if not 0 < low <= high:
        raise ValueError(f"need 0 < scales[0] <= scales[1], got {scales!r}")
    roughness = check_real(roughness, "roughness", 0.0)
    if roughness == 0:
        raise ValueError("roughness must be positive")
    waves = check_int(waves, "waves", minimum=1)
    if periods is not None:
        periods = _check_periods(periods, dim)
    octaves = np.log2(high / low) * (np.arange(waves) + 0.5) / waves  # evenly spread
    amplitude = roughness**octaves
    amplitude /= np.sqrt(np.sum(amplitude**2) / 2)  # each cosine has variance 1/2
    directions = rng.normal(size=(waves, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    wavevectors = directions.T * (low * 2.0**octaves)  # (dim, waves)
    if periods is not None and len(periods):
        wavevectors = _fit_periods(wavevectors, periods)
    phases = rng.uniform(0.0, 2 * np.pi, waves)
    flat = points.reshape(-1, dim)
    out = np.empty(len(flat))
    for start in range(0, len(flat), _CHUNK):
        block = flat[start : start + _CHUNK]
        out[start : start + _CHUNK] = np.cos(block @ wavevectors + phases) @ amplitude
    return out.reshape(points.shape[:-1])


def _check_periods(periods: ArrayLike, dim: int) -> NDArray[np.float64]:
    """Return ``periods`` as float64, shape (n_periods, dim), after checking them."""
    periods = finite_floats(periods, "periods")
    if periods.ndim != 2 or periods.shape[1] != dim or len(periods) > dim:
        raise ValueError(
            f"periods must have shape (n_periods, {dim}) with n_periods <= {dim}, "
            f"got {periods.shape}"
        )
    if len(periods) and np.linalg.matrix_rank(periods) < len(periods):
        raise ValueError("periods must be linearly independent")
    return periods


def _fit_periods(
    wavevectors: NDArray[np.float64], periods: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Move each wavevector (a column) so that whole waves fit into every period.

    Wave ``k`` repeats over period ``P`` when ``dot(k, P)`` is a multiple of ``2 * pi``. Round
    that number of turns along each period, and keep the wavevector's components orthogonal to
    every period.
    """
    n_periods = len(periods)
    across = np.linalg.svd(periods)[2][n_periods:]  # orthonormal rows, orthogonal to them
    frame = np.concatenate([periods, across])  # (dim, dim), invertible
    turns = frame @ wavevectors / (2 * np.pi)
    turns[:n_periods] = np.round(turns[:n_periods])
    return np.linalg.solve(frame, 2 * np.pi * turns)


def _relief(
    points: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    areas: NDArray[np.float64],
    land: float,
    roughness: float,
    periods: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the elevation of each cell, and its continents' height above sea level.

    Height is the sum of two `noise` fields: continents, with wavelengths from the size of the
    world down to about a quarter of it, plus half as much detail at smaller scales. Sea level
    leaves the fraction ``land`` of the total ``areas`` above it. Elevation is the height
    relative to sea level, stretched to reach -1 at the lowest cell and 1 at the highest. The
    continents alone, relative to sea level, are larger far inland and more negative far out
    at sea.

    Draws two fields from ``rng``. Returns two arrays of float64, shape (n_cells,).
    """
    continents = noise(points, rng, scales=(1.0, 4.0), roughness=0.7, periods=periods)
    detail = noise(points, rng, scales=(4.0, 128.0), roughness=roughness, periods=periods)
    height = continents + 0.5 * detail
    sea_level = _sea_level(height, areas, land)
    above = height > sea_level
    elevation = np.zeros(len(height))
    tiny = np.finfo(np.float64).tiny
    np.divide(height - sea_level, max(height.max() - sea_level, tiny), out=elevation, where=above)
    np.divide(height - sea_level, max(sea_level - height.min(), tiny), out=elevation, where=~above)
    return np.clip(elevation, -1, 1), continents - sea_level


def _sea_level(height: NDArray[np.float64], areas: NDArray[np.float64], land: float) -> float:
    """Return the height that leaves the area fraction nearest ``land`` above it.

    ``height`` and ``areas`` are float64 of shape (n_cells,), and ``land`` a fraction in
    [0, 1]. The level lies halfway between the lowest cell above and the highest below.
    """
    order = np.argsort(height)[::-1]  # highest first
    covered = np.concatenate([[0.0], np.cumsum(areas[order])]) / areas.sum()
    above = int(np.argmin(np.abs(covered - land)))  # how many cells are land
    if above == 0:
        return float(height[order[0]])
    if above == len(height):
        return float(height[order[-1]]) - 1.0
    return float(height[order[above - 1]] + height[order[above]]) / 2
