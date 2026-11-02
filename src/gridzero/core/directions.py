"""Directions on a 2-D lattice: facing, snapping a vector to a direction, front halves.

A piece on squares or hexes faces one of the lattice's facet directions
(`LatticeSpace.directions`), numbered counterclockwise from the first lattice vector. Vectors
here are lattice offsets, such as ``space.unravel(b) - space.unravel(a)``, and every test is
exact: dot products are taken in an integer metric of the lattice, a positive multiple of the
Euclidean one, so a vector exactly between two directions is a tie, settled by a rule.
"""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core._arrays import check_int, int_array
from gridzero.core.lattice import LatticeSpace


def snap(space: LatticeSpace, vector: ArrayLike, previous: SupportsIndex) -> int:
    """Return the direction nearest a vector, breaking a tie toward the previous facing.

    A vector exactly between two directions, such as ``(1, 1)`` on squares or toward a corner
    of a hexagon, goes to the one fewer turns from ``previous``. That always settles it: two
    neighboring directions are never equally many turns from a third.

    Parameters
    ----------
    space : LatticeSpace
        Squares or hexes in 2-D.
    vector : array_like of int, shape (2,)
        A lattice offset. The zero vector ties every direction, so it keeps ``previous``.
    previous : int
        The direction faced before, ``0 <= previous < n_directions``.

    Returns
    -------
    int
        A direction.

    Raises
    ------
    TypeError
        If ``vector`` is not of integers or ``previous`` not an integer.
    ValueError
        If the lattice has no directions (see `LatticeSpace.directions`), ``vector`` is not of
        shape (2,), or ``previous`` is not a direction.
    """
    directions, metric = space._directions
    previous = _check_direction(previous, len(directions), "previous")
    v = int_array(vector, "vector")
    if v.shape != (2,):
        raise ValueError(f"vector must have shape (2,), got {v.shape}")
    dots = directions @ (metric @ v)
    candidates = np.flatnonzero(dots == dots.max())  # all of them for the zero vector
    n = len(directions)
    away = [_turns(int(d), previous, n) for d in candidates]
    return int(candidates[int(np.argmin(away))])


def turns(space: LatticeSpace, a: SupportsIndex, b: SupportsIndex) -> int:
    """Return how many steps of direction separate two directions, the shorter way round.

    Parameters
    ----------
    space : LatticeSpace
        Squares or hexes in 2-D.
    a, b : int
        Directions, ``0 <= a, b < n_directions``.

    Returns
    -------
    int
        From 0 to ``n_directions // 2``: on hexes, sixths of a full turn.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` is not an integer.
    ValueError
        If the lattice has no directions, or ``a`` or ``b`` is not a direction.
    """
    n = len(space._directions[0])
    return _turns(_check_direction(a, n, "a"), _check_direction(b, n, "b"), n)


def in_front(space: LatticeSpace, facing: SupportsIndex, vectors: ArrayLike) -> NDArray[np.bool_]:
    """Return whether vectors from a piece point into the half of the plane it faces.

    A vector at a right angle to the facing, along the edge of the half, counts as in front.

    Parameters
    ----------
    space : LatticeSpace
        Squares or hexes in 2-D.
    facing : int
        The piece's direction, ``0 <= facing < n_directions``.
    vectors : array_like of int, shape (..., 2)
        Lattice offsets from the piece, such as toward an attacker. Leading axes are kept.

    Returns
    -------
    ndarray of bool, shape (...)
        True where the vector is in front, or zero.

    Raises
    ------
    TypeError
        If ``vectors`` is not of integers or ``facing`` not an integer.
    ValueError
        If the lattice has no directions, ``vectors`` does not end in an axis of 2, or
        ``facing`` is not a direction.
    """
    directions, metric = space._directions
    facing = _check_direction(facing, len(directions), "facing")
    v = int_array(vectors, "vectors")
    if v.ndim == 0 or v.shape[-1] != 2:
        raise ValueError(f"vectors must end in an axis of 2, got shape {v.shape}")
    return v @ (metric @ directions[facing]) >= 0


def _turns(a: int, b: int, n: int) -> int:
    """Return the steps between directions ``a`` and ``b`` of ``n``, the shorter way round."""
    step = (a - b) % n
    return min(step, n - step)


def _check_direction(value: SupportsIndex, n: int, name: str) -> int:
    """Return ``value`` as an int after checking that it is one of ``n`` directions."""
    direction = check_int(value, name)
    if not 0 <= direction < n:
        raise ValueError(f"{name} must be a direction, 0 to {n - 1}, got {direction}")
    return direction
