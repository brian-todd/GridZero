"""Making boards fair and playable: symmetric values, top shares, joined cells, zones per seat.

A board game's board is a `core.State` on a space (often a 2-D `core.LatticeSpace`). These
helpers do what making one fair to every seat and playable needs, whatever the game:

- `symmetric` makes per-cell values, such as noise for obstacles, the same under maps of the
  cells that swap the seats, such as `LatticeSpace.opposite`, exactly;
- `highest` picks the cells holding the largest values, cut at a value, so that cells with
  equal values (a cell and its mirror image) always fall on the same side;
- `joined` and `pockets` say whether cells are joined through open cells, and which open
  cells are cut off from them, to be closed;
- `zones` gives each seat the cells within some steps of its seed cell, such as homes.

Which values, shares, seeds and depths, and how often to draw a board again, are the game's
choices. They take any space that `core.regions` and `core.steps` take.
"""

from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import Space, regions, steps
from gridzero.core._arrays import (
    cell_mask,
    check_cell,
    check_cells,
    check_int,
    check_real,
    int_array,
)


def symmetric(values: ArrayLike, *maps: ArrayLike) -> NDArray[np.float64]:
    """Return values made the same under maps of the cells: each cell's sum over its orbit.

    Cell ``c`` gets ``values[c] + values[m[c]]`` summed over every map ``m``, the values added
    in ascending order, so that every cell of an orbit gets exactly the same number (floats
    don't add associatively). With one map that is its own inverse, such as
    `LatticeSpace.opposite`, it is ``values + values[m]``.

    Parameters
    ----------
    values : array_like of float, shape (n_cells,)
        Finite values per cell.
    *maps : array_like of int, shape (n_cells,)
        Permutations of the cells, ``m[c]`` being the cell ``c`` goes to: with the identity,
        a group (every product of two of them among them), such as the symmetries of a board
        that swap its seats.

    Returns
    -------
    ndarray of float64, shape (n_cells,)
        ``out[m] == out`` for every map, exactly.

    Raises
    ------
    TypeError
        If a map is not of integers.
    ValueError
        If no map is given, ``values`` is not 1-D and finite, a map is not a permutation of
        its cells, or the maps with the identity are not a group.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    if not maps:
        raise ValueError("symmetric needs at least one map")
    perms = [_permutation(m, v.size) for m in maps]
    orbit = np.stack([v] + [v[m] for m in perms])
    orbit.sort(axis=0)
    out = orbit[0].copy()
    for row in orbit[1:]:  # one row at a time: a fixed order of additions
        out += row
    if any(not np.array_equal(out[m], out) for m in perms):
        raise ValueError("the maps with the identity must be a group: values differ in an orbit")
    return out


def highest(values: ArrayLike, share: float, where: ArrayLike | None = None) -> NDArray[np.bool_]:
    """Return the cells holding the largest values: a share of the cells, cut at a value.

    The ``k = round(share * n)`` largest of the ``n`` values considered set the cut, and every
    cell considered with a value at least that large is picked: ``k`` cells, or more where
    values tie at the cut. Cells with equal values are picked together, so symmetric values
    (`symmetric`) give a symmetric pick. Layers, such as walls on the highest cells and rubble
    on the next, come from cumulative shares: ``highest(v, 0.13) & ~highest(v, 0.08)``.

    Parameters
    ----------
    values : array_like of float, shape (n_cells,)
        Values per cell, not NaN.
    share : float
        The share of the cells considered to pick, in [0, 1].
    where : array_like of bool, shape (n_cells,), optional
        The cells considered; all of them by default.

    Returns
    -------
    ndarray of bool, shape (n_cells,)
        The cells picked, only among those considered.

    Raises
    ------
    TypeError
        If ``share`` is not a real number, or ``where`` is not boolean.
    ValueError
        If ``share`` is outside [0, 1], ``values`` is not 1-D or holds NaN, or ``where`` has
        another shape.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {v.shape}")
    if np.isnan(v).any():
        raise ValueError("values must not be NaN")
    share = check_real(share, "share", 0.0, 1.0)
    considered = np.ones(v.size, dtype=bool) if where is None else cell_mask(where, v.size, "where")
    ranked = np.sort(v[considered])[::-1]
    k = round(share * ranked.size)
    if k == 0:
        return np.zeros(v.size, dtype=bool)  # no cut: +inf would still pick cells holding it
    return considered & (v >= ranked[k - 1])


def joined(space: Space, open_: ArrayLike, cells: ArrayLike) -> bool:
    """Return whether cells are all open and joined through open cells that touch.

    Parameters
    ----------
    space : Space
        The cells.
    open_ : array_like of bool, shape (n_cells,)
        The cells that can be crossed.
    cells : array_like of int, shape (k,)
        Cell ids, at least one, such as each seat's home.

    Returns
    -------
    bool
        True if they lie in one region of ``open_`` (`core.regions`, facet adjacency).

    Raises
    ------
    TypeError, ValueError, IndexError
        If ``open_`` is not a mask of the cells, or ``cells`` are not cell ids or are none.
    """
    ids = check_cells(cells, space.n_cells)
    if ids.size == 0:
        raise ValueError("joined needs at least one cell")
    pieces = regions(space, open_)
    return bool(pieces[ids[0]] >= 0 and (pieces[ids] == pieces[ids[0]]).all())


def pockets(space: Space, open_: ArrayLike, keep: SupportsIndex) -> NDArray[np.bool_]:
    """Return the open cells cut off from a cell: those to close so every open cell is reached.

    Parameters
    ----------
    space : Space
        The cells.
    open_ : array_like of bool, shape (n_cells,)
        The cells that can be crossed.
    keep : int
        An open cell, such as a seat's home, from which every open cell left must be reached.

    Returns
    -------
    ndarray of bool, shape (n_cells,)
        The cells of ``open_`` outside ``keep``'s region (`core.regions`, facet adjacency).

    Raises
    ------
    TypeError, IndexError
        If ``open_`` is not a mask of the cells, or ``keep`` is not a cell id.
    ValueError
        If ``keep`` is not open.
    """
    cell = check_cell(keep, space.n_cells)
    mask = cell_mask(open_, space.n_cells, "open_")
    if not mask[cell]:
        raise ValueError(f"cell {cell} to keep is not open")
    pieces = regions(space, mask)
    return mask & (pieces != pieces[cell])


def zones(space: Space, seeds: ArrayLike, within: SupportsIndex) -> NDArray[np.int8]:
    """Return each seat's zone: the cells within some steps of its seed cell, such as a home.

    Parameters
    ----------
    space : Space
        The cells.
    seeds : array_like of int, shape (n_seats,)
        Seat ``s``'s seed cell, at least one seat and at most 127.
    within : int
        Steps between adjacent cells (facet adjacency, `core.steps`) from a seed, at least 0:
        0 gives each seat its seed alone.

    Returns
    -------
    ndarray of int8, shape (n_cells,)
        The seat whose zone holds each cell, and -1 for cells in none.

    Raises
    ------
    TypeError, IndexError
        If ``seeds`` are not cell ids, or ``within`` is not an integer.
    ValueError
        If ``within`` is negative, there are no seeds or more than 127, or two zones would
        share a cell.
    """
    within = check_int(within, "within", minimum=0)
    ids = check_cells(seeds, space.n_cells)
    if not 1 <= ids.size <= 127:
        raise ValueError(f"zones needs 1 to 127 seats, got {ids.size}")
    zone = np.full(space.n_cells, -1, dtype=np.int8)
    for seat, seed in enumerate(ids):
        near = steps(space, int(seed), limit=within) <= within
        if np.any(zone[near] >= 0):
            raise ValueError(f"zones {within} steps around their seeds overlap")
        zone[near] = seat
    return zone


def _permutation(values: ArrayLike, n: int) -> NDArray[np.int64]:
    """Return ``values`` as int64 after checking that it is a permutation of ``n`` cells."""
    perm = int_array(values, "map")
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f"a map must be a permutation of the {n} cells")
    return perm
