"""Internal helpers for immutable array storage and checks of shared argument kinds.

Errors follow one convention: a value of the wrong type (a float, bool or string where an
integer belongs) raises TypeError, and a value of the right type but out of range raises
ValueError, or IndexError for a cell id.
"""

import math
import numbers
import operator

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

_INT64 = np.iinfo(np.int64)


def frozen_copy(array: ArrayLike, dtype: DTypeLike) -> NDArray:
    """Return a read-only, owned, C-contiguous copy of ``array`` converted to ``dtype``.

    Parameters
    ----------
    array : array_like, any shape
        Values to copy. The caller keeps its own array, which later writes cannot reach.
    dtype : data-type
        Dtype of the copy.

    Returns
    -------
    ndarray of ``dtype``, the shape of ``array``
        A new array that owns its memory, with its write flag off.
    """
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def read_only[A: np.ndarray](array: A) -> A:
    """Mark a freshly computed ``array`` read-only and return it.

    Parameters
    ----------
    array : ndarray, any shape and dtype
        An array no one else holds, marked in place.

    Returns
    -------
    ndarray
        ``array`` itself, with its write flag off.
    """
    array.setflags(write=False)
    return array


def as_rows(values: ArrayLike, dtype: DTypeLike | None = None, keep: int = 1) -> NDArray:
    """Return ``values`` C-contiguous, with every axis but the last ``keep`` merged into one.

    Values of shape ``(*lead, *kept)`` become ``(prod(lead), *kept)``, the layout the Numba
    kernels take. The size is computed rather than left to ``reshape(-1, ...)``, which cannot
    infer it when an axis has length 0.

    Parameters
    ----------
    values : array_like, shape (*lead, *kept)
        Values with at least ``keep`` axes.
    dtype : data-type, optional
        Dtype to convert to. None keeps the dtype of ``values``.
    keep : int, default 1
        Number of trailing axes kept as they are.

    Returns
    -------
    ndarray, shape (prod(lead), *kept)
        A view of ``values`` when it is already C-contiguous of ``dtype``, else a copy.
    """
    array = np.ascontiguousarray(values, dtype=dtype)
    split = array.ndim - keep
    return array.reshape(math.prod(array.shape[:split]), *array.shape[split:])


def check_int(value: object, name: str, minimum: int | None = None) -> int:
    """Return ``value`` as an int after checking that it is an integer of at least ``minimum``.

    Booleans are rejected even though Python treats them as integers: ``every=True`` is a
    mistake, not a request for 1.

    Parameters
    ----------
    value : object
        Value to check: a Python or NumPy integer, or anything with ``__index__``.
    name : str
        Name of the argument, for error messages.
    minimum : int, optional
        Smallest value allowed. None allows any.

    Returns
    -------
    int
        ``value`` as a Python int.

    Raises
    ------
    TypeError
        If ``value`` is a bool or not an integer (floats such as ``2.0`` included).
    ValueError
        If ``value`` is less than ``minimum``.
    """
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:  # value may be anything, so ty rightly doubts it has __index__
        index = operator.index(value)  # ty: ignore[invalid-argument-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if minimum is not None and index < minimum:
        bound = {0: "non-negative", 1: "positive"}.get(minimum, f"at least {minimum}")
        raise ValueError(f"{name} must be {bound}, got {index}")
    return index


def check_real(
    value: object, name: str, low: float | None = None, high: float | None = None
) -> float:
    """Return ``value`` as a float after checking that it is a finite real number in range.

    Parameters
    ----------
    value : object
        Value to check: a real number, such as a Python or NumPy integer or float.
    name : str
        Name of the argument, for error messages.
    low, high : float, optional
        Bounds allowed, both included. None leaves that side open.

    Returns
    -------
    float
        ``value`` as a Python float.

    Raises
    ------
    TypeError
        If ``value`` is a bool or not a real number (strings and arrays included).
    ValueError
        If ``value`` is NaN, infinite, below ``low`` or above ``high``.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    above, below = low is None or number >= low, high is None or number <= high
    if math.isfinite(number) and above and below:
        return number
    if low is not None and high is not None:
        bounds = f" and in [{low}, {high}]"
    elif low is not None:
        bounds = f" and at least {low}"
    else:
        bounds = "" if high is None else f" and at most {high}"
    raise ValueError(f"{name} must be finite{bounds}, got {value!r}")


def component_shape(shape: object, context: str = "") -> tuple[int, ...]:
    """Return a component shape, an int or a tuple of ints, as a tuple of non-negative ints.

    Parameters
    ----------
    shape : object
        Shape to check: an integer, or an iterable of integers.
    context : str, default ""
        Start of every error message, such as ``"field 'pop': "``.

    Returns
    -------
    tuple of int
        The shape's entries, each a non-negative Python int.

    Raises
    ------
    TypeError
        If ``shape`` is neither an int nor iterable, or an entry is not an integer.
    ValueError
        If an entry is negative.
    """
    try:  # shape may be anything, so ty rightly doubts it is iterable
        entries = (shape,) if isinstance(shape, int | np.integer) else tuple(shape)  # ty: ignore[invalid-argument-type]
    except TypeError:
        raise TypeError(f"{context}shape must be an int or a tuple, got {shape!r}") from None
    return tuple(check_int(n, f"{context}each shape entry", minimum=0) for n in entries)


def finite_floats(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``values`` as float64 after checking that they are finite numbers.

    Parameters
    ----------
    values : array_like of int or float, any shape
        Values to check.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    ndarray of float64, the shape of ``values``
        ``values`` itself when it is already a float64 array, else a converted copy.

    Raises
    ------
    TypeError
        If ``values`` does not hold numbers: booleans and strings are rejected.
    ValueError
        If a value is NaN or infinite.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numbers, got dtype {array.dtype}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} holds NaN or infinite values")
    return array


def cell_mask(mask: ArrayLike, n_cells: int, name: str) -> NDArray[np.bool_]:
    """Return ``mask`` as an array after checking that it holds one bool per cell.

    Parameters
    ----------
    mask : array_like of bool, shape (n_cells,)
        Mask to check.
    n_cells : int
        Number of cells.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    ndarray of bool, shape (n_cells,)
        ``mask`` as an array, not copied when it already is one.

    Raises
    ------
    TypeError
        If ``mask`` is not boolean.
    ValueError
        If ``mask`` does not have shape ``(n_cells,)``.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must be boolean, got dtype {array.dtype}")
    if array.shape != (n_cells,):
        raise ValueError(f"{name} must have shape ({n_cells},), got {array.shape}")
    return array


def int_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Return ``values`` as an int64 array, refusing conversions that would change them.

    Floats are not truncated and booleans are not read as 0 and 1. An empty sequence is
    accepted whatever its dtype, because ``np.asarray([])`` is float64.

    Parameters
    ----------
    values : array_like of int, any shape
        Integers to convert.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    ndarray of int64, the shape of ``values``
        May share memory with ``values``.

    Raises
    ------
    TypeError
        If ``values`` is not an array of integers.
    ValueError
        If a uint64 value is too large for int64.
    """
    array = np.asarray(values)
    if array.size == 0:
        return array.astype(np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers, got dtype {array.dtype}")
    if array.dtype == np.uint64 and array.max() > _INT64.max:
        raise ValueError(f"{name} holds {int(array.max())}, which does not fit in int64")
    return array.astype(np.int64, copy=False)


def check_cell(cell: object, n_cells: int) -> int:
    """Return ``cell`` as an int after checking that it is a valid cell id.

    Parameters
    ----------
    cell : object
        Cell id to check: a Python or NumPy integer.
    n_cells : int
        Number of cells.

    Returns
    -------
    int
        ``cell`` as a Python int in ``[0, n_cells)``.

    Raises
    ------
    TypeError
        If ``cell`` is not an integer.
    IndexError
        If ``cell`` is outside ``[0, n_cells)``. Negative ids are rejected rather than counted
        from the end, because -1 marks a missing neighbor in padded tables.
    """
    index = check_int(cell, "a cell id")
    if not 0 <= index < n_cells:
        raise IndexError(f"cell id {index} is out of range [0, {n_cells})")
    return index


def check_cells(cells: ArrayLike, n_cells: int) -> NDArray[np.int64]:
    """Return ``cells`` as a 1-D int64 array after checking that every entry is a cell id.

    Parameters
    ----------
    cells : array_like of int, shape (n,)
        Cell ids to check.
    n_cells : int
        Number of cells.

    Returns
    -------
    ndarray of int64, shape (n,)
        May share memory with ``cells``.

    Raises
    ------
    TypeError
        If ``cells`` does not hold integers.
    ValueError
        If ``cells`` is not 1-D.
    IndexError
        If an id is outside ``[0, n_cells)``. Negative ids are not counted from the end.
    """
    cells = int_array(cells, "cells")
    if cells.ndim != 1:
        raise ValueError(f"cells must be a 1-D array of cell ids, got shape {cells.shape}")
    return check_ids(cells, n_cells, "cells")


def check_ids(values: ArrayLike, n_cells: int, name: str) -> NDArray[np.int64]:
    """Return cell ids of any shape as an int64 array, after checking every entry.

    Parameters
    ----------
    values : array_like of int, any shape
        Cell ids to check.
    n_cells : int
        Number of cells.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    ndarray of int64, the shape of ``values``
        May share memory with ``values``.

    Raises
    ------
    TypeError
        If ``values`` does not hold integers.
    IndexError
        If an id is outside ``[0, n_cells)``. Negative ids are not counted from the end.
    """
    ids = int_array(values, name)
    if ((ids < 0) | (ids >= n_cells)).any():
        raise IndexError(f"{name} must hold cell ids in [0, {n_cells})")
    return ids


def cell_selection(cells: ArrayLike, n_cells: int, name: str) -> NDArray[np.int64]:
    """Return cells given as an id, ids, or a mask, as cell ids. Masks give ids in order.

    Parameters
    ----------
    cells : int, array_like of int, shape (k,), or array_like of bool, shape (n_cells,)
        A cell id, cell ids, or a mask of cells.
    n_cells : int
        Number of cells.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    ndarray of int64, shape (k,)
        C-contiguous cell ids.

    Raises
    ------
    TypeError
        If ``cells`` holds neither integers nor booleans.
    ValueError
        If ids are not 1-D, or a mask has the wrong shape.
    IndexError
        If an id is outside ``[0, n_cells)``.
    """
    array = np.asarray(cells)
    if array.dtype == np.bool_:
        return np.flatnonzero(cell_mask(array, n_cells, name))
    if array.ndim == 0:
        return np.array([check_cell(cells, n_cells)], dtype=np.int64)
    ids = int_array(array, name)
    if ids.ndim != 1:
        raise ValueError(
            f"{name} must be a cell id, a 1-D array of them, or a mask, got shape {ids.shape}"
        )
    return np.ascontiguousarray(check_ids(ids, n_cells, name))
