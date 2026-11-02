"""Choosing between the reference operations and their fast paths.

Every operation on operators, intents and action spaces has a reference implementation in
NumPy and SciPy. Some also have a fast path in Numba. Fast paths do the same arithmetic in the
same order, so they give bit-identical results (NaN payloads aside), and the tests check that
they do. Choosing an implementation therefore changes how fast a model runs, never what it
computes.

The choice is set for a block of code with `implementation`, for example to run a whole
simulation on the reference path:

>>> with implementation("reference"):
...     sim.run(100)
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

type Impl = Literal["auto", "reference", "numba"]
"""An implementation: ``"auto"`` (the fast paths), ``"reference"`` or ``"numba"``."""

_IMPLS = ("auto", "reference", "numba")

_current: ContextVar[Impl] = ContextVar("implementation", default="auto")


@contextmanager
def implementation(impl: Impl) -> Generator[None]:
    """Use one implementation of the operations inside a ``with`` block.

    The setting is per thread (a context variable), and is restored when the block ends, also
    on an error.

    Parameters
    ----------
    impl : {"auto", "reference", "numba"}
        ``"auto"``, the default outside any block, takes the fast paths: currently the Numba
        kernels wherever there is one, which were faster than the reference or about even in
        every case measured. It may later choose by operation or size. ``"reference"`` uses
        NumPy and SciPy only. ``"numba"`` uses Numba wherever there is a kernel, and the
        reference elsewhere.

    Raises
    ------
    TypeError
        If ``impl`` is not a string.
    ValueError
        If ``impl`` is not one of the names above.
    """
    if not isinstance(impl, str):
        raise TypeError(f"impl must be a string, got {type(impl).__name__}")
    if impl not in _IMPLS:
        raise ValueError(f"impl must be one of {_IMPLS}, got {impl!r}")
    token = _current.set(impl)
    try:
        yield
    finally:
        _current.reset(token)


def fast() -> bool:
    """Return whether operations should take their fast paths, under the current setting."""
    return _current.get() != "reference"
