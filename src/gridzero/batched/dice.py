"""Random draws for rules as arrays: a key per game or run, and a stream per step.

`core`'s loop draws every step and phase from its own NumPy stream of one root seed
(``SeedSequence(seed, spawn_key=(step, phase))``), so that a draw depends on nothing drawn
before it. These do the same with JAX's generators: `root` makes a run's key from its seed,
and `stream` the key of one of its streams. The numbers are JAX's own, not NumPy's.

Keys are kept as their data, uint32 arrays of shape (2,), so that a state holding one is plain
arrays: stacked, compared and saved like the rest.
"""

from typing import SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from gridzero.core import check_int


def root(seed: SupportsIndex | jax.Array) -> jax.Array:
    """Return the key of a game or run: the root of every draw in it.

    Parameters
    ----------
    seed : int, or Array of uint32, shape (2,)
        The seed, at least 0 and below ``2**63``. Or its low and its high 32 bits, which a
        state of arrays can hold: ``[seed & 0xFFFFFFFF, seed >> 32]``.

    Returns
    -------
    Array of uint32, shape (2,)
        The key's data. The same seed gives the same key, in either form.

    Raises
    ------
    TypeError
        If ``seed`` is neither an integer nor an array of integers.
    ValueError
        If ``seed`` is out of range, or an array of another shape.
    """
    if isinstance(seed, jax.Array | np.ndarray):
        if not jnp.issubdtype(seed.dtype, jnp.integer):
            raise TypeError(f"seed must hold integers, got dtype {seed.dtype}")
        if seed.shape != (2,):
            raise ValueError(f"a seed in two halves must have shape (2,), got {seed.shape}")
        low, high = jnp.asarray(seed, dtype=jnp.uint32)
    else:
        number = check_int(seed, "seed", minimum=0)
        if number >= 2**63:
            raise ValueError(f"seed must be below 2**63, got {number}")
        low, high = jnp.uint32(number & 0xFFFFFFFF), jnp.uint32(number >> 32)
    return jax.random.key_data(jax.random.fold_in(jax.random.key(low), high))


def stream(key: ArrayLike, *spawn: ArrayLike) -> jax.Array:
    """Return the key of one stream of a key: a step's, or a phase's of a step.

    Parameters
    ----------
    key : Array of uint32, shape (2,)
        A key's data, from `root` or from this function.
    *spawn : int or Array of int, shape ()
        What names the stream, such as the step and the phase: each at least 0 and below
        ``2**32``. Streams named differently draw independently.

    Returns
    -------
    Array of uint32, shape (2,)
        The stream's key data.
    """
    out = jax.random.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))
    for part in spawn:
        out = jax.random.fold_in(out, part)
    return jax.random.key_data(out)


def d(key: ArrayLike, sides: int, shape: tuple[int, ...] = ()) -> jax.Array:
    """Roll dice: integers from 1 to ``sides``, each as likely.

    Parameters
    ----------
    key : Array of uint32, shape (2,)
        A key's data. The same key gives the same rolls: take a `stream` per roll, or ask for
        all of a step's rolls at once.
    sides : int
        The die's sides, at least 1.
    shape : tuple of int, default ()
        The shape of the rolls.

    Returns
    -------
    Array of int32, shape ``shape``
        The rolls.

    Raises
    ------
    TypeError
        If ``sides`` is not an integer.
    ValueError
        If ``sides`` is below 1.
    """
    sides = check_int(sides, "sides", minimum=1)
    wrapped = jax.random.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))
    return jax.random.randint(wrapped, shape, 1, sides + 1, dtype=jnp.int32)


def permutation(key: ArrayLike, n: int) -> jax.Array:
    """Return the numbers from 0 to ``n - 1`` in a random order, each order as likely.

    Parameters
    ----------
    key : Array of uint32, shape (2,)
        A key's data. The same key gives the same order.
    n : int
        How many, at least 0.

    Returns
    -------
    Array of int32, shape (n,)
        Each number once.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n`` is negative.
    """
    n = check_int(n, "n", minimum=0)
    wrapped = jax.random.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))
    return jax.random.permutation(wrapped, n).astype(jnp.int32)
