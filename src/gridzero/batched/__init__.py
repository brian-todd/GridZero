"""Rules as arrays, for many games or runs at once on the GPU: `core`'s operations in JAX.

`core` runs one state at a time in NumPy and Numba, and defines every result. This package runs
the same operations as JAX arrays of fixed shape, so that a model's rules, written once more
with ``jax.numpy``, are compiled (``jax.jit``) and run for thousands of states together
(``jax.vmap``): self-play, searches, sweeps over seeds and parameters. **Every function here
takes one state.** A batch is that function under ``jax.vmap``.

It is a second engine, not a third value of `core.implementation`. What it promises:

- Integers and booleans equal `core`'s, bit for bit, while they fit in 32 bits (JAX computes
  in int32 and float32 unless told otherwise).
- Floats equal `core`'s to a tolerance: they are float32, and a GPU adds in another order.
- Random draws come from JAX's generators (`dice`), not NumPy's. A model in arrays is the same
  model with its own dice: equal to its reference given equal dice, not equal per seed.

The package imports JAX, so nothing else in `gridzero` imports it, as with Torch in `agents`.

Operators
---------
Operator, operator
    A `core.NeighborOperator`'s padded tables as JAX arrays, with its transpose's.
pull, push, gather
    Aggregate neighbor values into cells, send values out to neighbors, and collect raw
    neighbor values: `core.pull`, `core.push` and `core.gather`.

Queries on any space
--------------------
steps, reach
    Moves from the nearest source to each cell, around blocked cells, and the cells a budget
    reaches (all that can be reached, without one): `core.steps` and `core.reach` for moves
    that cost 1.
FAR
    The number of steps to a cell that isn't reached.

Queries around a cell of a lattice
----------------------------------
Patch, patch
    The cells around each cell of a 2-D lattice of squares or hexes, by offset: where each
    lies, which are adjacent, what blocks the line to each, and directions.
slot
    The slot of a patch at a lattice offset.
clear, sight
    Which of the cells around a cell it sees, and the cells some viewers see: `core.sight`.
steps_around, path_around
    The cheapest way to each cell around a cell in a few moves, and the path to one:
    `core.steps` and `core.path`.

Dice
----
root, stream, d, permutation
    A generator's key from a seed, the key of a stream of it (a step, a phase), rolls of a
    die, and a random order.

Games
-----
ArrayGame
    Protocol: what a game's rules as arrays provide, and how they map onto the game they
    copy.
"""

from gridzero.batched.dice import d, permutation, root, stream
from gridzero.batched.games import ArrayGame
from gridzero.batched.operators import Operator, gather, operator, pull, push
from gridzero.batched.queries import (
    FAR,
    Patch,
    clear,
    patch,
    path_around,
    reach,
    sight,
    slot,
    steps,
    steps_around,
)

__all__ = [
    "FAR",
    "ArrayGame",
    "Operator",
    "Patch",
    "clear",
    "d",
    "gather",
    "operator",
    "patch",
    "path_around",
    "permutation",
    "pull",
    "push",
    "reach",
    "root",
    "sight",
    "slot",
    "steps",
    "steps_around",
    "stream",
]
