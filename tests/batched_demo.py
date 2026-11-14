"""The checks every game's array rules run: `check_array_game` holds a
`gridzero.batched.ArrayGame` to its game, ply by ply, given the game's dice."""

from collections import Counter
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from gridzero.batched import ArrayGame

type Chooser = Callable[[Any, Any, list[int], np.random.Generator], int]
"""Picks a local action: ``choose(game, position, legal, rng)``, ``legal`` the legal ones."""


class Compiled[G: ArrayGame[Any, Any, Any]]:
    """An array game's functions, compiled once for all the games a test plays."""

    def __init__(self, arrays: G) -> None:
        self.arrays = arrays
        self.legal = jax.jit(arrays.legal)
        self.apply = jax.jit(arrays.apply)
        self.step = jax.jit(arrays.step)
        self.observe = jax.jit(arrays.observe)
        self.player = jax.jit(arrays.player)


def differences(found, wanted) -> list[str]:
    """Return the names of the fields of two states (named tuples of arrays) that differ.

    A field differs if its values, its shape or its dtype do.
    """
    out = []
    for name, a, b in zip(found._fields, found, wanted, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            out.append(name)
    return out


def random_choice(game, position, legal: list[int], rng: np.random.Generator) -> int:
    """Pick a legal local action at random."""
    return int(rng.choice(legal))


def check_array_game(
    game,
    compiled: Compiled[Any],
    position,
    choose: Chooser = random_choice,
    *,
    rng: np.random.Generator,
    plies: int | None = None,
    observe: bool = True,
    watch: Callable[[Any], None] | None = None,
) -> Counter[str]:
    """Play a game on from ``position`` in both engines, and compare them at every ply.

    The game's own `step` decides what happens. At each position, converted to arrays: the
    state the array rules reached equals it, field for field (values, shapes and dtypes); the
    player to move is the game's; the legal local actions are the game's legal actions, through
    `ArrayGame.local` and back through `ArrayGame.action`; and each player's observation is the
    game's, bit for bit. Then ``choose`` picks a local action, the game steps, and the array
    rules apply the same action with the game's dice (`ArrayGame.dice`).

    Parameters
    ----------
    game : Game
        The game, which must also have ``observe(position, player)``.
    compiled : Compiled
        Its array rules.
    position : Position
        Where to start: a position that hides nothing, or one a search would play on.
    choose : callable, default `random_choice`
        Picks each local action.
    rng : numpy.random.Generator
        What ``choose`` draws from.
    plies : int, optional
        How many plies to play. By default, to the game's end.
    observe : bool, default True
        Whether to compare observations.
    watch : callable, optional
        Called with each position the game steps to, for a test to count what happened.

    Returns
    -------
    Counter of str
        ``"plies"`` played, and every difference found by name, such as ``"legal"`` or a
        field of the state. A test asserts that only ``"plies"`` is there.
    """
    arrays = compiled.arrays
    found: Counter[str] = Counter()
    state = arrays.to_arrays(position)
    back = arrays.to_arrays(arrays.to_position(state))
    found.update(f"converted back: {name}" for name in differences(back, state))
    while not position.over and (plies is None or found["plies"] < plies):
        legal = game.legal(position)
        actions = [(int(verb), int(cell)) for verb, cell in np.argwhere(legal)]
        wanted = sorted(arrays.local(position, action) for action in actions)
        mask = np.asarray(compiled.legal(state))
        if mask.shape != (arrays.n_actions,) or mask.dtype != np.bool_:
            found["legal's shape or dtype"] += 1
        if np.flatnonzero(mask).tolist() != wanted:
            found["legal"] += 1
        if {tuple(arrays.action(position, local)) for local in wanted} != set(actions):
            found["action"] += 1
        if int(compiled.player(state)) != game.player(position):
            found["player"] += 1
        if observe:
            for player in range(game.n_players):
                planes = np.asarray(compiled.observe(state, player))
                if planes.tobytes() != game.observe(position, player).tobytes():
                    found["observe"] += 1
        local = choose(game, position, wanted, rng)
        after = game.step(position, arrays.action(position, local))
        state = compiled.apply(state, local, arrays.dice(position, after))
        found.update(differences(state, arrays.to_arrays(after)))
        if watch is not None:
            watch(after)
        position = after
        found["plies"] += 1
    if position.over and np.asarray(compiled.legal(state)).any():
        found["legal once over"] += 1
    return found
