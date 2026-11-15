"""The checks every game's array rules run: `check_array_game` holds a
`gridzero.batched.ArrayGame` to its game, ply by ply, given the game's dice. And `Wager`, a
fake game as arrays whose values are known, for what runs on any `gridzero.batched.ArrayRules`.
`check_array_model` holds a simulation's phases in arrays to its model, step by step."""

from collections import Counter
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gridzero import batched, core
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
        self.outcomes = jax.jit(arrays.outcomes)
        self.framed = jax.jit(arrays.framed)
        self.over = jax.jit(arrays.over)
        self.winner = jax.jit(arrays.winner)


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
    outcome: Callable[[Any, Any], tuple[int, float]] | None = None,
) -> Counter[str]:
    """Play a game on from ``position`` in both engines, and compare them at every ply.

    The game's own `step` decides what happens. At each position, converted to arrays: the
    state the array rules reached equals it, field for field (values, shapes and dtypes); the
    player to move is the game's; the legal local actions are the game's legal actions, through
    `ArrayGame.local` and back through `ArrayGame.action`; and each player's observation is the
    game's, bit for bit; `ArrayGame.framed` names each legal action as the mover's frame does
    (``game.frame``); and `ArrayGame.over` and `ArrayGame.winner` are the position's. Then
    ``choose`` picks a local action, the game steps, and the array rules apply the same action
    with the game's dice (`ArrayGame.dice`). The state they reach is one of the step's
    `ArrayGame.outcomes` with those dice, whose chances are probabilities.

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
    outcome : callable, optional
        ``outcome(before, after)``: which of the step's outcomes the game's dice made, as
        `ArrayGame.outcomes` numbers them, and the probability the game's rules give it, read
        from the two positions. That outcome must be the state reached, with that chance.

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
        if bool(compiled.over(state)) or int(compiled.winner(state)) != -1:
            found["over or winner"] += 1
        frame = game.frame(game.player(position))
        names = np.asarray(compiled.framed(state))
        n_cells = game.space.n_cells
        if names.shape != (arrays.n_actions,) or names.dtype != np.int32:
            found["framed's shape or dtype"] += 1
        for action in actions:
            name = int(names[arrays.local(position, action)])
            if divmod(name, n_cells) != tuple(frame.action_into(action)):
                found["framed"] += 1
        if observe:
            for player in range(game.n_players):
                planes = np.asarray(compiled.observe(state, player))
                if planes.tobytes() != game.observe(position, player).tobytes():
                    found["observe"] += 1
        local = choose(game, position, wanted, rng)
        after = game.step(position, arrays.action(position, local))
        dice = arrays.dice(position, after)
        ways, chances = compiled.outcomes(state, local, dice)
        state = compiled.apply(state, local, dice)
        found.update(differences(state, arrays.to_arrays(after)))
        chances = np.asarray(chances)
        if chances.shape != (arrays.n_outcomes,) or chances.dtype != np.float32:
            found["chances' shape or dtype"] += 1
        if (chances < 0).any() or abs(float(chances.sum()) - 1.0) > 1e-6:
            found["chances are not probabilities"] += 1
        reached = [
            o
            for o in np.flatnonzero(chances > 0)
            if not differences(jax.tree.map(lambda x, o=o: x[o], ways), state)
        ]
        if not reached:
            found["outcomes"] += 1
        if outcome is not None:
            which, chance = outcome(position, after)
            if which not in reached or abs(float(chances[which]) - chance) > 1e-6:
                found["the outcome reached"] += 1
        if watch is not None:
            watch(after)
        position = after
        found["plies"] += 1
    if position.over and np.asarray(compiled.legal(state)).any():
        found["legal once over"] += 1
    winner = -1 if position.winner is None else position.winner
    if bool(compiled.over(state)) != position.over or int(compiled.winner(state)) != winner:
        found["over or winner"] += 1
    return found


def check_array_model(model, phases, space, seed: int, steps: int, *, rtol: float) -> None:
    """Run a model's simulation and its phases in arrays from one start; compare every step.

    The reference is ``core.Simulation(model.start(space, seed), model.phases(), seed=seed)``.
    The arrays start from the same state (`gridzero.batched.fields`) and step with
    `gridzero.batched.Simulation`, compiled. After each step every field is the reference's:
    booleans and integers exactly, floats within ``rtol`` of the field's largest value, and no
    amount (an extensive floating field) is negative. For models whose rules draw nothing: the
    two engines have their own dice.

    Parameters
    ----------
    model : core.Model
        The reference.
    phases : sequence of batched.Phase
        Its phases as rules in arrays.
    space : Space
        The cells.
    seed : int
        The run's seed.
    steps : int
        Steps to compare.
    rtol : float
        The tolerance of floats, relative to each field's largest value: 0 for exact.
    """
    reference = core.Simulation(model.start(space, seed), model.phases(), seed=seed)
    arrays = batched.Simulation(space, phases)
    step = jax.jit(arrays.step)
    key = batched.root(seed)
    fields = batched.fields(reference.state)
    for _ in range(steps):
        wanted = reference.step()
        fields = step(fields, key)
        assert int(fields.step) == wanted.step
        for field in wanted:
            got, want = np.asarray(fields[field]), wanted[field]
            where = f"{field.name} at step {wanted.step}"
            assert got.shape == want.shape, where
            if want.dtype.kind != "f":
                np.testing.assert_array_equal(got, want, err_msg=where)
                continue
            scale = float(np.abs(want).max()) if want.size else 0.0
            np.testing.assert_allclose(got, want, rtol=0, atol=rtol * scale, err_msg=where)
            if field.extensive:
                assert got.min(initial=0.0) >= 0, where
    back = batched.state(fields, space)
    assert type(back) is type(reference.state)
    assert back.step == steps


class WagerState(NamedTuple):
    """A position of `Wager`."""

    ply: jax.Array
    """Decisions taken so far: int32, shape ()."""
    over: jax.Array
    """Whether the game has ended: bool, shape ()."""
    winner: jax.Array
    """The player that won, or -1: int32, shape ()."""


class Wager:
    """A fake game of two players as arrays (`gridzero.batched.ArrayRules`), with dice.

    Player 0 decides first: settle for a draw (action 0), gamble (1), or wait (2). A gamble
    ends the game: its player wins on a d20 of ``faces`` or less, and loses otherwise. After
    a wait, player 1 can only wait for ``waits`` plies, then decides between settling and
    gambling. So every value is known: a draw is worth 0.5, and a gamble ``faces / 20`` to
    its player.

    There is one cell and a verb per action, so a network's scores are a score per action,
    and an observation is one number: the ply, as a share of the last.
    """

    SETTLE, GAMBLE, WAIT = 0, 1, 2
    n_actions = 3
    n_outcomes = 2

    def __init__(self, faces: int = 14, waits: int = 2) -> None:
        self.faces = faces
        self.waits = waits

    def start(self, ply: int = 0) -> WagerState:
        """A position at a ply: 0 for player 0's decision, ``waits + 1`` for player 1's."""
        return WagerState(jnp.int32(ply), jnp.bool_(False), jnp.int32(-1))

    def player(self, state: WagerState) -> jax.Array:
        return jnp.where(state.ply == 0, 0, 1).astype(jnp.int32)

    def over(self, state: WagerState) -> jax.Array:
        return state.over

    def winner(self, state: WagerState) -> jax.Array:
        return state.winner

    def legal(self, state: WagerState) -> jax.Array:
        waiting = (state.ply > 0) & (state.ply <= self.waits)
        last = state.ply > self.waits
        mask = jnp.stack([~waiting, ~waiting, ~last])
        return mask & ~state.over

    def roll(self, key) -> jax.Array:
        return batched.d(key, 20)

    def draws(self, state: WagerState) -> jax.Array:
        return self.roll(batched.stream(batched.root(0), 1, state.ply))

    def apply(self, state: WagerState, action, dice) -> WagerState:
        action = jnp.asarray(action)
        mover = self.player(state)
        won = jnp.where(jnp.asarray(dice) <= self.faces, mover, 1 - mover)
        ends = action != self.WAIT
        winner = jnp.where(action == self.GAMBLE, won, -1).astype(jnp.int32)
        return WagerState(state.ply + 1, ends, winner)

    def step(self, state: WagerState, action) -> WagerState:
        return self.apply(state, action, self.draws(state))

    def outcomes(self, state: WagerState, action, dice) -> tuple[WagerState, jax.Array]:
        """Outcome 1 is the gamble won, outcome 0 anything else."""
        rolls = jnp.array([20, 1])
        ways = jax.vmap(lambda roll: self.apply(state, action, roll))(rolls)
        wins = jnp.float32(self.faces) / 20
        gamble = jnp.asarray(action) == self.GAMBLE
        return ways, jnp.where(gamble, jnp.stack([1 - wins, wins]), jnp.array([1.0, 0.0]))

    def observe(self, state: WagerState, player) -> jax.Array:
        return jnp.full((1, 1), state.ply / (self.waits + 1), jnp.float32)

    def framed(self, state: WagerState) -> jax.Array:
        return jnp.arange(3, dtype=jnp.int32)
