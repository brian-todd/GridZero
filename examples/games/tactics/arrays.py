"""Tactics' rules once more, as JAX arrays: for thousands of games at once on the GPU.

`Tactics` stays the game: players, ladders, pages and tests read it. `TacticsArrays` makes the
same moves on a `State` of arrays of fixed shape (`gridzero.batched.ArrayGame`), so that
``jax.jit`` compiles them and ``jax.vmap`` runs a batch. It plays **any position**, under fog
too: legal actions are then those of the mover's belief, a move stops before a unit it didn't
know of, and each side remembers where it last saw the other's units. `TacticsArrays.view`
gives a side's belief (`Tactics.view`), a position that hides nothing, which is what a search
plays on; ``TacticsArrays(game, hidden=False)`` plays only such positions, at less cost. Given
the dice the game rolled, its positions are the game's, field for field.

How it differs in form:

- Units are slots by uid (the order of deployment), not fields per cell: a state is about a
  kilobyte, small enough to keep one per node of a search.
- Actions are local to the unit that acts: a target is one of the cells around it, so there
  are 133 actions under the default rules, whatever the board's size, instead of 18 per cell.
  Deployment shares the slots, as a kind and a tile of the home.
- Every fact about hexes is a table that `core` filled (`gridzero.batched.patch`): the rules
  here only gather. Stats, caps and bonuses are read from the same `Tactics` value, so a
  variant by parameters needs no new code.
- The dice are JAX's own (`TacticsArrays.roll`). `TacticsArrays.apply` takes a step's dice as
  an argument: the game's, to copy its step. `TacticsArrays.outcomes` gives a search every way
  a step's attacks may hit and miss, each with its exact chance.
"""

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from gridzero import batched
from gridzero.batched import FAR, Patch
from gridzero.core import Field, hops
from gridzero.game import Action

from . import objective
from .board import center_cell, homes
from .events import Attacked
from .game import Tactics
from .state import Hold, TacticsPosition, TacticsState, Verb


class State(NamedTuple):
    """One position of Tactics as arrays: the units by uid, the terrain and the counters.

    ``U`` is twice the team size. A slot without a unit (not yet placed, or dead) holds each
    field's default, as an empty tile does in `TacticsState`, so equal positions are equal
    arrays. The fields of units are those of `state.Unit`.
    """

    cell: jax.Array
    """Where each unit stands, -1 for none: int16, shape (U,)."""
    kind: jax.Array
    """Index into the game's `UnitStats`, -1 for none: int8, shape (U,)."""
    team: jax.Array
    """0 or 1, -1 for none: int8, shape (U,)."""
    hp: jax.Array
    """Hit points left: int16, shape (U,)."""
    facing: jax.Array
    """Direction faced, 0 to 5: int8, shape (U,)."""
    initiative: jax.Array
    """Place in the order of turns: int16, shape (U,)."""
    moved: jax.Array
    """Tiles moved this round: int8, shape (U,)."""
    swapped: jax.Array
    """Has swapped this round: bool, shape (U,)."""
    attacked: jax.Array
    """Has attacked, deferred or worked on a tile this round: bool, shape (U,)."""
    faced: jax.Array
    """Has turned since its last move, attack or swap: bool, shape (U,)."""
    done: jax.Array
    """Has had its turn this round: bool, shape (U,)."""
    engaged: jax.Array
    """Was attacked since its last turn began, or this round: bool, shape (U,)."""
    watch: jax.Array
    """The tile it deferred its attack onto, or -1: int16, shape (U,)."""
    spotted: jax.Array
    """Under fog, where the other side believes it to be, or -1: int16, shape (U,)."""
    spotted_hp: jax.Array
    """Its hit points when the other side last saw it: int16, shape (U,)."""
    spotted_facing: jax.Array
    """Its facing when the other side last saw it: int8, shape (U,)."""
    spotted_round: jax.Array
    """The round in which the other side last saw it: int16, shape (U,)."""
    wall: jax.Array
    """Tiles that block movement and sight: bool, shape (n_cells,)."""
    rubble: jax.Array
    """Tiles that block movement only: bool, shape (n_cells,)."""
    round: jax.Array
    """0 during deployment, then the round being played: int16, shape ()."""
    owner: jax.Array
    """The team that holds the point, or -1: int8, shape ()."""
    clock: jax.Array
    """Round ends in a row the owner has held it: int16, shape ()."""
    winner: jax.Array
    """The team that won, or -1 while the game runs: int8, shape ()."""
    over: jax.Array
    """Whether the game has ended: bool, shape ()."""
    belief: jax.Array
    """The team whose belief this is, or -1 for a game without fog: int8, shape ()."""
    ply: jax.Array
    """Decisions taken so far: int32, shape ()."""
    seed: jax.Array
    """The game's seed, the root of the state's own dice, as its low and its high 32 bits
    (`gridzero.batched.root`): uint32, shape (2,)."""


class Dice(NamedTuple):
    """The random draws of one step, whether it uses them or not."""

    rolls: jax.Array
    """The d20 of each attack the step makes, in the order they are rolled: int32, shape
    (`TacticsArrays.n_rolls`,). The strike of a unit that braces comes before its attacker's
    roll, and archers fire at a tile in order of their cells."""
    first: jax.Array
    """The team that takes the battle's first turn, 0 or 1: int32, shape (). Read by the step
    that starts the battle."""
    order: jax.Array
    """A number per unit, all different: int32, shape (U,). The step that starts the battle
    lines each team up by them, lowest first, and under ``shuffle_each_round`` each step that
    starts a round lines up every unit."""


class _Rolled(NamedTuple):
    """The rolls of a step under way: those given, how many are used, and what each needed."""

    rolls: jax.Array
    """The d20 of each attack, in order: int32, shape (n_rolls,)."""
    used: jax.Array
    """How many of them the step has used so far: int32, shape ()."""
    faces: jax.Array
    """How many of a d20's faces hit, for each roll used, 1 to 19: int32, shape (n_rolls,)."""


class _Walk(NamedTuple):
    """The walk of the unit that acted in a step: what the other side may glimpse of it."""

    unit: jax.Array
    """The slot of the unit: int32, shape ()."""
    tiles: jax.Array
    """The tiles of its route, from where it started, then the last one repeated: int32,
    shape (moves + 1,)."""
    length: jax.Array
    """How many moves it made, 0 for none: int32, shape ()."""


_UNITS: dict[str, tuple[type[np.generic], int]] = {
    "cell": (np.int16, -1),
    "kind": (np.int8, -1),
    "team": (np.int8, -1),
    "hp": (np.int16, 0),
    "facing": (np.int8, 0),
    "initiative": (np.int16, 0),
    "moved": (np.int8, 0),
    "swapped": (np.bool_, 0),
    "attacked": (np.bool_, 0),
    "faced": (np.bool_, 0),
    "done": (np.bool_, 0),
    "engaged": (np.bool_, 0),
    "watch": (np.int16, -1),
    "spotted": (np.int16, -1),
    "spotted_hp": (np.int16, 0),
    "spotted_facing": (np.int8, 0),
    "spotted_round": (np.int16, 0),
}
"""The fields of `State` per unit: the dtype each is kept in, and its value without a unit."""

_REST: dict[str, type[np.generic]] = {
    "wall": np.bool_,
    "rubble": np.bool_,
    "round": np.int16,
    "owner": np.int8,
    "clock": np.int16,
    "winner": np.int8,
    "over": np.bool_,
    "belief": np.int8,
    "ply": np.int32,
    "seed": np.uint32,
}
"""The other fields of `State`, and the dtype each is kept in."""

_DTYPES = {**{name: dtype for name, (dtype, _) in _UNITS.items()}, **_REST}
_COUNTED = tuple(name for name, dtype in _DTYPES.items() if np.issubdtype(dtype, np.signedinteger))
"""The fields the rules count with: widened to int32 while a step is worked out."""


class TacticsArrays:
    """The rules of a `Tactics` value as JAX functions of one `State`.

    A `gridzero.batched.ArrayGame`. `legal`, `apply`, `step`, `observe`, `player` and `draws`
    take one state and compile under ``jax.jit``; ``jax.vmap`` runs them for a batch.
    `to_arrays`, `to_position`, `action`, `local` and `dice` are NumPy's, between this engine
    and the game.

    Parameters
    ----------
    game : Tactics
        The rules.
    hidden : bool, default True
        Whether to play positions that hide units from a side (a battle under ``fog``, not a
        belief) as the game does: legal actions are those of the mover's belief, a move stops
        before a unit it didn't know of, and each step brings each side's memory of the other
        up to date. False leaves that out, for what plays only positions that hide nothing,
        such as a search, which plays beliefs: it is a sixth of a search's time. Without fog
        it changes nothing.

    Attributes
    ----------
    game : Tactics
        The rules.
    hidden : bool
        Whether it plays positions that hide units: under ``fog``, and asked to.
    n_cells, n_units : int
        The board's tiles, and the slots of units: twice the team size.
    n_actions : int
        The local actions: the larger of a battle's and a deployment's.
    n_rolls : int
        The d20 rolls a step may make: 2 for an attack against a unit that braces, or one
        per unit of a side that may defer its attack.
    n_outcomes : int
        The ways a step's rolls may hit and miss that `outcomes` tells apart: ``2 **
        n_rolls``. In outcome ``o``, the step's ``i``-th roll hits if bit ``i`` of ``o`` is
        set.
    targets : int
        The cells around a unit that MOVE, ATTACK and DEFER may name.
    move0, attack0, defer0, swap0, build0, break0, face0, end0 : int
        The first local action of each verb in battle: a verb's actions name the cells around
        the acting unit in the order of ``near``'s slots, from slot 1, and FACE the directions.
    home_size : int
        The tiles of a home: in deployment, action ``kind * home_size + h`` places a unit of a
        kind on the ``h``-th tile of the mover's home (for team 1, the opposite of team 0's).
    near, far : Patch
        The tables of the cells around a unit: as far as an action or an aura reaches, and as
        far as any kind sees (``far`` only under ``fog``).

    Raises
    ------
    TypeError
        If ``game`` is not a `Tactics`.
    ValueError
        If the board has too many tiles to count in int16.
    """

    def __init__(self, game: Tactics, *, hidden: bool = True) -> None:
        if not isinstance(game, Tactics):
            raise TypeError(f"game must be a Tactics, got {type(game).__name__}")
        if not isinstance(hidden, bool):
            raise TypeError(f"hidden must be a bool, got {type(hidden).__name__}")
        self.game = game
        self.hidden = game.fog and hidden
        space = game.space
        kinds = game.units
        n = self.n_cells = space.n_cells
        if n > np.iinfo(np.int16).max:
            raise ValueError(f"a board of {n} tiles is too large: cells are kept as int16")
        self.n_units = 2 * game.team_size
        self._steps = max(s.move * (2 if s.double_move else 1) for s in kinds)
        reach = max(s.reach for s in kinds)
        local = max(float(self._steps), reach)
        builds = any(s.builds for s in kinds)
        self.near = near = batched.patch(space, max(local, game.aura_radius, 2.0 * builds, 1.0))
        self.targets = near.within(local) - 1
        self._beside = near.within(1.0) - 1  # the tiles beside a unit: slots 1 to this
        _check_moves(near, self._steps, self.targets)
        directions = len(space.directions())
        t = self.targets
        self.move0, self.attack0, self.defer0, self.swap0 = 0, t, 2 * t, 3 * t
        self.build0 = self.swap0 + self._beside
        self.break0 = self.build0 + self._beside
        self.face0 = self.break0 + self._beside
        self.end0 = self.face0 + directions
        home = homes(space, game.home_depth)
        tiles = np.flatnonzero(home == 0)
        opposite = np.asarray(space.opposite())
        assert sorted(opposite[tiles].tolist()) == np.flatnonzero(home == 1).tolist()
        self.home_size = len(tiles)
        self._home_tiles = np.stack([tiles, opposite[tiles]])
        self.n_actions = max(self.end0 + 1, len(kinds) * self.home_size)
        defers = sum(cap for cap, s in zip(game.caps, kinds, strict=True) if s.defers)
        self._watchers = min(game.team_size, defers)
        self.n_rolls = max(2, self._watchers)
        self.n_outcomes = 1 << self.n_rolls
        self._entries = max(self._steps, 2)  # tiles a unit may enter in a step: a walk, a swap
        self._home = home
        self._point = center_cell(space)
        self._np_at = np.asarray(near.at)
        self.far: Patch | None = None

        j = jnp.asarray
        coords = np.full((n + 1, 2), 4 * max(space.shape), dtype=np.int32)
        coords[:n] = space.coords
        self._coords = j(coords)
        self._opposite = j(opposite.astype(np.int32))
        self._home_of = j(home.astype(np.int32))
        self._tiles = j(self._home_tiles.astype(np.int32))
        self._first_facing = j(np.array([game._first_facing(space, c) for c in range(n)]))
        last = [0]  # what kind -1 indexes: a slot without a unit
        self._atk = j(np.array([s.atk for s in kinds] + last))
        self._defense = j(np.array([s.defense for s in kinds] + last))
        self._move = j(np.array([s.move for s in kinds] + last))
        self._charge = j(np.array([s.charge for s in kinds] + last))
        self._hp = j(np.array([s.hp for s in kinds] + last))
        flags = ("formation", "defers", "double_move", "aura", "builds", "braces")
        self._flags = {f: j(np.array([getattr(s, f) for s in kinds] + [False])) for f in flags}
        self._reach = j(np.array([near.within(s.reach) for s in kinds] + last))
        self._near_only = j(np.array([s.reach < 1.5 for s in kinds] + [False]))
        self._aura_count = near.within(game.aura_radius)
        self._caps = j(np.array(game.caps))
        slots = self.n_units + 2
        places = [[s.places_for(count) for count in range(slots)] for s in kinds]
        self._places = j(np.array(places))
        # Shares as float32 tables, so that planes hold the bits of the game's own.
        most_hp = max(s.hp for s in kinds)
        hp_share = [np.arange(most_hp + 1) / s.hp for s in kinds] + [np.zeros(most_hp + 1)]
        self._hp_share = j(np.array(hp_share, dtype=np.float32))
        units = np.arange(self.n_units)
        alive = np.maximum(1, np.arange(self.n_units + 1))[:, None]
        self._order_share = j(((units + 1) / alive).astype(np.float32))
        rounds = np.arange(game.max_rounds + 2)
        self._round_share = j((rounds / game.max_rounds).astype(np.float32))
        self._clock_share = j(np.minimum(1.0, rounds / game.hold_rounds).astype(np.float32))
        self._age_share = j(np.minimum(1.0, rounds / game.max_rounds).astype(np.float32))
        places_left = np.arange(game.team_size + 1) / game.team_size
        self._place_share = j(places_left.astype(np.float32))
        caps = np.array(game.caps, dtype=np.float64)[:, None]
        to_deploy = (caps - np.arange(self.n_units + 1)) / np.maximum(caps, 1)
        self._deploy_share = j(to_deploy.astype(np.float32))
        if game.fog:
            self.far = batched.patch(space, max(s.sight for s in kinds))
            self._sight = j(np.array([self.far.within(s.sight) for s in kinds] + last))
            ids = np.arange(n)
            self._hops = j(hops(space, ids[:, None], ids[None, :]).astype(np.int32))
        self._hits = j((np.arange(self.n_outcomes)[:, None] >> np.arange(self.n_rolls)) & 1 > 0)
        # Each local action as a team's frame names it (`Tactics.frame`): its verb there,
        # and for a battle the slot around the acting unit of the tile it names.
        frames = [game.frame(team) for team in (0, 1)]
        verb_in = np.array(
            [[f.action_into((v, 0)).verb for v in range(game.n_verbs)] for f in frames]
        )
        cell_in = np.array([[f.action_into((0, c)).cell for c in range(n)] + [n] for f in frames])
        verbs, slots = self._battle_actions()
        self._verb_in = j(verb_in[:, verbs].astype(np.int32))
        self._cell_in = j(cell_in.astype(np.int32))
        self._slot_of = j(slots)
        placed = Verb.PLACE + np.repeat(np.arange(len(kinds)), self.home_size)
        deploying = np.zeros((2, self.n_actions), dtype=np.int32)
        for team in (0, 1):
            placed_on = cell_in[team, np.tile(self._home_tiles[team], len(kinds))]
            deploying[team, : placed.size] = verb_in[team, placed] * n + placed_on
        self._deploying = j(deploying)

    def _battle_actions(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Return the game's verb of each local action of a battle, and the slot it names.

        Two int32 arrays of shape (end0 + 1,): the slot is that of the target among the cells
        around the acting unit (``near``), 0 (the unit's own tile) for FACE and END.
        """
        t, b, turns = self.targets, self._beside, self.end0 - self.face0
        verbs = np.concatenate(
            [
                np.repeat([Verb.MOVE, Verb.ATTACK, Verb.DEFER], t),
                np.repeat([Verb.SWAP, Verb.BUILD, Verb.BREAK], b),
                Verb.FACE + np.arange(turns),
                [Verb.END],
            ]
        )
        around, beside = np.arange(1, t + 1), np.arange(1, b + 1)
        slots = np.concatenate(
            [around, around, around, beside, beside, beside, np.zeros(turns + 1, dtype=np.int64)]
        )
        return verbs.astype(np.int32), slots.astype(np.int32)

    # Between the game and its arrays (NumPy).

    def to_arrays(self, position: TacticsPosition) -> State:
        """Return a position of the game as a `State`.

        Parameters
        ----------
        position : TacticsPosition
            A position on a board of this game: its homes and its point at the center.

        Returns
        -------
        State
            NumPy arrays in the dtypes of `State`, which JAX takes as they are.

        Raises
        ------
        ValueError
            If the board is not one of this game's, a unit's uid is not below ``n_units`` or
            is held twice, or a side has more units that may defer than the caps allow.
        """
        game, state = self.game, position.state
        game._check_space(state)
        if not (
            np.array_equal(state.home, self._home)
            and np.flatnonzero(state.point).tolist() == [self._point]
        ):
            raise ValueError("the board must have this game's homes, and its point at the center")
        u = state.units
        cells = state.occupied.nonzero()[0]
        uid = u.uid[cells].astype(np.int64)
        if uid.size and (
            uid.min() < 0 or uid.max() >= self.n_units or np.unique(uid).size < uid.size
        ):
            raise ValueError(f"units need uids of their own below {self.n_units}, got {uid}")
        defers = np.array([s.defers for s in game.units])[u.kind[cells]]
        for team in (0, 1):
            if np.count_nonzero(defers & (u.team[cells] == team)) > self._watchers:
                raise ValueError(f"a side may field {self._watchers} units that defer at most")
        fields: dict[str, object] = {}
        for name, (dtype, default) in _UNITS.items():
            values = np.full(self.n_units, default, dtype=dtype)
            values[uid] = cells if name == "cell" else getattr(u, name)[cells]
            fields[name] = values
        fields |= {
            "wall": state.wall,
            "rubble": state.rubble,
            "round": position.round,
            "owner": position.hold.owner,
            "clock": position.hold.clock,
            "winner": -1 if position.winner is None else position.winner,
            "over": position.over,
            "belief": position.belief,
            "ply": position.ply,
            "seed": [position.seed & 0xFFFFFFFF, position.seed >> 32],
        }
        arrays = {name: np.asarray(value, dtype=_DTYPES[name]) for name, value in fields.items()}
        return State(**cast("dict[str, jax.Array]", arrays))  # NumPy arrays stand in for JAX's

    def to_position(self, state: State) -> TacticsPosition:
        """Return a `State` as a position of the game: the inverse of `to_arrays`.

        Parameters
        ----------
        state : State
            One state, not a batch.

        Returns
        -------
        TacticsPosition
            A position with the state's units, terrain, counters and seed, and no events.
            Played on by `Tactics.step`, it rolls the game's dice, not this engine's.
        """
        n = self.n_cells
        cell = np.asarray(state.cell).astype(np.int64)
        here = cell >= 0
        values: dict[Field[Any], NDArray[Any]] = {
            TacticsState.wall: np.asarray(state.wall),
            TacticsState.rubble: np.asarray(state.rubble),
            TacticsState.home: self._home,
            TacticsState.point: np.arange(n) == self._point,
        }
        uid = TacticsState.units.uid
        per_cell = np.full(n, uid.default, dtype=uid.dtype)
        per_cell[cell[here]] = np.flatnonzero(here)
        values[uid] = per_cell
        for name in _UNITS:
            if name == "cell":
                continue
            field = getattr(TacticsState.units, name)
            per_cell = np.full(n, field.default or 0, dtype=field.dtype)
            per_cell[cell[here]] = np.asarray(getattr(state, name))[here]
            values[field] = per_cell
        winner = int(state.winner)
        low, high = np.asarray(state.seed).tolist()
        return TacticsPosition(
            TacticsState(self.game.space, values),
            low + (high << 32),
            ply=int(state.ply),
            over=bool(state.over),
            winner=None if winner < 0 else winner,
            round=int(state.round),
            hold=Hold(int(state.owner), int(state.clock)),
            belief=int(state.belief),
        )

    def action(self, position: TacticsPosition, local: int) -> Action:
        """Return the game's action for a local action in a position.

        Parameters
        ----------
        position : TacticsPosition
            A position not over.
        local : int
            A local action, legal in ``position``.

        Returns
        -------
        Action
            ``(verb, cell)``.

        Raises
        ------
        ValueError
            If the game is over, or ``local`` names nothing in this position.
        """
        verbs, cells = self._meanings(position)
        if not 0 <= local < self.n_actions or verbs[local] < 0 or cells[local] >= self.n_cells:
            raise ValueError(f"local action {local} names nothing at ply {position.ply}")
        return Action(int(verbs[local]), int(cells[local]))

    def local(self, position: TacticsPosition, action: tuple[int, int]) -> int:
        """Return the local action of a game's action in a position: the inverse of `action`.

        Parameters
        ----------
        position : TacticsPosition
            A position not over.
        action : Action or tuple of int
            ``(verb, cell)``, legal in ``position``.

        Returns
        -------
        int
            The local action.

        Raises
        ------
        ValueError
            If the game is over, or no local action names ``action``: its target lies too far
            from the acting unit to be legal.
        """
        verbs, cells = self._meanings(position)
        found = np.flatnonzero((verbs == action[0]) & (cells == action[1]))
        if found.size != 1:
            raise ValueError(f"no local action names {tuple(action)} at ply {position.ply}")
        return int(found[0])

    def _meanings(self, position: TacticsPosition) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the game's verb and cell of every local action: two (n_actions,) int64 arrays.

        The verb is -1 for a local action that names nothing, and the cell ``n_cells`` for a
        target off the board.
        """
        game = self.game
        verbs = np.full(self.n_actions, -1, dtype=np.int64)
        cells = np.full(self.n_actions, self.n_cells, dtype=np.int64)
        if position.deploying:
            team = game.player(position)
            count = len(game.units) * self.home_size
            verbs[:count] = Verb.PLACE + np.repeat(np.arange(len(game.units)), self.home_size)
            cells[:count] = np.tile(self._home_tiles[team], len(game.units))
            return verbs, cells
        a = game.active(position)
        if a < 0:
            raise ValueError("the game is over")
        named, slots = self._battle_actions()
        verbs[: self.end0 + 1] = named
        cells[: self.end0 + 1] = self._np_at[a, slots]
        return verbs, cells

    def dice(self, before: TacticsPosition, after: TacticsPosition) -> Dice:
        """Return the dice the game rolled in a step, read from the position it made.

        Parameters
        ----------
        before, after : TacticsPosition
            A position and the one `Tactics.step` made of it.

        Returns
        -------
        Dice
            NumPy arrays with which `apply` makes ``after`` of ``before``: the rolls of the
            step's `Attacked` events in order, and the order of turns as it stands after.
        """
        rolls = [event.roll for event in after.events if isinstance(event, Attacked)]
        rolls = (rolls + [1] * self.n_rolls)[: self.n_rolls]
        order = np.arange(self.n_units, dtype=np.int32)
        first = 0
        if after.round > before.round:  # a round began: the order as the game rolled it
            units = after.state.units
            cells = after.state.occupied.nonzero()[0]
            order = order + self.n_units  # the dead after the living, each different
            order[units.uid[cells]] = units.initiative[cells]
            first = int(units.team[cells[np.argmin(units.initiative[cells])]])
        draws = (np.array(rolls, dtype=np.int32), np.int32(first), order)
        return Dice(*cast("tuple[jax.Array, ...]", draws))  # NumPy arrays stand in for JAX's

    # What a decision is about.

    def player(self, state: State) -> jax.Array:
        """Return the team that decides next.

        Parameters
        ----------
        state : State
            A state not over.

        Returns
        -------
        Array of int32, shape ()
            In deployment the team that places next, and in battle the acting unit's team.
        """
        s = _wide(state)
        return jnp.where(s.round == 0, self._placer(s), s.team[self._acting(s)])

    def over(self, state: State) -> jax.Array:
        """Return whether the game has ended.

        Parameters
        ----------
        state : State
            One state.

        Returns
        -------
        Array of bool, shape ()
            ``state.over``.
        """
        return jnp.asarray(state.over)

    def winner(self, state: State) -> jax.Array:
        """Return the team that won.

        Parameters
        ----------
        state : State
            One state.

        Returns
        -------
        Array of int32, shape ()
            The winner, or -1 while the game runs. Tactics has no draws.
        """
        return jnp.asarray(state.winner).astype(jnp.int32)

    def legal(self, state: State) -> jax.Array:
        """Return the mask of legal local actions, as `Tactics.legal` finds them.

        Parameters
        ----------
        state : State
            One state.

        Returns
        -------
        Array of bool, shape (n_actions,)
            All False once the game is over.
        """
        s = self._as_mover_believes(_wide(state))
        none = jnp.zeros(self.n_actions, jnp.bool_)
        placing, acting = self._deployable(s), self._battle_legal(s)
        placing = none.at[: placing.shape[0]].set(placing)
        acting = none.at[: acting.shape[0]].set(acting)
        return jnp.where(s.round == 0, placing, acting) & ~s.over

    def view(self, state: State, player: ArrayLike) -> State:
        """Return what a team is shown of a state, as `Tactics.view` does.

        Parameters
        ----------
        state : State
            One state.
        player : Array of int, shape ()
            0 or 1: the team.

        Returns
        -------
        State
            The state itself without ``fog``, once the game is over, and when it is a belief
            already. Otherwise the team's belief (``belief == player``): its own units as they
            are, and each of the other side's on the tile it was last seen on, with the hit
            points and facing it showed then, as a unit that has done nothing this round. A
            belief's seed is drawn from the state's (`gridzero.batched.stream` 3 of its root),
            as the game gives its beliefs a seed of their own: another than the game's, as
            this engine's dice are.
        """
        if not self.game.fog:
            return jax.tree.map(jnp.asarray, state)
        s = _wide(state)
        shown = (s.belief < 0) & ~s.over
        return _narrow(_select(shown, self._believed(s, jnp.asarray(player, jnp.int32)), s))

    def _hidden(self, s: State) -> jax.Array:
        """Return whether a state hides units from a side: a battle under fog, not a belief."""
        return jnp.bool_(self.hidden) & (s.belief < 0) & (s.round > 0) & ~s.over

    def _as_mover_believes(self, s: State) -> State:
        """Return a state as its mover decides on it: its belief where the state hides units."""
        if not self.hidden:
            return s
        return _select(self._hidden(s), self._believed(s, s.team[self._acting(s)]), s)

    def _believed(self, s: State, team: jax.Array) -> State:
        """Return a team's belief of a state of a game under fog, as `Tactics._believed`."""
        alive, battle = s.cell >= 0, s.round > 0
        ghost = alive & (s.team != team) & (s.spotted != s.cell) & battle
        cell = jnp.where(ghost, s.spotted, s.cell)
        hp = jnp.where(ghost, s.spotted_hp, s.hp)
        facing = jnp.where(ghost, s.spotted_facing, s.facing)
        shown = alive & battle  # in a belief every unit is where it is shown
        key = batched.stream(batched.root(s.seed), 3)
        return s._replace(
            cell=cell,
            hp=hp,
            facing=facing,
            moved=jnp.where(ghost, 0, s.moved),
            swapped=s.swapped & ~ghost,
            attacked=s.attacked & ~ghost,
            faced=s.faced & ~ghost,
            engaged=s.engaged & ~ghost,
            watch=jnp.where(ghost, -1, s.watch),
            spotted=jnp.where(shown, cell, s.spotted),
            spotted_hp=jnp.where(shown, hp, s.spotted_hp),
            spotted_facing=jnp.where(shown, facing, s.spotted_facing),
            spotted_round=jnp.where(shown & (s.team == team), s.round, s.spotted_round),
            belief=team,
            seed=key.at[1].set(key[1] & 0x7FFFFFFF),  # below 2**63, as a game's seed
        )

    def _acting(self, s: State) -> jax.Array:
        """Return the slot of the unit whose turn it is: the lowest initiative not done."""
        waiting = (s.cell >= 0) & ~s.done
        return jnp.argmin(jnp.where(waiting, s.initiative, FAR)).astype(jnp.int32)

    def _unit_at(self, s: State) -> jax.Array:
        """Return the slot of the unit on each tile, -1 for none: int32, shape (n_cells + 1,).

        The last entry is off the board: no unit.
        """
        where = jnp.where(s.cell >= 0, s.cell, self.n_cells)
        slots = jnp.arange(self.n_units, dtype=jnp.int32)
        return jnp.full(self.n_cells + 1, -1, jnp.int32).at[where].set(slots).at[-1].set(-1)

    def _fielded(self, s: State, team: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return a side's units by kind, its places left, and the places each kind's next takes.

        As `Tactics._fielded`: int32 arrays of shape (n_kinds,), and a scalar.
        """
        kinds = jnp.arange(len(self.game.units))
        mine = (s.cell >= 0) & (s.team == team)
        have = ((kinds[:, None] == s.kind[None, :]) & mine).sum(axis=1)
        spent = self._places[kinds, have]
        return have, self.game.team_size - spent.sum(), self._places[kinds, have + 1] - spent

    def _placer(self, s: State) -> jax.Array:
        """Return the team that places next, or -1 once neither can, as `Tactics._placer`."""
        can, placed = [], []
        for team in (0, 1):
            have, left, cost = self._fielded(s, jnp.int32(team))
            can.append(((have < self._caps) & (cost <= left)).any())
            placed.append(have.sum())
        first = can[0] & (~can[1] | (placed[0] <= placed[1]))
        return jnp.where(first, 0, jnp.where(can[1], 1, -1)).astype(jnp.int32)

    def _deployable(self, s: State) -> jax.Array:
        """Return the legal actions of a deployment: bool, shape (n_kinds * home_size,)."""
        team = self._placer(s)
        have, left, cost = self._fielded(s, team)
        tiles = self._tiles[jnp.maximum(team, 0)]
        free = ~(s.wall | s.rubble)[tiles] & (self._unit_at(s)[tiles] < 0)
        fits = (have < self._caps) & (cost <= left)
        return (fits[:, None] & free[None, :]).reshape(-1) & (team >= 0)

    def _battle_legal(self, s: State) -> jax.Array:
        """Return the legal actions of a battle's decision: bool, shape (end0 + 1,)."""
        game, near = self.game, self.near
        u = self._acting(s)
        k, start = s.kind[u], s.cell[u]
        most, six = self.targets + 1, self._beside + 1
        cells = near.at[start]
        holds = self._unit_at(s)
        occupant = holds[cells]
        open_ = jnp.append(~(s.wall | s.rubble), False)[cells]  # by slot: neither wall nor rubble
        on_board = cells < self.n_cells
        twice = self._flags["double_move"][k] & ~s.attacked[u]
        left = jnp.where(s.swapped[u], 0, jnp.where(twice, 2, 1) * self._move[k] - s.moved[u])
        blocked = s.wall | s.rubble | (holds[:-1] >= 0)
        cost = batched.steps_around(near, start, blocked, self._steps)[1:most]
        move = (cost >= 1) & (cost <= left)
        can = ~s.attacked[u] & ~(self._flags["double_move"][k] & (s.moved[u] > self._move[k]))
        slots = jnp.arange(1, most)
        seen = self._near_only[k] | batched.clear(near, s.wall, start)[1:most]
        in_range = (slots < self._reach[k]) & seen & on_board[1:most]
        other = occupant[1:most]
        attack = can & in_range & (other >= 0) & (s.team[other] != s.team[u])
        defer = can & self._flags["defers"][k] & in_range & open_[1:most]
        friend = occupant[1:six]
        swap = (
            (s.moved[u] == 0)
            & ~s.swapped[u]
            & (friend >= 0)
            & (s.team[friend] == s.team[u])
            & (s.moved[friend] == 0)
            & ~s.swapped[friend]
        )
        works = can & self._flags["builds"][k]
        wall = jnp.append(s.wall, False)[cells[1:six]]
        point = cells[1:six] == self._point
        around = jnp.append(open_, False)[near.beside[1:six]]  # the tiles around each, in order
        arcs = (around & ~jnp.roll(around, -1, axis=1)).sum(axis=1)
        clear = ~open_[1:six] | (arcs <= 1)
        build = works & on_board[1:six] & ~wall & (friend < 0) & ~point & clear
        break_ = works & on_board[1:six] & ~open_[1:six] & ~point
        turns = jnp.arange(self.end0 - self.face0)
        face = game.free_facing & ~s.faced[u] & (turns != s.facing[u])
        end = jnp.ones(1, jnp.bool_)
        return jnp.concatenate([move, attack, defer, swap, build, break_, face, end])

    # A step.

    def roll(self, key: ArrayLike) -> Dice:
        """Return the dice of one step, drawn from a key.

        Parameters
        ----------
        key : Array of uint32, shape (2,)
            A generator's key (`gridzero.batched.root`, `gridzero.batched.stream`).

        Returns
        -------
        Dice
            A d20 per roll a step may make, the team to go first, and a random order of the
            units. The same key gives the same dice.
        """
        return Dice(
            batched.d(batched.stream(key, 0), 20, (self.n_rolls,)),
            batched.d(batched.stream(key, 1), 2) - 1,
            batched.permutation(batched.stream(key, 2), self.n_units),
        )

    def draws(self, state: State) -> Dice:
        """Return the dice of the step a state takes next, from its own stream.

        Parameters
        ----------
        state : State
            One state.

        Returns
        -------
        Dice
            Drawn from the stream of the state's seed named by its ply, as the game draws from
            a stream of its seed named by the ply. The same state gives the same dice.
        """
        return self.roll(batched.stream(batched.root(state.seed), 1, state.ply))

    def step(self, state: State, action: ArrayLike) -> State:
        """Return the state after a legal action, rolling the state's own dice.

        Parameters
        ----------
        state : State
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``.

        Returns
        -------
        State
            ``apply(state, action, draws(state))``.
        """
        return self.apply(state, action, self.draws(state))

    def apply(self, state: State, action: ArrayLike, dice: Dice) -> State:
        """Return the state after a legal action, given the step's dice, as `Tactics.step` does.

        Parameters
        ----------
        state : State
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``. An illegal one gives a state that means
            nothing: nothing is checked.
        dice : Dice
            The step's draws. A step uses only the first rolls it needs.

        Returns
        -------
        State
            The next state, one ply later.
        """
        return self._applied(state, action, dice)[0]

    def outcomes(self, state: State, action: ArrayLike, dice: Dice) -> tuple[State, jax.Array]:
        """Return every way a step's attacks may hit and miss, each with its chance.

        Parameters
        ----------
        state : State
            A state not over.
        action : Array of int, shape ()
            A local action, legal in ``state``.
        dice : Dice
            The step's other draws: who goes first and the order of turns, where the step
            starts the battle or a round. Its rolls are not read.

        Returns
        -------
        states : State
            The state after each outcome: every array has a leading axis of ``n_outcomes``.
            In outcome ``o`` the step's ``i``-th roll hits if bit ``i`` of ``o`` is set.
        chances : Array of float32, shape (n_outcomes,)
            The probability of each: the product of its rolls' chances, each the share of a
            d20's faces that hit, or miss. They sum to 1. An outcome that names a roll the
            step never makes (the unit died first, or the step rolls less) has chance 0.
        """
        rolls = jnp.arange(self.n_rolls)

        def one(hits: jax.Array) -> tuple[State, jax.Array]:
            forced = dice._replace(rolls=jnp.where(hits, 20, 1))
            after, rolled = self._applied(state, action, forced)
            made = rolls < rolled.used
            share = jnp.where(hits, rolled.faces, 20 - rolled.faces).astype(jnp.float32) / 20
            return after, jnp.where(made, share, jnp.where(hits, 0.0, 1.0)).prod()

        return jax.vmap(one)(self._hits)

    def _applied(self, state: State, action: ArrayLike, dice: Dice) -> tuple[State, _Rolled]:
        """Return the state after a legal action, and the rolls the step made of its dice."""
        s = _wide(state)
        action = jnp.asarray(action, jnp.int32)
        dice = Dice(*(jnp.asarray(draw, jnp.int32) for draw in dice))
        deployed = self._deploy(s, action, dice)
        battle, rolled, walk = self._battle(s, action, dice)
        after = _select(s.round == 0, deployed, battle)
        if self.hidden:
            after = self._spotted(s, after, walk)
        gone = after.cell < 0  # a slot without a unit holds the defaults
        cleared = {
            name: jnp.where(gone, default, getattr(after, name))
            for name, (_, default) in _UNITS.items()
        }
        after = _narrow(after._replace(**cleared, ply=s.ply + 1))
        return after, rolled._replace(used=jnp.where(s.round == 0, 0, rolled.used))

    def _spotted(self, s: State, after: State, walk: _Walk) -> State:
        """Return the state after a step under fog, each side's memory brought up to date.

        From the step that starts the battle, where the fog falls, and every step of it.
        """
        alive = after.cell >= 0
        falls = (s.round == 0) & (after.round == 1) & (s.belief < 0)  # each side saw it all
        fallen = after._replace(
            spotted=jnp.where(alive, after.cell, -1),
            spotted_hp=after.hp,
            spotted_facing=after.facing,
            spotted_round=jnp.where(alive, 1, after.spotted_round),
        )
        before = _select(falls, fallen, after)
        mover = jnp.where(falls, -1, s.team[self._acting(s)])
        spotted = self._spot(before, jnp.where(falls, 1, s.round), mover, walk)
        return _select(falls | self._hidden(s), spotted, after)

    def _spot(self, s: State, round_: jax.Array, mover: jax.Array, walk: _Walk) -> State:
        """Bring each side's memory of the other's units up to date, as `Tactics._spot`.

        A unit on a tile the other side sees is spotted where it stands. One that isn't keeps
        the tile it was last seen on until the other side sees that tile without it: then it is
        believed on the nearest tile that side doesn't see (by hops, then cell id), walls,
        rubble and the tiles of other ghosts aside, in order of the units' cells. The team
        ``mover`` (-1 for none) also sees what its unit saw along ``walk``, and that unit is
        last seen on the last tile of its walk the other side sees.
        """
        far, n, none = cast("Patch", self.far), self.n_cells, self.n_units
        alive = s.cell >= 0
        open_ = ~(s.wall | s.rubble)
        sees = [
            batched.sight(
                far, s.wall, jnp.where(alive & (s.team == t), s.cell, -1), self._sight[s.kind]
            )
            for t in (0, 1)
        ]
        u = walk.unit
        walks = (mover >= 0) & (walk.length >= 1) & (s.cell[u] >= 0)  # it lived through it
        along = jnp.arange(walk.tiles.shape[0])
        on_way = jnp.where(walks & (along < walk.length), walk.tiles, -1)
        saw = batched.sight(far, s.wall, on_way, self._sight[s.kind[u]])
        theirs = jnp.where(mover == 0, sees[1], sees[0])
        glimpsed = walks & (along <= walk.length) & jnp.append(theirs, False)[walk.tiles]
        last = walk.tiles[jnp.argmax(jnp.where(glimpsed, along, -1))]
        mark = jnp.where(glimpsed.any(), u, none)
        s = s._replace(
            spotted=s.spotted.at[mark].set(last, mode="drop"),
            spotted_hp=s.spotted_hp.at[mark].set(s.hp[u], mode="drop"),
            spotted_facing=s.spotted_facing.at[mark].set(s.facing[u], mode="drop"),
            spotted_round=s.spotted_round.at[mark].set(round_, mode="drop"),
        )
        sees = [sees[t] | (saw & (mover == t)) for t in (0, 1)]
        spotted = s.spotted
        for side in (0, 1):
            other = sees[1 - side]
            mine = alive & (s.team == side)
            seen = mine & jnp.append(other, False)[jnp.where(alive, s.cell, n)]
            spotted = jnp.where(seen, s.cell, spotted)
            s = s._replace(
                spotted_hp=jnp.where(seen, s.hp, s.spotted_hp),
                spotted_facing=jnp.where(seen, s.facing, s.spotted_facing),
                spotted_round=jnp.where(seen, round_, s.spotted_round),
            )
            lost = mine & ~seen
            ghost = jnp.clip(spotted, 0, n - 1)
            gone = lost & other[ghost]
            free = open_ & ~other
            free = free.at[jnp.where(lost & ~gone, ghost, n)].set(False, mode="drop")
            in_order = jnp.argsort(jnp.where(gone, s.cell, FAR))
            for nth in range(self.game.team_size):
                unit = in_order[nth]
                moves = gone[unit]
                there = jnp.argmin(jnp.where(free, self._hops[ghost[unit]], FAR))
                spotted = spotted.at[jnp.where(moves, unit, none)].set(there, mode="drop")
                free = free.at[jnp.where(moves, there, n)].set(False, mode="drop")
        return s._replace(spotted=spotted)

    def _deploy(self, s: State, action: jax.Array, dice: Dice) -> State:
        """Place a unit for the team deploying, and start the battle after the last one."""
        team = self._placer(s)
        kind = action // self.home_size
        tile = self._tiles[jnp.maximum(team, 0), action % self.home_size]
        slot = (s.cell >= 0).sum()
        s = s._replace(
            cell=s.cell.at[slot].set(tile),
            kind=s.kind.at[slot].set(kind),
            team=s.team.at[slot].set(team),
            hp=s.hp.at[slot].set(self._hp[kind]),
            facing=s.facing.at[slot].set(self._first_facing[tile]),
        )
        starts = self._placer(s) < 0
        order = self._order(s, dice, first_round=True)
        return s._replace(
            initiative=jnp.where(starts, order, s.initiative),
            round=jnp.where(starts, 1, 0),
        )

    def _battle(self, s: State, action: jax.Array, dice: Dice) -> tuple[State, _Rolled, _Walk]:
        """Carry out the acting unit's action and finish the step, as `Tactics.step` in battle.

        Returns the state, the rolls the step made and the acting unit's walk.
        """
        game, near, none = self.game, self.near, self.n_units
        u = self._acting(s)
        start, facing = s.cell[u], s.facing[u]
        cells = near.at[start]
        holds = self._unit_at(s)
        firsts = (self.attack0, self.defer0, self.swap0, self.build0, self.break0, self.face0)
        verb = sum((action >= first).astype(jnp.int32) for first in firsts)  # 0 MOVE .. 6 FACE, END
        moves, attacks, defers, swaps, builds, breaks, turns = (
            jnp.equal(verb, v) for v in range(7)
        )
        ends = action == self.end0
        turns = turns & ~ends
        base = jnp.asarray((0, *firsts))[verb]
        target = jnp.where(verb < 6, action - base + 1, 0)  # the slot it names, 0 for its own
        tile = cells[target]

        # What the verb costs the unit, and a tile worked on.
        spent = attacks | defers | builds | breaks
        works = jnp.where(builds | breaks, tile, self.n_cells)
        was_wall, was_rubble = s.wall[tile], s.rubble[tile]
        s = s._replace(
            attacked=s.attacked.at[u].set(s.attacked[u] | spent),
            faced=s.faced.at[u].set(jnp.where(spent | swaps, False, s.faced[u] | turns)),
            facing=s.facing.at[u].set(jnp.where(turns, action - self.face0, facing)),
            done=s.done.at[u].set(s.done[u] | ends),
            watch=s.watch.at[u].set(jnp.where(defers, tile, s.watch[u])),
            wall=s.wall.at[works].set(builds & was_rubble, mode="drop"),
            rubble=s.rubble.at[works].set(jnp.where(builds, ~was_rubble, was_wall), mode="drop"),
        )

        # A move's route, around the tiles enemies watch where another way is as short: as
        # the mover believes the board to be, and it stops before a unit it didn't know of.
        plan = self._as_mover_believes(s)
        alive = plan.cell >= 0
        blocked = s.wall | s.rubble | (self._unit_at(plan)[:-1] >= 0)
        watching = alive & (plan.team != s.team[u]) & (plan.watch >= 0)
        watched = jnp.where(watching, plan.watch, self.n_cells)
        cost = jnp.full(self.n_cells + 1, _STEP, jnp.int32).at[watched].set(_STEP + 1)[:-1]
        goal = jnp.where(moves, target, 0)
        route, length = batched.path_around(near, start, goal, blocked, self._steps, cost)
        if self.hidden:
            ahead = jnp.arange(1, self._steps + 1)
            met = (holds[cells[route[1:]]] >= 0) & (ahead <= length) & self._hidden(s)
            length = jnp.where(met.any(), jnp.argmax(met), length).astype(jnp.int32)
        walks_on = moves & (length > 0)
        s = s._replace(faced=s.faced.at[u].set(s.faced[u] & ~walks_on))

        # A swap: both units change places before any archer fires.
        partner = jnp.where(swaps, holds[tile], u)
        toward = near.turned[target, facing]
        a, b = jnp.where(swaps, u, none), jnp.where(swaps, partner, none)
        s = s._replace(
            cell=s.cell.at[a].set(tile, mode="drop").at[b].set(start, mode="drop"),
            swapped=s.swapped.at[a].set(True, mode="drop").at[b].set(True, mode="drop"),
        )
        if game.turn_on_move:
            turned = s.facing.at[a].set(toward, mode="drop")
            s = s._replace(facing=turned.at[b].set((toward + 3) % 6, mode="drop"))

        # The tiles units enter, one at a time: each springs the attacks deferred onto it.
        rolled = _Rolled(dice.rolls, jnp.int32(0), jnp.zeros(self.n_rolls, jnp.int32))
        for entry in range(self._entries):
            here, there = route[min(entry, self._steps)], route[min(entry + 1, self._steps)]
            walks = moves & (entry < length) & (s.cell[u] >= 0)
            mover = jnp.where(walks, u, none)
            s = s._replace(
                cell=s.cell.at[mover].set(cells[there], mode="drop"),
                moved=s.moved.at[mover].add(1, mode="drop"),
            )
            if game.turn_on_move:
                way = batched.slot(near, near.step[there] - near.step[here])
                faces = near.turned[way, s.facing[u]]
                s = s._replace(facing=s.facing.at[mover].set(faces, mode="drop"))
            enters = walks | (swaps & (entry < 2))
            who = jnp.where(swaps & (entry == 1), partner, u)
            s, rolled = self._spring(s, who, enters, rolled)
        end = route[jnp.clip(length, 0, self._steps)]
        if game.turn_on_move:  # a walk ends facing the way it went, from start to end
            walked = jnp.where(walks_on & (s.cell[u] >= 0), u, none)
            s = s._replace(facing=s.facing.at[walked].set(near.turned[end, facing], mode="drop"))

        other = jnp.where(attacks, holds[tile], u)
        s, rolled = self._attack(s, u, other, attacks, rolled, deferred=False)
        walk = _Walk(u, cells[route], jnp.where(moves, length, 0))
        return self._next(s, u, dice), rolled, walk

    def _spring(
        self, s: State, who: jax.Array, on: jax.Array, rolled: _Rolled
    ) -> tuple[State, _Rolled]:
        """Fire the attacks enemy archers deferred onto the tile of unit ``who``, if ``on``.

        Archers fire in order of their cells, each once, if the tile is within its reach and
        sight, until the unit dies. Returns the state and the rolls made so far.
        """
        near, none = self.near, self.n_units
        tile = s.cell[who]
        waits = on & (tile >= 0) & (s.cell >= 0) & (s.team != s.team[who]) & (s.watch == tile)
        in_order = jnp.argsort(jnp.where(waits, s.cell, FAR))
        for nth in range(self._watchers):
            archer = in_order[nth]
            at = jnp.where(s.cell[archer] >= 0, s.cell[archer], self.n_cells)
            way = batched.slot(near, self._coords[tile] - self._coords[at])
            seen = batched.clear(near, s.wall, at)[jnp.minimum(way, near.size - 1)]
            fires = waits[archer] & (s.cell[who] >= 0) & seen & (way < self._reach[s.kind[archer]])
            s = s._replace(watch=s.watch.at[jnp.where(fires, archer, none)].set(-1, mode="drop"))
            s, rolled = self._attack(s, archer, who, fires, rolled, deferred=True)
        return s, rolled

    def _attack(
        self,
        s: State,
        attacker: jax.Array,
        target: jax.Array,
        on: jax.Array,
        rolled: _Rolled,
        *,
        deferred: bool,
    ) -> tuple[State, _Rolled]:
        """Roll one attack between two units, if ``on``, as `Tactics._attack` does.

        A target bracing against the charge strikes first, and may kill the attacker. Returns
        the state and the rolls made so far.
        """
        atk, defense, braced = self._stats(s, attacker, target, deferred=deferred, strike=False)
        if not deferred:
            back, guard, _ = self._stats(s, target, attacker, deferred=False, strike=True)
            s, rolled = self._blow(s, target, attacker, on & braced, back, guard, rolled)
            on = on & (s.cell[attacker] >= 0)
        return self._blow(s, attacker, target, on, atk, defense, rolled)

    def _blow(
        self,
        s: State,
        attacker: jax.Array,
        target: jax.Array,
        on: jax.Array,
        atk: jax.Array,
        defense: jax.Array,
        rolled: _Rolled,
    ) -> tuple[State, _Rolled]:
        """Roll for one blow of given Atk and Def, if ``on``: engage the target, hurt or kill it."""
        game = self.game
        nth = jnp.minimum(rolled.used, self.n_rolls - 1)
        roll = rolled.rolls[nth]
        needed = game.hit_base - game.hit_step * (atk - defense)
        faces = jnp.clip(20 - needed, 0, 18) + 1  # the 20, and 2 to 19 from the roll needed
        hit = on & ((roll == 20) | ((roll != 1) & (roll >= needed)))
        hp = s.hp[target]
        damage = jnp.minimum(atk, hp)
        dies = hit & (damage == hp)
        struck = jnp.where(on, target, self.n_units)
        s = s._replace(
            hp=s.hp.at[struck].set(jnp.where(hit, hp - damage, hp), mode="drop"),
            cell=s.cell.at[struck].set(jnp.where(dies, -1, s.cell[target]), mode="drop"),
            engaged=s.engaged.at[struck].set(True, mode="drop"),
        )
        made = jnp.where(on, nth, self.n_rolls)
        faces = rolled.faces.at[made].set(faces, mode="drop")
        return s, _Rolled(rolled.rolls, rolled.used + on, faces)

    def _stats(
        self, s: State, attacker: jax.Array, target: jax.Array, *, deferred: bool, strike: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the effective Atk and Def, and whether the target braces, as `Tactics._stats`."""
        game, near = self.game, self.near
        alive = s.cell >= 0
        where = self._coords[jnp.where(alive, s.cell, self.n_cells)]
        slots = batched.slot(near, where - where[target])  # every unit, from the target
        beside = alive & (slots >= 1) & (slots <= self._beside)
        enemies = (beside & (s.team == s.team[attacker])).sum()
        friends = beside & (s.team == s.team[target]) & self._flags["formation"][s.kind]
        from_ = jnp.minimum(slots[attacker], near.size - 1)
        behind = ~near.front[s.facing[target], from_]
        ka, kt = s.kind[attacker], s.kind[target]
        charges = self._charge[ka] > 0
        if strike or deferred:
            walled = charged = jnp.bool_(False)
        else:
            walled = charges & self._flags["braces"][kt] & ~s.engaged[target] & friends.any()
            charged = charges & (s.moved[attacker] >= self._move[ka])
        flank = jnp.bool_(False) if strike else ~walled & (behind | (enemies >= 2))
        atk = self._atk[ka] + self._aura(s, attacker) + game.engaged_bonus * s.engaged[target]
        atk = atk + jnp.where(charged & ~walled, self._charge[ka], 0)
        atk = jnp.where(flank, atk * game.flank_factor, atk)
        if deferred:
            atk = atk + game.overwatch_bonus
        formed = jnp.minimum(game.formation_cap, game.formation_bonus * friends.sum())
        defense = self._defense[kt] + self._aura(s, target)
        defense = defense + jnp.where(self._flags["formation"][kt], formed, 0)
        return atk, defense, charged & walled

    def _aura(self, s: State, unit: jax.Array) -> jax.Array:
        """Return a unit's aura bonus: its own aura's, or a friend's within reach, or 0."""
        alive = s.cell >= 0
        where = self._coords[jnp.where(alive, s.cell, self.n_cells)]
        near = batched.slot(self.near, where - where[unit]) < self._aura_count
        gives = alive & self._flags["aura"][s.kind] & (s.team == s.team[unit]) & near
        return jnp.where(gives.any(), self.game.aura_bonus, 0)

    def _order(self, s: State, dice: Dice, *, first_round: bool) -> jax.Array:
        """Return the order of turns a round's start rolls: each unit's place, int32 (U,).

        Rolled once (``first_round``), the teams alternate from ``dice.first``, each in the
        order of its units' numbers, and the larger team takes the last turns. Shuffled each
        round, every unit takes its number's place among the living. 0 for a slot with no unit.
        """
        alive = s.cell >= 0
        before = alive[None, :] & (dice.order[None, :] < dice.order[:, None])
        if self.game.shuffle_each_round:
            return jnp.where(alive, before.sum(axis=1), 0)
        if not first_round:
            return s.initiative
        same = s.team[None, :] == s.team[:, None]
        place = (before & same).sum(axis=1)
        others = (alive[None, :] & ~same).sum(axis=1)
        leads = s.team == dice.first
        place = place + jnp.minimum(jnp.where(leads, place, place + 1), others)
        return jnp.where(alive, place, 0)

    def _next(self, s: State, u: jax.Array, dice: Dice) -> State:
        """Finish a step: the end of the game, of the round, or of the acting unit's turn."""
        game, none = self.game, self.n_units
        alive = s.cell >= 0
        sides = [(alive & (s.team == team)).any() for team in (0, 1)]
        killed = ~(sides[0] & sides[1])
        waiting = alive & ~s.done
        round_ends = ~killed & ~waiting.any()
        on_point = alive & (s.cell == self._point)
        standing = jnp.where(on_point.any(), s.team[jnp.argmax(on_point)], -1)
        owner = jnp.where(standing >= 0, standing, s.owner)
        clock = jnp.where(owner < 0, s.clock, jnp.where(owner == s.owner, s.clock + 1, 1))
        late = s.round >= game.capture_by
        ends = jnp.where(
            owner < 0, late, (clock >= game.hold_rounds) | (s.round >= game.max_rounds)
        )
        decided = round_ends & ends
        holder = jnp.where(owner < 0, 1 - objective.DEADLINE_LOSER, owner)
        winner = jnp.where(killed, jnp.where(sides[0], 0, 1), jnp.where(decided, holder, -1))
        over = killed | decided
        fresh = round_ends & ~ends  # a new round begins
        initiative = jnp.where(fresh, self._order(s, dice, first_round=False), s.initiative)
        first = jnp.argmin(jnp.where(jnp.where(fresh, alive, waiting), initiative, FAR))
        begins = jnp.where(~over & (fresh | (first != u)), first, none)  # a turn begins
        engaged = s.engaged.at[begins].set(False, mode="drop")
        return s._replace(
            initiative=initiative,
            done=jnp.where(fresh, False, s.done),
            swapped=jnp.where(fresh, False, s.swapped),
            attacked=jnp.where(fresh, False, s.attacked),
            faced=jnp.where(fresh, False, s.faced).at[begins].set(False, mode="drop"),
            moved=jnp.where(fresh, 0, s.moved),
            watch=s.watch.at[begins].set(-1, mode="drop"),
            engaged=engaged if game.engaged_until_turn else jnp.where(fresh, False, s.engaged),
            round=s.round + fresh,
            owner=jnp.where(round_ends, owner, s.owner),
            clock=jnp.where(round_ends, clock, s.clock),
            winner=winner,
            over=over,
        )

    # What a network reads.

    def framed(self, state: State) -> jax.Array:
        """Return where a network scores each local action, in the mover's frame.

        Parameters
        ----------
        state : State
            A state not over.

        Returns
        -------
        Array of int32, shape (n_actions,)
            ``verb * n_cells + cell`` of each local action as the frame of the team to move
            names it (`Tactics.frame`): for team 1, the opposite tile, and a facing turned half
            way round. 0 for a local action that names nothing, or a tile off the board.
        """
        s = _wide(state)
        n = self.n_cells
        team = jnp.clip(self.player(state), 0, 1)
        tiles = self.near.at[s.cell[self._acting(s)]][self._slot_of]
        battle = jnp.where(tiles < n, self._verb_in[team] * n + self._cell_in[team, tiles], 0)
        acting = jnp.zeros(self.n_actions, jnp.int32).at[: battle.shape[0]].set(battle)
        return jnp.where(s.round == 0, self._deploying[team], acting)

    def observe(self, state: State, player: ArrayLike) -> jax.Array:
        """Return what a team sees of a state: `Tactics.observe`, bit for bit.

        Parameters
        ----------
        state : State
            One state.
        player : Array of int, shape ()
            0 or 1: the team.

        Returns
        -------
        Array of float32, shape (len(game.planes), n_cells)
            The planes of `Tactics.planes`, in the team's frame.
        """
        game, n, slots = self.game, self.n_cells, self.n_units
        s = _wide(state)
        team = jnp.asarray(player, jnp.int32)
        kinds = jnp.arange(len(game.units))[:, None] == s.kind[None, :]
        alive = s.cell >= 0
        where = jnp.where(alive, s.cell, n)
        own, enemy = alive & (s.team == team), alive & (s.team == 1 - team)
        battle = s.round > 0
        facing = (s.facing + 3 * team) % 6  # the frame turns half a turn for team 1
        place = (alive[None, :] & (s.initiative[None, :] < s.initiative[:, None])).sum(axis=1)
        acts = (jnp.arange(slots) == self._acting(s)) & battle & ~s.over
        deployed = (own[None, :] & kinds).sum(axis=1)
        left = jnp.clip(self._fielded(s, team)[1], 0, game.team_size)
        to_place = jnp.where(battle, 0.0, self._place_share[left])
        to_deploy = self._deploy_share[jnp.arange(len(game.units)), deployed] * (to_place > 0)

        def on_tiles(values: jax.Array) -> jax.Array:
            """Lay rows of values per unit on the units' tiles: float32, (rows, n_cells)."""
            values = jnp.where(alive[None, :], values, 0).astype(jnp.float32)
            return jnp.zeros((values.shape[0], n + 1), jnp.float32).at[:, where].set(values)[:, :n]

        def marked(tiles: jax.Array, which: jax.Array) -> jax.Array:
            """Mark the tiles of the units in a mask: bool, (n_cells + 1,)."""
            return jnp.zeros(n + 1, jnp.bool_).at[jnp.where(which, tiles, n)].max(which)

        def filled(value: jax.Array) -> jax.Array:
            """Fill a plane with one value: float32, (1, n_cells)."""
            return jnp.full((1, n), value, jnp.float32)

        watches = alive & (s.watch >= 0)
        tiles = jnp.arange(n)
        planes = [
            s.wall[None],
            s.rubble[None],
            (self._home_of == team)[None],
            (self._home_of == 1 - team)[None],
            on_tiles(own[None] & kinds),
            on_tiles(enemy[None] & kinds),
            on_tiles(self._hp_share[s.kind, jnp.clip(s.hp, 0, self._hp_share.shape[1] - 1)][None]),
            on_tiles(jnp.arange(6)[:, None] == facing[None, :]),
            on_tiles(jnp.stack([s.moved > 0, s.swapped, s.attacked, s.done, s.engaged])),
            marked(s.watch, watches & (s.team == team))[None, :n],
            marked(s.watch, watches & (s.team != team))[None, :n],
            on_tiles(acts[None]),
            on_tiles(jnp.where(battle, self._order_share[alive.sum(), place], 0.0)[None]),
            on_tiles((s.moved >= jnp.maximum(self._move[s.kind], 1))[None]),
            filled(~battle),
            filled(self._round_share[s.round]),
            jnp.broadcast_to(to_deploy[:, None], (len(game.units), n)),
            filled(to_place),
            (tiles == self._point)[None],
            filled(s.owner == team),
            filled(s.owner == 1 - team),
            filled(self._clock_share[jnp.clip(s.clock, 0, game.max_rounds + 1)]),
            filled(team == objective.DEADLINE_LOSER),
        ]
        if self.far is not None:
            looks = jnp.where(own, s.cell, -1)
            seen = batched.sight(self.far, s.wall, looks, self._sight[s.kind]) | ~battle
            ghost = enemy & ~jnp.append(seen, False)[where] & battle
            age = jnp.clip(s.round - s.spotted_round + 1, 0, game.max_rounds + 1)
            planes += [seen[None], on_tiles(jnp.where(ghost, self._age_share[age], 0.0)[None])]
        out = jnp.concatenate([plane.astype(jnp.float32) for plane in planes])
        return jnp.where(team == 1, out[:, self._opposite], out)


_STEP = 1024
"""What a step costs a route, where entering a watched tile costs 1 more, as in the game."""


def _wide(state: State) -> State:
    """Return a state with its counts as int32, for the rules to count with."""
    wide = {name: jnp.asarray(getattr(state, name)).astype(jnp.int32) for name in _COUNTED}
    return state._replace(**wide)


def _narrow(s: State) -> State:
    """Return a state with every field in the dtype `State` keeps it in: `_wide` undone."""
    return State(*(jnp.asarray(value).astype(_DTYPES[name]) for name, value in s._asdict().items()))


def _select(which: jax.Array, a: State, b: State) -> State:
    """Return state ``a`` where ``which``, else ``b``: each field chosen whole."""
    return jax.tree.map(lambda x, y: jnp.where(which, x, y), a, b)


def _check_moves(near: Patch, steps: int, targets: int) -> None:
    """Raise unless every tile a unit may walk to is a target of its actions.

    ValueError if a walk of ``steps`` tiles may leave the cells an action names.
    """
    beside = np.asarray(near.beside)
    reached = np.zeros(near.size + 1, dtype=bool)
    reached[0] = True
    for _ in range(min(steps, near.size)):
        reached[:-1] |= reached[beside].any(axis=1)
    if steps > near.moves or reached[targets + 1 : -1].any():
        raise ValueError(f"a move of {steps} tiles leaves the cells an action may name")
