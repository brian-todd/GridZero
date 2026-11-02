"""The game of Tactics: its rules as an immutable value, and pure steps between positions.

`Tactics` is an environment in miniature: its fields are the rules, and it holds no position.
``start(seed)`` makes the first `TacticsPosition`, ``legal`` gives the mask of legal actions,
and ``step`` makes the next position from any position and one action. Each step is one
decision: placing a unit, or one action of the acting unit's turn. It is a `game.Game`, so
`game.play` and `game.series` play it.
"""

import functools
from dataclasses import dataclass, field
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents.frame import Frame
from gridzero.core import (
    Context,
    LatticeSpace,
    check_int,
    check_real,
    in_front,
    path,
    reach,
    sight,
    snap,
)
from gridzero.core._arrays import check_cell, read_only
from gridzero.game import Action, check_action, winner

from .board import board_space, homes, make_board
from .events import (
    Attacked,
    Deferred,
    Died,
    Ended,
    Event,
    GameOver,
    Moved,
    Placed,
    RoundStarted,
    Swapped,
    Turned,
)
from .rules import DEFAULT_COUNTS, DEFAULT_UNITS, UnitStats, hit_chance, hits, roll_needed
from .state import TacticsPosition, TacticsState, Verb

_WATCHED = 1e-3
"""What entering a watched tile adds to a route's cost: enough to steer around it among routes of
the fewest steps, too little to make a route longer (moves are at most 126 steps)."""


class Odds(NamedTuple):
    """What an attack would be, as `Tactics.odds` works it out."""

    atk: int
    """The attacker's effective Atk: also the damage of a hit."""
    defense: int
    """The target's effective Def."""
    flank: bool
    """Whether the attack flanks the target, which multiplies Atk."""
    chance: float
    """The chance it hits, 0.05 to 0.95."""


@dataclass(frozen=True)
class Tactics:
    """A two-player skirmish on a hex board: the rules, as an immutable value.

    Each side deploys its units in its home, alternately, then every living unit takes one
    turn per round in a shuffled order. On its turn a unit may move, attack (or defer its
    attack) and turn, in any order, splitting its movement around the attack. A side wins when
    the other has no units left. The example's README gives the rules in full.

    Different rules are a different value: ``dataclasses.replace(game, flank_factor=3)``.

    Parameters
    ----------
    units : tuple of UnitStats, default `rules.DEFAULT_UNITS`
        The kinds of unit, by kind number.
    counts : tuple of int, default `rules.DEFAULT_COUNTS`
        How many of each kind each side deploys.
    shape : tuple of int, default (24, 24)
        The board's axial shape.
    walls, rubble : float, default 0.08 and 0.05
        The shares of walls and rubble on boards that `start` makes (see `board.make_board`).
    home_depth : int, default 5
        Rows of each home. Each home must hold a side's units.
    hit_base : int, default 8
        The d20 roll an attack needs when Atk equals Def.
    hit_step : int, default 2
        How much each point of Atk over Def lowers the roll needed.
    flank_factor : int, default 3
        What flanking multiplies Atk by.
    engaged_bonus : int, default 1
        Atk added against a unit already attacked this round.
    aura_bonus : int, default 1
        Atk and Def added by a friendly aura.
    aura_radius : float, default 3.0
        How far an aura reaches, between centers.
    formation_bonus : int, default 1
        Def added to a unit with `formation` per adjacent friendly unit with it.
    formation_cap : int, default 2
        The most Def that formation adds.
    turn_on_move : bool, default True
        Whether a move turns the unit toward its movement vector.
    free_facing : bool, default True
        Whether a unit may turn freely on its turn (the `Verb.FACE` actions).
    face_center : bool, default True
        Whether units are deployed facing the board's center, rather than direction 0.
    max_rounds : int, default 30
        The game ends once this many rounds have passed.
    decide_at_cap : bool, default True
        Whether a game that reaches ``max_rounds`` goes to the side with more hit points left,
        rather than being a draw. Equal hit points are a draw either way. Without it, a beaten
        side gains a draw by keeping its last units out of reach until the cap.

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a parameter is out of range, ``counts`` doesn't match ``units``, or a home is too
        small for a side's units.
    """

    units: tuple[UnitStats, ...] = DEFAULT_UNITS
    counts: tuple[int, ...] = DEFAULT_COUNTS
    shape: tuple[int, int] = (24, 24)
    walls: float = 0.08
    rubble: float = 0.05
    home_depth: int = 5
    hit_base: int = 8
    hit_step: int = 2
    flank_factor: int = 3
    engaged_bonus: int = 1
    aura_bonus: int = 1
    aura_radius: float = 3.0
    formation_bonus: int = 1
    formation_cap: int = 2
    turn_on_move: bool = True
    free_facing: bool = True
    face_center: bool = True
    max_rounds: int = 30
    decide_at_cap: bool = True
    _last: list[tuple[TacticsPosition, _Turn]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """The last position asked about and its decision: callers ask one position several things."""

    def __post_init__(self) -> None:
        if not isinstance(self.units, tuple) or not self.units:
            raise TypeError("units must be a non-empty tuple of UnitStats")
        for stats in self.units:
            if not isinstance(stats, UnitStats):
                raise TypeError(f"units must hold UnitStats, got {stats!r}")
            if 2 * stats.move > np.iinfo(np.int8).max:
                raise ValueError(f"{stats.name} moves too far to count: {stats.move}")
            if stats.hp > np.iinfo(np.int16).max:
                raise ValueError(f"{stats.name} has too many hit points: {stats.hp}")
        if not isinstance(self.counts, tuple) or len(self.counts) != len(self.units):
            raise ValueError("counts must be a tuple with one count per kind of unit")
        for count in self.counts:
            check_int(count, "each count", minimum=0)
        if not 1 <= 2 * self.team_size <= np.iinfo(np.int8).max:
            raise ValueError(f"each side needs 1 to 63 units, got {self.team_size}")
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise TypeError(f"shape must be a tuple of two ints, got {self.shape!r}")
        for length in self.shape:
            check_int(length, "each shape entry", minimum=2)
        check_real(self.walls, "walls", 0.0, 1.0)
        check_real(self.rubble, "rubble", 0.0, 1.0)
        for name in ("hit_base", "engaged_bonus", "aura_bonus", "formation_bonus"):
            check_int(getattr(self, name), name, minimum=0 if name != "hit_base" else None)
        check_int(self.hit_step, "hit_step", minimum=0)
        check_int(self.flank_factor, "flank_factor", minimum=1)
        check_int(self.formation_cap, "formation_cap", minimum=0)
        check_int(self.max_rounds, "max_rounds", minimum=1)
        check_real(self.aura_radius, "aura_radius", 0.0)
        for flag in ("turn_on_move", "free_facing", "face_center", "decide_at_cap"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
        home = homes(board_space(self.shape), self.home_depth)
        if min(np.count_nonzero(home == 0), np.count_nonzero(home == 1)) < self.team_size:
            raise ValueError(
                f"homes {self.home_depth} rows deep can't hold {self.team_size} units a side"
            )

    @property
    def team_size(self) -> int:
        """How many units each side deploys."""
        return sum(self.counts)

    @property
    def n_players(self) -> int:
        """The number of players: 2, one per team."""
        return 2

    @property
    def n_verbs(self) -> int:
        """The number of verbs: ``Verb.PLACE`` plus one per kind of unit."""
        return Verb.PLACE + len(self.units)

    @property
    def space(self) -> LatticeSpace:
        """The board's hex lattice, shared by every board of this shape (`board.board_space`)."""
        return board_space(self.shape)

    @property
    def planes(self) -> tuple[str, ...]:
        """The names of the planes of `observe`, in order (`agents.frame.Learnable`).

        One name per plane. Planes of units are 1 on the unit's cell; planes of scalars, such
        as ``"round"``, hold the same value on every cell. There are planes per kind of unit.
        """
        kinds = [stats.name for stats in self.units]
        return (
            "wall",
            "rubble",
            "own home",
            "enemy home",
            *(f"own {name}" for name in kinds),
            *(f"enemy {name}" for name in kinds),
            "hp",
            *(f"facing {d}" for d in range(6)),
            "moved",
            "swapped",
            "attacked",
            "done",
            "engaged",
            "watched by own",
            "watched by enemy",
            "active",
            "deploying",
            "round",
            *(f"to deploy {name}" for name in kinds),
        )

    @functools.cached_property
    def _flags(self) -> dict[str, NDArray[np.bool_]]:
        """Each passive by kind, with an extra False at the end, which kind -1 indexes."""
        names = ("formation", "defers", "double_move", "aura")
        return {name: np.array([getattr(s, name) for s in self.units] + [False]) for name in names}

    def start(self, seed: SupportsIndex) -> TacticsPosition:
        """Return the first position of a game: an empty board, before deployment.

        Parameters
        ----------
        seed : int
            The root of every random draw in the game, non-negative. Its stream ``(0,)`` makes
            the board (`board.make_board`).

        Returns
        -------
        TacticsPosition
            Ply 0, round 0: team 0 deploys first.

        Raises
        ------
        TypeError
            If ``seed`` is not an integer.
        ValueError
            If ``seed`` is negative.
        """
        seed = check_int(seed, "seed", minimum=0)
        stream = Context(seed=seed, spawn_key=(0,)).rng
        board = make_board(
            self.shape,
            seed=int(stream.integers(2**63)),
            walls=self.walls,
            rubble=self.rubble,
            home_depth=self.home_depth,
        )
        return TacticsPosition(board, seed)

    def player(self, position: TacticsPosition) -> int:
        """Return the team that decides next.

        Parameters
        ----------
        position : TacticsPosition
            A position of a game not over.

        Returns
        -------
        int
            0 or 1: during deployment the teams alternate, team 0 first, and in battle it is the
            acting unit's team.

        Raises
        ------
        ValueError
            If the game is over.
        """
        turn = self._turn(position)
        if position.over:
            raise ValueError("the game is over")
        return turn.team

    def view(self, position: TacticsPosition, player: int) -> TacticsPosition:
        """Return what a player is shown of a position: all of it, since nothing is hidden.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.
        player : int
            The team looking.

        Returns
        -------
        TacticsPosition
            ``position`` itself.
        """
        return position

    def observe(self, position: TacticsPosition, team: SupportsIndex, /) -> NDArray[np.float32]:
        """Return what a team sees of a position: planes per cell, in its own frame.

        The game has full information, so an observation holds the whole position, laid out for
        networks (`agents.frame.Learnable`): one plane of floats per feature, each a value per
        cell, in the player's frame. Units are "own" and "enemy" rather than team 0 and 1, and
        for team 1 the board is turned half a turn (`LatticeSpace.opposite`, directions turned
        by 3), so the player's home is always team 0's corner. One policy then plays both
        sides.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.
        team : int
            0 or 1: the player.

        Returns
        -------
        ndarray of float32, shape (len(planes), n_cells)
            Values in [0, 1]. ``hp`` is the share of the kind's full hit points left, ``facing
            d`` is one-hot per unit in the player's frame, ``round`` is the round over
            ``max_rounds``, and ``to deploy`` is the share of each kind the player has yet to
            deploy. Cell ``c`` of the planes is cell ``space.opposite()[c]`` of the board for
            team 1, and cell ``c`` for team 0 (`frame`).

        Raises
        ------
        TypeError
            If ``team`` is not an integer.
        ValueError
            If ``team`` is neither 0 nor 1, or the position is not on this game's board.
        """
        team = _check_team(team)
        self.active(position)  # checks the position
        state = position.state
        u = state.units
        n_kinds = len(self.units)
        present = state.occupied
        own = present & (u.team == team)
        enemy = present & (u.team == 1 - team)
        facing = (u.facing + 3 * team) % 6  # the frame turns half a turn for team 1
        full_hp = np.array([stats.hp for stats in self.units] + [1])  # kind -1 indexes the last
        watched_by = np.zeros((2, state.n_cells), dtype=np.bool_)
        archers = np.flatnonzero(present & (u.watch >= 0))
        for archer in archers:
            watched_by[int(u.team[archer] != team), u.watch[archer]] = True
        active = np.zeros(state.n_cells, dtype=np.bool_)
        if (cell := self.active(position)) >= 0:
            active[cell] = True
        deployed = np.array(
            [np.count_nonzero(own & (u.kind == kind)) for kind in range(n_kinds)], dtype=np.float64
        )
        counts = np.array(self.counts, dtype=np.float64)
        left = (counts - deployed) / np.maximum(counts, 1)  # 0 for a kind with no units
        rows = [
            state.wall,
            state.rubble,
            state.home == team,
            state.home == 1 - team,
            *(own & (u.kind == kind) for kind in range(n_kinds)),
            *(enemy & (u.kind == kind) for kind in range(n_kinds)),
            np.where(present, u.hp / full_hp[u.kind], 0.0),
            *(present & (facing == d) for d in range(6)),
            present & (u.moved > 0),
            u.swapped,
            u.attacked,
            u.done,
            u.engaged,
            watched_by[0],
            watched_by[1],
            active,
            np.full(state.n_cells, float(position.deploying)),
            np.full(state.n_cells, position.round / self.max_rounds),
            *(np.full(state.n_cells, left[k]) for k in range(n_kinds)),
        ]
        out = np.stack(rows).astype(np.float32)
        return out[:, state.space.opposite()] if team == 1 else out

    def frame(self, team: SupportsIndex, /) -> Frame:
        """Return a team's frame, where `observe` shows the board to it (`agents.frame.Frame`).

        Team 0 sees the board as it is. Team 1 sees it turned half a turn: cells through
        `LatticeSpace.opposite`, and ``Verb.FACE + d`` as ``Verb.FACE + (d + 3) % 6``. Each is
        its own inverse, so a network's choice in a team's frame goes back by the same map.

        Parameters
        ----------
        team : int
            0 or 1.

        Returns
        -------
        Frame
            The same object on every call.

        Raises
        ------
        TypeError
            If ``team`` is not an integer.
        ValueError
            If ``team`` is neither 0 nor 1.
        """
        return self._frames[_check_team(team)]

    @functools.cached_property
    def _frames(self) -> tuple[Frame, Frame]:
        """The frames of team 0 and team 1."""
        board = Frame.board(self.n_verbs, self.space.n_cells)
        return board, Frame.opposite(self.space, self.n_verbs, directions=Verb.FACE)

    def active(self, position: TacticsPosition) -> int:
        """Return the cell of the unit whose turn it is, or -1 during deployment or after the end.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.

        Returns
        -------
        int
            A cell id, or -1.
        """
        return self._turn(position).active

    def legal(self, position: TacticsPosition) -> NDArray[np.bool_]:
        """Return the mask of legal actions: the actions `step` accepts.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.

        Returns
        -------
        ndarray of bool, shape (n_verbs, n_cells)
            ``mask[verb, cell]`` is True where ``Action(verb, cell)`` is legal. All False once
            the game is over.
        """
        turn = self._turn(position)
        mask = np.zeros((self.n_verbs, position.state.n_cells), dtype=np.bool_)
        if not position.over:  # each phase has its own verbs, and the others are never legal
            verbs = range(Verb.PLACE, self.n_verbs) if position.deploying else range(Verb.PLACE)
            for verb in verbs:
                mask[verb] = turn.targets(verb)
        return mask

    def step(
        self, position: TacticsPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> TacticsPosition:
        """Return the position after one action. The position given is unchanged.

        Random draws (the order of turns, and the d20 of each attack) come from a stream of the
        position's seed keyed by its ply, so the same position and action always give the same
        result.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game, not over.
        action : Action or tuple of int
            ``(verb, cell)``, legal in ``position``.

        Returns
        -------
        TacticsPosition
            The next position, one ply later.

        Raises
        ------
        TypeError
            If the verb or the cell is not an integer.
        ValueError
            If the game is over, the verb is unknown or the action is illegal.
        IndexError
            If the cell is not a cell id.
        """
        turn = self._turn(position)
        verb, cell = check_action(position, action, self.n_verbs)
        if not turn.targets(verb)[cell]:
            raise ValueError(f"illegal action {Action(verb, cell)} at ply {position.ply}")
        state = position.state
        work = _Work(state)
        dice = Context(seed=position.seed, spawn_key=(1, position.ply))
        if position.deploying:
            return self._deploy(position, work, verb - Verb.PLACE, cell, dice)
        a = turn.active
        before = int(work.uid[a])
        if verb == Verb.MOVE:
            self._move(work, state, a, cell, dice)
        elif verb == Verb.ATTACK:
            work.attacked[a] = True
            work.faced[a] = False
            self._attack(work, state.space, a, cell, dice, deferred=False)
        elif verb == Verb.DEFER:
            work.watch[a] = cell
            work.attacked[a] = True
            work.faced[a] = False
            work.events.append(Deferred(before, cell))
        elif verb == Verb.SWAP:
            self._swap(work, state, a, cell, dice)
        elif verb == Verb.END:
            work.done[a] = True
            work.events.append(Ended(before))
        else:  # Verb.FACE + d
            work.facing[a] = verb - Verb.FACE
            work.faced[a] = True
            work.events.append(Turned(before, verb - Verb.FACE))
        return self._next(position, work, dice, before)

    def odds(
        self,
        position: TacticsPosition,
        attacker: SupportsIndex,
        target: SupportsIndex,
        at: SupportsIndex | None = None,
    ) -> Odds:
        """Return what an attack between two units would be, without rolling.

        It counts every bonus as an attack would, whether or not the attack is legal: auras,
        formation, flanking and engagement.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.
        attacker, target : int
            The cells of two units.
        at : int, optional
            An empty cell to attack from instead, as if the attacker had moved there first:
            what a move then an attack would give, when the move turns no one else.

        Returns
        -------
        Odds
            Effective Atk and Def, whether it flanks, and the chance of a hit.

        Raises
        ------
        ValueError
            If either cell holds no unit, or ``at`` holds one other than the attacker.
        IndexError
            If a cell is not a cell id.
        """
        state = position.state
        a = check_cell(attacker, state.n_cells)
        t = check_cell(target, state.n_cells)
        u = state.units
        if u.kind[a] < 0 or u.kind[t] < 0:
            raise ValueError(f"cells {a} and {t} must both hold units")
        kind, team, facing, engaged = u.kind, u.team, u.facing, u.engaged
        if at is not None and (c := check_cell(at, state.n_cells)) != a:
            if u.kind[c] >= 0:
                raise ValueError(f"cell {c} must be empty to attack from it")
            kind, team, facing, engaged = kind.copy(), team.copy(), facing.copy(), engaged.copy()
            kind[c], team[c], facing[c], engaged[c] = kind[a], team[a], facing[a], engaged[a]
            kind[a], team[a] = -1, -1
            a = c
        atk, defense, flank = self._stats(state.space, kind, team, facing, engaged, a, t)
        return Odds(atk, defense, flank, hit_chance(atk, defense, self.hit_base, self.hit_step))

    # Steps.

    def _deploy(
        self, position: TacticsPosition, work: _Work, kind: int, cell: int, dice: Context
    ) -> TacticsPosition:
        """Place a unit for the team deploying, and start the battle after the last one."""
        state = position.state
        placed = int(np.count_nonzero(work.kind >= 0))
        work.uid[cell] = placed
        work.kind[cell] = kind
        work.team[cell] = placed % 2
        work.hp[cell] = self.units[kind].hp
        work.facing[cell] = self._first_facing(state.space, cell)
        work.events.append(Placed(placed, kind, placed % 2, cell))
        if placed + 1 < 2 * self.team_size:
            return work.position(position, position.ply + 1, drew=dice.drew)
        self._new_round(work, dice, 1)
        return work.position(position, position.ply + 1, round=1, drew=dice.drew)

    def _at_cap(self, work: _Work, alive: NDArray[np.bool_]) -> int | None:
        """Return the winner when the rounds run out: more hit points left, or None for a draw."""
        if not self.decide_at_cap:
            return None
        return winner([int(work.hp[alive & (work.team == team)].sum()) for team in (0, 1)])

    def _next(
        self, position: TacticsPosition, work: _Work, dice: Context, before: int
    ) -> TacticsPosition:
        """Finish a step: the end of the game, of the round, or of the acting unit's turn."""
        ply, round_ = position.ply + 1, position.round
        alive = work.kind >= 0
        sides = [bool(np.any(alive & (work.team == team))) for team in (0, 1)]
        if not all(sides):
            winner = 0 if sides[0] else 1
            work.events.append(GameOver(winner))
            return work.position(position, ply, round_, winner=winner, over=True, drew=dice.drew)
        waiting = alive & ~work.done
        if not waiting.any():
            round_ += 1
            if round_ > self.max_rounds:
                winner = self._at_cap(work, alive)
                work.events.append(GameOver(winner))
                return work.position(
                    position, ply, self.max_rounds, winner=winner, over=True, drew=dice.drew
                )
            self._new_round(work, dice, round_)
        else:
            nxt = _first(work.initiative, waiting)
            if work.uid[nxt] != before:  # a new turn begins
                work.watch[nxt] = -1  # a deferred attack lapses at its archer's next turn
                work.faced[nxt] = False
        return work.position(position, ply, round_, drew=dice.drew)

    def _new_round(self, work: _Work, dice: Context, round_: int) -> None:
        """Reset what lasts a round, and shuffle the order of turns of the living units."""
        alive = work.kind >= 0
        for flags in (work.done, work.swapped, work.attacked, work.faced, work.engaged):
            flags[:] = False
        work.moved[:] = 0
        work.initiative[:] = 0
        work.initiative[alive] = dice.rng.permutation(np.count_nonzero(alive))
        work.watch[_first(work.initiative, alive)] = -1  # the first turn begins
        work.events.append(RoundStarted(round_))

    def _move(self, work: _Work, state: TacticsState, a: int, goal: int, dice: Context) -> None:
        """Walk the acting unit to ``goal``, tile by tile, springing deferred attacks."""
        space = state.space
        cost = np.ones(space.n_cells)
        enemy = (work.kind >= 0) & (work.team != work.team[a]) & (work.watch >= 0)
        cost[work.watch[enemy]] += _WATCHED
        route = path(space, a, goal, cost=cost, blocked=~state.passable | (work.kind >= 0))
        coords = space.unravel(route)
        facing = int(work.facing[a])
        uid = int(work.uid[a])
        start = 0  # where the piece of the route not yet recorded starts
        for k in range(1, route.size):
            here, there = int(route[k - 1]), int(route[k])
            work.units.move(here, there)
            work.moved[there] += 1
            if self.turn_on_move:
                work.facing[there] = snap(space, coords[k] - coords[k - 1], work.facing[there])
            if self._watched(work, there):
                work.events.append(Moved(uid, tuple(int(c) for c in route[start : k + 1])))
                start = k
                if not self._spring(work, state, there, dice):
                    return  # the mover died
        if start < route.size - 1:
            work.events.append(Moved(uid, tuple(int(c) for c in route[start:])))
        if self.turn_on_move:
            work.facing[goal] = snap(space, coords[-1] - coords[0], facing)
        work.faced[goal] = False

    def _swap(self, work: _Work, state: TacticsState, a: int, b: int, dice: Context) -> None:
        """Swap the acting unit at ``a`` with its partner at ``b``; both use up their movement."""
        coords = state.space.unravel(np.array([a, b]))
        direction = snap(state.space, coords[1] - coords[0], work.facing[a])
        work.events.append(Swapped(int(work.uid[a]), int(work.uid[b]), a, b))
        work.units.swap(a, b)
        work.swapped[[a, b]] = True
        if self.turn_on_move:
            work.facing[b] = direction
            work.facing[a] = (direction + 3) % 6
        work.faced[b] = False
        self._spring(work, state, b, dice)  # the acting unit enters first
        if work.kind[a] >= 0:
            self._spring(work, state, a, dice)

    def _spring(self, work: _Work, state: TacticsState, cell: int, dice: Context) -> bool:
        """Fire the enemy archers' attacks deferred onto ``cell``; return whether its unit lives.

        Archers fire in order of cell id. Each fires once, if the tile is still within its reach
        and sight. One out of reach keeps its designation until its next turn.
        """
        for archer in self._watchers(work, cell):
            radius = self.units[work.kind[archer]].reach
            if sight(state.space, archer, state.wall, radius=radius)[cell]:
                work.watch[archer] = -1
                self._attack(work, state.space, int(archer), cell, dice, deferred=True)
                if work.kind[cell] < 0:
                    return False
        return True

    def _watchers(self, work: _Work, cell: int) -> NDArray[np.intp]:
        """Return the cells of the enemy archers watching ``cell``, in order of id."""
        team = work.team[cell]
        return np.flatnonzero((work.watch == cell) & (work.kind >= 0) & (work.team != team))

    def _watched(self, work: _Work, cell: int) -> bool:
        """Return whether enemies of the unit on ``cell`` watch it."""
        return self._watchers(work, cell).size > 0

    def _attack(
        self,
        work: _Work,
        space: LatticeSpace,
        attacker: int,
        target: int,
        dice: Context,
        *,
        deferred: bool,
    ) -> None:
        """Roll one attack, engage the target, and remove it if it dies."""
        atk, defense, flank = self._stats(
            space, work.kind, work.team, work.facing, work.engaged, attacker, target
        )
        roll = int(dice.rng.integers(1, 21))  # a d20
        needed = roll_needed(atk, defense, self.hit_base, self.hit_step)
        hit = hits(roll, needed)
        work.engaged[target] = True
        hp = int(work.hp[target])
        damage = min(atk, hp) if hit else 0
        work.events.append(
            Attacked(
                attacker=int(work.uid[attacker]),
                attacker_kind=int(work.kind[attacker]),
                attacker_team=int(work.team[attacker]),
                target=int(work.uid[target]),
                target_kind=int(work.kind[target]),
                cell=target,
                roll=roll,
                needed=needed,
                atk=atk,
                defense=defense,
                flank=flank,
                hit=hit,
                damage=damage,
                deferred=deferred,
            )
        )
        if hit and damage == hp:
            uid, kind, team = int(work.uid[target]), int(work.kind[target]), int(work.team[target])
            work.units.clear(target)
            work.events.append(Died(uid, kind, team, target))
        elif hit:
            work.hp[target] = hp - damage

    # Rules about units.

    def _stats(
        self,
        space: LatticeSpace,
        kind: NDArray[np.int8],
        team: NDArray[np.int8],
        facing: NDArray[np.int8],
        engaged: NDArray[np.bool_],
        attacker: int,
        target: int,
    ) -> tuple[int, int, bool]:
        """Return the effective Atk, the effective Def and whether the attack flanks."""
        near = _near(space, target, 1.0)
        enemies = np.count_nonzero((kind[near] >= 0) & (team[near] == team[attacker]))
        coords = space.unravel(np.array([attacker, target]))
        behind = not bool(in_front(space, facing[target], coords[0] - coords[1]))
        flank = behind or bool(enemies >= 2)
        atk = self.units[kind[attacker]].atk + self._aura(space, kind, team, attacker)
        atk += self.engaged_bonus * bool(engaged[target])
        if flank:
            atk *= self.flank_factor
        defense = self.units[kind[target]].defense + self._aura(space, kind, team, target)
        if self.units[kind[target]].formation:
            friends = (team[near] == team[target]) & self._flags["formation"][kind[near]]
            defense += min(self.formation_cap, self.formation_bonus * np.count_nonzero(friends))
        return int(atk), int(defense), flank

    def _aura(
        self, space: LatticeSpace, kind: NDArray[np.int8], team: NDArray[np.int8], cell: int
    ) -> int:
        """Return the aura bonus of the unit at ``cell``: given by a friend within reach, or 0."""
        near = np.append(_near(space, cell, self.aura_radius), cell)
        givers = self._flags["aura"][kind[near]] & (team[near] == team[cell])
        return self.aura_bonus if givers.any() else 0

    def _first_facing(self, space: LatticeSpace, cell: int) -> int:
        """Return the facing of a unit deployed on ``cell``: toward the center, or direction 0."""
        if not self.face_center:
            return 0
        q, r = space.unravel(cell)
        toward = np.array([space.shape[0] - 1 - 2 * q, space.shape[1] - 1 - 2 * r])
        return snap(space, toward, 0)  # twice the vector to the center, in integers

    # Checks.

    def _check_space(self, state: TacticsState) -> None:
        """Raise unless the state lies on a hex board of this game's shape.

        TypeError if its space is not a lattice, ValueError if the lattice is not such a board.
        """
        space = state.space
        if not (space.tiling == "hex" and not any(space.periodic) and space.shape == self.shape):
            raise ValueError(
                f"the board must be a hex lattice of shape {self.shape} that doesn't wrap"
            )

    def _turn(self, position: TacticsPosition) -> _Turn:
        """Check a position and return what its decision is about."""
        if self._last and self._last[0][0] is position:
            return self._last[0][1]
        if not isinstance(position, TacticsPosition):
            raise TypeError(f"position must be a TacticsPosition, got {type(position).__name__}")
        self._check_space(position.state)
        turn = _Turn(self, position)
        self._last[:] = [(position, turn)]  # holds the position, so `is` can't match another
        return turn


def _near(space: LatticeSpace, cell: int, radius: float) -> NDArray[np.int64]:
    """Return the cells within ``radius`` of ``cell``, as `within` finds them, but not ``cell``.

    Looked up in the graph the space keeps for the radius: shape (n,), int64, ascending.
    """
    graph = space.radius_graph(radius)
    return graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]]


def _first(initiative: NDArray[np.int16], mask: NDArray[np.bool_]) -> int:
    """Return the cell in ``mask`` with the lowest initiative."""
    cells = np.flatnonzero(mask)
    return int(cells[np.argmin(initiative[cells])])


class _Turn:
    """What a position's decision is about: who decides, and which targets each verb has."""

    def __init__(self, game: Tactics, position: TacticsPosition) -> None:
        self.game = game
        self.position = position
        self.state = state = position.state
        units = state.units
        self.none = read_only(np.zeros(state.n_cells, dtype=np.bool_))  # returned as it is
        if position.deploying:
            self.active = -1
            self.team = int(np.count_nonzero(state.occupied)) % 2
        elif position.over:
            self.active, self.team = -1, -1
        else:
            self.active = _first(units.initiative, state.occupied & ~units.done)
            self.team = int(units.team[self.active])

    @functools.cached_property
    def stats(self) -> UnitStats:
        """The acting unit's stats."""
        return self.game.units[self.state.units.kind[self.active]]

    @functools.cached_property
    def in_range(self) -> NDArray[np.bool_]:
        """The cells within the acting unit's reach and sight."""
        space, a, reach = self.state.space, self.active, self.stats.reach
        if reach < 1.5:  # at most the six neighbors, which always see each other
            mask = self.none.copy()
            mask[_near(space, a, reach)] = mask[a] = True
            return mask
        return sight(space, a, self.state.wall, radius=reach)

    def movement_left(self) -> int:
        """Return the tiles the acting unit may still move this turn."""
        units, a, stats = self.state.units, self.active, self.stats
        if units.swapped[a]:
            return 0
        double = stats.double_move and not units.attacked[a]
        return (2 if double else 1) * stats.move - int(units.moved[a])

    def can_attack(self) -> bool:
        """Whether the acting unit may still attack or defer its attack."""
        units, a, stats = self.state.units, self.active, self.stats
        return not units.attacked[a] and not (stats.double_move and units.moved[a] > stats.move)

    def targets(self, verb: int) -> NDArray[np.bool_]:
        """Return where an action with ``verb`` is legal: shape (n_cells,), bool."""
        game, state, a = self.game, self.state, self.active
        if self.position.over or not 0 <= verb < game.n_verbs:
            return self.none
        if self.position.deploying:
            kind = verb - Verb.PLACE
            if kind < 0:
                return self.none
            units = state.units
            deployed = np.count_nonzero((units.kind == kind) & (units.team == self.team))
            if deployed >= game.counts[kind]:
                return self.none
            return (state.home == self.team) & state.passable & ~state.occupied
        units = state.units
        mask = self.none.copy()
        if verb == Verb.MOVE:
            left = self.movement_left()
            if left > 0:
                blocked = ~state.passable | state.occupied
                mask = reach(state.space, a, left, blocked=blocked)
                mask[a] = False
        elif verb == Verb.ATTACK:
            if self.can_attack():
                mask = self.in_range & state.occupied & (units.team != self.team)
        elif verb == Verb.DEFER:
            if self.stats.defers and self.can_attack():
                mask = self.in_range & state.passable
                mask[a] = False
        elif verb == Verb.SWAP:
            if units.moved[a] == 0 and not units.swapped[a]:
                near = _near(state.space, a, 1.0)
                ok = (units.team[near] == self.team) & (units.moved[near] == 0)
                mask[near] = ok & ~units.swapped[near]
        elif verb == Verb.END:
            mask[a] = True
        elif verb < Verb.END:  # Verb.FACE + d
            if game.free_facing and not units.faced[a] and units.facing[a] != verb - Verb.FACE:
                mask[a] = True
        return mask


class _Work:
    """A step's writable copy of the fields of `Unit` (`core.State.edit`), and its events.

    Each field's array is an attribute of the same name, for the rules to write in place;
    ``units`` moves, swaps and clears whole units, and `position` freezes them.
    """

    def __init__(self, state: TacticsState) -> None:
        U = TacticsState.units
        self.units = state.edit(U)
        units = self.units
        self.uid = units[U.uid]
        self.kind = units[U.kind]
        self.team = units[U.team]
        self.hp = units[U.hp]
        self.facing = units[U.facing]
        self.initiative = units[U.initiative]
        self.moved = units[U.moved]
        self.swapped = units[U.swapped]
        self.attacked = units[U.attacked]
        self.faced = units[U.faced]
        self.done = units[U.done]
        self.engaged = units[U.engaged]
        self.watch = units[U.watch]
        self.events: list[Event] = []

    def position(
        self,
        before: TacticsPosition,
        ply: int,
        round: int = 0,
        winner: int | None = None,
        over: bool = False,
        *,
        drew: bool,
    ) -> TacticsPosition:
        """Return the position these values make after ``before``, with the step's events."""
        return TacticsPosition(
            self.units.freeze(),
            before.seed,
            ply=ply,
            over=over,
            winner=winner,
            drew=drew,
            events=tuple(self.events),
            round=round,
        )


def _check_team(team: SupportsIndex) -> int:
    """Return ``team`` as an int, checked to be 0 or 1."""
    team = check_int(team, "team")
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")
    return team
