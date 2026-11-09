"""The game of Tactics: its rules as an immutable value, and pure steps between positions.

`Tactics` is an environment in miniature: its fields are the rules, and it holds no position.
``start(seed)`` makes the first `TacticsPosition`, ``legal`` gives the mask of legal actions,
and ``step`` makes the next position from any position and one action. Each step is one
decision: placing a unit, or one action of the acting unit's turn. It is a `game.Game`, so
`game.play` and `game.series` play it.

Every search spends most of its time here and in the players, so these modules write NumPy for
speed: ``mask.nonzero()[0]`` for ``np.flatnonzero(mask)`` on a 1-D mask, and methods such as
``x.any()`` for ``np.any(x)``, which goes through a Python wrapper. What depends only on the space
or an offset is cached (`_near_graph`, `_snap`), and what a decision needs twice is worked out once
(`_Turn.targets`). None of this changes a rule, and ``test_play_unchanged`` pins that it doesn't
change a game.
"""

import functools
from dataclasses import dataclass, field
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents.frame import Frame
from gridzero.core import (
    Context,
    Field,
    LatticeSpace,
    check_int,
    check_real,
    hops,
    in_front,
    path,
    reach,
    sight,
    snap,
)
from gridzero.core._arrays import check_cell, read_only
from gridzero.game import Action, check_action, joined

from . import objective
from .board import board_space, homes, make_board, rings
from .events import (
    Attacked,
    Broke,
    Built,
    Captured,
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
from .rules import DEFAULT_CAPS, DEFAULT_UNITS, UnitStats, hit_chance, hits, roll_needed
from .state import Hold, TacticsPosition, TacticsState, Verb

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
    struck: float = 0.0
    """The hit points the attacker can expect to lose first, to a unit bracing against a charge."""


@dataclass(frozen=True)
class Tactics:
    """A two-player skirmish on a hex board: the rules, as an immutable value.

    Each side picks and deploys its units in its home, alternately, then every living unit takes
    one turn per round, in an order rolled once when the battle starts. On its turn a unit may
    move, attack (or defer its attack, or work on a tile) and turn, in any order, splitting its
    movement around the attack.
    A side wins when the other has no units left, or by holding the point at the center
    (`objective`). The example's README gives the rules in full.

    Different rules are a different value: ``dataclasses.replace(game, flank_factor=3)``.

    Parameters
    ----------
    units : tuple of UnitStats, default `rules.DEFAULT_UNITS`
        The kinds of unit, by kind number.
    caps : tuple of int, default `rules.DEFAULT_CAPS`
        The most of each kind a side may field.
    team_size : int, default 8
        How many places each side fills, kinds chosen as it places them: units take places by
        `UnitStats.places_for`. A side places until the next unit of no kind it may still field
        fits in what it has left. At most what the caps can field.
    shape : tuple of int, default (25, 25)
        The board's axial shape, both odd so that it has a center for the point.
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
        Atk added against an engaged unit: one attacked since its last turn began (or this
        round, see `engaged_until_turn`).
    aura_bonus : int, default 1
        Atk and Def added by a friendly aura.
    aura_radius : float, default 3.0
        How far an aura reaches, between centers.
    overwatch_bonus : int, default 1
        Atk added to a deferred attack after flanking multiplies it, so a lone flanked arrow can't
        kill a full-health captain.
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
    shuffle_each_round : bool, default False
        Whether the order of turns is shuffled at the start of every round, rather than rolled
        once when the battle starts. Rolled once, the teams alternate, the team that starts by a
        coin, each team's units in a random order; the side with more units takes the last
        turns, and the dead drop out. Shuffled, every living unit gets a random place.
    engaged_until_turn : bool, default True
        Whether a unit stays engaged until its next turn begins, rather than until the round
        ends.
    max_rounds : int, default 20
        The game ends with this round, and the point's owner wins.
    hold_rounds : int, default 4
        Rounds in a row a side must own the point to win.
    capture_by : int, default 10
        If nobody has captured the point by the end of this round, team 1 loses. At most
        ``max_rounds``, so that the point has an owner at the end of the last round.

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    ValueError
        If a parameter is out of range, ``caps`` doesn't match ``units`` or can't fill a team, or
        a home is too small for a side's units.
    """

    units: tuple[UnitStats, ...] = DEFAULT_UNITS
    caps: tuple[int, ...] = DEFAULT_CAPS
    team_size: int = 8
    shape: tuple[int, int] = (25, 25)
    walls: float = 0.08
    rubble: float = 0.05
    home_depth: int = 5
    hit_base: int = 8
    hit_step: int = 2
    flank_factor: int = 3
    engaged_bonus: int = 1
    aura_bonus: int = 1
    aura_radius: float = 3.0
    overwatch_bonus: int = 1
    formation_bonus: int = 1
    formation_cap: int = 2
    turn_on_move: bool = True
    free_facing: bool = True
    face_center: bool = True
    shuffle_each_round: bool = False
    engaged_until_turn: bool = True
    max_rounds: int = 20
    hold_rounds: int = 4
    capture_by: int = 10
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
        if not isinstance(self.caps, tuple):
            raise TypeError(f"caps must be a tuple of ints, got {self.caps!r}")
        if len(self.caps) != len(self.units):
            raise ValueError(f"caps must have one cap per kind of unit, got {self.caps}")
        for cap in self.caps:
            check_int(cap, "each cap", minimum=0)
        check_int(self.team_size, "team_size")
        if not 1 <= 2 * self.team_size <= np.iinfo(np.int8).max:
            raise ValueError(f"each side needs 1 to 63 units, got {self.team_size}")
        most = sum(s.places_for(c) for c, s in zip(self.caps, self.units, strict=True))
        if most < self.team_size:
            raise ValueError(f"caps {self.caps} can't fill a team of {self.team_size} places")
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise TypeError(f"shape must be a tuple of two ints, got {self.shape!r}")
        for length in self.shape:
            check_int(length, "each shape entry", minimum=3)
            if length % 2 == 0:
                raise ValueError(f"shape must be odd, to have a center, got {self.shape}")
        check_real(self.walls, "walls", 0.0, 1.0)
        check_real(self.rubble, "rubble", 0.0, 1.0)
        for name in (
            "hit_base",
            "engaged_bonus",
            "aura_bonus",
            "overwatch_bonus",
            "formation_bonus",
        ):
            check_int(getattr(self, name), name, minimum=0 if name != "hit_base" else None)
        check_int(self.hit_step, "hit_step", minimum=0)
        check_int(self.flank_factor, "flank_factor", minimum=1)
        check_int(self.formation_cap, "formation_cap", minimum=0)
        check_int(self.max_rounds, "max_rounds", minimum=1)
        check_int(self.hold_rounds, "hold_rounds", minimum=1)
        check_int(self.capture_by, "capture_by", minimum=1)
        if self.capture_by > self.max_rounds:
            raise ValueError(
                f"capture_by must be at most max_rounds, got {self.capture_by} and "
                f"{self.max_rounds}: the point must have an owner at the end"
            )
        check_real(self.aura_radius, "aura_radius", 0.0)
        for flag in (
            "turn_on_move",
            "free_facing",
            "face_center",
            "shuffle_each_round",
            "engaged_until_turn",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
        home = homes(board_space(self.shape), self.home_depth)
        if min(np.count_nonzero(home == 0), np.count_nonzero(home == 1)) < self.team_size:
            raise ValueError(
                f"homes {self.home_depth} rows deep can't hold {self.team_size} units a side"
            )

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
            "order",
            "moved its move",
            "deploying",
            "round",
            *(f"to deploy {name}" for name in kinds),
            "to place",
            "point",
            "own point",
            "enemy point",
            "clock",
            "loses the deadline",
        )

    @functools.cached_property
    def _flags(self) -> dict[str, NDArray[np.bool_]]:
        """Each passive by kind, with an extra False at the end, which kind -1 indexes."""
        names = ("formation", "defers", "double_move", "aura", "braces")
        return {name: np.array([getattr(s, name) for s in self.units] + [False]) for name in names}

    def _fielded(
        self, kind: NDArray[np.int8], team: NDArray[np.int8], side: int
    ) -> tuple[NDArray[np.int64], int, NDArray[np.int64]]:
        """Return a side's units by kind, its places left, and the places each kind's next takes.

        Arrays of shape (n_kinds,), int64.
        """
        have = np.bincount(kind[(kind >= 0) & (team == side)], minlength=len(self.units))
        spent = sum(s.places_for(int(n)) for s, n in zip(self.units, have, strict=True))
        cost = np.array(
            [
                s.places_for(int(n) + 1) - s.places_for(int(n))
                for s, n in zip(self.units, have, strict=True)
            ]
        )
        return have, self.team_size - spent, cost

    def _placer(self, kind: NDArray[np.int8], team: NDArray[np.int8]) -> int:
        """Return the team that places next, or -1 once neither can.

        The team with fewer units placed, team 0 on a tie, among those that may still place a
        unit: a kind under its cap whose next unit fits in the places left.
        """
        caps = np.array(self.caps)
        best, fewest = -1, 0
        for side in (0, 1):
            have, left, cost = self._fielded(kind, team, side)
            if ((have < caps) & (cost <= left)).any() and (best < 0 or have.sum() < fewest):
                best, fewest = side, int(have.sum())
        return best

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
            0 or 1: during deployment the team with fewer units placed, team 0 on a tie, among
            those with room left, and in battle it is the
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
            d`` is one-hot per unit in the player's frame, ``order`` is a unit's place in the
            order of turns over the living units (``(k + 1) / n``), ``moved its move`` marks a
            unit that has moved at least its kind's ``move`` this round, ``round`` is the round over
            ``max_rounds``, ``to deploy`` is the share of each kind's cap the player may still
            deploy, ``to place`` is the share of its team it has yet to place, ``own point`` and
            ``enemy point`` fill their planes when that side owns the point, and ``clock`` is the
            owner's rounds in a row over ``hold_rounds``, at most 1. ``loses the deadline`` fills
            its plane for the seat that does: the frame hides the seat, and the deadline tells
            them apart. Cell ``c`` of
            the planes is cell ``space.opposite()[c]`` of the board for team 1, and cell ``c``
            for team 0 (`frame`).

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
        archers = (present & (u.watch >= 0)).nonzero()[0]
        for archer in archers:
            watched_by[int(u.team[archer] != team), u.watch[archer]] = True
        active = np.zeros(state.n_cells, dtype=np.bool_)
        if (cell := self.active(position)) >= 0:
            active[cell] = True
        order = np.zeros(state.n_cells)
        living = present.nonzero()[0]
        ranks = np.argsort(np.argsort(u.initiative[living], kind="stable"), kind="stable")
        order[living] = (ranks + 1) / max(1, living.size)
        moves = np.array([stats.move for stats in self.units] + [0])
        deployed = np.array(
            [np.count_nonzero(own & (u.kind == kind)) for kind in range(n_kinds)], dtype=np.float64
        )
        caps = np.array(self.caps, dtype=np.float64)
        left_places = self._fielded(u.kind, u.team, team)[1]
        to_place = left_places / self.team_size if position.deploying else 0.0
        # 0 for a kind with no units, once its team is full, and in battle
        left = (caps - deployed) / np.maximum(caps, 1) * (to_place > 0)
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
            order * (not position.deploying),
            present & (u.moved >= np.maximum(moves[u.kind], 1)),
            np.full(state.n_cells, float(position.deploying)),
            np.full(state.n_cells, position.round / self.max_rounds),
            *(np.full(state.n_cells, left[k]) for k in range(n_kinds)),
            np.full(state.n_cells, to_place),
            state.point,
            np.full(state.n_cells, position.hold.owner == team),
            np.full(state.n_cells, position.hold.owner == 1 - team),
            np.full(state.n_cells, min(1.0, position.hold.clock / self.hold_rounds)),
            np.full(state.n_cells, team == objective.DEADLINE_LOSER),
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
                targets = turn.targets(verb)
                if targets is not turn.none:
                    mask[verb] = targets
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
        work = _Work(state, position.hold)
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
        elif verb in (Verb.BUILD, Verb.BREAK):
            work.attacked[a] = True
            work.faced[a] = False
            work.change_terrain(state, before, cell, build=verb == Verb.BUILD)
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
        *,
        deferred: bool = False,
    ) -> Odds:
        """Return what an attack between two units would be, without rolling.

        It counts every bonus as an attack would, whether or not the attack is legal: auras,
        formation, flanking, engagement, overwatch, and a charge or the bracing that stops it.

        Parameters
        ----------
        position : TacticsPosition
            A position of this game.
        attacker, target : int
            The cells of two units.
        at : int, optional
            An empty cell to attack from instead, as if the attacker had moved there first:
            what a move then an attack would give, when the move turns no one else.
        deferred : bool, default False
            Whether it is a deferred attack, which gets ``overwatch_bonus``.

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
        a = first = check_cell(attacker, state.n_cells)
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
        walked = 0
        if self.units[kind[a]].charge:  # the tiles it would have moved, on the shortest way there
            walked = int(u.moved[first]) + int(hops(state.space, np.array([a]), first)[0])
        atk, defense, flank, braced = self._stats(
            state.space, kind, team, facing, engaged, a, t, deferred=deferred, walked=walked
        )
        chance = hit_chance(atk, defense, self.hit_base, self.hit_step)
        struck = 0.0
        if braced:
            back, guard, _, _ = self._stats(
                state.space, kind, team, facing, engaged, t, a, deferred=False, strike=True
            )
            hp = int(u.hp[first])
            struck = hit_chance(back, guard, self.hit_base, self.hit_step) * min(back, hp)
        return Odds(atk, defense, flank, chance, struck)

    # Steps.

    def _deploy(
        self, position: TacticsPosition, work: _Work, kind: int, cell: int, dice: Context
    ) -> TacticsPosition:
        """Place a unit for the team deploying, and start the battle after the last one."""
        state = position.state
        placed = int(np.count_nonzero(work.kind >= 0))
        team = self._placer(work.kind, work.team)
        work.uid[cell] = placed
        work.kind[cell] = kind
        work.team[cell] = team
        work.hp[cell] = self.units[kind].hp
        work.facing[cell] = self._first_facing(state.space, cell)
        work.events.append(Placed(placed, kind, team, cell))
        if self._placer(work.kind, work.team) >= 0:
            return work.position(position, position.ply + 1, drew=dice.drew)
        self._new_round(work, dice, 1)
        return work.position(position, position.ply + 1, round=1, drew=dice.drew)

    def _next(
        self, position: TacticsPosition, work: _Work, dice: Context, before: int
    ) -> TacticsPosition:
        """Finish a step: the end of the game, of the round, or of the acting unit's turn."""
        ply, round_ = position.ply + 1, position.round
        alive = work.kind >= 0
        sides = [bool((alive & (work.team == team)).any()) for team in (0, 1)]
        if not all(sides):
            winner = 0 if sides[0] else 1
            work.events.append(GameOver(winner, "killed"))
            return work.position(position, ply, round_, winner=winner, over=True, drew=dice.drew)
        waiting = alive & ~work.done
        if not waiting.any():
            ended = self._end_round(position, work, round_)
            if ended is not None:
                return work.position(position, ply, round_, winner=ended, over=True, drew=dice.drew)
            round_ += 1
            self._new_round(work, dice, round_)
        else:
            nxt = _first(work.initiative, waiting)
            if work.uid[nxt] != before:  # a new turn begins
                work.watch[nxt] = -1  # a deferred attack lapses at its archer's next turn
                work.faced[nxt] = False
                if self.engaged_until_turn:
                    work.engaged[nxt] = False
        return work.position(position, ply, round_, drew=dice.drew)

    def _end_round(self, position: TacticsPosition, work: _Work, round_: int) -> int | None:
        """Apply the objective at a round's end: return the winner if the game ends with it."""
        before = position.hold
        work.hold = objective.after_round(before, position.state.point, work.team)
        if work.hold.owner != before.owner:
            work.events.append(Captured(work.hold.owner, round_, before.owner >= 0))
        result = objective.ending(
            work.hold,
            round_,
            hold_rounds=self.hold_rounds,
            capture_by=self.capture_by,
            max_rounds=self.max_rounds,
        )
        if result is None:
            return None
        work.events.append(GameOver(*result))
        return result[0]

    def _new_round(self, work: _Work, dice: Context, round_: int) -> None:
        """Reset what lasts a round, and roll the order of turns (`shuffle_each_round`)."""
        alive = work.kind >= 0
        for flags in (work.done, work.swapped, work.attacked, work.faced):
            flags[:] = False
        if not self.engaged_until_turn:
            work.engaged[:] = False
        work.moved[:] = 0
        if self.shuffle_each_round:
            work.initiative[:] = 0
            work.initiative[alive] = dice.rng.permutation(np.count_nonzero(alive))
        elif round_ == 1:
            first = int(dice.rng.integers(2))
            lines = [dice.rng.permutation((alive & (work.team == t)).nonzero()[0]) for t in (0, 1)]
            lines = [lines[first], lines[1 - first]]
            order = [
                line[k] for k in range(max(map(len, lines))) for line in lines if k < len(line)
            ]
            work.initiative[:] = 0
            work.initiative[np.array(order)] = np.arange(len(order))
        starts = _first(work.initiative, alive)  # the first turn begins
        work.watch[starts] = -1
        if self.engaged_until_turn:
            work.engaged[starts] = False
        work.events.append(RoundStarted(round_))

    def _move(self, work: _Work, state: TacticsState, a: int, goal: int, dice: Context) -> None:
        """Walk the acting unit to ``goal``, tile by tile, springing deferred attacks."""
        space = state.space
        cost = None
        enemy = (work.kind >= 0) & (work.team != work.team[a]) & (work.watch >= 0)
        if enemy.any():
            cost = np.ones(space.n_cells)
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
                dq, dr = (coords[k] - coords[k - 1]).tolist()
                work.facing[there] = _snap(space, dq, dr, int(work.facing[there]))
            if self._watched(work, there):
                work.events.append(Moved(uid, tuple(int(c) for c in route[start : k + 1])))
                start = k
                if not self._spring(work, state, there, dice):
                    return  # the mover died
        if start < route.size - 1:
            work.events.append(Moved(uid, tuple(int(c) for c in route[start:])))
        if self.turn_on_move:
            dq, dr = (coords[-1] - coords[0]).tolist()
            work.facing[goal] = _snap(space, dq, dr, facing)
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
        return ((work.watch == cell) & (work.kind >= 0) & (work.team != team)).nonzero()[0]

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
        strike: bool = False,
    ) -> None:
        """Roll one attack, engage the target, and remove it if it dies.

        A target bracing against the charge strikes first (``strike``), and may kill the attacker.
        """
        atk, defense, flank, braced = self._stats(
            space,
            work.kind,
            work.team,
            work.facing,
            work.engaged,
            attacker,
            target,
            deferred=deferred,
            walked=int(work.moved[attacker]),
            strike=strike,
        )
        if braced:
            self._attack(work, space, target, attacker, dice, deferred=False, strike=True)
            if work.kind[attacker] < 0:
                return
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
                braced=strike,
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
        *,
        deferred: bool,
        walked: int = 0,
        strike: bool = False,
    ) -> tuple[int, int, bool, bool]:
        """Return the effective Atk and Def, whether it flanks, and whether the target braces.

        ``walked`` is the tiles the attacker has moved this round, and ``strike`` says this is a
        bracing unit's strike, which never flanks. A unit that charges never flanks a unit that
        braces, has a friend with `formation` beside it and hasn't been attacked this round: that
        unit braces against its charge.
        """
        near = _near(space, target, 1.0)
        enemies = np.count_nonzero((kind[near] >= 0) & (team[near] == team[attacker]))
        coords = space.unravel(np.array([attacker, target]))
        behind = not bool(in_front(space, facing[target], coords[0] - coords[1]))
        stats = self.units[kind[attacker]]
        walled = bool(
            stats.charge
            and not (strike or deferred)
            and self._flags["braces"][kind[target]]
            and not engaged[target]
            and ((team[near] == team[target]) & self._flags["formation"][kind[near]]).any()
        )
        flank = not (strike or walled) and (behind or bool(enemies >= 2))
        atk = stats.atk + self._aura(space, kind, team, attacker)
        atk += self.engaged_bonus * bool(engaged[target])
        charged = bool(stats.charge) and walked >= stats.move and not (strike or deferred)
        braced = charged and walled
        if charged and not walled:
            atk += stats.charge
        if flank:
            atk *= self.flank_factor
        if deferred:
            atk += self.overwatch_bonus
        defense = self.units[kind[target]].defense + self._aura(space, kind, team, target)
        if self.units[kind[target]].formation:
            friends = (team[near] == team[target]) & self._flags["formation"][kind[near]]
            defense += min(self.formation_cap, self.formation_bonus * np.count_nonzero(friends))
        return int(atk), int(defense), flank, bool(braced)

    def _aura(
        self, space: LatticeSpace, kind: NDArray[np.int8], team: NDArray[np.int8], cell: int
    ) -> int:
        """Return the aura bonus of the unit at ``cell``: given by a friend within reach, or 0."""
        aura = self._flags["aura"]
        if aura[kind[cell]]:  # it counts its own
            return self.aura_bonus
        near = _near(space, cell, self.aura_radius)
        givers = aura[kind[near]] & (team[near] == team[cell])
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


@functools.cache
def _near_graph(space: LatticeSpace, radius: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the CSR rows and neighbors of the space's graph for a radius.

    `LatticeSpace.radius_graph` checks its radius on every call: cached here for the hot paths,
    for as long as the process runs, like `board.board_space`. Both are int64.
    """
    graph = space.radius_graph(radius)
    return graph.indptr, graph.indices


@functools.cache
def _snap(space: LatticeSpace, dq: int, dr: int, previous: int) -> int:
    """Return `core.snap` of the lattice offset ``(dq, dr)``, cached: it depends on nothing else.

    Plain ints, so that they can be a key. Few offsets occur, so the cache stays small.
    """
    return snap(space, (dq, dr), previous)


def _near(space: LatticeSpace, cell: int, radius: float) -> NDArray[np.int64]:
    """Return the cells within ``radius`` of ``cell``, as `within` finds them, but not ``cell``.

    Looked up in the graph the space keeps for the radius: shape (n,), int64, ascending.
    """
    indptr, indices = _near_graph(space, radius)
    return indices[indptr[cell] : indptr[cell + 1]]


@functools.cache
def _none(n_cells: int) -> NDArray[np.bool_]:
    """Return a mask of no cells, shared and read-only: shape (n_cells,), bool."""
    return read_only(np.zeros(n_cells, dtype=np.bool_))


def _first(initiative: NDArray[np.int16], mask: NDArray[np.bool_]) -> int:
    """Return the cell in ``mask`` with the lowest initiative."""
    cells = mask.nonzero()[0]
    return int(cells[initiative[cells].argmin()])


class _Turn:
    """What a position's decision is about: who decides, and which targets each verb has."""

    def __init__(self, game: Tactics, position: TacticsPosition) -> None:
        self.game = game
        self.position = position
        self.state = state = position.state
        units = state.units
        self.none = _none(state.n_cells)  # returned as it is
        self._targets: dict[int, NDArray[np.bool_]] = {}
        if position.deploying:
            self.active = -1
            self.team = game._placer(units.kind, units.team)
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

    @functools.cached_property
    def buildable(self) -> NDArray[np.bool_]:
        """The tiles the acting unit may build on: shape (n_cells,), bool.

        Adjacent, neither a wall, a unit nor the point. Open tiles also must not cut any living
        unit or the point off from the rest, counting walls and rubble as blocked and ignoring
        units. Rubble to wall blocks no more walking, so it needs no check.
        """
        state, space = self.state, self.state.space
        near = _near(space, self.active, 1.0)
        near = near[~(state.wall[near] | state.occupied[near] | state.point[near])]
        mask = self.none.copy()
        mask[near] = True
        tiles = near[state.passable[near]]
        # Closing a tile whose open neighbors make one arc around it leaves them joined by that
        # arc, so only a tile between two arcs can cut anything off. Most tiles need no search.
        around = np.append(state.passable, False)[rings(self.game.shape)[tiles]]
        arcs = np.count_nonzero(around & ~around[:, [1, 2, 3, 4, 5, 0]], axis=1)
        anchors = (state.occupied | state.point).nonzero()[0]
        for tile in tiles[arcs > 1]:
            open_ = state.passable.copy()
            open_[tile] = False
            mask[tile] = joined(space, open_, anchors)
        return mask

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
        """Return where an action with ``verb`` is legal: shape (n_cells,), bool, read-only.

        Worked out once a decision, so `Tactics.step` needn't repeat what `Tactics.legal` found.
        """
        known = self._targets.get(verb)
        if known is None:
            known = self._find_targets(verb)
            known.setflags(write=False)
            self._targets[verb] = known
        return known

    def _find_targets(self, verb: int) -> NDArray[np.bool_]:
        """Work out `targets`: ``self.none`` if there are none, which a verb must copy to write."""
        game, state, a = self.game, self.state, self.active
        if self.position.over or not 0 <= verb < game.n_verbs:
            return self.none
        if self.position.deploying:
            kind = verb - Verb.PLACE
            if kind < 0:
                return self.none
            units = state.units
            have, left, cost = game._fielded(units.kind, units.team, self.team)
            if have[kind] >= game.caps[kind] or cost[kind] > left:
                return self.none
            return (state.home == self.team) & state.passable & ~state.occupied
        units = state.units
        mask = self.none
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
        elif verb == Verb.BUILD:
            if self.stats.builds and self.can_attack():
                mask = self.buildable
        elif verb == Verb.BREAK:
            if self.stats.builds and self.can_attack():
                near = _near(state.space, a, 1.0)
                mask = self.none.copy()
                mask[near] = ~state.passable[near] & ~state.point[near]
        elif verb == Verb.SWAP:
            if units.moved[a] == 0 and not units.swapped[a]:
                near = _near(state.space, a, 1.0)
                ok = (units.team[near] == self.team) & (units.moved[near] == 0)
                mask = self.none.copy()
                mask[near] = ok & ~units.swapped[near]
        elif verb == Verb.END:
            mask = self.none.copy()
            mask[a] = True
        elif verb < Verb.END:  # Verb.FACE + d
            if game.free_facing and not units.faced[a] and units.facing[a] != verb - Verb.FACE:
                mask = self.none.copy()
                mask[a] = True
        return mask


class _Work:
    """A step's writable copy of the fields of `Unit` (`core.State.edit`), and its events.

    Each field's array is an attribute of the same name, for the rules to write in place;
    ``units`` moves, swaps and clears whole units, and `position` freezes them. ``hold`` is who
    holds the point in the position it makes. Terrain isn't part of the edit, whose moves and
    swaps act on every field: `change_terrain` keeps a new wall and rubble for `position` to set.
    """

    def __init__(self, state: TacticsState, hold: Hold) -> None:
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
        self.hold = hold
        self.terrain: dict[Field[np.bool_], NDArray[np.bool_]] = {}
        self.events: list[Event] = []

    def change_terrain(self, state: TacticsState, uid: int, tile: int, *, build: bool) -> None:
        """Move ``tile`` one step up the ladder or down, and record who did it."""
        wall, rubble = state.wall.copy(), state.rubble.copy()
        if build:
            was_rubble = bool(rubble[tile])
            wall[tile], rubble[tile] = was_rubble, not was_rubble
            self.events.append(Built(uid, tile, "wall" if was_rubble else "rubble"))
        else:
            was_wall = bool(wall[tile])
            wall[tile], rubble[tile] = False, was_wall
            self.events.append(Broke(uid, tile, "rubble" if was_wall else "open"))
        self.terrain = {TacticsState.wall: wall, TacticsState.rubble: rubble}

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
        state = self.units.freeze()
        return TacticsPosition(
            state.replace(self.terrain) if self.terrain else state,
            before.seed,
            ply=ply,
            over=over,
            winner=winner,
            drew=drew,
            events=tuple(self.events),
            round=round,
            hold=self.hold,
        )


def _check_team(team: SupportsIndex) -> int:
    """Return ``team`` as an int, checked to be 0 or 1."""
    team = check_int(team, "team")
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")
    return team
