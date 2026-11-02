"""The rules of Colonies: orders given in turn, then resolved together by a simulation.

The board is an icosphere, partly sea, and each land cell supports a number of people, its
capacity. Each player starts with a colony of settlers in one land cell, far from the others.
A round has two parts. First every player gives one order, in seat order: gather its people
near a land cell toward it, or hold. Then the round resolves as a few steps of a
`core.Simulation`: people march toward their player's target over land, or drift, fight
wherever players meet, and grow toward the capacity of their cell. A player left with fewer
than `MIN_PEOPLE` people is out, and the game ends when one player is left, who wins, or after
the last round, when the most people wins.

Fog of war: a player sees only the cells within a few moves over land of its own people, and
none of the orders given before its own in a round. Its view (`ColoniesView`) and its legal
orders depend on nothing else.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import Frame
from gridzero.core import (
    Adjacency,
    Context,
    Field,
    MeshSpace,
    Phase,
    Simulation,
    Source,
    Spread,
    State,
    check_int,
    check_real,
    icosphere,
    phase,
    regions,
    steps,
)
from gridzero.game import Action, Position, check_action, winner

GATHER, HOLD = 0, 1
"""The verbs: march toward a land cell, or stay put (naming cell 0)."""

MAX_PLAYERS = 4
"""The most players a game may have: the rows of `ColoniesState.people`."""

GRADIENT = 0.1
"""How much less a marching player is drawn to a cell one move further from its target."""

MIN_PEOPLE = 0.5
"""People a player needs in a cell to see from it, and in all to stay in the game."""

TIE = 1e-6
"""Totals of people closer than this are equal."""


class ColoniesState(State):
    """The land, what each cell supports, and each player's people in it."""

    space: MeshSpace
    land = Field(np.bool_, static=True, default=True)
    """Whether the cell is land: people live and move only on land."""
    capacity = Field(np.float64, static=True)
    """The people the cell supports, all players together: 0 at sea."""
    people = Field(np.float64, (MAX_PLAYERS,), extensive=True)
    """Each player's people in the cell: rows past the number of players stay 0."""


class Ordered(NamedTuple):
    """A player gave its order for the round, which stays hidden."""

    player: int


class Resolved(NamedTuple):
    """A round resolved, leaving each player these people in all."""

    round: int
    people: tuple[float, ...]


class Out(NamedTuple):
    """A player was left with fewer than `MIN_PEOPLE` people, and is out of the game."""

    player: int


type Event = Ordered | Resolved | Out
"""What a step of Colonies did."""


class ColoniesView(NamedTuple):
    """What a player is shown: the map, and the people in the cells it sees.

    Parameters
    ----------
    seen : ndarray of bool, shape (n_cells,)
        The cells the player sees: every cell without fog.
    land : ndarray of bool, shape (n_cells,)
        Whether each cell is land: the map is no secret.
    capacity : ndarray of float64, shape (n_cells,)
        The people each cell supports.
    people : ndarray of float64, shape (n_players, n_cells)
        Each player's people in the cells seen, and 0 in the others.
    round : int
        Rounds resolved so far.
    out : tuple of bool
        Whether each player is out.
    """

    seen: NDArray[np.bool_]
    land: NDArray[np.bool_]
    capacity: NDArray[np.float64]
    people: NDArray[np.float64]
    round: int
    out: tuple[bool, ...]


@dataclass(frozen=True, eq=False)
class ColoniesPosition(Position[ColoniesState, Event]):
    """The state, the seed, the ply, the result and the step's events, with the round's orders.

    Parameters
    ----------
    round : int, default 0
        Rounds resolved so far.
    orders : tuple of Action, default ()
        The orders given this round, in seat order, by the players still in.
    out : tuple of bool, default ()
        Whether each player is out: one per player, set by `Colonies.start`.
    """

    round: int = 0
    orders: tuple[Action, ...] = ()
    out: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Colonies:
    """Colonies on an icosphere (see the module).

    Parameters
    ----------
    frequency : int, default 4
        The icosphere's frequency, at least 1: ``10 * frequency**2 + 2`` cells.
    players : int, default 2
        How many players, from 2 to `MAX_PLAYERS`.
    sea : float, default 0.3
        The share of cells that are sea, in [0, 1), drawn anew for each start. At least one
        land cell per player must be left.
    settlers : float, default 10.0
        The people each player starts with, positive.
    richness : float, default 10.0
        The mean capacity of a land cell of mean area, positive. Each land cell's is drawn
        between half and one and a half times that, times its area over the mean.
    rounds : int, default 30
        Rounds before the game ends, at least 1.
    steps : int, default 3
        Steps of the simulation that resolves a round, at least 1.
    reach : int, default 3
        How many moves over land from its target a gathering player's people march, at
        least 1. Its people further away drift, as if it held.
    march : float, default 0.5
        The share of a gathering player's people within `reach` of its target, but not on
        it, that leaves each cell in a step, toward the target, in [0, 1].
    drift : float, default 0.05
        The share of a player's people that drifts to neighboring land in a step when it
        holds, or beyond `reach` when it gathers, in [0, 1].
    fight : float, default 0.2
        People a cell's player loses in a step per enemy there, times a random factor between
        0.5 and 1.5 drawn per cell, non-negative.
    growth : float, default 0.3
        The logistic growth rate of people toward their cell's capacity per step,
        non-negative. A cell holding more than its capacity loses people instead, at most
        a share `growth` of them per step.
    vision : int, default 2
        How many moves over land a player sees beyond each cell where it has at least
        `MIN_PEOPLE` people, non-negative.
    fog : bool, default True
        Whether players see only that far. Without fog, every view shows the whole board.

    Raises
    ------
    TypeError, ValueError
        If a parameter is not a number of its kind, or out of range.
    """

    frequency: int = 4
    players: int = 2
    sea: float = 0.3
    settlers: float = 10.0
    richness: float = 10.0
    rounds: int = 30
    steps: int = 3
    reach: int = 3
    march: float = 0.5
    drift: float = 0.05
    fight: float = 0.2
    growth: float = 0.3
    vision: int = 2
    fog: bool = True

    def __post_init__(self) -> None:
        check_int(self.frequency, "frequency", minimum=1)
        check_int(self.players, "players", minimum=2)
        if self.players > MAX_PLAYERS:
            raise ValueError(f"players must be at most {MAX_PLAYERS}, got {self.players}")
        check_int(self.rounds, "rounds", minimum=1)
        check_int(self.steps, "steps", minimum=1)
        check_int(self.reach, "reach", minimum=1)
        check_int(self.vision, "vision", minimum=0)
        if not isinstance(self.fog, bool):
            raise TypeError(f"fog must be a bool, got {type(self.fog).__name__}")
        if not 0 <= check_real(self.sea, "sea") < 1:
            raise ValueError(f"sea must be in [0, 1), got {self.sea}")
        n_cells = 10 * self.frequency**2 + 2
        if n_cells - round(self.sea * n_cells) < self.players:
            raise ValueError(f"sea {self.sea} leaves fewer land cells than {self.players} players")
        for name in ("settlers", "richness"):
            if not check_real(getattr(self, name), name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("march", "drift"):
            if not 0 <= check_real(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        for name in ("fight", "growth"):
            if not check_real(getattr(self, name), name) >= 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)}")

    @property
    def n_players(self) -> int:
        """How many players."""
        return self.players

    @property
    def n_verbs(self) -> int:
        """Two verbs, `GATHER` and `HOLD`."""
        return 2

    @property
    def space(self) -> MeshSpace:
        """The board: the icosphere of the game's frequency."""
        return board(self.frequency)

    @property
    def planes(self) -> tuple[str, ...]:
        """The planes of an observation (`observe`), for a learner (`agents.Observable`)."""
        return ("own", "others", "seen", "land", "capacity", "round")

    @property
    def revealed(self) -> tuple[str, ...]:
        """The planes `reveal` returns: the people of the player and of the others."""
        return ("own", "others")

    def observe(self, view: ColoniesView, player: int) -> NDArray[np.float32]:
        """Return a player's view as planes per cell, for a learner (`agents.Observable`).

        People and capacities are squashed into [0, 1) as ``x / (x + richness)``: 0.5 at the
        mean capacity of a land cell of mean area.

        Parameters
        ----------
        view : ColoniesView
            What the player is shown (`view`).
        player : int
            The player looking.

        Returns
        -------
        ndarray of float32, shape (6, n_cells)
            The planes of `planes`: the player's people and everyone else's together in the
            cells it sees, 1 where it sees, 1 on land, the capacity, and the share of the rounds
            resolved in every cell. The board's own cells, since every player's frame is the
            board.
        """
        own = view.people[player]
        planes = np.empty((6, view.seen.size), dtype=np.float32)
        planes[0] = self._squash(own)
        planes[1] = self._squash(view.people.sum(axis=0) - own)
        planes[2] = view.seen
        planes[3] = view.land
        planes[4] = self._squash(view.capacity)
        planes[5] = view.round / self.rounds
        return planes

    def reveal(self, position: ColoniesPosition, player: int) -> NDArray[np.float32]:
        """Return everyone's people as a player would see them without fog (`agents.Revealable`).

        Parameters
        ----------
        position : ColoniesPosition
            A position of this game.
        player : int
            The player shown it.

        Returns
        -------
        ndarray of float32, shape (2, n_cells)
            The planes of `revealed`, squashed as in `observe`, in every cell.
        """
        people = position.state.people[: self.players]
        own = people[player]
        return np.stack([self._squash(own), self._squash(people.sum(axis=0) - own)])

    def frame(self, player: int) -> Frame:
        """Return a player's frame: the board itself, for every player (`agents.Observable`)."""
        return _frame(self.n_verbs, self.space.n_cells)

    def _squash(self, x: NDArray[np.float64]) -> NDArray[np.float32]:
        """Return ``x / (x + richness)`` as float32, for amounts ``x >= 0``."""
        return (x / (x + self.richness)).astype(np.float32)

    def start(self, seed: SupportsIndex) -> ColoniesPosition:
        """Return the first position, drawn from the seed's stream ``(0,)``.

        The sea is drawn as in Voronoi, then each land cell's capacity. The first colony goes
        to a random cell of the largest landmass, and each next one to the land cell fewest
        moves over land from the colonies so far, farthest from them. Another landmass is used
        only once this one is full, and ties go to the lowest id. Then the colonies are dealt
        to the players in a random order: the later ones lie on the landmass's far ends, and
        would otherwise always go to the last seats.

        Parameters
        ----------
        seed : int
            The root of the game's draws, non-negative.

        Returns
        -------
        ColoniesPosition
            Round 0, with no orders, nobody out, and `settlers` people in each player's
            colony.

        Raises
        ------
        TypeError
            If ``seed`` is not an integer.
        ValueError
            If ``seed`` is negative.
        """
        seed = check_int(seed, "seed", minimum=0)
        space = self.space
        rng = Context(seed=seed, spawn_key=(0,)).rng
        land = np.ones(space.n_cells, dtype=np.bool_)
        land[rng.choice(space.n_cells, round(self.sea * space.n_cells), replace=False)] = False
        area = space.measures / space.measures.mean()
        capacity = rng.uniform(0.5, 1.5, space.n_cells) * self.richness * area * land
        largest = np.flatnonzero(regions(space, land) == 0)
        colonies = [int(largest[rng.integers(largest.size)])]
        for _ in range(1, self.players):
            moves = steps(space, colonies, blocked=~land)
            far = np.where(np.isfinite(moves), moves, -0.5)  # other landmasses come last
            far[~land] = -1.0
            far[colonies] = -1.0
            colonies.append(int(np.argmax(far)))
        people = np.zeros((MAX_PLAYERS, space.n_cells))
        people[np.arange(self.players), rng.permutation(colonies)] = self.settlers
        state = ColoniesState(
            space,
            {
                ColoniesState.land: land,
                ColoniesState.capacity: capacity,
                ColoniesState.people: people,
            },
        )
        return ColoniesPosition(state, seed, out=(False,) * self.players)

    def player(self, position: ColoniesPosition) -> int:
        """Return the player to move: the next player still in without an order this round.

        Raises
        ------
        ValueError
            If the game is over.
        """
        if position.over:
            raise ValueError("the game is over")
        return _seats(position)[len(position.orders)]

    def view(self, position: ColoniesPosition, player: int) -> ColoniesView:
        """Return what a player is shown: the map, and the people in the cells it sees.

        Parameters
        ----------
        position : ColoniesPosition
            A position of this game.
        player : int
            The player shown it.

        Returns
        -------
        ColoniesView
            Nothing in it depends on the people in cells the player doesn't see, or on the
            orders given this round.
        """
        state = position.state
        seen = self.sight(state, player)
        people = state.people[: self.players] * seen
        return ColoniesView(seen, state.land, state.capacity, people, position.round, position.out)

    def sight(self, state: ColoniesState, player: int) -> NDArray[np.bool_]:
        """Return the cells a player sees.

        Parameters
        ----------
        state : ColoniesState
            A state of this game.
        player : int
            The player who looks.

        Returns
        -------
        ndarray of bool, shape (n_cells,)
            The land within `vision` moves over land of a cell where the player has at least
            `MIN_PEOPLE` people, or every cell without fog.
        """
        if not self.fog:
            return np.ones(state.n_cells, dtype=np.bool_)
        own = state.people[player] >= MIN_PEOPLE
        if not own.any():
            return np.zeros(state.n_cells, dtype=np.bool_)
        return np.isfinite(steps(state.space, own, blocked=~state.land, limit=self.vision))

    def legal(self, position: ColoniesPosition) -> NDArray[np.bool_]:
        """Return the legal orders, bool of shape (2, n_cells), none once the game is over.

        `GATHER` toward any land cell the player to move sees, or `HOLD` on cell 0, so that
        the mask tells it nothing its view doesn't.
        """
        state = position.state
        mask = np.zeros((2, state.n_cells), dtype=np.bool_)
        if not position.over:
            mask[GATHER] = state.land & self.sight(state, self.player(position))
            mask[HOLD, 0] = True
        return mask

    def step(
        self, position: ColoniesPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> ColoniesPosition:
        """Return the position after an order (see `game.Game.step`).

        An order before the round's last is only kept, and draws nothing. The last one
        resolves the round: a simulation of `steps` steps, seeded from the step's stream
        ``(1, ply)``. Then each player left with fewer than `MIN_PEOPLE` people is out, and
        its people are removed. The game ends when at most one player is left, or after the
        last round.
        """
        verb, cell = check_action(position, action, self.n_verbs)
        if not self.legal(position)[verb, cell]:
            raise ValueError(f"illegal action {Action(verb, cell)} at ply {position.ply}")
        seats = _seats(position)
        player = seats[len(position.orders)]
        orders = (*position.orders, Action(verb, cell))
        ply = position.ply + 1
        if len(orders) < len(seats):
            return ColoniesPosition(
                position.state,
                position.seed,
                ply=ply,
                events=(Ordered(player),),
                round=position.round,
                orders=orders,
                out=position.out,
            )
        dice = Context(seed=position.seed, spawn_key=(1, position.ply))
        seed = int(dice.rng.integers(2**63))
        state = self.resolve(position.state, dict(zip(seats, orders, strict=True)), seed)
        dying = [p for p in seats if self.totals(state)[p] < MIN_PEOPLE]
        if dying:
            people = state.people.copy()
            people[dying] = 0.0
            state = state.replace({ColoniesState.people: people})
        out = tuple(gone or p in dying for p, gone in enumerate(position.out))
        totals = self.totals(state)
        round_ = position.round + 1
        over = round_ == self.rounds or out.count(False) <= 1
        return ColoniesPosition(
            state,
            position.seed,
            ply=ply,
            over=over,
            winner=winner(totals, TIE) if over else None,  # those out have none
            drew=dice.drew,
            events=(
                Ordered(player),
                Resolved(round_, tuple(float(t) for t in totals)),
                *(Out(p) for p in dying),
            ),
            round=round_,
            out=out,
        )

    def resolve(
        self, state: ColoniesState, orders: Mapping[int, Action], seed: SupportsIndex
    ) -> ColoniesState:
        """Return the state after a round's orders, all carried out together.

        Parameters
        ----------
        state : ColoniesState
            The state before the round.
        orders : mapping of int to Action
            The order of each player still in.
        seed : int
            The seed of the round's simulation, non-negative.

        Returns
        -------
        ColoniesState
            The state after `steps` steps of `phases`.
        """
        simulation = Simulation(state, self.phases(state, orders), seed=seed)
        simulation.run(self.steps)
        return simulation.state

    def phases(
        self, state: ColoniesState, orders: Mapping[int, Action]
    ) -> list[Phase[ColoniesState]]:
        """Return the phases of a step of a round: people march, fight and grow.

        Parameters
        ----------
        state : ColoniesState
            The state the round starts from: its land sets how far each cell is from a target.
        orders : mapping of int to Action
            The order of each player still in.

        Returns
        -------
        list of Phase
            ``march``, ``fight`` and ``grow``, in that order.
        """
        n_cells = state.n_cells
        leave = np.zeros((MAX_PLAYERS, n_cells))
        toward = np.ones((MAX_PLAYERS, n_cells))
        for p, (verb, cell) in orders.items():
            if verb == HOLD:
                leave[p] = self.drift
                continue
            moves = steps(state.space, cell, blocked=~state.land)
            toward[p] = np.where(np.isfinite(moves), GRADIENT ** np.minimum(moves, 1e3), 0.0)
            leave[p] = np.where(moves <= self.reach, self.march, self.drift)
            leave[p, cell] = 0.0  # the people at the target stay

        @phase(writes=ColoniesState.people, hoods={"near": Adjacency()})
        def march(s: ColoniesState, ctx: Context) -> Spread:
            """Move each player's people toward its target, or drift those who hold."""
            near = ctx.hoods["near"]
            return Spread(ColoniesState.people, near, leave=leave, toward=toward, domain=s.land)

        @phase(writes=ColoniesState.people)
        def fight(s: ColoniesState, ctx: Context) -> Source | None:
            """Take people from each player in proportion to its enemies in the cell."""
            enemies = s.people.sum(axis=0) - s.people
            if not ((enemies > 0) & (s.people > 0)).any():  # draw only when players meet
                return None
            luck = ctx.rng.uniform(0.5, 1.5, n_cells)
            return Source(ColoniesState.people, -self.fight * enemies * luck)

        @phase(writes=ColoniesState.people)
        def grow(s: ColoniesState, ctx: Context) -> Source:
            """Grow each player's people toward the cell's capacity, shared by all in it."""
            crowding = np.divide(
                s.people.sum(axis=0), s.capacity, out=np.ones(n_cells), where=s.land
            )
            shortfall = np.maximum(1 - crowding, -1.0)  # crowds starve, but never all at once
            return Source(ColoniesState.people, self.growth * s.people * shortfall)

        return [march, fight, grow]

    def totals(self, state: ColoniesState) -> NDArray[np.float64]:
        """Return each player's people in all.

        Parameters
        ----------
        state : ColoniesState
            A state of this game.

        Returns
        -------
        ndarray of float64, shape (n_players,)
            The sum of each player's people over the cells.
        """
        return state.people[: self.players].sum(axis=1)


def _seats(position: ColoniesPosition) -> list[int]:
    """Return the players still in, in seat order."""
    return [p for p, gone in enumerate(position.out) if not gone]


@functools.cache
def _frame(n_verbs: int, n_cells: int) -> Frame:
    """Return the frame that is the board itself, made once per size."""
    return Frame.board(n_verbs, n_cells)


@functools.cache
def board(frequency: int) -> MeshSpace:
    """Return the icosphere of a frequency, made once and shared by every position."""
    return icosphere(frequency)
