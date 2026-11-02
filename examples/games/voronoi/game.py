"""The rules of the Voronoi game on a planet: stones placed in turns, and the land they hold.

The board is an icosphere, partly sea. Players take turns placing a stone on a free land cell,
a fixed number each. Each land cell belongs to the player whose nearest stone is fewest moves
away over land, and to nobody when two players are as near or no stone reaches it. Once every
stone is down, the player holding the most land area wins.
"""

import functools
from dataclasses import dataclass
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Context,
    Field,
    MeshSpace,
    State,
    check_int,
    check_real,
    derived,
    icosphere,
    steps,
)
from gridzero.game import Action, Position, check_action, winner

PLACE = 0
"""The one verb: place a stone on a free land cell."""

TIE = 1e-9
"""Areas closer than this share of the sphere's area are equal. Cells that a symmetry of the
icosphere maps onto each other have areas that differ only by rounding, so a symmetric split is
a draw."""


class VoronoiState(State):
    """The land, the stones on it, and who holds each cell."""

    space: MeshSpace
    land = Field(np.bool_, static=True, default=True)
    """Whether the cell is land: stones go only on land, and only land is held."""
    stone = Field(np.int8, default=-1)
    """The player whose stone is on the cell, -1 for none."""

    @derived(land, stone)
    def territory(self) -> NDArray[np.int8]:
        """Return the player whose stone is nearest each cell, over land, int8 of shape (n_cells,).

        -1 for sea, for a cell two players are as near, and for one no stone reaches.
        """
        players = int(self.stone.max()) + 1
        territory = np.full(self.n_cells, -1, dtype=np.int8)
        if players == 0:
            return territory
        moves = np.stack(
            [steps(self.space, self.stone == p, blocked=~self.land) for p in range(players)]
        )
        nearest = moves.min(axis=0)
        alone = ((moves == nearest).sum(axis=0) == 1) & np.isfinite(nearest) & self.land
        territory[alone] = np.argmin(moves, axis=0)[alone]
        return territory


class Placed(NamedTuple):
    """A player placed a stone."""

    player: int
    cell: int


@dataclass(frozen=True, eq=False)
class VoronoiPosition(Position[VoronoiState, Placed]):
    """The state, the seed, the ply, the result and the step's events."""


@dataclass(frozen=True)
class Voronoi:
    """The Voronoi game on an icosphere (see the module).

    Parameters
    ----------
    frequency : int, default 4
        The icosphere's frequency, at least 1: ``10 * frequency**2 + 2`` cells.
    players : int, default 2
        How many players, at least 2.
    stones : int, default 3
        Stones each player places, at least 1. The game ends sooner if no free land is left.
    sea : float, default 0.25
        The share of cells that are sea, in [0, 1), drawn anew for each start.

    Raises
    ------
    TypeError, ValueError
        If a parameter is not a number of its kind, or out of range.
    """

    frequency: int = 4
    players: int = 2
    stones: int = 3
    sea: float = 0.25

    def __post_init__(self) -> None:
        check_int(self.frequency, "frequency", minimum=1)
        check_int(self.players, "players", minimum=2)
        check_int(self.stones, "stones", minimum=1)
        if not 0 <= check_real(self.sea, "sea") < 1:
            raise ValueError(f"sea must be in [0, 1), got {self.sea}")

    @property
    def n_players(self) -> int:
        """How many players."""
        return self.players

    @property
    def n_verbs(self) -> int:
        """One verb, `PLACE`."""
        return 1

    @property
    def space(self) -> MeshSpace:
        """The board: the icosphere of the game's frequency."""
        return board(self.frequency)

    def start(self, seed: SupportsIndex) -> VoronoiPosition:
        """Return the first position: no stones, and the sea drawn from the seed's stream ``(0,)``.

        Parameters
        ----------
        seed : int
            The root of the game's draws, non-negative.

        Returns
        -------
        VoronoiPosition
            Ply 0, with ``round(sea * n_cells)`` cells of sea.

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
        return VoronoiPosition(VoronoiState(space, {VoronoiState.land: land}), seed)

    def player(self, position: VoronoiPosition) -> int:
        """Return the player to move: the players take turns, 0 first.

        Raises
        ------
        ValueError
            If the game is over.
        """
        if position.over:
            raise ValueError("the game is over")
        return position.ply % self.players

    def view(self, position: VoronoiPosition, player: int) -> VoronoiPosition:
        """Return the position: nothing is hidden."""
        return position

    def legal(self, position: VoronoiPosition) -> NDArray[np.bool_]:
        """Return the free land cells, bool of shape (1, n_cells), none once the game is over."""
        state = position.state
        return (state.land & (state.stone < 0) & (not position.over))[None]

    def step(
        self, position: VoronoiPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> VoronoiPosition:
        """Return the position after a stone is placed (see `game.Game.step`)."""
        verb, cell = check_action(position, action, self.n_verbs)
        state = position.state
        if not state.land[cell] or state.stone[cell] >= 0:
            raise ValueError(f"illegal action {Action(verb, cell)}: cell {cell} is sea or taken")
        player = position.ply % self.players
        stone = state.stone.copy()
        stone[cell] = player
        state = state.replace({VoronoiState.stone: stone})
        ply = position.ply + 1
        over = ply == self.stones * self.players or not (state.land & (stone < 0)).any()
        return VoronoiPosition(
            state,
            position.seed,
            ply=ply,
            over=over,
            winner=winner(self.areas(state), TIE * state.space.measures.sum()) if over else None,
            events=(Placed(player, cell),),
        )

    def areas(self, state: VoronoiState) -> NDArray[np.float64]:
        """Return the land area each player holds.

        Parameters
        ----------
        state : VoronoiState
            A state of this game.

        Returns
        -------
        ndarray of float64, shape (n_players,)
            The sum of the areas of the cells in each player's territory.
        """
        held = state.territory >= 0
        weights = state.space.measures[held]
        return np.bincount(state.territory[held], weights=weights, minlength=self.players)


@functools.cache
def board(frequency: int) -> MeshSpace:
    """Return the icosphere of a frequency, made once and shared by every position."""
    return icosphere(frequency)
