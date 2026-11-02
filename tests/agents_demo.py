"""A small game for the tests of `agents`, in place of a second real game: `Patches`.

Two players take turns on a small square board. The one to move either places a stone on a
free cell (``PLACE``), or gambles for one (``GAMBLE``): a coin decides whether the cell and the
free neighbor of lowest id become the gambler's, or the cell goes to the other player. When the
board is full, the larger connected group of stones wins, then the more stones; equal is a draw.

It differs from Tactics where a learner's code could hide a bug: squares rather than hexes, a
board that is not square, two verbs, a coin on some plies only, and games of 12 plies at most.
It is `Learnable`: three planes, and player 1 sees the board turned half a turn. It is also
`Revealable`, though nothing is hidden: `reveal` gives the observation, so that a belief has
something to learn.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents.frame import Frame, Learnable, Observable, Revealable
from gridzero.core import Context, Field, LatticeSpace, State, check_int, regions
from gridzero.game import Action, Position, Record, check_action, replay, winner

PLACE, GAMBLE = 0, 1
"""The verbs of `Patches`."""


class PatchesState(State):
    """Who owns each cell: -1 while free."""

    owner = Field(np.int8, default=-1)


class Took(NamedTuple):
    """A player took cells."""

    player: int
    cells: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class PatchesPosition(Position[PatchesState, Took]):
    """A position of `Patches`."""


@functools.cache
def _space(shape: tuple[int, int]) -> LatticeSpace:
    return LatticeSpace(shape, periodic=False)


@dataclass(frozen=True)
class Patches:
    """Place stones or gamble for them; the larger connected group wins (see the module)."""

    shape: tuple[int, int] = (3, 4)

    @property
    def n_players(self) -> int:
        return 2

    @property
    def n_verbs(self) -> int:
        return 2

    @property
    def space(self) -> LatticeSpace:
        return _space(self.shape)

    @property
    def planes(self) -> tuple[str, ...]:
        return ("own", "enemy", "free")

    def start(self, seed: SupportsIndex) -> PatchesPosition:
        return PatchesPosition(PatchesState(self.space), check_int(seed, "seed", minimum=0))

    def player(self, position: PatchesPosition) -> int:
        if position.over:
            raise ValueError("the game is over")
        return position.ply % 2

    def view(self, position: PatchesPosition, player: int) -> PatchesPosition:
        return position

    def legal(self, position: PatchesPosition) -> NDArray[np.bool_]:
        mask = np.zeros((2, position.state.n_cells), dtype=np.bool_)
        if not position.over:
            mask[:] = position.state.owner < 0
        return mask

    def step(
        self, position: PatchesPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> PatchesPosition:
        verb, cell = check_action(position, action, self.n_verbs)
        state = position.state
        if state.owner[cell] >= 0:
            raise ValueError(f"illegal action {Action(verb, cell)} at ply {position.ply}")
        player = position.ply % 2
        ctx = Context(seed=position.seed, spawn_key=(1, position.ply))
        work = state.edit(PatchesState.owner)
        owner = work[PatchesState.owner]
        taken = [cell]
        if verb == GAMBLE and ctx.rng.random() >= 0.5:
            player = 1 - player  # tails: the cell is the other player's
        elif verb == GAMBLE:
            graph = self.space.adjacency()
            free = [int(n) for n in graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]]]
            free = [n for n in free if owner[n] < 0]
            taken += free[:1]
        owner[taken] = player
        state = work.freeze()
        over = bool((state.owner >= 0).all())
        return PatchesPosition(
            state,
            position.seed,
            ply=position.ply + 1,
            over=over,
            winner=self._winner(state) if over else None,
            drew=ctx.drew,
            events=(Took(player, tuple(taken)),),
        )

    def observe(self, position: PatchesPosition, player: int) -> NDArray[np.float32]:
        owner = position.state.owner
        planes = np.stack([owner == player, owner == 1 - player, owner < 0]).astype(np.float32)
        return planes[:, self.frame(player).cells]

    @property
    def revealed(self) -> tuple[str, ...]:
        return self.planes

    def reveal(self, position: PatchesPosition, player: int) -> NDArray[np.float32]:
        return self.observe(position, player)

    def frame(self, player: int) -> Frame:
        if player == 0:
            return Frame.board(2, self.space.n_cells)
        return Frame.opposite(self.space, 2)

    def _winner(self, state: PatchesState) -> int | None:
        """Return the player with the larger group, then more stones, or None for a draw."""
        scores = []
        for player in (0, 1):
            mine = state.owner == player
            groups = regions(self.space, mine)
            largest = int(np.bincount(groups[mine]).max()) if mine.any() else 0
            scores.append(largest * (state.n_cells + 1) + int(mine.sum()))  # group, then stones
        return winner(scores)


@dataclass(frozen=True)
class Lowest:
    """Places on the lowest free cell, after a first stone anywhere: a teacher to learn from.

    Every choice but the first follows from the board, so a network can learn them, and the
    first, drawn from the player's generator, sets each game apart. Here rather than in a test
    module, so that processes of `agents.imitate.collect` find it.
    """

    def __call__(
        self,
        game: Patches,
        player: int,
        position: PatchesPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        free = np.flatnonzero(legal[PLACE])
        return Action(PLACE, int(free[rng.integers(free.size)] if position.ply == 0 else free[0]))


def check_observable[P: Position[State, object]](
    game: Observable[P, Any], records: Sequence[Record[P]]
) -> None:
    """Hold a game to the promises of `agents.frame.Observable`, at every position of ``records``.

    Checks that every position lies on `space`; that `planes` are distinct names; that each
    player's observation of its view is float32 of shape ``(len(planes), n_cells)`` with values
    in [0, 1], and leaves the position unchanged; and that each player's frame maps the legal
    actions into its frame and back.

    Parameters
    ----------
    game : Observable
        The rules.
    records : sequence of Record
        Games of it, such as those `check_game` returns.
    """
    planes = game.planes
    assert len(set(planes)) == len(planes) > 0
    for record in records:
        for position in replay(game, record):
            assert position.state.space is game.space
            if position.over:
                continue
            legal = game.legal(position)
            for player in range(game.n_players):
                before = position.key()
                obs = game.observe(game.view(position, player), player)
                assert position.key() == before, "observe changed the position"
                assert obs.dtype == np.float32
                assert obs.shape == (len(planes), game.space.n_cells)
                assert ((obs >= 0) & (obs <= 1)).all()
                frame = game.frame(player)
                assert frame.cells.shape == (game.space.n_cells,)
                assert frame.verbs.shape == (game.n_verbs,)
                np.testing.assert_array_equal(frame.back(frame.into(legal)), legal)


def check_learnable[P: Position[State, object]](
    game: Learnable[P], records: Sequence[Record[P]]
) -> None:
    """Hold a game to the promises of `agents.frame.Learnable`: `check_observable`'s.

    Its view is its position, so `check_observable` reads the position's observations.

    Parameters
    ----------
    game : Learnable
        The rules.
    records : sequence of Record
        Games of it, such as those `check_game` returns.
    """
    check_observable(game, records)


def check_revealable[P: Position[State, object]](
    game: Revealable[P, Any], records: Sequence[Record[P]]
) -> None:
    """Hold a game to the promises of `agents.frame.Revealable`, at every position of ``records``.

    Checks `check_observable`'s promises, that `revealed` are distinct names, and that what is
    revealed to each player is float32 of shape ``(len(revealed), n_cells)`` with values in
    [0, 1], and leaves the position unchanged.

    Parameters
    ----------
    game : Revealable
        The rules.
    records : sequence of Record
        Games of it, such as those `check_game` returns.
    """
    check_observable(game, records)
    revealed = game.revealed
    assert len(set(revealed)) == len(revealed) > 0
    for record in records:
        for position in replay(game, record):
            for player in range(game.n_players):
                before = position.key()
                shown = game.reveal(position, player)
                assert position.key() == before, "reveal changed the position"
                assert shown.dtype == np.float32
                assert shown.shape == (len(revealed), game.space.n_cells)
                assert ((shown >= 0) & (shown <= 1)).all()
