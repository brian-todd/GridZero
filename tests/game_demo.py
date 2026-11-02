"""A small fake game for the tests of `game`, and the checks that every game's tests run.

`Claims` is played on a strip of cells: players take turns claiming a free cell, outright or by
a coin flip that may give it to the next player, and whoever claims the most cells wins. Cells won
by a flip stay hidden from the players' views until the end, so the view is not the position.
`check_game` holds a game to the promises of the `game.Game` protocol.
"""

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any, NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import Context, Field, LatticeSpace, State, check_int
from gridzero.game import (
    Action,
    Game,
    Player,
    Position,
    Record,
    check_action,
    play,
    replay,
    winner,
)

CLAIM, FLIP = 0, 1
"""The verbs of `Claims`: claim a free cell, or flip a coin for it."""

HIDDEN = -2
"""What a view shows for a cell won by a flip, before the end."""


class StripState(State):
    """Who owns each cell, and which cells went by a flip."""

    space: LatticeSpace
    owner = Field(np.int8, default=-1)
    flipped = Field(np.bool_)


class Claimed(NamedTuple):
    """A cell changed hands."""

    player: int
    cell: int
    flipped: bool


@dataclass(frozen=True, eq=False)
class ClaimsPosition(Position[StripState, Claimed]):
    """A position of `Claims`, with the number of flips so far."""

    flips: int = 0


@dataclass(frozen=True)
class Claims:
    """Players claim the cells of a strip in turns; the most cells wins, and a tie is a draw.

    ``FLIP`` on a free cell flips a coin: heads, the cell is the flipper's, tails the next
    player's. Cells won by a flip are hidden in the views until the game is over.
    """

    length: int = 5
    players: int = 2

    @property
    def n_players(self) -> int:
        return self.players

    @property
    def n_verbs(self) -> int:
        return 2

    @property
    def space(self) -> LatticeSpace:
        return _strip(self.length)

    def start(self, seed: SupportsIndex) -> ClaimsPosition:
        return ClaimsPosition(StripState(self.space), check_int(seed, "seed", minimum=0))

    def player(self, position: ClaimsPosition) -> int:
        if position.over:
            raise ValueError("the game is over")
        return position.ply % self.players

    def view(self, position: ClaimsPosition, player: int) -> NDArray[np.int8]:
        state = position.state
        if position.over:
            return state.owner
        return np.where(state.flipped, np.int8(HIDDEN), state.owner)

    def legal(self, position: ClaimsPosition) -> NDArray[np.bool_]:
        free = position.state.owner < 0
        mask = np.zeros((2, position.state.n_cells), dtype=np.bool_)
        if not position.over:
            mask[:] = free
        return mask

    def step(
        self, position: ClaimsPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> ClaimsPosition:
        verb, cell = check_action(position, action, self.n_verbs)
        state = position.state
        if state.owner[cell] >= 0:
            raise ValueError(f"illegal action {Action(verb, cell)} at ply {position.ply}")
        player = position.ply % self.players
        ctx = Context(seed=position.seed, spawn_key=(1, position.ply))
        owner = state.owner.copy()
        flipped = state.flipped.copy()
        if verb == FLIP:
            owner[cell] = player if ctx.rng.random() < 0.5 else (player + 1) % self.players
            flipped[cell] = True
        else:
            owner[cell] = player
        state = state.replace({StripState.owner: owner, StripState.flipped: flipped})
        over = bool((owner >= 0).all())
        counts = np.bincount(owner[owner >= 0], minlength=self.players)
        return ClaimsPosition(
            state,
            position.seed,
            ply=position.ply + 1,
            over=over,
            winner=winner(counts) if over else None,
            drew=ctx.drew,
            events=(Claimed(int(owner[cell]), cell, verb == FLIP),),
            flips=position.flips + (verb == FLIP),
        )


@functools.cache
def _strip(length: int) -> LatticeSpace:
    """Return the strip of a length, made once."""
    return LatticeSpace((length,), periodic=False)


@dataclass(frozen=True)
class RandomClaimer:
    """Picks a legal action at random, flipping with probability ``flip``."""

    flip: float = 0.5

    def __call__(
        self,
        game: Claims,
        player: int,
        view: NDArray[np.int8],
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        verb = FLIP if rng.random() < self.flip else CLAIM
        cells = np.flatnonzero(legal[verb])
        return Action(verb, int(cells[rng.integers(cells.size)]))


@dataclass(frozen=True)
class FirstFree:
    """Claims the free cell of lowest id, outright, from what it is shown."""

    def __call__(
        self,
        game: Claims,
        player: int,
        view: NDArray[np.int8],
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        return Action(CLAIM, int(np.flatnonzero(view == -1)[0]))


def check_game[P: Position[State, object], V](
    game: Game[P, V], player: Player[Any, V], seeds: range = range(4), max_plies: int = 10_000
) -> list[Record[P]]:
    """Hold a game to the promises of `game.Game`, over games of ``player`` against itself.

    Checks, at every position of each game: `legal`'s shape and dtype, that a position not over
    has a legal action and one over has none, no player to move, and refuses a step, that the
    player to move is a player, that `step` leaves its input unchanged, gives the same key for
    the same position and action, gives the same key under another seed when it did not draw,
    and raises on an illegal action. Then that the winner is a player or None, and that
    replaying the record gives its final position.

    Parameters
    ----------
    game : Game
        The rules.
    player : Player
        A player that picks among all legal actions, such as a random one.
    seeds : range, default range(4)
        One game per seed, from ``game.start(seed)``.
    max_plies : int, default 10000
        A game longer than this fails the check, as one that may never end.

    Returns
    -------
    list of Record
        The games, for further checks.
    """
    records = []
    for seed in seeds:
        _check_ends(game, player, game.start(seed), max_plies)
        record = play(game, [player] * game.n_players, game.start(seed))
        assert record.final.over
        assert record.final.winner in (None, *range(game.n_players))
        assert record.final.ply == len(record.actions)
        positions = list(replay(game, record))
        assert positions[-1].key() == record.final.key()
        assert [e for p in positions[1:] for e in p.events] == list(record.events)
        for position, action in zip(positions, (*record.actions, None), strict=True):
            _check_position(game, position, action)
        records.append(record)
    return records


def _check_ends[P: Position[State, object], V](
    game: Game[P, V], player: Player[Any, V], position: P, max_plies: int
) -> None:
    """Check that a game of ``player`` against itself ends within ``max_plies``."""
    rng = np.random.default_rng(0)
    for _ in range(max_plies):
        if position.over:
            return
        p = game.player(position)
        position = game.step(
            position, player(game, p, game.view(position, p), game.legal(position), rng)
        )
    assert position.over, f"the game didn't end within {max_plies} plies"


def _check_position[P: Position[State, object], V](
    game: Game[P, V], position: P, action: Action | None
) -> None:
    """Check one position of a game, and the step its record took from it (None at the end)."""
    legal = game.legal(position)
    assert legal.dtype == np.bool_
    assert legal.shape == (game.n_verbs, position.state.n_cells)
    if action is None:
        assert position.over
        assert not legal.any()
        try:
            game.step(position, (0, 0))
        except ValueError:
            pass
        else:
            raise AssertionError("a step from a position over was accepted")
        try:
            game.player(position)
        except ValueError:
            pass
        else:
            raise AssertionError("a position over has a player to move")
        return
    assert not position.over
    assert legal[action.verb, action.cell]
    assert 0 <= game.player(position) < game.n_players
    before = position.key()
    after = game.step(position, action)
    assert position.key() == before, "step changed its input"
    assert game.step(position, action).key() == after.key(), "the same step gave another key"
    assert after.ply == position.ply + 1
    if not after.drew:
        other = dataclasses.replace(position, seed=position.seed + 1)
        assert game.step(other, action).key() == after.key(), "a step that drew nothing varied"
    illegal = np.argwhere(~legal)
    if illegal.size:
        verb, cell = illegal[0]
        try:
            game.step(position, (int(verb), int(cell)))
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal action {(int(verb), int(cell))} was accepted")


def is_symmetric(
    state: State,
    cells: NDArray[np.int64],
    seats: tuple[Any, ...] = (),
    swap: tuple[int, ...] = (1, 0),
) -> bool:
    """Return whether a board looks the same through a map of its cells that swaps the seats.

    Every field at cell ``c`` must equal that field at ``cells[c]``, with the fields in
    ``seats``, which hold a seat (or -1 for none), relabeled by ``swap``: seat ``s`` there is
    seat ``swap[s]`` here. For a board of two seats facing each other, ``cells`` is
    `LatticeSpace.opposite`.
    """
    relabel = np.asarray(swap)
    for field in state:
        values = state[field]
        if field in seats:
            values = np.where(values >= 0, relabel[np.maximum(values, 0)], values)
        if not np.array_equal(state[field], values[..., cells]):
            return False
    return True
