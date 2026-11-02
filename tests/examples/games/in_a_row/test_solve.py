"""Tests of exact values of small games of k in a row, and of the search judged by them.

The values come from `agents.exact`; these tests hold them to what is known of the games.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from examples.games.in_a_row import (
    PLACE,
    InARow,
    InARowPosition,
    InARowState,
    ThreatPlayer,
)
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.agents.exact import optimal, solve
from gridzero.game import series

TIC_TAC_TOE = InARow((3, 3), 3, gravity=False)
SMALL = InARow((4, 3), 3)
TTT_VALUES = solve(TIC_TAC_TOE)


@dataclass(frozen=True, eq=False)
class Exact:
    """Judges positions by their solved values, with a uniform prior."""

    values: dict[bytes, float]

    def __call__(self, game, positions, legal):
        logits = [np.zeros(int(mask.sum())) for mask in legal]
        return logits, np.array([self.values[p.key()] for p in positions])


@dataclass(frozen=True)
class Blind:
    """Judges every position even, with a uniform prior: the search alone decides."""

    def __call__(self, game, positions, legal):
        return [np.zeros(int(mask.sum())) for mask in legal], np.full(len(positions), 0.5)


def at(game, rows, ply=None):
    """Return a position from rows drawn top down ("X", "O", ".")."""
    owner = np.full(game.space.n_cells, -1, dtype=np.int8)
    for height, marks in enumerate(reversed(rows)):
        for column, mark in enumerate(marks):
            if mark != ".":
                owner[game.space.ravel([column, height])] = "XO".index(mark)
    assert (owner == 0).sum() - (owner == 1).sum() in (0, 1), "X moves first"
    state = InARowState(game.space, {InARowState.owner: owner})
    return InARowPosition(state, 0, ply=int((owner >= 0).sum()) if ply is None else ply)


def test_tic_tac_toe_is_a_draw_with_perfect_play():
    assert len(TTT_VALUES) == 5478
    assert TTT_VALUES[TIC_TAC_TOE.start(0).key()] == 0.5


def test_small_boards_with_gravity_are_won_by_the_first_player():
    assert solve(SMALL)[SMALL.start(0).key()] == 1.0


def test_values_of_finished_and_one_move_positions():
    before = at(TIC_TAC_TOE, ["...", "OO.", "XX."])
    won = TIC_TAC_TOE.step(before, (PLACE, int(TIC_TAC_TOE.space.ravel([2, 0]))))
    assert solve(TIC_TAC_TOE, won) == {won.key(): 1.0}
    # O to move, and O wins at once at the top middle.
    position = at(TIC_TAC_TOE, ["X.X", ".OX", "XOO"])
    assert TIC_TAC_TOE.player(position) == 1
    assert solve(TIC_TAC_TOE, position)[position.key()] == 0.0


def test_optimal_moves_are_those_that_keep_the_best_value():
    # O to move: the top middle wins at once; anything else lets X win.
    position = at(TIC_TAC_TOE, ["X.X", ".OX", "XOO"])
    values = solve(TIC_TAC_TOE, position)
    best = optimal(TIC_TAC_TOE, position, values)
    assert np.flatnonzero(best[PLACE]).tolist() == [int(TIC_TAC_TOE.space.ravel([1, 2]))]
    # X to move in an even position: every move but the one block loses.
    position = at(TIC_TAC_TOE, ["...", ".O.", "XOX"])
    values = solve(TIC_TAC_TOE, position)
    best = optimal(TIC_TAC_TOE, position, values)
    assert values[position.key()] == 0.5
    assert np.flatnonzero(best[PLACE]).tolist() == [int(TIC_TAC_TOE.space.ravel([1, 2]))]


def test_every_first_move_of_tic_tac_toe_keeps_the_draw():
    best = optimal(TIC_TAC_TOE, TIC_TAC_TOE.start(0), TTT_VALUES)
    assert best.all()


def test_a_search_on_even_values_finds_a_win_in_one_and_blocks_one():
    player = AlphaZeroPlayer(Blind(), simulations=256)  # 64 missed the block 2 times in 10
    rng = np.random.default_rng(0)
    game = InARow()
    win = at(game, ["X......", "X..O...", "X..O..O"])  # X to move, wins on top of column 0
    assert game.player(win) == 0
    action = player(game, game.player(win), win, game.legal(win), rng)
    assert game.space.unravel(action.cell)[0] == 0
    block = at(game, ["...O...", "...O...", "XX.O..X"])
    assert game.player(block) == 0  # O wins on top of column 3
    action = player(game, game.player(block), block, game.legal(block), rng)
    assert game.space.unravel(action.cell)[0] == 3


@dataclass(frozen=True, eq=False)
class OneVisitEach:
    """Searches with exact values and one simulation per legal action.

    Each root action is then judged by its own exact value, and the search ranks them. More
    simulations go deeper, where a mean over replies chosen by a uniform prior is not the exact
    value (findings G7 and G9).
    """

    values: dict[bytes, float]

    def __call__(self, game, player, position, legal, rng):
        n = int(legal.sum())
        return AlphaZeroPlayer(Exact(self.values), simulations=n, considered=n)(
            game, player, position, legal, rng
        )


def reachable(game):
    """Return every position of ``game`` reachable from the start and not over."""
    positions = {}
    frontier = [game.start(0)]
    while frontier:
        position = frontier.pop()
        if position.key() in positions or position.over:
            continue
        positions[position.key()] = position
        moves = np.flatnonzero(game.legal(position)[PLACE])
        frontier.extend(game.step(position, (PLACE, int(c))) for c in moves)
    return list(positions.values())


def test_a_search_on_exact_values_picks_an_optimal_move_everywhere_in_tic_tac_toe():
    player = OneVisitEach(TTT_VALUES)
    rng = np.random.default_rng(0)
    positions = reachable(TIC_TAC_TOE)
    assert len(positions) == 4520  # 5,478 less the 958 over
    for position in positions:
        action = player(
            TIC_TAC_TOE, TIC_TAC_TOE.player(position), position, TIC_TAC_TOE.legal(position), rng
        )
        assert optimal(TIC_TAC_TOE, position, TTT_VALUES)[action.verb, action.cell]


@pytest.mark.parametrize("rival", [UniformPlayer(), ThreatPlayer()])
def test_a_search_on_exact_values_wins_a_small_gravity_game_it_starts(rival):
    # The first player wins 4x3 with three in a row. Second, the exact player is lost from the
    # start: every move keeps the same (lost) value, so it doesn't count here.
    records = series(SMALL, [("exact", OneVisitEach(solve(SMALL))), ("rival", rival)], 10)
    first = [r for r in records if r.names[0] == "exact"]
    assert len(first) == 10
    assert all(r.final.winner == 0 for r in first)
