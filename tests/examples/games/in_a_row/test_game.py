"""Tests of the rules of k in a row: the board, lines, gravity, results, errors and oracles."""

import numpy as np
import pytest

from examples.games.in_a_row import (
    PLACE,
    InARow,
    InARowPosition,
    InARowState,
    Placed,
)
from game_demo import check_game
from gridzero.agents import UniformPlayer

CONNECT_FOUR = InARow()
TIC_TAC_TOE = InARow((3, 3), 3, gravity=False)
SMALL = InARow((4, 4), 3)

A212693 = (1, 7, 49, 238, 1120, 4263, 16422, 54859)
"""Positions after 0, 1, 2, ... plies of Connect Four (OEIS A212693, checked 2026-10-28)."""


def at(game, owner, ply=None):
    """Return a position of ``game`` with ``owner`` per cell, at the ply its stones imply."""
    owner = np.asarray(owner, dtype=np.int8)
    state = InARowState(game.space, {InARowState.owner: owner})
    return InARowPosition(state, 0, ply=int((owner >= 0).sum()) if ply is None else ply)


def cell(game, column, row):
    return int(game.space.ravel([column, row]))


def test_connect_four_has_69_lines_and_tic_tac_toe_8():
    assert CONNECT_FOUR.lines.shape == (69, 4)
    assert TIC_TAC_TOE.lines.shape == (8, 3)


@pytest.mark.parametrize("game", [CONNECT_FOUR, TIC_TAC_TOE, SMALL, InARow((5, 2), 2)])
def test_every_line_is_a_straight_run_in_lattice_coordinates(game):
    coords = game.space.unravel(game.lines)  # (n_lines, length, 2)
    steps = np.diff(coords, axis=1)
    assert (np.abs(steps) <= 1).all()
    assert (steps == steps[:, :1]).all()  # one direction along each line
    assert (np.abs(steps).sum(axis=-1) > 0).all()
    assert len({tuple(sorted(line)) for line in game.lines.tolist()}) == len(game.lines)


def test_lines_through_each_cell_match_connect_fours_known_table():
    # The classic table of winning lines per cell, row by row from the bottom.
    table = [
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3],
    ]
    counts = np.bincount(CONNECT_FOUR.lines.ravel(), minlength=42)
    np.testing.assert_array_equal(CONNECT_FOUR.space.grid_view(counts), np.array(table).T)


def test_a_run_across_two_columns_in_flat_ids_is_not_a_line():
    game = InARow((7, 6), 4, gravity=False)
    owner = np.full(42, -1)
    owner[[4, 5, 6]] = 0  # column 0, rows 4 and 5; column 1, row 0
    owner[[20, 30, 40]] = 1
    after = game.step(at(game, owner), (PLACE, 7))  # column 1, row 1
    assert not after.over


@pytest.mark.parametrize("game", [InARow((7, 6), 4, gravity=False), TIC_TAC_TOE])
def test_completing_any_line_wins_on_any_cell_of_it(game):
    for line in game.lines:
        for last in line:
            owner = np.full(game.space.n_cells, -1)
            owner[line[line != last]] = 0
            position = at(game, owner, ply=0)
            after = game.step(position, (PLACE, int(last)))
            assert after.over
            assert after.winner == 0


def test_a_stone_falls_to_the_lowest_free_cell_of_its_column():
    position = CONNECT_FOUR.start(0)
    bottom = [cell(CONNECT_FOUR, c, 0) for c in range(7)]
    assert np.flatnonzero(CONNECT_FOUR.legal(position)[PLACE]).tolist() == bottom
    position = CONNECT_FOUR.step(position, (PLACE, cell(CONNECT_FOUR, 3, 0)))
    legal = CONNECT_FOUR.legal(position)[PLACE]
    assert legal[cell(CONNECT_FOUR, 3, 1)]
    assert not legal[cell(CONNECT_FOUR, 3, 0)]
    assert legal.sum() == 7
    with pytest.raises(ValueError, match="illegal"):
        CONNECT_FOUR.step(position, (PLACE, cell(CONNECT_FOUR, 4, 1)))  # a stone in the air


def test_a_full_column_takes_no_stone():
    position = CONNECT_FOUR.start(0)
    for _ in range(6):
        position = CONNECT_FOUR.step(position, (PLACE, int(np.argmax(position.state.owner < 0))))
    assert not CONNECT_FOUR.legal(position)[PLACE, :6].any()
    assert CONNECT_FOUR.legal(position)[PLACE].sum() == 6
    with pytest.raises(ValueError, match="illegal"):
        CONNECT_FOUR.step(position, (PLACE, 5))


def test_without_gravity_every_free_cell_is_legal():
    owner = np.array([0, -1, 1, -1, -1, 0, -1, -1, -1])
    legal = TIC_TAC_TOE.legal(at(TIC_TAC_TOE, owner, ply=4))
    np.testing.assert_array_equal(legal[PLACE], owner < 0)


def test_vertical_and_diagonal_wins_with_gravity():
    def play(columns):
        position = CONNECT_FOUR.start(0)
        for c in columns:
            legal = CONNECT_FOUR.legal(position)[PLACE]
            cells = [i for i in np.flatnonzero(legal) if CONNECT_FOUR.space.unravel(i)[0] == c]
            position = CONNECT_FOUR.step(position, (PLACE, cells[0]))
        return position

    vertical = play([2, 3, 2, 3, 2, 3, 2])
    assert vertical.over
    assert vertical.winner == 0
    assert vertical.ply == 7
    diagonal = play([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])  # player 0 up from (0, 0) to (3, 3)
    assert diagonal.over
    assert diagonal.winner == 0
    assert not play([0, 1, 1, 2, 2, 3, 2, 3, 3, 6]).over


def test_a_full_board_without_a_line_is_a_draw():
    # X O X / X O O / O X X, rows from the top; the last stone is X's at the top right.
    owner = np.full(9, -1)
    rows = {2: "XOX", 1: "XOO", 0: "OXX"}
    for row, marks in rows.items():
        for column, mark in enumerate(marks):
            owner[cell(TIC_TAC_TOE, column, row)] = "XO".index(mark)
    last = cell(TIC_TAC_TOE, 2, 2)
    owner[last] = -1
    after = TIC_TAC_TOE.step(at(TIC_TAC_TOE, owner), (PLACE, last))
    assert after.over
    assert after.winner is None
    assert not TIC_TAC_TOE.legal(after).any()


def test_a_line_on_the_last_free_cell_wins():
    owner = np.full(9, -1)
    rows = {2: "XOX", 1: "OXO", 0: "OX."}
    for row, marks in rows.items():
        for column, mark in enumerate(marks):
            if mark != ".":
                owner[cell(TIC_TAC_TOE, column, row)] = "XO".index(mark)
    after = TIC_TAC_TOE.step(at(TIC_TAC_TOE, owner), (PLACE, cell(TIC_TAC_TOE, 2, 0)))
    assert after.over
    assert after.winner == 0


def test_a_step_records_who_placed_where():
    position = CONNECT_FOUR.step(CONNECT_FOUR.start(0), (PLACE, 18))
    assert position.events == (Placed(0, 18),)
    assert CONNECT_FOUR.player(position) == 1
    assert not position.drew


def test_every_seed_starts_from_the_empty_board():
    a, b = CONNECT_FOUR.start(0), CONNECT_FOUR.start(12345)
    assert a.key() == b.key()
    assert (a.state.owner == -1).all()
    assert a.seed == 0
    assert b.seed == 12345


@pytest.mark.parametrize("game", [CONNECT_FOUR, TIC_TAC_TOE, SMALL])
def test_the_game_keeps_the_promises_of_a_game(game):
    check_game(game, UniformPlayer(), seeds=range(6))


def test_tic_tac_toe_reaches_5478_positions():
    seen = {}
    frontier = [TIC_TAC_TOE.start(0)]
    while frontier:
        nxt = []
        for position in frontier:
            key = position.state.owner.tobytes()
            if key in seen:
                continue
            seen[key] = position
            moves = np.flatnonzero(TIC_TAC_TOE.legal(position)[PLACE])
            nxt.extend(TIC_TAC_TOE.step(position, (PLACE, int(c))) for c in moves)
        frontier = nxt
    assert len(seen) == 5478


@pytest.mark.parametrize("plies", [6, pytest.param(7, marks=pytest.mark.slow)])
def test_connect_four_positions_by_ply_match_oeis(plies):
    level = {CONNECT_FOUR.start(0).state.owner.tobytes(): CONNECT_FOUR.start(0)}
    counts = [1]
    for _ in range(plies):
        nxt = {}
        for position in level.values():
            for c in np.flatnonzero(CONNECT_FOUR.legal(position)[PLACE]):
                after = CONNECT_FOUR.step(position, (PLACE, int(c)))
                nxt.setdefault(after.state.owner.tobytes(), after)
        level = nxt
        counts.append(len(level))
    assert tuple(counts) == A212693[: plies + 1]


def test_a_finished_game_takes_no_step():
    owner = np.full(9, -1)
    owner[[0, 1]] = 0
    owner[[3, 4]] = 1
    over = TIC_TAC_TOE.step(at(TIC_TAC_TOE, owner), (PLACE, 2))
    assert over.over
    with pytest.raises(ValueError, match="over"):
        TIC_TAC_TOE.step(over, (PLACE, 5))
    with pytest.raises(ValueError, match="over"):
        TIC_TAC_TOE.player(over)


def test_an_unknown_verb_is_refused():
    with pytest.raises(ValueError, match="verb"):
        CONNECT_FOUR.step(CONNECT_FOUR.start(0), (1, 0))
    with pytest.raises(ValueError, match="verb"):
        CONNECT_FOUR.step(CONNECT_FOUR.start(0), (-1, 0))


def test_an_action_of_the_wrong_type_raises_type_error(not_an_integer):
    position = CONNECT_FOUR.start(0)
    with pytest.raises(TypeError):  # a verb that isn't an integer
        CONNECT_FOUR.step(position, (not_an_integer, 0))
    with pytest.raises(TypeError):  # a cell that isn't an integer
        CONNECT_FOUR.step(position, (PLACE, not_an_integer))


@pytest.mark.parametrize("bad", [-1, 42])
def test_a_cell_off_the_board_raises_index_error(bad):
    with pytest.raises(IndexError):
        CONNECT_FOUR.step(CONNECT_FOUR.start(0), (PLACE, bad))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"shape": [7, 6]}, TypeError),
        ({"shape": (7,)}, TypeError),
        ({"shape": (7.0, 6)}, TypeError),
        ({"shape": (0, 6)}, ValueError),
        ({"length": 0}, ValueError),
        ({"length": 4.0}, TypeError),
        ({"length": True}, TypeError),
        ({"gravity": 1}, TypeError),
        ({"shape": (3, 2), "length": 4}, ValueError),
    ],
)
def test_bad_rules_raise(kwargs, error):
    with pytest.raises(error):
        InARow(**kwargs)
