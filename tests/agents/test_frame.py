import dataclasses

import numpy as np
import pytest

from agents_demo import GAMBLE, PLACE, Patches, Took, check_learnable
from game_demo import check_game
from gridzero.agents.frame import Frame
from gridzero.agents.random import UniformPlayer
from gridzero.core import LatticeSpace
from gridzero.game import Action

GAME = Patches()


def test_into_gathers_values_at_the_frames_verbs_and_cells(rng):
    frame = Frame([2, 0, 1], [1, 0])
    values = rng.random((4, 2, 3))
    turned = frame.into(values)
    for v in range(2):
        for c in range(3):
            np.testing.assert_array_equal(
                turned[:, v, c], values[:, frame.verbs[v], frame.cells[c]]
            )


def test_back_undoes_into_also_for_frames_that_are_not_their_own_inverse(rng):
    frame = Frame([2, 0, 1], [1, 0])  # a rotation of three cells: its inverse differs
    values = rng.random((2, 3))
    np.testing.assert_array_equal(frame.back(frame.into(values)), values)
    np.testing.assert_array_equal(frame.into(frame.back(values)), values)
    assert not np.array_equal(frame.into(frame.into(values)), values)


def test_one_action_goes_where_its_values_go():
    frame = Frame([2, 0, 1], [1, 0])
    for verb in range(2):
        for cell in range(3):
            one = np.zeros((2, 3), dtype=bool)
            one[verb, cell] = True
            turned = frame.action_into(Action(verb, cell))
            assert np.argwhere(frame.into(one)).tolist() == [list(turned)]
            assert frame.action_back(turned) == (verb, cell)


def test_the_board_frame_returns_values_and_actions_unchanged():
    frame = Frame.board(2, 3)
    values = np.ones((2, 3))
    assert frame.identity
    assert frame.into(values) is values
    assert frame.back(values) is values
    assert frame.action_into((1, 2)) == (1, 2)
    assert not Frame([0, 1, 2], [1, 0]).identity
    assert not Frame([0, 2, 1], [0, 1]).identity


def test_frames_are_read_only_copies():
    cells = np.array([1, 0])
    frame = Frame(cells, [0])
    cells[0] = 0
    assert frame.cells.tolist() == [1, 0]
    assert not frame.cells.flags.writeable


def test_frames_need_permutations_and_values_of_their_shape(not_an_integer):
    with pytest.raises(ValueError, match="permutation"):
        Frame([0, 0, 1], [0])
    with pytest.raises(ValueError, match="permutation"):
        Frame([0, 1], [1, 2])
    with pytest.raises(ValueError, match="1-D"):
        Frame([[0, 1]], [0])
    with pytest.raises(TypeError, match="cells"):
        Frame([0.0, 1.0], [0])
    frame = Frame([1, 0], [0])
    with pytest.raises(ValueError, match="shape"):
        frame.into(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        frame.back(np.zeros((2, 2)))
    with pytest.raises(IndexError, match="cell id 2"):
        frame.action_into((0, 2))
    for convert in (frame.action_into, frame.action_back):
        with pytest.raises(IndexError, match=r"verb 1 is out of range \[0, 1\)"):
            convert((1, 0))
        with pytest.raises(IndexError, match="verb -1"):
            convert((-1, 0))
        with pytest.raises(TypeError, match="verb"):
            convert((not_an_integer, 0))
    with pytest.raises(TypeError):
        frame.action_into((0, not_an_integer))
    with pytest.raises(TypeError):
        Frame.board(not_an_integer, 3)


def test_patches_is_a_game_and_learnable():
    records = check_game(GAME, UniformPlayer(), range(6), max_plies=12)
    check_learnable(GAME, records)
    assert {r.final.winner for r in records} >= {0, 1}


def test_a_gamble_draws_the_coin_and_a_placement_does_not():
    start = GAME.start(0)
    assert not GAME.step(start, (PLACE, 5)).drew
    assert GAME.step(start, (GAMBLE, 5)).drew
    outcomes = {GAME.step(GAME.start(s), (GAMBLE, 5)).events for s in range(20)}
    # heads: cell 5 and its free neighbor of lowest id, 1; tails: cell 5 to player 1
    assert outcomes == {(Took(0, (5, 1)),), (Took(1, (5,)),)}


def test_player_one_sees_the_board_turned_half_a_turn():
    position = GAME.step(GAME.start(0), (PLACE, 0))
    own, enemy, free = GAME.observe(position, 0)
    assert (own[0], enemy.sum(), free[0]) == (1, 0, 0)
    own, enemy, free = GAME.observe(position, 1)
    corner = GAME.space.n_cells - 1  # cell 0 turned half a turn
    assert (enemy[corner], own.sum()) == (1, 0)
    assert GAME.frame(1).action_into((PLACE, 0)) == (PLACE, corner)


@pytest.mark.parametrize(
    ("rows", "winner"),
    [
        # Cell (i, j) is row i, column j. Player 0: a group of 5 and 5 stones; player 1: a
        # group of 4 and 7 stones.
        (["0001", "1101", "1101"], 0),
        # Groups of 3 for both; player 1 has 7 stones against 5.
        (["0110", "1001", "1011"], 1),
        # Groups of 2 and 6 stones each.
        (["0011", "1100", "0011"], None),
    ],
)
def test_the_larger_group_wins_then_more_stones(rows, winner):
    position = GAME.start(0)
    for i, row in enumerate(rows):
        for j, owner in enumerate(row):
            if (i, j) != (2, 3):
                position = dataclasses.replace(position, ply=int(owner))
                position = GAME.step(position, (PLACE, 4 * i + j))
    last = dataclasses.replace(position, ply=int(rows[2][3]))
    final = GAME.step(last, (PLACE, 11))
    assert final.over
    assert final.winner == winner


def test_the_opposite_frame_turns_cells_and_direction_verbs_half_way_round():
    space = LatticeSpace((3, 5), periodic=False)  # squares: 4 directions, verbs 2 to 5
    frame = Frame.opposite(space, 7, directions=2)
    np.testing.assert_array_equal(frame.cells, space.opposite())
    np.testing.assert_array_equal(frame.verbs, [0, 1, 4, 5, 2, 3, 6])
    for d, step in enumerate(space.directions()):
        cell = space.ravel([1, 2])  # the middle, which the half turn keeps
        verb, there = frame.action_back((2 + d, space.ravel(space.unravel(cell) + step)))
        assert there == space.ravel(space.unravel(cell) - step)
        np.testing.assert_array_equal(space.directions()[verb - 2], -step)


def test_the_opposite_frame_is_its_own_inverse_and_keeps_other_verbs():
    space = LatticeSpace((4, 4), tiling="hex", periodic=False)
    frame = Frame.opposite(space, 9, directions=1)
    np.testing.assert_array_equal(frame.verbs[frame.verbs], np.arange(9))
    np.testing.assert_array_equal(frame.cells[frame.cells], np.arange(space.n_cells))
    np.testing.assert_array_equal(frame.verbs[[0, 7, 8]], [0, 7, 8])
    np.testing.assert_array_equal(Frame.opposite(space, 3).verbs, [0, 1, 2])


def test_the_opposite_frame_needs_direction_verbs_that_fit_and_a_lattice_with_them(
    not_an_integer,
):
    hexes = LatticeSpace((4, 4), tiling="hex", periodic=False)
    with pytest.raises(ValueError, match="don't fit"):
        Frame.opposite(hexes, 6, directions=1)
    triangles = LatticeSpace((5, 4), tiling="triangle", periodic=False)
    Frame.opposite(triangles, 3)  # opposite cells, and no direction verbs
    with pytest.raises(ValueError, match="squares or hexes"):
        Frame.opposite(triangles, 3, directions=0)
    if not_an_integer is not None:  # None means no direction verbs
        with pytest.raises(TypeError):  # verb numbers are integers
            Frame.opposite(hexes, 9, directions=not_an_integer)
