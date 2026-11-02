import numpy as np
import pytest

from examples.games.tactics import (
    TacticsState,
    board_space,
    homes,
    make_board,
)
from examples.games.tactics.board import home_corners
from game_demo import is_symmetric
from gridzero.core import regions, steps


@pytest.fixture(scope="module")
def board():
    return make_board((32, 32), seed=7)


def test_board_space_is_a_shared_hex_lattice_that_does_not_wrap():
    space = board_space((6, 8))
    assert space.tiling == "hex"
    assert space.shape == (6, 8)
    assert not any(space.periodic)
    assert board_space((6, 8)) is space


def test_opposite_swaps_the_home_corners():
    space = board_space((6, 8))
    m = space.opposite()
    np.testing.assert_array_equal(m[m], np.arange(space.n_cells))
    a, b = home_corners(space)
    assert (m[a], m[b]) == (b, a)


def test_home_corners_are_the_short_diagonal():
    space = board_space((32, 32))
    a, b = home_corners(space)
    assert steps(space, a)[b] == 31


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_homes_hold_the_tiles_fewer_than_depth_steps_from_their_corner(depth):
    space = board_space((32, 32))
    home = homes(space, depth)
    for team, corner in enumerate(home_corners(space)):
        near = steps(space, corner) < depth
        np.testing.assert_array_equal(home == team, near)
        assert near.sum() == depth**2
    np.testing.assert_array_equal(home[space.opposite()], np.where(home >= 0, 1 - home, -1))


def test_homes_that_would_overlap_raise():
    with pytest.raises(ValueError, match="overlap"):
        homes(board_space((6, 6)), 4)


def test_homes_reject_depths_below_one_and_non_integers(not_an_integer):
    with pytest.raises(ValueError, match="depth"):
        homes(board_space((6, 6)), 0)
    with pytest.raises(TypeError, match="depth"):
        homes(board_space((6, 6)), not_an_integer)


def test_board_is_the_same_from_both_sides(board):
    assert is_symmetric(board, board.space.opposite(), seats=(TacticsState.home,))


def test_board_homes_are_clear_and_every_open_tile_is_reachable(board):
    assert not (board.wall | board.rubble)[board.home >= 0].any()
    assert (board.home == 0).sum() == (board.home == 1).sum() == 25
    pieces = regions(board.space, board.passable)
    assert pieces.max() == 0


def test_crowded_boards_join_their_homes_and_close_every_pocket():
    # Many walls on a small board: pockets to close, and draws whose homes are cut off.
    space = board_space((12, 12))
    a, b = home_corners(space)
    for seed in range(40):
        board = make_board((12, 12), seed=seed, walls=0.3, rubble=0.0, home_depth=3)
        pieces = regions(space, board.passable)
        assert pieces.max() == 0, seed
        assert pieces[a] == pieces[b] == 0
        assert not board.wall[board.home >= 0].any()


def test_board_has_about_the_shares_of_walls_and_rubble_asked_for(board):
    outside = board.home < 0
    assert 0.08 <= board.wall[outside].mean() <= 0.12  # cut-off pockets become walls
    assert board.rubble[outside].mean() == pytest.approx(0.05, abs=0.01)
    assert not (board.wall & board.rubble).any()


def test_board_has_no_units(board):
    assert not board.occupied.any()


def test_same_seed_gives_the_same_board_and_another_seed_another(board):
    again = make_board((32, 32), seed=7)
    other = make_board((32, 32), seed=8)
    np.testing.assert_array_equal(again.wall, board.wall)
    np.testing.assert_array_equal(again.rubble, board.rubble)
    assert not np.array_equal(other.wall, board.wall)


def test_board_without_obstacles_is_open():
    b = make_board((9, 9), seed=0, walls=0.0, rubble=0.0, home_depth=3)
    assert b.passable.all()


def test_board_rejects_bad_shares_and_seeds(not_an_integer):
    with pytest.raises(ValueError, match="below 1"):
        make_board((9, 9), seed=0, walls=0.6, rubble=0.4, home_depth=3)
    with pytest.raises(ValueError, match="walls"):
        make_board((9, 9), seed=0, walls=-0.1, home_depth=3)
    with pytest.raises(ValueError, match="seed"):
        make_board((9, 9), seed=-1, home_depth=3)
    with pytest.raises(TypeError, match="seed"):
        make_board((9, 9), seed=not_an_integer, home_depth=3)
