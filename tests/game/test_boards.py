import numpy as np
import pytest

from game_demo import is_symmetric
from gridzero.core import Field, LatticeSpace, State, icosphere, regions, steps
from gridzero.game import highest, joined, pockets, symmetric, zones

HEXES = LatticeSpace((6, 6), tiling="hex", periodic=False)
SQUARES = LatticeSpace((4, 6), periodic=False)


def mirror(space):
    """The map that turns a square board's cells left to right."""
    i, j = space.unravel(np.arange(space.n_cells)).T
    return space.ravel(np.stack([i, space.shape[1] - 1 - j], axis=1))


def test_symmetric_values_are_the_same_through_the_map(rng):
    values = rng.random(HEXES.n_cells)
    m = HEXES.opposite()
    out = symmetric(values, m)
    np.testing.assert_array_equal(out[m], out)
    np.testing.assert_array_equal(out, values + values[m])  # one involution: the plain sum


def test_symmetric_values_over_a_group_are_exact_in_every_orbit(rng):
    # The half turn and the two mirrors of a rectangle, with the identity, are a group of four.
    half, left_right = SQUARES.opposite(), mirror(SQUARES)
    up_down = left_right[half]
    values = rng.random(SQUARES.n_cells) * 10.0 ** rng.integers(-8, 8, SQUARES.n_cells)
    out = symmetric(values, half, left_right, up_down)
    for m in (half, left_right, up_down):
        np.testing.assert_array_equal(out[m], out)  # bit for bit, though floats don't associate
    orbit = np.sort(np.stack([values, values[half], values[left_right], values[up_down]]), axis=0)
    np.testing.assert_allclose(out, orbit.sum(axis=0), rtol=1e-15)


def test_symmetric_needs_a_group_of_permutations_and_finite_values(rng):
    values = rng.random(SQUARES.n_cells)
    with pytest.raises(ValueError, match="at least one map"):
        symmetric(values)
    with pytest.raises(ValueError, match="group"):
        symmetric(values, mirror(SQUARES), SQUARES.opposite())  # their product is missing
    with pytest.raises(ValueError, match="permutation"):
        symmetric(values, np.zeros(SQUARES.n_cells, dtype=np.int64))
    with pytest.raises(ValueError, match="permutation"):
        symmetric(values, SQUARES.opposite()[:-1])
    with pytest.raises(TypeError):  # cell ids are integers
        symmetric(values, SQUARES.opposite().astype(float))
    with pytest.raises(ValueError, match="finite"):
        symmetric(np.full(SQUARES.n_cells, np.nan), SQUARES.opposite())
    with pytest.raises(ValueError, match="1-D"):
        symmetric(values[None], SQUARES.opposite())


def test_highest_picks_a_share_of_the_cells_considered_by_value():
    values = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.0])
    np.testing.assert_array_equal(highest(values, 0.5), [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(highest(values, 0.0), np.zeros(6, dtype=bool))
    np.testing.assert_array_equal(highest(values, 1.0), np.ones(6, dtype=bool))
    where = np.array([False, True, True, True, True, True])
    np.testing.assert_array_equal(highest(values, 0.4, where), [0, 0, 1, 0, 1, 0])


def test_highest_rounds_the_count_to_the_nearest():
    values = np.arange(6.0)
    assert highest(values, 0.6).sum() == 4  # 3.6 cells
    assert highest(values, 0.55).sum() == 3  # 3.3 cells


def test_highest_of_no_share_picks_nothing_even_at_infinity():
    values = np.array([np.inf, 1.0, 2.0, np.inf])
    np.testing.assert_array_equal(highest(values, 0.0), np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(highest(values, 0.4), [1, 0, 0, 1])  # 1.6 cells: ties at inf


def test_highest_picks_ties_at_the_cut_together():
    values = np.array([3.0, 2.0, 2.0, 1.0])
    np.testing.assert_array_equal(highest(values, 0.5), [1, 1, 1, 0])  # 2 asked, 3 picked


def test_highest_of_symmetric_values_is_symmetric(rng):
    m = HEXES.opposite()
    values = symmetric(rng.random(HEXES.n_cells), m)
    for share in (0.1, 0.33, 0.5, 0.9):
        picked = highest(values, share)
        np.testing.assert_array_equal(picked[m], picked)


def test_highest_checks_its_inputs(not_a_real):
    values = np.arange(4.0)
    with pytest.raises(ValueError, match="share"):
        highest(values, 1.5)
    with pytest.raises(TypeError):  # a share is a real number
        highest(values, not_a_real)
    with pytest.raises(ValueError, match="NaN"):
        highest(np.array([1.0, np.nan]), 0.5)
    with pytest.raises((TypeError, ValueError)):  # a mask of the cells, of booleans
        highest(values, 0.5, np.ones(3, dtype=bool))


def strip(open_cells):
    """A row of squares and which of them are open."""
    space = LatticeSpace((1, len(open_cells)), periodic=False)
    return space, np.array([c == "." for c in open_cells])


def test_cells_are_joined_through_open_cells_that_touch():
    space, open_ = strip("..#..")
    assert joined(space, open_, [0, 1])
    assert not joined(space, open_, [0, 4])
    assert not joined(space, open_, [2])  # a closed cell is joined to nothing
    assert joined(space, open_, [4])


def test_pockets_are_the_open_cells_cut_off_from_the_cell_kept():
    space, open_ = strip("..#..#.")
    np.testing.assert_array_equal(pockets(space, open_, 0), [0, 0, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(pockets(space, open_, 4), [1, 1, 0, 0, 0, 0, 1])
    closed = open_ & ~pockets(space, open_, 0)
    assert regions(space, closed).max() == 0


def test_joined_and_pockets_work_on_meshes(rng):
    space = icosphere(2)
    open_ = rng.random(space.n_cells) < 0.6
    keep = int(np.flatnonzero(open_)[0])
    left = open_ & ~pockets(space, open_, keep)
    assert joined(space, left, np.flatnonzero(left))
    assert regions(space, left).max() == 0


def test_joined_and_pockets_check_their_cells():
    space, open_ = strip("..#")
    with pytest.raises(ValueError, match="at least one"):
        joined(space, open_, [])
    with pytest.raises(IndexError):
        joined(space, open_, [3])
    with pytest.raises(ValueError, match="not open"):
        pockets(space, open_, 2)
    with pytest.raises(IndexError):
        pockets(space, open_, 3)


@pytest.mark.parametrize("within", [0, 1, 2, 3])
def test_zones_hold_the_cells_within_steps_of_each_seed(within):
    seeds = [0, HEXES.n_cells - 1]
    zone = zones(HEXES, seeds, within)
    assert zone.dtype == np.int8
    for seat, seed in enumerate(seeds):
        np.testing.assert_array_equal(zone == seat, steps(HEXES, seed) <= within)
    assert (zone >= -1).all()


def test_zones_of_seeds_opposite_each_other_are_symmetric():
    seeds = [0, int(HEXES.opposite()[0])]
    zone = zones(HEXES, seeds, 2)
    np.testing.assert_array_equal(zone[HEXES.opposite()], np.where(zone >= 0, 1 - zone, -1))


def test_zones_take_any_number_of_seats_and_refuse_overlaps(not_an_integer):
    corners = [0, 5, 30, 35]
    zone = zones(HEXES, corners, 1)
    assert set(zone[corners].tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="overlap"):
        zones(HEXES, [0, 1], 1)
    with pytest.raises(ValueError, match="1 to 127"):
        zones(HEXES, [], 1)
    many = LatticeSpace((12, 12), periodic=False)
    assert zones(many, np.arange(127), 0).max() == 126
    with pytest.raises(ValueError, match="1 to 127"):
        zones(many, np.arange(128), 0)  # seats are int8
    with pytest.raises(ValueError, match="within"):
        zones(HEXES, [0], -1)
    with pytest.raises(TypeError, match="within"):
        zones(HEXES, [0], not_an_integer)
    with pytest.raises(IndexError):
        zones(HEXES, [36], 1)


class BoardState(State):
    wall = Field(np.bool_)
    home = Field(np.int8, default=-1)


def test_a_board_made_from_these_is_the_same_for_both_seats(rng):
    # The recipe of Tactics' boards on squares: symmetric noise, walls on the highest share
    # away from the homes, pockets closed.
    m = SQUARES.opposite()
    home = zones(SQUARES, [0, int(m[0])], 1)
    for _ in range(20):
        wall = highest(symmetric(rng.random(SQUARES.n_cells), m), 0.3, home < 0)
        if joined(SQUARES, ~wall, [0, int(m[0])]):
            break
    else:
        pytest.fail("no board joined the homes")
    wall |= pockets(SQUARES, ~wall, 0)
    board = BoardState(SQUARES, {BoardState.wall: wall, BoardState.home: home})
    assert is_symmetric(board, m, seats=(BoardState.home,))
    uneven = BoardState(SQUARES, {BoardState.wall: wall, BoardState.home: zones(SQUARES, [0], 1)})
    assert not is_symmetric(uneven, m, seats=(BoardState.home,))
    assert not is_symmetric(board, m)  # without relabeling, the homes differ
