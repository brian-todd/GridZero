import numpy as np
import pytest

from examples.games.tactics import TacticsState, units
from gridzero.core import icosphere
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, piece


def test_units_lists_the_units_on_the_board_by_uid(make_battle, at):
    position = make_battle(
        [piece(4, 4, KNIGHT, 1), piece(0, 0, FOOT, 0, facing=3), piece(8, 8, CAPTAIN, 0)]
    )
    listed = units(position.state)
    assert len(listed) == 3
    np.testing.assert_array_equal(listed.uid, [0, 1, 2])
    np.testing.assert_array_equal(listed.cell, [at(4, 4), at(0, 0), at(8, 8)])
    np.testing.assert_array_equal(listed.kind, [KNIGHT, FOOT, CAPTAIN])
    np.testing.assert_array_equal(listed.team, [1, 0, 0])
    np.testing.assert_array_equal(listed.hp, [7, 5, 9])
    np.testing.assert_array_equal(listed.facing, [0, 3, 0])
    with pytest.raises(ValueError, match="read-only"):
        listed.hp[0] = 1


def test_units_of_an_empty_board_is_empty(make_battle):
    assert len(units(make_battle([]).state)) == 0


def test_occupied_and_passable_follow_units_walls_and_rubble(make_battle, at):
    position = make_battle([piece(1, 1, ARCHER, 0)], walls=[(2, 2)], rubble=[(3, 3)])
    state = position.state
    assert np.flatnonzero(state.occupied).tolist() == [at(1, 1)]
    assert set(np.flatnonzero(~state.passable).tolist()) == {at(2, 2), at(3, 3)}


def test_a_board_is_made_only_on_a_lattice():
    planet = icosphere(1)
    values = {
        TacticsState.wall: np.zeros(planet.n_cells, bool),
        TacticsState.rubble: np.zeros(planet.n_cells, bool),
        TacticsState.home: np.full(planet.n_cells, -1),
    }
    with pytest.raises(TypeError, match="LatticeSpace"):
        TacticsState(planet, values)
