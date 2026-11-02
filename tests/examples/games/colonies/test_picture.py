"""Tests of Colonies' colors."""

import numpy as np
from matplotlib import colormaps

from examples.games.colonies import (
    FOG,
    MAX_PLAYERS,
    PLAYERS,
    Colonies,
    ColoniesState,
    ColoniesView,
    Raider,
    Settler,
    colors,
    view_colors,
)
from examples.games.colonies.picture import EMPTY, SEA
from gridzero.agents import UniformPlayer
from gridzero.game import play


def test_each_cell_is_colored_by_the_sea_empty_land_or_its_most_people():
    game = Colonies(frequency=2, players=3, rounds=5)
    state = play(game, [Raider(), UniformPlayer(), Settler()], game.start(1)).final.state
    out = colors(state)
    assert out.shape == (42, 4)
    assert out.dtype == np.float64
    assert ((out >= 0) & (out <= 1)).all()
    assert (out[~state.land] == SEA).all()
    total = state.people.sum(axis=0)
    assert (out[state.land & (total == 0)] == EMPTY).all()
    full = state.land & (total >= state.capacity)
    most = np.argmax(state.people, axis=0)
    for p in range(3):  # a full cell is its player's color alone, one per player
        mine = {tuple(c) for c in out[full & (most == p)]}
        assert len(mine) <= 1
        assert not mine & {SEA, EMPTY}
    assert len({tuple(c) for c in out[full]}) == len(set(most[full].tolist()))


def test_a_cell_takes_the_color_of_its_most_people_by_how_full_it_is():
    game = Colonies(frequency=1, players=3)
    people = np.zeros((MAX_PLAYERS, 12))
    people[1, 5] = 10.0  # full, alone
    people[[0, 2], 7] = [1.0, 4.0]  # half full, mostly player 2's
    capacity = np.full(12, 10.0)
    state = ColoniesState(
        game.space, {ColoniesState.capacity: capacity, ColoniesState.people: people}
    )
    out = colors(state)
    palette = colormaps["tab10"](np.arange(1, 10))[:, :3]
    np.testing.assert_allclose(out[5, :3], palette[1])
    np.testing.assert_allclose(out[7, :3], 0.5 * palette[2] + 0.5 * np.asarray(EMPTY[:3]))
    assert (out[[0, 1, 2]] == EMPTY).all()


def test_players_colors_are_tab10_from_its_second():
    np.testing.assert_array_equal(PLAYERS, colormaps["tab10"](np.arange(1, 10))[:, :3])


def test_a_view_shows_only_the_people_seen_and_darkens_the_cells_unseen():
    game = Colonies(frequency=1, players=2)
    people = np.zeros((MAX_PLAYERS, 12))
    people[1, 5] = 10.0  # an enemy, full, unseen
    people[0, 7] = 10.0  # player 0, full, seen
    capacity = np.full(12, 10.0)
    state = ColoniesState(
        game.space, {ColoniesState.capacity: capacity, ColoniesState.people: people}
    )
    seen = np.isin(np.arange(12), [7, 8])
    view = ColoniesView(seen, state.land, capacity, people[:2] * seen, 0, (False, False))
    out = view_colors(state, view)
    empty = np.asarray(EMPTY[:3])
    np.testing.assert_allclose(out[7, :3], PLAYERS[0])
    np.testing.assert_allclose(out[8, :3], empty)
    np.testing.assert_allclose(out[~seen, :3], np.tile(FOG * empty, (10, 1)))  # cell 5 too
    assert (out[:, 3] == 1).all()
