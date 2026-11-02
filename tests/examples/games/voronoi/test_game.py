"""Tests of the rules of the Voronoi game: the board, the sea, territory, turns, results, errors."""

import numpy as np
import pytest

from examples.games.voronoi import (
    PLACE,
    Placed,
    Voronoi,
    VoronoiPosition,
    VoronoiState,
    board,
)
from game_demo import check_game
from gridzero.agents import UniformPlayer
from gridzero.core import MeshSpace, steps
from gridzero.game import play

ICOSAHEDRON = Voronoi(frequency=1, stones=1, sea=0.0)
"""12 cells, each with 5 neighbors, 5 cells two moves away and 1 opposite, three away."""


def at(game, stone, land=None, ply=None):
    """Return a position of ``game`` with ``stone`` per cell, and all land by default."""
    stone = np.asarray(stone, dtype=np.int8)
    land = np.ones(stone.size, dtype=np.bool_) if land is None else np.asarray(land)
    state = VoronoiState(game.space, {VoronoiState.stone: stone, VoronoiState.land: land})
    return VoronoiPosition(state, 0, ply=int((stone >= 0).sum()) if ply is None else ply)


def stones(n_cells, **cells):
    """Return a stone array with player ``int(name[1:])`` on each named cell: ``p0=3, p1=7``."""
    stone = np.full(n_cells, -1, dtype=np.int8)
    for name, cell in cells.items():
        stone[cell] = int(name[1:])
    return stone


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize(("frequency", "sea"), [(1, 0.0), (1, 0.25), (2, 0.25)])
def test_the_game_keeps_the_promises_of_a_game(players, frequency, sea):
    game = Voronoi(frequency=frequency, players=players, stones=2, sea=sea)
    check_game(game, UniformPlayer(), seeds=range(3))


def test_the_board_is_an_icosphere_shared_by_every_position():
    game = Voronoi(frequency=2)
    assert isinstance(game.space, MeshSpace)
    assert game.space.n_cells == 42
    assert game.space is board(2) is game.start(0).state.space is game.start(5).state.space


def test_the_sea_is_drawn_from_the_seed_and_covers_its_share():
    game = Voronoi(frequency=2, sea=0.25)
    land = [game.start(seed).state.land for seed in range(6)]
    assert all((~mask).sum() == round(0.25 * 42) for mask in land)
    np.testing.assert_array_equal(game.start(3).state.land, land[3])
    assert len({mask.tobytes() for mask in land}) == 6
    assert Voronoi(frequency=2, sea=0.0).start(0).state.land.all()


def test_the_sea_comes_from_the_seeds_stream_0():
    game = Voronoi(frequency=2, sea=0.25)
    rng = np.random.default_rng(np.random.SeedSequence(9, spawn_key=(0,)))  # Context's stream
    sea = rng.choice(42, round(0.25 * 42), replace=False)
    np.testing.assert_array_equal(np.flatnonzero(~game.start(9).state.land), np.sort(sea))


def test_only_free_land_is_legal():
    game = Voronoi(frequency=1, sea=0.25)
    position = game.start(1)
    position = game.step(position, (PLACE, int(np.flatnonzero(position.state.land)[0])))
    legal = game.legal(position)
    assert legal.shape == (1, 12)
    np.testing.assert_array_equal(legal[0], position.state.land & (position.state.stone < 0))
    sea = int(np.flatnonzero(~position.state.land)[0])
    with pytest.raises(ValueError, match="sea or taken"):
        game.step(position, (PLACE, sea))


def test_a_stone_holds_the_cells_nearer_to_it_than_to_any_other():
    space = ICOSAHEDRON.space
    moves = steps(space, 0)
    opposite = int(np.flatnonzero(moves == 3)[0])
    state = at(ICOSAHEDRON, stones(12, p0=0, p1=opposite)).state
    np.testing.assert_array_equal(state.territory[moves < 1.5], 0)  # the stone and its ring
    np.testing.assert_array_equal(state.territory[moves > 1.5], 1)  # the opposite half
    assert (state.territory == 0).sum() == (state.territory == 1).sum() == 6


def test_cells_as_near_to_two_players_belong_to_nobody():
    space = ICOSAHEDRON.space
    a, b = 0, int(space.adjacency().neighbors(0)[0])
    state = at(ICOSAHEDRON, stones(12, p0=a, p1=b)).state
    to_a, to_b = steps(space, a), steps(space, b)
    np.testing.assert_array_equal(state.territory == -1, to_a == to_b)
    np.testing.assert_array_equal(state.territory == 0, to_a < to_b)
    assert (to_a == to_b).sum() > 0


def test_the_sea_is_held_by_nobody_and_blocks_the_way():
    space = Voronoi(frequency=2).space
    cut_off = 5
    land = np.ones(42, dtype=np.bool_)
    land[space.adjacency().neighbors(cut_off)] = False  # an island of one cell
    stone = stones(42, p0=int(np.flatnonzero(land & (steps(space, cut_off) > 2))[0]))
    state = at(Voronoi(frequency=2), stone, land).state
    assert (state.territory[~land] == -1).all()
    assert state.territory[cut_off] == -1  # no stone reaches the island
    assert (state.territory[land] == 0).sum() == land.sum() - 1


def test_areas_add_up_with_the_land_nobody_holds_and_the_sea_to_the_sphere():
    game = Voronoi(frequency=2, players=3, stones=2, sea=0.25)
    record = play(game, [UniformPlayer()] * 3, game.start(4))
    state = record.final.state
    areas = game.areas(state)
    assert areas.shape == (3,)
    assert areas.dtype == np.float64
    nobody = state.space.measures[state.territory < 0].sum()
    assert areas.sum() + nobody == pytest.approx(4 * np.pi)
    held = [state.space.measures[state.territory == p].sum() for p in range(3)]
    np.testing.assert_allclose(areas, held)


def test_the_players_take_turns_until_every_stone_is_down():
    game = Voronoi(frequency=2, players=3, stones=2)
    record = play(game, [UniformPlayer()] * 3, game.start(0))
    assert record.final.ply == 6
    assert all(isinstance(e, Placed) for e in record.events)
    assert [e.player for e in record.events if isinstance(e, Placed)] == [0, 1, 2, 0, 1, 2]
    assert np.bincount(record.final.state.stone[record.final.state.stone >= 0]).tolist() == [2] * 3


def test_the_game_ends_when_the_land_runs_out():
    game = Voronoi(frequency=1, players=4, stones=3, sea=0.25)  # 9 land cells, 12 stones
    record = play(game, [UniformPlayer()] * 4, game.start(2))
    assert record.final.ply == 9
    assert not game.legal(record.final).any()


def test_the_most_area_wins_and_a_symmetric_split_is_a_draw():
    space = ICOSAHEDRON.space
    opposite = int(np.flatnonzero(steps(space, 0) == 3)[0])
    split = ICOSAHEDRON.step(at(ICOSAHEDRON, stones(12, p0=0), ply=1), (PLACE, opposite))
    assert split.over
    assert split.winner is None  # 6 cells each: equal areas but for rounding
    two = Voronoi(frequency=2, stones=1, sea=0.0)
    graph = two.space.adjacency()
    degree = np.diff(graph.indptr)
    pentagon = int(np.flatnonzero(degree == 5)[0])  # no symmetry maps it to a hexagon
    hexagon = int(np.flatnonzero((degree == 6) & (steps(two.space, pentagon) == 2))[0])
    lopsided = two.step(at(two, stones(42, p0=pentagon), ply=1), (PLACE, hexagon))
    areas = two.areas(lopsided.state)
    assert abs(areas[0] - areas[1]) > 1e-3
    assert lopsided.winner == int(np.argmax(areas))


def test_no_step_draws_from_its_stream():
    game = Voronoi(frequency=2, players=3, stones=2)
    record = play(game, [UniformPlayer()] * 3, game.start(0))
    position = record.start
    for action in record.actions:
        position = game.step(position, action)
        assert not position.drew


def test_start_refuses_a_seed_that_is_negative_or_not_an_integer(not_an_integer):
    with pytest.raises(TypeError, match="seed"):
        Voronoi().start(not_an_integer)
    with pytest.raises(ValueError, match="seed"):
        Voronoi().start(-1)


def test_the_rules_refuse_bad_parameters(not_an_integer, not_a_real):
    for name in ("frequency", "players", "stones"):
        with pytest.raises(TypeError, match=name):
            Voronoi(**{name: not_an_integer})
    with pytest.raises(ValueError, match="frequency"):
        Voronoi(frequency=0)
    with pytest.raises(ValueError, match="players"):
        Voronoi(players=1)
    with pytest.raises(ValueError, match="stones"):
        Voronoi(stones=0)
    with pytest.raises(TypeError, match="sea"):
        Voronoi(sea=not_a_real)
    for sea in (-0.1, 1.0):
        with pytest.raises(ValueError, match="sea"):
            Voronoi(sea=sea)
