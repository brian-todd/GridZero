"""The engineer's work: the ladder of terrain, what work spends, and the rule against sealing."""

import dataclasses

import numpy as np
import pytest

from agents_demo import check_learnable
from examples.games.tactics import (
    DEFAULT_UNITS,
    Action,
    Broke,
    Built,
    GameOver,
    GreedyPlayer,
    Hold,
    RandomPlayer,
    Tactics,
    Verb,
    summarize,
)
from examples.games.tactics.text import event_text
from game_demo import check_game
from gridzero.core import hops, regions
from gridzero.game import Record, joined, play, series
from tactics_demo import ARCHER, ENGINEER, FOOT, KNIGHT, WORKS, mirrored, piece, play_random


def take(position, verb, cell):
    return WORKS.step(position, (verb, cell))


def terrain_at(position, cell):
    state = position.state
    return "wall" if state.wall[cell] else "rubble" if state.rubble[cell] else "open"


def test_the_engineer_has_the_archers_numbers_and_builds_instead_of_deferring():
    engineer, archer = DEFAULT_UNITS[ENGINEER], DEFAULT_UNITS[ARCHER]
    numbers = lambda u: (u.hp, u.atk, u.defense, u.move, u.reach)  # noqa: E731
    assert numbers(engineer) == numbers(archer)
    assert (engineer.builds, engineer.defers, archer.builds, archer.defers) == (
        True,
        False,
        False,
        True,
    )


def test_the_ladder_goes_up_a_step_at_a_time(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces, game=WORKS)
    assert WORKS.legal(position)[Verb.BUILD, at(4, 5)]
    after = take(position, Verb.BUILD, at(4, 5))
    assert terrain_at(after, at(4, 5)) == "rubble"
    assert after.events == (Built(0, at(4, 5), "rubble"),)
    # A new turn, on the tile it just raised.
    position = make_battle(pieces, game=WORKS, rubble=[(4, 5)])
    after = take(position, Verb.BUILD, at(4, 5))
    assert terrain_at(after, at(4, 5)) == "wall"
    assert after.events == (Built(0, at(4, 5), "wall"),)


def test_the_ladder_goes_down_a_step_at_a_time(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=[(4, 5)])
    after = take(position, Verb.BREAK, at(4, 5))
    assert terrain_at(after, at(4, 5)) == "rubble"
    assert after.events == (Broke(0, at(4, 5), "rubble"),)
    position = make_battle(pieces, game=WORKS, rubble=[(4, 5)])
    after = take(position, Verb.BREAK, at(4, 5))
    assert terrain_at(after, at(4, 5)) == "open"
    assert after.events == (Broke(0, at(4, 5), "open"),)


def test_a_wall_cannot_go_higher_and_open_ground_cannot_go_lower(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=[(4, 5)])
    legal = WORKS.legal(position)
    assert not legal[Verb.BUILD, at(4, 5)]
    assert legal[Verb.BREAK, at(4, 5)]
    assert not legal[Verb.BREAK, at(3, 4)]  # open


def test_the_position_given_is_unchanged_by_work(make_battle, at):
    position = make_battle([piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)], game=WORKS)
    take(position, Verb.BUILD, at(4, 5))
    assert not position.state.rubble.any()
    assert position.state.passable[at(4, 5)]


def test_work_spends_the_attack_but_not_the_movement(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(4, 6, FOOT, 1)]
    position = make_battle(pieces, game=WORKS)
    assert WORKS.legal(position)[Verb.ATTACK].any()
    after = take(position, Verb.BUILD, at(3, 4))
    legal = WORKS.legal(after)
    assert not legal[[Verb.ATTACK, Verb.BUILD, Verb.BREAK]].any()
    assert legal[Verb.MOVE].any()


def test_attacking_spends_the_work(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(4, 6, FOOT, 1)]
    after = take(make_battle(pieces, game=WORKS), Verb.ATTACK, at(4, 6))
    legal = WORKS.legal(after)
    assert not legal[[Verb.BUILD, Verb.BREAK]].any()


def test_an_engineer_may_move_then_work_and_reaches_only_adjacent_tiles(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, rubble=[(5, 5)])
    assert not WORKS.legal(position)[Verb.BUILD, at(5, 5)]  # two steps away
    moved = take(position, Verb.MOVE, at(4, 5))
    assert WORKS.legal(moved)[Verb.BUILD, at(5, 5)]
    build = WORKS.legal(moved)[Verb.BUILD]
    near = position.state.space.radius_graph(1.0)
    ring = near.indices[near.indptr[at(4, 5)] : near.indptr[at(4, 5) + 1]]
    assert set(np.flatnonzero(build).tolist()) <= set(ring.tolist())


@pytest.mark.parametrize("kind", [FOOT, ARCHER, KNIGHT])
def test_only_a_unit_that_builds_may_work(make_battle, at, kind):
    position = make_battle([piece(4, 4, kind, 0), piece(8, 8, FOOT, 1)], game=WORKS, walls=[(4, 5)])
    legal = WORKS.legal(position)
    assert not legal[[Verb.BUILD, Verb.BREAK]].any()
    with pytest.raises(ValueError, match="illegal"):
        take(position, Verb.BREAK, at(4, 5))


def test_no_work_on_a_tile_with_a_unit_or_on_the_point(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(4, 5, FOOT, 0), piece(5, 4, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, point=(3, 4))
    build = WORKS.legal(position)[Verb.BUILD]
    assert not build[at(4, 5)]
    assert not build[at(5, 4)]
    assert not build[at(3, 4)]
    assert build[at(4, 3)]


def test_no_work_at_all_on_a_point_a_hand_made_board_blocked(make_battle, at):
    # Rubble to wall needs no check for sealing, so only the rule about the point forbids it.
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, point=(3, 4), rubble=[(3, 4)])
    assert not WORKS.legal(position)[Verb.BREAK, at(3, 4)]
    assert not WORKS.legal(position)[Verb.BUILD, at(3, 4)]


# Sealing. A line of walls at q = 4 with one gap at (4, 4) cuts the board in two: every step from
# q = 3 to q = 5 crosses q = 4, and (4, 4) is the only way across.
LINE = [(4, r) for r in range(9) if r != 4]


def test_a_build_that_would_cut_an_enemy_off_is_illegal(make_battle, at):
    pieces = [piece(3, 4, ENGINEER, 0), piece(6, 4, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=LINE)
    assert not WORKS.legal(position)[Verb.BUILD, at(4, 4)]
    assert WORKS.legal(position)[Verb.BUILD, at(2, 4)]
    with pytest.raises(ValueError, match="illegal"):
        take(position, Verb.BUILD, at(4, 4))


def test_a_build_that_would_cut_the_point_off_is_illegal(make_battle, at):
    pieces = [piece(3, 4, ENGINEER, 0), piece(1, 1, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=LINE)
    assert WORKS.legal(position)[Verb.BUILD, at(4, 4)]  # nothing beyond it
    position = make_battle(pieces, game=WORKS, walls=LINE, point=(6, 6))
    assert not WORKS.legal(position)[Verb.BUILD, at(4, 4)]


def test_a_build_that_would_cut_off_a_pocket_one_tile_wide_is_illegal(make_battle, at):
    # Around (4, 4) one open neighbor, a wall, the open rest, a wall: the arcs are short.
    pocket = [(6, 4), (5, 5), (4, 5), (5, 3), (6, 3)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(5, 4, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=pocket)
    assert not WORKS.legal(position)[Verb.BUILD, at(4, 4)]


def test_a_build_between_two_arcs_is_legal_when_there_is_another_way_around(make_battle, at):
    # Walls on two sides of the tile split its neighbors in two arcs, joined the long way round.
    pieces = [piece(3, 4, ENGINEER, 0), piece(6, 4, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=[(4, 3), (4, 5)])
    assert WORKS.legal(position)[Verb.BUILD, at(4, 4)]


def test_units_do_not_count_as_blocking_when_checking_for_a_seal(make_battle, at):
    # The gap is two tiles wide; a unit in one of them doesn't make the other the last way across.
    line = [(4, r) for r in range(9) if r not in (4, 5)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(4, 5, FOOT, 1), piece(6, 4, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=line)
    assert WORKS.legal(position)[Verb.BUILD, at(4, 4)]


def test_rubble_to_wall_needs_no_check_since_rubble_already_blocks_the_way(make_battle, at):
    line = [*LINE, (4, 4)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(1, 1, FOOT, 1)]
    walls = [p for p in line if p != (4, 4)]
    position = make_battle(pieces, game=WORKS, walls=walls, rubble=[(4, 4)])
    assert WORKS.legal(position)[Verb.BUILD, at(4, 4)]


@pytest.mark.parametrize(
    ("rubble", "walls", "sees"),
    [((), (), True), (((4, 4),), (), True), ((), ((4, 4),), False)],
    ids=["open", "rubble", "wall"],
)
def test_a_wall_stops_an_arrow_and_rubble_does_not(make_battle, at, rubble, walls, sees):
    pieces = [piece(4, 3, ARCHER, 0), piece(4, 6, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, rubble=rubble, walls=walls)
    assert WORKS.legal(position)[Verb.ATTACK, at(4, 6)] == sees


def test_a_wall_just_built_stops_the_next_arrow(make_battle, at):
    pieces = [
        piece(3, 4, ENGINEER, 0),
        piece(4, 3, ARCHER, 1),
        piece(4, 6, FOOT, 0, done=True),
    ]
    position = make_battle(pieces, game=WORKS, rubble=[(4, 4)])
    position = take(position, Verb.BUILD, at(4, 4))
    position = take(position, Verb.END, at(3, 4))
    assert WORKS.active(position) == at(4, 3)
    assert not WORKS.legal(position)[Verb.ATTACK, at(4, 6)]


def test_new_rubble_blocks_the_way_for_the_next_move(make_battle, at):
    pieces = [piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)]
    after = take(make_battle(pieces, game=WORKS), Verb.BUILD, at(4, 5))
    assert not WORKS.legal(after)[Verb.MOVE, at(4, 5)]


# Whole games.


@pytest.mark.parametrize("seed", range(3))
def test_buildable_tiles_are_exactly_those_a_search_of_the_whole_board_allows(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    for position, _ in play_random(WORKS, seed, rng, end_share=0.02):
        legal = WORKS.legal(position)[Verb.BUILD]
        if not legal.any():
            continue
        state = position.state
        anchors = np.flatnonzero(state.occupied | state.point)
        around = hops(state.space, np.arange(state.n_cells), WORKS.active(position)) == 1
        for tile in np.flatnonzero(around & state.passable & ~state.occupied & ~state.point):
            open_ = state.passable.copy()
            open_[tile] = False
            assert legal[tile] == joined(state.space, open_, anchors)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", range(3))
def test_living_units_and_the_point_stay_joined_through_random_games(seed):
    rng = np.random.default_rng(seed)
    worked = 0
    for position, action in play_random(WORKS, seed, rng, end_share=0.02):
        state = position.state
        if not position.deploying:
            pieces = regions(state.space, state.passable)
            where = np.flatnonzero(state.occupied | state.point)
            assert (pieces[where] == pieces[where[0]]).all()
        worked += action[0] in (Verb.BUILD, Verb.BREAK)
    assert worked > 0


def test_the_game_keeps_every_rule_of_a_game_and_is_learnable():
    records = check_game(WORKS, RandomPlayer(), seeds=range(2))
    check_learnable(WORKS, records)


def test_a_mirrored_position_with_terrain_work_looks_the_same_to_the_other_team():
    rng = np.random.default_rng(5)
    for position, _ in play_random(WORKS, 5, rng):
        if position.over:
            break
        same = [k for k, name in enumerate(WORKS.planes) if name != "loses the deadline"]
        np.testing.assert_array_equal(
            WORKS.observe(mirrored(WORKS, position), 1)[same], WORKS.observe(position, 0)[same]
        )


def test_terrain_shows_in_the_observation_after_work(make_battle, at):
    position = make_battle([piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)], game=WORKS)
    after = take(position, Verb.BUILD, at(4, 5))
    planes = WORKS.planes
    obs = WORKS.observe(after, 0)
    assert obs[planes.index("rubble"), at(4, 5)] == 1
    assert obs[planes.index("own engineer"), at(4, 4)] == 1
    assert WORKS.observe(after, 1)[planes.index("enemy engineer")].sum() == 1


def test_work_is_counted_and_told(make_battle, at):
    position = make_battle([piece(4, 4, ENGINEER, 0), piece(8, 8, FOOT, 1)], game=WORKS)
    state = position.state
    assert (
        event_text(WORKS, state, Built(0, at(4, 5), "rubble"))
        == "unit 0 builds (4, 5) up to rubble"
    )
    assert (
        event_text(WORKS, state, Broke(0, at(4, 5), "open")) == "unit 0 breaks (4, 5) down to open"
    )
    events = (
        Built(0, at(4, 5), "rubble"),
        Built(0, at(4, 5), "wall"),
        Broke(0, at(4, 5), "rubble"),
        GameOver(0, "killed"),
    )
    summary = summarize(WORKS, [Record(("a", "b"), position, (), position, events)])
    assert (summary.built, summary.broke) == (2, 1)


# Greedy's work.


def test_greedy_blocks_the_only_short_way_to_the_point_for_the_enemy(make_battle, at):
    # A wall with one gap at (4, 4) and another far away, so that a block doesn't seal: the
    # engineer stands by the near gap, and the enemy would have to walk around.
    walls = [(4, r) for r in range(9) if r not in (4, 0)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(8, 6, FOOT, 1)]  # out of its reach, even moving
    position = make_battle(pieces, game=WORKS, walls=walls, point=(2, 4), hold=Hold(0, 1))
    legal = WORKS.legal(position)
    action = GreedyPlayer(risk=0.0)(WORKS, 0, position, legal, np.random.default_rng(0))
    assert action == Action(Verb.BUILD, at(4, 4))


def test_greedy_breaks_a_way_for_friends_to_the_point(make_battle, at):
    walls = [(4, r) for r in range(9) if r not in (0, 8)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(8, 8, FOOT, 1, done=True)]
    position = make_battle(pieces, game=WORKS, rubble=[(4, 4)], walls=walls, point=(6, 4))
    action = GreedyPlayer(risk=0.0, advance=0.0)(
        WORKS, 0, position, WORKS.legal(position), np.random.default_rng(0)
    )
    assert action == Action(Verb.BREAK, at(4, 4))


def test_greedy_walls_off_an_enemy_archers_line_to_a_friend(make_battle, at):
    pieces = [
        piece(3, 5, ENGINEER, 0),
        piece(4, 4, FOOT, 0, done=True),
        piece(4, 7, ARCHER, 1, done=True),
    ]
    position = make_battle(pieces, game=WORKS, rubble=[(4, 5)])
    legal = WORKS.legal(position)
    value, action = GreedyPlayer()._best_work(WORKS, position, legal, at(3, 5))
    assert (value, action) == (1.0, Action(Verb.BUILD, at(4, 5)))


def test_a_greedy_with_no_weight_on_work_never_builds(make_battle, at):
    walls = [(4, r) for r in range(9) if r not in (4, 0)]
    pieces = [piece(3, 4, ENGINEER, 0), piece(8, 6, FOOT, 1)]
    position = make_battle(pieces, game=WORKS, walls=walls, point=(2, 4), hold=Hold(0, 1))
    action = GreedyPlayer(risk=0.0, work=0.0)(
        WORKS, 0, position, WORKS.legal(position), np.random.default_rng(0)
    )
    assert action.verb not in (Verb.BUILD, Verb.BREAK)


@pytest.mark.parametrize("seed", range(2))
def test_greedy_engineers_play_legal_whole_games(seed):
    game = dataclasses.replace(Tactics(), shape=(13, 13), home_depth=4)
    record = play(game, [GreedyPlayer(), GreedyPlayer()], game.start(seed))
    assert record.final.over
    assert record.final.winner in (0, 1)


def test_greedy_beats_random_with_engineers():
    records = series(WORKS, [("g", GreedyPlayer()), ("r", RandomPlayer())], 3, seed=9)
    assert summarize(WORKS, records).tally.wins["g"] >= 5
