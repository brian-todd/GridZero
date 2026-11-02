import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    GameOver,
    RandomPlayer,
    Tactics,
    TacticsPosition,
    Verb,
    board_space,
    units,
)
from game_demo import check_game
from gridzero.core import path
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, SMALL, TINY, mirrored, piece, play_random

PLACE = Verb.PLACE


def cells_of(mask):
    return set(np.flatnonzero(mask).tolist())


def hex_reach(game, start, blocked, budget):
    """The tiles a breadth-first search over hex neighbors reaches in ``budget`` moves."""
    shape = game.shape
    space = board_space(shape)
    q0, r0 = space.unravel(start)
    seen = {(int(q0), int(r0)): 0}
    frontier = [(int(q0), int(r0))]
    for depth in range(1, budget + 1):
        nxt = []
        for q, r in frontier:
            for dq, dr in space.directions():
                p = (q + int(dq), r + int(dr))
                inside = 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                if inside and p not in seen and not blocked[space.ravel(p)]:
                    seen[p] = depth
                    nxt.append(p)
        frontier = nxt
    return {int(space.ravel(p)) for p in seen} - {start}


def take(game, position, verb, cell):
    return game.step(position, (verb, cell))


def deploy_all(game, position):
    """Deploy every unit: each side fills its home's free tiles in id order, kinds in order."""
    while position.deploying:
        legal = game.legal(position)
        verb = int(np.flatnonzero(legal.any(axis=1))[0])
        position = take(game, position, verb, int(np.flatnonzero(legal[verb])[0]))
    return position


def hit_seed(make, attack, wanted):
    """Return the first seed whose position, after ``attack``, satisfies ``wanted``."""
    for seed in range(200):
        after = attack(make(seed))
        if wanted(after):
            return after
    raise AssertionError("no seed gave the outcome")


# Rules as a value.


def test_rules_are_a_hashable_value_and_replace_makes_new_rules():
    assert Tactics() == Tactics()
    assert hash(Tactics()) == hash(Tactics())
    harder = dataclasses.replace(Tactics(), flank_factor=4)
    assert harder != Tactics()
    assert harder.n_verbs == Verb.PLACE + 4
    assert harder.team_size == 12


def test_rules_reject_inconsistent_parameters():
    with pytest.raises(ValueError, match="counts"):
        Tactics(counts=(1, 2))
    with pytest.raises(ValueError, match="hold"):
        Tactics(home_depth=3)  # 9 tiles for 12 units
    with pytest.raises(TypeError, match="free_facing"):
        Tactics(free_facing=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="decide_at_cap"):
        Tactics(decide_at_cap=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="flank_factor"):
        Tactics(flank_factor=0)
    with pytest.raises(TypeError, match="UnitStats"):
        Tactics(units=("archer",), counts=(1,))  # ty: ignore[invalid-argument-type]


def test_rules_reject_counts_that_are_not_integers(not_an_integer):
    with pytest.raises(TypeError, match="count"):
        Tactics(counts=(6, 3, 2, not_an_integer))


# Deployment.


def test_start_starts_deployment_with_team_zero_on_its_home():
    position = TINY.start(1)
    assert (position.ply, position.round, position.deploying, position.over) == (0, 0, True, False)
    assert TINY.player(position) == 0
    assert TINY.active(position) == -1
    legal = TINY.legal(position)
    assert legal.shape == (TINY.n_verbs, 81)
    assert not legal[:PLACE].any()
    home = cells_of((position.state.home == 0) & position.state.passable)
    for kind in range(4):
        assert cells_of(legal[PLACE + kind]) == home


def test_deployment_alternates_and_each_side_deploys_its_counts():
    position = TINY.start(1)
    players = []
    while position.deploying:
        players.append(TINY.player(position))
        legal = TINY.legal(position)
        verb = int(np.flatnonzero(legal.any(axis=1))[-1])  # the highest kind left first
        position = take(TINY, position, verb, int(np.flatnonzero(legal[verb])[0]))
    assert players == [0, 1] * 5
    listed = units(position.state)
    for team in (0, 1):
        kinds = np.bincount(listed.kind[listed.team == team], minlength=4)
        np.testing.assert_array_equal(kinds, TINY.counts)
        assert (position.state.home[listed.cell[listed.team == team]] == team).all()


def test_a_kind_is_no_longer_offered_once_its_count_is_deployed():
    position = TINY.start(1)
    position = take(
        TINY,
        position,
        PLACE + CAPTAIN,
        int(np.flatnonzero(TINY.legal(position)[PLACE + CAPTAIN])[0]),
    )
    position = take(
        TINY, position, PLACE + FOOT, int(np.flatnonzero(TINY.legal(position)[PLACE + FOOT])[0])
    )
    legal = TINY.legal(position)  # team 0 again, its captain deployed
    assert not legal[PLACE + CAPTAIN].any()
    assert legal[PLACE + FOOT].any()


def test_battle_begins_after_the_last_unit_with_turns_shuffled():
    position = deploy_all(TINY, TINY.start(1))
    assert (position.round, position.ply, position.deploying) == (1, 10, False)
    state = position.state
    order = np.sort(state.units.initiative[state.occupied])
    np.testing.assert_array_equal(order, np.arange(10))
    first = TINY.active(position)
    assert state.units.initiative[first] == 0
    assert TINY.player(position) == state.units.team[first]


def test_units_deploy_facing_the_center():
    game = Tactics()
    position = game.start(2)
    space = position.state.space
    last = game.shape[0] - 1  # the homes' far corners: (last, 0) and (0, last)
    position = take(game, position, PLACE + FOOT, int(space.ravel([last, 0])))
    position = take(game, position, PLACE + FOOT, int(space.ravel([0, last])))
    facing = position.state.units.facing
    assert facing[space.ravel([last, 0])] == 2  # toward (-1, 1): up and to the left
    assert facing[space.ravel([0, last])] == 5
    flat_game = dataclasses.replace(game, face_center=False)
    flat = take(flat_game, flat_game.start(2), PLACE + FOOT, int(space.ravel([last, 0])))
    assert flat.state.units.facing[space.ravel([last, 0])] == 0


def test_start_rejects_a_seed_that_is_negative_or_not_an_integer():
    with pytest.raises(ValueError, match="seed"):
        SMALL.start(-1)
    with pytest.raises(TypeError, match="seed"):
        SMALL.start(1.0)  # ty: ignore[invalid-argument-type]  # a float, on purpose


# Movement.


def test_moves_reach_the_tiles_within_movement_around_walls_rubble_and_units(make_battle, at):
    pieces = [piece(4, 4, FOOT, 0), piece(3, 4, FOOT, 0), piece(4, 3, FOOT, 1)]
    position = make_battle(pieces, walls=[(5, 4), (4, 5)], rubble=[(3, 5)])
    blocked = ~position.state.passable | position.state.occupied
    expected = hex_reach(SMALL, at(4, 4), blocked, 2)
    assert cells_of(SMALL.legal(position)[Verb.MOVE]) == expected
    assert at(6, 4) not in expected  # behind the wall


def test_units_cannot_move_through_friends(make_battle):
    position = make_battle([piece(0, 0, FOOT, 0), piece(1, 0, FOOT, 0)], walls=[(0, 1)])
    assert not SMALL.legal(position)[Verb.MOVE].any()


def test_a_unit_may_move_attack_and_move_again(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(6, 4, FOOT, 1, hp=5)])
    assert not SMALL.legal(position)[Verb.ATTACK].any()
    position = take(SMALL, position, Verb.MOVE, at(5, 4))
    assert cells_of(SMALL.legal(position)[Verb.ATTACK]) == {at(6, 4)}
    position = take(SMALL, position, Verb.ATTACK, at(6, 4))
    legal = SMALL.legal(position)
    assert not legal[Verb.ATTACK].any()
    assert cells_of(legal[Verb.MOVE]) == hex_reach(SMALL, at(5, 4), position.state.occupied, 1)
    position = take(SMALL, position, Verb.MOVE, at(5, 5))
    assert not SMALL.legal(position)[Verb.MOVE].any()


def test_a_knight_moves_eight_without_attacking_and_cannot_attack_past_four(make_battle, at):
    position = make_battle([piece(0, 4, KNIGHT, 0), piece(7, 4, FOOT, 1)])
    blocked = position.state.occupied
    assert cells_of(SMALL.legal(position)[Verb.MOVE]) == hex_reach(SMALL, at(0, 4), blocked, 8)
    position = take(SMALL, position, Verb.MOVE, at(6, 4))  # 6 tiles, now beside the enemy
    legal = SMALL.legal(position)
    assert not legal[Verb.ATTACK].any()
    assert cells_of(legal[Verb.MOVE]) == hex_reach(SMALL, at(6, 4), position.state.occupied, 2)


@pytest.mark.parametrize(("tiles", "may_attack"), [(4, True), (5, False)])
def test_a_knight_may_attack_after_four_tiles_but_not_five(make_battle, at, tiles, may_attack):
    position = make_battle([piece(0, 4, KNIGHT, 0), piece(tiles + 1, 4, FOOT, 1)])
    position = take(SMALL, position, Verb.MOVE, at(tiles, 4))
    assert SMALL.legal(position)[Verb.ATTACK].any() == may_attack


def test_a_knight_that_attacks_moves_only_four_in_all(make_battle, at):
    position = make_battle([piece(4, 4, KNIGHT, 0), piece(5, 4, CAPTAIN, 1)])
    position = take(SMALL, position, Verb.MOVE, at(4, 5))
    position = take(SMALL, position, Verb.ATTACK, at(5, 4))
    expected = hex_reach(SMALL, at(4, 5), position.state.occupied, 3)
    assert cells_of(SMALL.legal(position)[Verb.MOVE]) == expected


def test_a_swap_trades_places_uses_both_movements_and_leaves_the_attack(make_battle, at):
    pieces = [piece(4, 4, FOOT, 0), piece(5, 4, ARCHER, 0, done=True), piece(6, 4, FOOT, 1)]
    position = make_battle(pieces)
    assert cells_of(SMALL.legal(position)[Verb.SWAP]) == {at(5, 4)}
    position = take(SMALL, position, Verb.SWAP, at(5, 4))
    u = position.state.units
    assert (u.kind[at(5, 4)], u.kind[at(4, 4)]) == (FOOT, ARCHER)
    assert u.swapped[at(5, 4)]
    assert u.swapped[at(4, 4)]
    assert SMALL.active(position) == at(5, 4)
    legal = SMALL.legal(position)
    assert not legal[Verb.MOVE].any()
    assert cells_of(legal[Verb.ATTACK]) == {at(6, 4)}
    assert u.facing[at(5, 4)] == 0
    assert u.facing[at(4, 4)] == 3


@pytest.mark.parametrize("who", ["active", "partner"])
def test_no_swap_once_either_unit_has_moved(make_battle, at, who):
    moved = {"active": (1, 0), "partner": (0, 1)}[who]
    pieces = [piece(4, 4, FOOT, 0, moved=moved[0]), piece(5, 4, FOOT, 0, moved=moved[1])]
    assert not SMALL.legal(make_battle(pieces))[Verb.SWAP].any()


def test_no_swap_with_a_friend_two_tiles_away(make_battle):
    pieces = [piece(4, 4, FOOT, 0), piece(6, 4, FOOT, 0), piece(5, 2, FOOT, 0)]
    assert not SMALL.legal(make_battle(pieces))[Verb.SWAP].any()


def test_no_swap_with_an_enemy_or_a_unit_that_already_swapped(make_battle):
    pieces = [piece(4, 4, FOOT, 0), piece(5, 4, FOOT, 1), piece(3, 4, FOOT, 0, swapped=True)]
    assert not SMALL.legal(make_battle(pieces))[Verb.SWAP].any()


# Attacks.


def test_archers_shoot_three_tiles_where_walls_do_not_block(make_battle, at):
    pieces = [
        piece(1, 1, ARCHER, 0),
        piece(4, 1, FOOT, 1),  # 3 away, past rubble at (2, 1) and a friend at (3, 1)
        piece(3, 1, FOOT, 0),
        piece(4, 0, FOOT, 1),  # 3 away, clear
        piece(1, 5, FOOT, 1),  # 4 away
        piece(3, 3, FOOT, 1),  # 4 away
        piece(1, 4, FOOT, 1),  # 3 away, behind a wall
    ]
    position = make_battle(pieces, walls=[(1, 3)], rubble=[(2, 1)])
    assert cells_of(SMALL.legal(position)[Verb.ATTACK]) == {at(4, 1), at(4, 0)}


def test_footsoldiers_attack_only_adjacent_enemies(make_battle, at):
    pieces = [
        piece(4, 4, FOOT, 0),
        piece(5, 4, FOOT, 1),
        piece(6, 4, FOOT, 1),
        piece(3, 4, FOOT, 0),
    ]
    assert cells_of(SMALL.legal(make_battle(pieces))[Verb.ATTACK]) == {at(5, 4)}


def test_odds_of_a_plain_attack_from_the_front(make_battle, at):
    position = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1, facing=0)])
    odds = SMALL.odds(position, at(5, 4), at(4, 4))
    assert (odds.atk, odds.defense, odds.flank, odds.chance) == (2, 2, False, 0.65)


@pytest.mark.parametrize(
    ("q", "r", "flank"),
    [(5, 3, False), (5, 4, False), (4, 5, False), (3, 5, True), (3, 4, True), (4, 3, True)],
)
def test_attacks_from_behind_flank_and_double_atk(make_battle, at, q, r, flank):
    position = make_battle([piece(q, r, FOOT, 0), piece(4, 4, FOOT, 1, facing=0)])
    odds = SMALL.odds(position, at(q, r), at(4, 4))
    assert odds.flank is flank
    assert odds.atk == (4 if flank else 2)


def test_a_distant_archer_behind_its_target_flanks(make_battle, at):
    position = make_battle([piece(1, 4, ARCHER, 0), piece(4, 4, FOOT, 1, facing=0)])
    assert SMALL.odds(position, at(1, 4), at(4, 4)).flank


def test_two_adjacent_enemies_surround_a_unit_even_in_front(make_battle, at):
    pieces = [piece(5, 4, FOOT, 0), piece(5, 3, FOOT, 0), piece(4, 4, FOOT, 1, facing=0)]
    odds = SMALL.odds(make_battle(pieces), at(5, 4), at(4, 4))
    assert odds.flank
    assert odds.atk == 4


def test_enemies_two_tiles_away_do_not_surround(make_battle, at):
    pieces = [piece(5, 4, FOOT, 0), piece(6, 3, FOOT, 0), piece(4, 4, FOOT, 1, facing=0)]
    odds = SMALL.odds(make_battle(pieces), at(5, 4), at(4, 4))
    assert not odds.flank
    assert odds.atk == 2


def test_formation_counts_adjacent_footsoldiers_up_to_the_cap(make_battle, at):
    def defense(*friends):
        pieces = [piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1, facing=0)]
        pieces += [piece(q, r, kind, 1) for q, r, kind in friends]
        return SMALL.odds(make_battle(pieces), at(5, 4), at(4, 4)).defense

    assert defense() == 2
    assert defense((3, 4, FOOT)) == 3
    assert defense((3, 4, FOOT), (4, 3, FOOT), (4, 5, FOOT)) == 2 + SMALL.formation_cap
    assert defense((3, 4, ARCHER)) == 2  # only units in formation count
    assert defense((2, 4, FOOT), (6, 2, FOOT)) == 2  # two tiles away


def test_an_engaged_target_gives_one_more_atk_before_doubling(make_battle, at):
    plain = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1, engaged=True)])
    assert SMALL.odds(plain, at(5, 4), at(4, 4)).atk == 3
    behind = make_battle([piece(3, 4, FOOT, 0), piece(4, 4, FOOT, 1, engaged=True)])
    assert SMALL.odds(behind, at(3, 4), at(4, 4)).atk == 6


def test_a_captain_lifts_friends_within_three_tiles_and_itself(make_battle, at):
    near = make_battle(
        [
            piece(5, 4, FOOT, 0),
            piece(8, 4, CAPTAIN, 0),
            piece(4, 4, FOOT, 1),
            piece(1, 4, CAPTAIN, 1),
        ]
    )
    odds = SMALL.odds(near, at(5, 4), at(4, 4))
    assert (odds.atk, odds.defense) == (3, 3)
    far = make_battle([piece(5, 4, FOOT, 0), piece(8, 5, CAPTAIN, 0), piece(4, 4, FOOT, 1)])
    assert SMALL.odds(far, at(5, 4), at(4, 4)).atk == 2  # 4 steps away
    alone = make_battle([piece(5, 4, CAPTAIN, 0), piece(4, 4, FOOT, 1)])
    assert SMALL.odds(alone, at(5, 4), at(4, 4)).atk == 5


def test_footsoldiers_hold_together_up_to_two_def(make_battle, at):
    def defense(friends):
        pieces = [piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1), *friends]
        return SMALL.odds(make_battle(pieces), at(5, 4), at(4, 4)).defense

    assert defense([]) == 2
    assert defense([piece(3, 4, FOOT, 1)]) == 3
    assert defense([piece(3, 4, FOOT, 1), piece(4, 3, FOOT, 1), piece(3, 5, FOOT, 1)]) == 4
    assert defense([piece(3, 4, ARCHER, 1), piece(4, 3, FOOT, 0)]) == 2  # neither counts


def test_attacks_hit_as_often_as_the_odds_say_and_always_engage(make_battle, at):
    hits = 0
    n = 400
    for seed in range(n):
        position = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1)], seed=seed)
        after = take(SMALL, position, Verb.ATTACK, at(4, 4))
        u = after.state.units
        assert u.engaged[at(4, 4)]
        assert u.attacked[at(5, 4)]
        assert u.hp[at(4, 4)] in (5, 3)
        hits += u.hp[at(4, 4)] == 3
    assert abs(hits / n - 0.65) < 4 * np.sqrt(0.65 * 0.35 / n)


def test_a_unit_brought_to_zero_hp_leaves_and_the_last_one_ends_the_game(make_battle, at):
    def make(seed):
        return make_battle([piece(5, 4, CAPTAIN, 0), piece(4, 4, ARCHER, 1, hp=2)], seed=seed)

    after = hit_seed(make, lambda p: take(SMALL, p, Verb.ATTACK, at(4, 4)), lambda p: p.over)
    assert not after.state.occupied[at(4, 4)]
    assert after.winner == 0
    assert not SMALL.legal(after).any()
    with pytest.raises(ValueError, match="over"):
        take(SMALL, after, Verb.END, at(5, 4))
    with pytest.raises(ValueError, match="over"):
        SMALL.player(after)


# Deferred attacks.


def test_archers_may_defer_onto_open_tiles_in_reach_and_sight(make_battle, at):
    position = make_battle([piece(1, 1, ARCHER, 0)], walls=[(1, 3)], rubble=[(2, 1)])
    targets = cells_of(SMALL.legal(position)[Verb.DEFER])
    assert at(4, 1) in targets
    assert at(1, 4) not in targets
    assert at(2, 1) not in targets
    assert at(1, 3) not in targets
    assert at(1, 1) not in targets
    assert not SMALL.legal(make_battle([piece(1, 1, FOOT, 0)]))[Verb.DEFER].any()


def test_deferring_uses_the_attack(make_battle, at):
    position = make_battle([piece(1, 1, ARCHER, 0), piece(2, 1, FOOT, 1)])
    position = take(SMALL, position, Verb.DEFER, at(4, 1))
    assert position.state.units.watch[at(1, 1)] == at(4, 1)
    legal = SMALL.legal(position)
    assert not legal[Verb.ATTACK].any()
    assert not legal[Verb.DEFER].any()


def test_an_enemy_entering_a_watched_tile_is_attacked_once(make_battle, at):
    pieces = [piece(6, 4, FOOT, 1), piece(1, 4, ARCHER, 0, watch=at(4, 4), done=True)]
    position = take(SMALL, make_battle(pieces), Verb.MOVE, at(4, 4))
    u = position.state.units
    assert u.watch[at(1, 4)] == -1
    assert u.engaged[at(4, 4)]  # hit or missed, it was attacked


def test_a_friend_entering_a_watched_tile_is_not_attacked(make_battle, at):
    pieces = [piece(6, 4, FOOT, 0), piece(1, 4, ARCHER, 0, watch=at(5, 4), done=True)]
    position = take(SMALL, make_battle(pieces), Verb.MOVE, at(5, 4))
    assert position.state.units.watch[at(1, 4)] == at(5, 4)
    assert not position.state.units.engaged[at(5, 4)]


def test_a_move_takes_an_equally_short_route_around_a_watched_tile(make_battle, at):
    plain = make_battle([piece(6, 4, FOOT, 1)])
    route = path(plain.state.space, at(6, 4), at(4, 5))
    assert route.size == 3
    watched = int(route[1])
    archer = piece(5, 2, ARCHER, 0, watch=watched, done=True)  # both routes' tiles in reach
    position = take(SMALL, make_battle([piece(6, 4, FOOT, 1), archer]), Verb.MOVE, at(4, 5))
    u = position.state.units
    assert u.watch[at(5, 2)] == watched
    assert not u.engaged[at(4, 5)]


def test_a_deferred_attack_lapses_at_its_archers_next_turn(make_battle, at):
    pieces = [piece(6, 4, FOOT, 1), piece(1, 4, ARCHER, 0, watch=at(8, 8))]
    position = take(SMALL, make_battle(pieces), Verb.END, at(6, 4))
    assert SMALL.active(position) == at(1, 4)
    assert position.state.units.watch[at(1, 4)] == -1


def test_a_watched_tile_out_of_reach_does_not_fire(make_battle, at):
    pieces = [piece(6, 4, FOOT, 1), piece(0, 8, ARCHER, 0, watch=at(5, 4), done=True)]
    position = take(SMALL, make_battle(pieces), Verb.MOVE, at(5, 4))
    u = position.state.units
    assert u.watch[at(0, 8)] == at(5, 4)
    assert not u.engaged[at(5, 4)]


def test_a_mover_killed_by_a_deferred_attack_stops_and_its_turn_ends(make_battle, at):
    def make(seed):
        pieces = [
            piece(7, 4, FOOT, 1, hp=1),
            piece(4, 6, FOOT, 1),
            piece(3, 4, ARCHER, 0, watch=at(6, 4), done=True),
        ]
        return make_battle(pieces, seed=seed)

    after = hit_seed(
        make,
        lambda p: take(SMALL, p, Verb.MOVE, at(5, 4)),
        lambda p: not p.state.occupied[at(6, 4)] and not p.state.occupied[at(5, 4)],
    )
    assert after.state.occupied.sum() == 2
    assert SMALL.active(after) == at(4, 6)


# Facing.


def test_a_move_turns_a_unit_toward_its_movement_vector(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0, facing=3)])
    position = take(SMALL, position, Verb.MOVE, at(6, 4))
    assert position.state.units.facing[at(6, 4)] == 0


@pytest.mark.parametrize(("before", "after"), [(2, 1), (4, 0), (0, 0), (1, 1)])
def test_a_move_toward_a_corner_turns_the_nearer_way(make_battle, at, before, after):
    position = make_battle([piece(4, 4, FOOT, 0, facing=before)])
    position = take(SMALL, position, Verb.MOVE, at(5, 5))  # (1, 1): 30 degrees
    assert position.state.units.facing[at(5, 5)] == after


def test_without_turning_on_moves_a_unit_keeps_its_facing(make_battle, at):
    game = dataclasses.replace(SMALL, turn_on_move=False)
    position = make_battle([piece(4, 4, FOOT, 0, facing=3)], game=game)
    position = take(game, position, Verb.MOVE, at(6, 4))
    assert position.state.units.facing[at(6, 4)] == 3


def test_a_unit_may_turn_once_between_its_other_actions(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0, facing=2), piece(8, 8, FOOT, 1)])
    legal = SMALL.legal(position)
    turns = [d for d in range(6) if legal[Verb.FACE + d, at(4, 4)]]
    assert turns == [0, 1, 3, 4, 5]
    assert legal[Verb.FACE : Verb.END].sum() == 5  # only on its own cell
    position = take(SMALL, position, Verb.FACE + 4, at(4, 4))
    assert position.state.units.facing[at(4, 4)] == 4
    assert not SMALL.legal(position)[Verb.FACE : Verb.END].any()
    position = take(SMALL, position, Verb.MOVE, at(5, 4))
    assert SMALL.legal(position)[Verb.FACE : Verb.END].any()


def test_without_free_facing_units_never_turn_at_will(make_battle):
    game = dataclasses.replace(SMALL, free_facing=False)
    position = make_battle([piece(4, 4, FOOT, 0)], game=game)
    assert not game.legal(position)[Verb.FACE : Verb.END].any()


# Turns and rounds.


def test_ending_a_turn_passes_to_the_next_unit_in_initiative(make_battle, at):
    pieces = [piece(1, 1, FOOT, 0, initiative=2), piece(7, 7, FOOT, 1, initiative=0)]
    pieces.append(piece(4, 4, KNIGHT, 0, initiative=1))
    position = make_battle(pieces)
    assert SMALL.active(position) == at(7, 7)
    assert SMALL.player(position) == 1
    position = take(SMALL, position, Verb.END, at(7, 7))
    assert SMALL.active(position) == at(4, 4)
    assert SMALL.player(position) == 0


def test_a_new_round_resets_the_rounds_flags_and_reshuffles(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(6, 4, FOOT, 1)])
    position = take(SMALL, position, Verb.MOVE, at(5, 4))
    position = take(SMALL, position, Verb.ATTACK, at(6, 4))
    position = take(SMALL, position, Verb.END, at(5, 4))
    position = take(SMALL, position, Verb.END, at(6, 4))
    assert (position.round, position.ply) == (2, 4)
    u = position.state.units
    occupied = position.state.occupied
    for flags in (u.done, u.attacked, u.engaged, u.swapped, u.faced):
        assert not flags.any()
    assert not u.moved.any()
    assert sorted(u.initiative[occupied].tolist()) == [0, 1]


def test_the_game_is_a_draw_after_the_last_round(make_battle, at):
    game = dataclasses.replace(SMALL, max_rounds=1)
    position = make_battle([piece(4, 4, FOOT, 0), piece(6, 4, FOOT, 1)], game=game)
    position = take(game, position, Verb.END, at(4, 4))
    position = take(game, position, Verb.END, at(6, 4))
    assert position.over
    assert position.winner is None
    assert position.round == 1


@pytest.mark.parametrize(("hp", "winner"), [((5, 4), 0), ((2, 3), 1), ((4, 4), None)])
def test_at_the_cap_the_side_with_more_hit_points_left_wins(make_battle, at, hp, winner):
    game = dataclasses.replace(SMALL, max_rounds=1, decide_at_cap=True)
    pieces = [
        piece(1, 1, FOOT, 0, hp=hp[0]),
        piece(4, 4, ARCHER, 1, hp=1),
        piece(6, 6, FOOT, 1, hp=hp[1] - 1),
    ]
    position = make_battle(pieces, game=game)
    for q, r in ((1, 1), (4, 4), (6, 6)):
        position = take(game, position, Verb.END, at(q, r))
    assert position.over
    assert position.winner == winner
    assert position.events[-1] == GameOver(winner)


def test_without_deciding_at_the_cap_more_hit_points_still_draw(make_battle, at):
    game = dataclasses.replace(SMALL, max_rounds=1, decide_at_cap=False)
    position = make_battle([piece(4, 4, FOOT, 0, hp=5), piece(6, 4, FOOT, 1, hp=1)], game=game)
    position = take(game, position, Verb.END, at(4, 4))
    position = take(game, position, Verb.END, at(6, 4))
    assert position.over
    assert position.winner is None


def test_the_default_rules_decide_at_the_cap():
    assert Tactics().decide_at_cap


# Errors.


def test_illegal_actions_raise(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    with pytest.raises(ValueError, match="illegal"):
        take(SMALL, position, Verb.ATTACK, at(8, 8))
    with pytest.raises(ValueError, match="illegal"):
        take(SMALL, position, PLACE + FOOT, at(0, 0))
    with pytest.raises(ValueError, match="verb"):
        take(SMALL, position, SMALL.n_verbs, at(4, 4))
    with pytest.raises(IndexError):
        take(SMALL, position, Verb.END, 81)


def test_actions_need_integers(make_battle, at, not_an_integer):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    with pytest.raises(TypeError, match="verb"):
        SMALL.step(position, (not_an_integer, at(4, 4)))
    with pytest.raises(TypeError, match="cell"):
        SMALL.step(position, (Verb.END, not_an_integer))


def test_positions_of_other_boards_raise(make_battle):
    with pytest.raises(TypeError, match="TacticsPosition"):
        SMALL.legal("position")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="shape"):
        TINY.legal(Tactics().start(0))


# Whole games.


def test_a_step_leaves_the_position_it_was_given_unchanged(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(5, 4, FOOT, 1)])
    before = {f: position.state[f].copy() for f in position.state}
    take(SMALL, position, Verb.ATTACK, at(5, 4))
    for f, values in before.items():
        np.testing.assert_array_equal(position.state[f], values)
    assert position.ply == 0


def test_the_same_seed_and_choices_give_the_same_game():
    def final(seed):
        *_, (position, _) = play_random(TINY, seed, np.random.default_rng(seed))
        return position

    a, b, c = final(3), final(3), final(4)
    assert a.ply == b.ply
    assert a.winner == b.winner
    for f in a.state:
        np.testing.assert_array_equal(a.state[f], b.state[f])
    assert a.ply != c.ply or not all(np.array_equal(a.state[f], c.state[f]) for f in a.state)


@pytest.mark.parametrize("seed", range(3))
def test_random_games_keep_the_invariants_and_end(seed):
    rng = np.random.default_rng(seed)
    hp: dict[int, int] = {}
    previous = None
    for position, action in play_random(TINY, seed, rng):
        state = position.state
        u = state.units
        occupied = state.occupied
        assert position.ply == (0 if previous is None else previous.ply + 1)
        assert not (occupied & ~state.passable).any()
        assert len(set(u.uid[occupied].tolist())) == occupied.sum()
        assert (u.hp[occupied] >= 1).all()
        for uid, h in zip(u.uid[occupied].tolist(), u.hp[occupied].tolist(), strict=True):
            assert h <= hp.get(uid, h)
            hp[uid] = h
        if not position.deploying:
            assert hp.keys() >= set(u.uid[occupied].tolist())
            gone = set(hp) - set(u.uid[occupied].tolist())
            assert all(uid not in u.uid[occupied] for uid in gone)
        if not position.over:
            legal = TINY.legal(position)
            assert legal[action]
            illegal = np.argwhere(~legal)
            verb, cell = illegal[rng.integers(len(illegal))]
            with pytest.raises(ValueError, match="illegal"):
                TINY.step(position, (int(verb), int(cell)))
        previous = position
    assert previous is not None
    assert previous.over
    assert previous.round <= TINY.max_rounds


def test_legal_actions_mirror_with_the_board():
    perm = list(range(TINY.n_verbs))
    for d in range(6):
        perm[Verb.FACE + d] = Verb.FACE + (d + 3) % 6
    checked = 0
    rng = np.random.default_rng(5)
    for position, _ in play_random(TINY, 5, rng):
        if position.deploying or position.over or rng.random() > 0.2:
            continue
        flipped = mirrored(TINY, position)
        m = position.state.space.opposite()
        legal, legal_flipped = TINY.legal(position), TINY.legal(flipped)
        np.testing.assert_array_equal(legal_flipped[perm][:, m], legal)
        assert TINY.player(flipped) == 1 - TINY.player(position)
        a = TINY.active(position)
        for t in np.flatnonzero(legal[Verb.ATTACK]):
            assert TINY.odds(flipped, m[a], m[t]) == TINY.odds(position, a, t)
        checked += 1
    assert checked > 10


def test_a_position_records_its_seed_and_board():
    position = TINY.start(9)
    assert isinstance(position, TacticsPosition)
    assert position.seed == 9
    np.testing.assert_array_equal(TINY.start(9).state.wall, position.state.wall)


def test_odds_from_another_cell_are_the_odds_after_moving_there(make_battle, at):
    pieces = [
        piece(3, 4, FOOT, 0),
        piece(5, 4, FOOT, 1, facing=3),
        piece(1, 3, CAPTAIN, 0, done=True),
        piece(6, 2, FOOT, 0, done=True),
    ]
    position = make_battle(pieces)
    target = at(5, 4)
    checked = 0
    for cell in np.flatnonzero(SMALL.legal(position)[Verb.MOVE]):
        moved = take(SMALL, position, Verb.MOVE, int(cell))
        if moved.state.units.kind[target] < 0:
            continue
        expected = SMALL.odds(moved, int(cell), target)
        assert SMALL.odds(position, at(3, 4), target, at=int(cell)) == expected
        checked += 1
    assert checked > 10
    assert SMALL.odds(position, at(3, 4), target, at=at(3, 4)) == SMALL.odds(
        position, at(3, 4), target
    )
    with pytest.raises(ValueError, match="empty"):
        SMALL.odds(position, at(3, 4), target, at=at(1, 3))


def test_odds_from_another_cell_leave_the_old_cell_empty(make_battle, at):
    pieces = [piece(6, 4, FOOT, 0), piece(5, 4, FOOT, 1, facing=0)]  # attacker in front
    position = make_battle(pieces)
    odds = SMALL.odds(position, at(6, 4), at(5, 4), at=at(5, 5))  # still in front, and alone
    assert not odds.flank


# The promises of every game.


def test_tactics_keeps_the_promises_of_a_game():
    records = check_game(TINY, RandomPlayer(), range(2))
    assert all(isinstance(r.final, TacticsPosition) for r in records)


def test_tactics_shows_players_the_whole_position():
    position = TINY.start(0)
    assert TINY.view(position, 0) is position
    assert TINY.view(position, 1) is position
    assert TINY.n_players == 2
