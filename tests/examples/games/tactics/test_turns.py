"""The order of turns, how long a unit stays engaged, and the places a team fills."""

import dataclasses

import numpy as np
import pytest

from examples.games.tactics import GreedyPlayer, Placed, RoundStarted, Tactics, Verb, default_roster
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, WORKS, piece

TWO_KNIGHTS = (0, 1, 2, 1, 0)  # 1 + (1 + 2) + 1 places: 4 units
FIVE_UNITS = (2, 1, 0, 1, 1)  # 5 places: 5 units


def deployed(game, seed, rosters=(TWO_KNIGHTS, FIVE_UNITS)):
    """Deploy by greedy players fielding ``rosters``; return the first position of the battle."""
    players = [GreedyPlayer(roster=r) for r in rosters]
    position = game.start(seed)
    rng = np.random.default_rng(0)
    while position.deploying:
        team = game.player(position)
        action = players[team](game, team, position, game.legal(position), rng)
        position = game.step(position, action)
    return position


def turn_order(position):
    """Return the uids of the living units, in their order of turns."""
    units = position.state.units
    cells = position.state.occupied.nonzero()[0]
    return units.uid[cells[np.argsort(units.initiative[cells])]].tolist()


def teams_in_order(position):
    units = position.state.units
    cells = position.state.occupied.nonzero()[0]
    return units.team[cells[np.argsort(units.initiative[cells])]].tolist()


# The order of turns.


def test_the_teams_alternate_and_the_side_with_more_units_takes_the_last_turns():
    teams = teams_in_order(deployed(WORKS, 1))
    assert len(teams) == 9
    assert teams[8:] == [1]
    assert all(teams[k] != teams[k + 1] for k in range(7))


def test_either_team_may_take_the_first_turn():
    firsts = {
        teams_in_order(deployed(WORKS, seed, (FIVE_UNITS, FIVE_UNITS)))[0] for seed in range(12)
    }
    assert firsts == {0, 1}


def test_a_teams_own_units_come_in_a_random_order():
    orders = {
        tuple(turn_order(deployed(WORKS, seed, (FIVE_UNITS, FIVE_UNITS)))) for seed in range(6)
    }
    assert len(orders) > 2


def test_the_order_is_rolled_once_and_the_dead_drop_out():
    game = WORKS
    rounds = 0
    for seed in range(4):
        position = deployed(game, seed)
        first = turn_order(position)
        rng = np.random.default_rng(0)
        player = GreedyPlayer()
        while not position.over:
            legal = game.legal(position)
            position = game.step(
                position, player(game, game.player(position), position, legal, rng)
            )
            if any(isinstance(e, RoundStarted) for e in position.events) and not position.over:
                rounds += 1
                alive = set(turn_order(position))
                assert turn_order(position) == [uid for uid in first if uid in alive]
    assert rounds >= 4


def test_shuffled_each_round_the_order_changes_between_rounds():
    game = dataclasses.replace(WORKS, shuffle_each_round=True)
    position = deployed(game, 2)
    orders = [turn_order(position)]
    rng = np.random.default_rng(0)
    player = GreedyPlayer()
    while not position.over and len(orders) < 4:
        position = game.step(
            position, player(game, game.player(position), position, game.legal(position), rng)
        )
        if any(isinstance(e, RoundStarted) for e in position.events):
            orders.append(turn_order(position))
    alive = set(orders[-1])
    assert len({tuple(uid for uid in o if uid in alive) for o in orders}) > 1


# How long a unit stays engaged.


def hit_and_end_round(make_battle, at, game):
    """An archer, acting first, hits a captain that has already acted; its turn ends."""
    pieces = [
        piece(2, 2, ARCHER, 0, initiative=0),
        piece(4, 2, CAPTAIN, 1, initiative=1, done=True),
    ]
    position = make_battle(pieces, game=game)
    for seed in range(40):
        after = game.step(make_battle(pieces, game=game, seed=seed), (Verb.ATTACK, at(4, 2, game)))
        if after.state.units.hp[at(4, 2, game)] < game.units[CAPTAIN].hp:
            break
    else:
        pytest.fail("no arrow hit")
    assert after.state.units.engaged[at(4, 2, game)]
    assert position.round == 1
    return game.step(after, (Verb.END, at(2, 2, game)))


def test_a_unit_hit_after_its_turn_stays_engaged_into_the_next_round_until_its_turn(
    make_battle, at
):
    next_round = hit_and_end_round(make_battle, at, WORKS)
    assert next_round.round == 2
    assert next_round.state.units.engaged[at(4, 2, WORKS)]
    assert WORKS.active(next_round) == at(2, 2, WORKS)
    its_turn = WORKS.step(next_round, (Verb.END, at(2, 2, WORKS)))
    assert WORKS.active(its_turn) == at(4, 2, WORKS)
    assert not its_turn.state.units.engaged[at(4, 2, WORKS)]


def test_engaged_until_the_round_ends_a_unit_recovers_with_the_new_round(make_battle, at):
    game = dataclasses.replace(WORKS, engaged_until_turn=False)
    next_round = hit_and_end_round(make_battle, at, game)
    assert next_round.round == 2
    assert not next_round.state.units.engaged[at(4, 2, game)]


def test_a_unit_engaged_at_the_start_of_its_own_turn_recovers_as_the_round_begins(make_battle, at):
    pieces = [
        piece(4, 2, FOOT, 1, initiative=0, done=True, engaged=True),
        piece(2, 2, ARCHER, 0, initiative=1),
    ]
    position = make_battle(pieces, game=WORKS)
    next_round = WORKS.step(position, (Verb.END, at(2, 2, WORKS)))
    assert next_round.round == 2
    assert WORKS.active(next_round) == at(4, 2, WORKS)
    assert not next_round.state.units.engaged[at(4, 2, WORKS)]


# Places.


def test_a_first_knight_takes_one_place_and_a_second_two():
    position = WORKS.start(1)
    to_place = []
    for kind in (KNIGHT, FOOT, KNIGHT):  # team 0, team 1, team 0
        cell = int(WORKS.legal(position)[Verb.PLACE + kind].nonzero()[0][0])
        position = WORKS.step(position, (Verb.PLACE + kind, cell))
        to_place.append(WORKS.observe(position, 0)[WORKS.planes.index("to place")][0])
    assert to_place == pytest.approx([4 / 5, 4 / 5, 2 / 5])


def test_the_side_with_fewer_units_places_next_and_the_other_finishes():
    position = deployed(WORKS, 1)
    placed = [e for e in position_events(WORKS, 1) if isinstance(e, Placed)]
    assert [e.team for e in placed] == [0, 1, 0, 1, 0, 1, 0, 1, 1]
    assert position.round == 1


def position_events(game, seed):
    players = [GreedyPlayer(roster=r) for r in (TWO_KNIGHTS, FIVE_UNITS)]
    position = game.start(seed)
    rng = np.random.default_rng(0)
    events = []
    while position.deploying:
        team = game.player(position)
        position = game.step(
            position, players[team](game, team, position, game.legal(position), rng)
        )
        events.extend(position.events)
    return events


def test_a_second_knight_cannot_take_the_last_place():
    game = dataclasses.replace(WORKS, caps=(2, 0, 2, 0, 0), team_size=3)
    position = game.start(1)
    for kind in (KNIGHT, FOOT, FOOT, FOOT):  # team 0 a knight and a footsoldier: 1 place left
        cell = int(game.legal(position)[Verb.PLACE + kind].nonzero()[0][0])
        position = game.step(position, (Verb.PLACE + kind, cell))
    legal = game.legal(position)
    assert game.player(position) == 0
    assert not legal[Verb.PLACE + KNIGHT].any()
    assert legal[Verb.PLACE + FOOT].any()


def test_a_side_with_no_kind_that_fits_is_passed_over():
    game = dataclasses.replace(WORKS, caps=(3, 0, 2, 0, 0), team_size=3)
    position = game.start(1)
    for team, kind in ((0, KNIGHT), (1, FOOT), (0, KNIGHT), (1, FOOT), (1, FOOT)):
        # team 0 is full after two knights, so team 1 places twice at the end
        assert game.player(position) == team
        cell = int(game.legal(position)[Verb.PLACE + kind].nonzero()[0][0])
        position = game.step(position, (Verb.PLACE + kind, cell))
    assert position.round == 1
    teams = position.state.units.team[position.state.occupied]
    assert np.bincount(teams).tolist() == [2, 3]


def test_a_place_too_few_for_any_kind_left_is_left_empty():
    game = dataclasses.replace(WORKS, caps=(1, 0, 2, 0, 0), team_size=3)
    position = game.start(1)
    for team, kind in ((0, KNIGHT), (1, FOOT), (0, FOOT), (1, KNIGHT)):
        assert game.player(position) == team
        cell = int(game.legal(position)[Verb.PLACE + kind].nonzero()[0][0])
        position = game.step(position, (Verb.PLACE + kind, cell))
    # each side has 2 places filled and 1 left: too few for a second knight, and no footsoldier
    assert position.round == 1


def test_caps_must_fill_a_team_by_places():
    Tactics(caps=(0, 0, 2, 0, 0), team_size=3)
    with pytest.raises(ValueError, match="places"):
        Tactics(caps=(0, 0, 2, 0, 0), team_size=4)


def test_the_default_roster_fills_the_places():
    game = Tactics()
    assert default_roster(game.caps, game.team_size, game.units) == (3, 2, 1, 1, 1)
    assert default_roster((1, 0, 2, 0, 0), 4) == (1, 0, 2, 0, 0)
    assert default_roster((1, 0, 2, 0, 0), 4, game.units) == (1, 0, 2, 0, 0)


def test_the_default_roster_leaves_out_a_unit_that_does_not_fit():
    units = Tactics().units
    # a footsoldier and a knight leave 1 place, too few for the second knight
    assert default_roster((1, 0, 2, 0, 0), 3, units) == (1, 0, 1, 0, 0)
    # one of each in order while it fits: no room for a first knight of 2 places
    wide = (units[FOOT], dataclasses.replace(units[KNIGHT], places=2))
    assert default_roster((1, 1), 2, wide) == (1, 0)


def test_places_grow_by_the_extra_for_each_further_unit():
    knight = Tactics().units[KNIGHT]
    assert [knight.places_for(n) for n in range(4)] == [0, 1, 3, 6]
    with pytest.raises(ValueError, match="extra_places"):
        dataclasses.replace(knight, extra_places=-1)


# What networks see of it.


def test_the_order_plane_ranks_the_living_units(make_battle, at):
    pieces = [
        piece(2, 2, FOOT, 0, initiative=5),
        piece(4, 4, FOOT, 1, initiative=2),
        piece(6, 6, FOOT, 0, initiative=9),
    ]
    position = make_battle(pieces, game=WORKS)
    order = WORKS.observe(position, 0)[WORKS.planes.index("order")]
    assert order[at(4, 4, WORKS)] == pytest.approx(1 / 3)
    assert order[at(2, 2, WORKS)] == pytest.approx(2 / 3)
    assert order[at(6, 6, WORKS)] == pytest.approx(1.0)
    assert order.sum() == pytest.approx(2.0)


def test_the_moved_its_move_plane_marks_a_full_move(make_battle, at):
    pieces = [
        piece(2, 2, KNIGHT, 0, moved=3),
        piece(4, 4, KNIGHT, 1, moved=2),
        piece(6, 6, FOOT, 0, moved=2),
    ]
    position = make_battle(pieces, game=WORKS)
    full = WORKS.observe(position, 0)[WORKS.planes.index("moved its move")]
    assert full.nonzero()[0].tolist() == sorted([at(2, 2, WORKS), at(6, 6, WORKS)])
