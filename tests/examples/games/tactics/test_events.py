import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    Attacked,
    Deferred,
    Died,
    Ended,
    GameOver,
    Moved,
    Placed,
    RoundStarted,
    Swapped,
    Turned,
    Verb,
)
from gridzero.core import steps
from tactics_demo import ARCHER, CAPTAIN, FOOT, SMALL, TINY, piece, play_random


def take(game, position, verb, cell):
    return game.step(position, (verb, cell))


def test_a_placement_records_the_unit_and_the_last_one_starts_round_one():
    position = TINY.start(1)
    cell = int(np.flatnonzero(TINY.legal(position)[Verb.PLACE + CAPTAIN])[0])
    after = take(TINY, position, Verb.PLACE + CAPTAIN, cell)
    assert after.events == (Placed(0, CAPTAIN, 0, cell),)
    while after.state.occupied.sum() < 2 * TINY.team_size - 1:
        legal = TINY.legal(after)
        verb = int(np.flatnonzero(legal.any(axis=1))[0])
        after = take(TINY, after, verb, int(np.flatnonzero(legal[verb])[0]))
    legal = TINY.legal(after)
    verb = int(np.flatnonzero(legal.any(axis=1))[0])
    last = take(TINY, after, verb, int(np.flatnonzero(legal[verb])[0]))
    assert isinstance(last.events[0], Placed)
    assert last.events[1:] == (RoundStarted(1),)


def test_a_move_records_its_route_step_by_step(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    after = take(SMALL, position, Verb.MOVE, at(6, 4))
    (moved,) = after.events
    assert isinstance(moved, Moved)
    assert moved.uid == 0
    assert moved.route[0] == at(4, 4)
    assert moved.route[-1] == at(6, 4)
    space = position.state.space
    for a, b in zip(moved.route, moved.route[1:], strict=False):
        assert steps(space, a)[b] == 1


def test_an_attack_records_its_numbers_and_the_damage_done(make_battle, at):
    for seed in range(20):
        position = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1)], seed=seed)
        odds = SMALL.odds(position, at(5, 4), at(4, 4))
        after = take(SMALL, position, Verb.ATTACK, at(4, 4))
        (event,) = after.events
        assert isinstance(event, Attacked)
        assert (event.attacker, event.target, event.cell) == (0, 1, at(4, 4))
        assert (event.attacker_kind, event.attacker_team, event.target_kind) == (FOOT, 0, FOOT)
        assert (event.atk, event.defense, event.flank) == (odds.atk, odds.defense, odds.flank)
        assert event.hit == (event.roll == 20 or (event.roll != 1 and event.roll >= event.needed))
        assert event.damage == 5 - after.state.units.hp[at(4, 4)]
        assert not event.deferred


def test_a_kill_records_the_death_and_the_last_one_the_end(make_battle, at):
    for seed in range(50):
        pieces = [piece(5, 4, CAPTAIN, 0), piece(4, 4, ARCHER, 1, hp=2)]
        after = take(SMALL, make_battle(pieces, seed=seed), Verb.ATTACK, at(4, 4))
        if after.over:
            break
    attack, died, over = after.events
    assert isinstance(attack, Attacked)
    assert attack.hit
    assert attack.damage == 2  # at most what it had
    assert died == Died(1, ARCHER, 1, at(4, 4))
    assert over == GameOver(0)


def test_a_deferred_attack_splits_the_move_around_it(make_battle, at):
    pieces = [piece(7, 4, FOOT, 1), piece(3, 4, ARCHER, 0, watch=at(6, 4), done=True)]
    after = take(SMALL, make_battle(pieces), Verb.MOVE, at(5, 4))
    first, attack, *rest = after.events
    assert first == Moved(0, (at(7, 4), at(6, 4)))
    assert isinstance(attack, Attacked)
    assert attack.deferred
    assert attack.cell == at(6, 4)
    if after.state.occupied[at(5, 4)]:
        assert rest == [Moved(0, (at(6, 4), at(5, 4)))]


def test_swaps_turns_deferrals_and_ends_are_recorded(make_battle, at):
    pieces = [piece(4, 4, ARCHER, 0), piece(5, 4, FOOT, 0), piece(8, 8, FOOT, 1)]
    position = make_battle(pieces)
    swapped = take(SMALL, position, Verb.SWAP, at(5, 4))
    assert swapped.events == (Swapped(0, 1, at(4, 4), at(5, 4)),)
    turned = take(SMALL, swapped, Verb.FACE + 3, at(5, 4))
    assert turned.events == (Turned(0, 3),)
    deferred = take(SMALL, turned, Verb.DEFER, at(6, 6))
    assert deferred.events == (Deferred(0, at(6, 6)),)
    ended = take(SMALL, deferred, Verb.END, at(5, 4))
    assert ended.events == (Ended(0),)


def test_the_last_turn_of_a_round_starts_the_next(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1, done=True)])
    after = take(SMALL, position, Verb.END, at(4, 4))
    assert after.events == (Ended(0), RoundStarted(2))


def test_a_draw_is_recorded_as_a_game_over_without_a_winner(make_battle, at):
    game = dataclasses.replace(SMALL, max_rounds=1)
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1, done=True)], game=game)
    after = take(game, position, Verb.END, at(4, 4))
    assert after.events == (Ended(0), GameOver(None))


def test_a_position_holds_only_the_events_of_its_own_step(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    moved = take(SMALL, position, Verb.MOVE, at(5, 4))
    ended = take(SMALL, moved, Verb.END, at(5, 4))
    assert position.events == ()
    assert ended.events == (Ended(0),)


def same_step(a, b):
    return a.events == b.events and all(np.array_equal(a.state[f], b.state[f]) for f in a.state)


@pytest.mark.parametrize("seed", range(3))
def test_only_steps_that_drew_dice_depend_on_the_seed(seed):
    rng = np.random.default_rng(seed)
    kinds = {"plain": 0, "round": 0, "attack": 0, "deferred": 0, "differ": 0}
    for position, _ in play_random(TINY, seed, rng):
        if position.over:
            continue
        other = dataclasses.replace(position, seed=seed + 1000)
        for verb, cell in np.argwhere(TINY.legal(position)):
            a = TINY.step(position, (verb, cell))
            b = TINY.step(other, (verb, cell))
            assert a.drew == b.drew == any(isinstance(e, RoundStarted | Attacked) for e in a.events)
            if not a.drew:
                assert same_step(a, b)
                kinds["plain"] += 1
                continue
            kinds["differ"] += not same_step(a, b)
            kinds["round"] += any(isinstance(e, RoundStarted) for e in a.events)
            attacks = [e for e in a.events if isinstance(e, Attacked)]
            kinds["attack"] += bool(attacks)
            kinds["deferred"] += any(e.deferred for e in attacks)
    assert min(kinds.values()) > 0, kinds
