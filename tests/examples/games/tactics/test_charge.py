"""Knights charge, footsoldiers brace against them, and an attack breaks the wall."""

import dataclasses

import pytest

from examples.games.tactics import (
    Attacked,
    Died,
    GreedyPlayer,
    RandomPlayer,
    Tactics,
    UnitStats,
    Verb,
    earlier_rules,
    event_text,
)
from examples.games.tactics.code_player import _attack
from game_demo import check_game
from gridzero.game import Action
from tactics_demo import ARCHER, FOOT, KNIGHT, piece

GAME = dataclasses.replace(Tactics(), shape=(9, 9), home_depth=4, team_size=5, capture_by=8)

# Facing 0, the unit at (4, 4) has (5, 4), (5, 3) and (4, 5) in front of it, and behind it (3, 4).
RIDER = (5, 4)
TARGET = (4, 4)
BESIDE = (3, 5)  # next to the target and not next to the rider


def battle(make_battle, target=FOOT, friend=True, rider=(), **extra):
    pieces = [piece(*RIDER, KNIGHT, 0, moved=3, **dict(rider))]
    pieces.append(piece(*TARGET, target, 1, facing=0, **extra))
    if friend:
        pieces.append(piece(*BESIDE, FOOT, 1, facing=0))
    return make_battle(pieces, game=GAME)


def attack_events(position, at):
    """Return the attacks and deaths of the first attacker's attack on the target."""
    after = GAME.step(position, Action(Verb.ATTACK, at(*TARGET)))
    return after, [e for e in after.events if isinstance(e, Attacked | Died)]


def test_a_charge_adds_its_bonus_once_the_knight_has_moved_its_full_move(make_battle, at):
    position = make_battle(
        [piece(*RIDER, KNIGHT, 0, moved=3), piece(*TARGET, ARCHER, 1, facing=0)], game=GAME
    )
    assert GAME.odds(position, at(*RIDER), at(*TARGET)).atk == 4


def test_a_knight_short_of_its_full_move_does_not_charge(make_battle, at):
    position = make_battle(
        [piece(*RIDER, KNIGHT, 0, moved=2), piece(*TARGET, ARCHER, 1, facing=0)], game=GAME
    )
    assert GAME.odds(position, at(*RIDER), at(*TARGET)).atk == 2


def test_odds_count_the_tiles_a_move_to_the_attacking_tile_would_add(make_battle, at):
    position = make_battle(
        [piece(7, 2, KNIGHT, 0, moved=1), piece(*TARGET, ARCHER, 1, facing=0)], game=GAME
    )
    assert GAME.odds(position, at(7, 2), at(*TARGET), at=at(*RIDER)).atk == 4
    position = make_battle(
        [piece(6, 3, KNIGHT, 0, moved=1), piece(*TARGET, ARCHER, 1, facing=0)], game=GAME
    )
    assert GAME.odds(position, at(6, 3), at(*TARGET), at=at(*RIDER)).atk == 2


def test_flanking_multiplies_the_charge(make_battle, at):
    position = make_battle(
        [piece(3, 4, KNIGHT, 0, moved=3), piece(*TARGET, ARCHER, 1, facing=0)], game=GAME
    )
    odds = GAME.odds(position, at(3, 4), at(*TARGET))
    assert (odds.flank, odds.atk) == (True, 12)


def test_a_formed_footsoldier_stops_a_charge_and_says_what_the_knight_can_expect(make_battle, at):
    odds = GAME.odds(battle(make_battle), at(*RIDER), at(*TARGET))
    assert odds.atk == 2
    # Atk 4 against the knight's Def 2, unflanked.
    assert odds.struck == pytest.approx(0.85 * 4)


def test_the_expected_strike_is_no_more_than_the_knight_has_left(make_battle, at):
    odds = GAME.odds(battle(make_battle, rider={"hp": 1}), at(*RIDER), at(*TARGET))
    assert odds.struck == pytest.approx(0.85)


def test_a_lone_footsoldier_does_not_brace(make_battle, at):
    odds = GAME.odds(battle(make_battle, friend=False), at(*RIDER), at(*TARGET))
    assert (odds.atk, odds.struck) == (4, 0.0)


def test_only_a_formation_friend_makes_a_footsoldier_brace(make_battle, at):
    position = make_battle(
        [
            piece(*RIDER, KNIGHT, 0, moved=3),
            piece(*TARGET, FOOT, 1, facing=0),
            piece(*BESIDE, ARCHER, 1, facing=0),
        ],
        game=GAME,
    )
    assert GAME.odds(position, at(*RIDER), at(*TARGET)).struck == 0.0


def test_a_charger_with_formation_of_its_own_does_not_form_the_wall_it_charges(make_battle, at):
    units = list(GAME.units)
    units[KNIGHT] = dataclasses.replace(units[KNIGHT], formation=True)
    game = dataclasses.replace(GAME, units=tuple(units))
    position = make_battle(
        [piece(*RIDER, KNIGHT, 0, moved=3), piece(*TARGET, FOOT, 1, facing=0)], game=game
    )
    assert game.odds(position, at(*RIDER), at(*TARGET)).struck == 0.0


def test_a_footsoldier_attacked_this_round_has_lost_its_footing(make_battle, at):
    odds = GAME.odds(battle(make_battle, engaged=True), at(*RIDER), at(*TARGET))
    assert (odds.atk, odds.struck) == (5, 0.0)  # 2 + 1 engaged + 2 charge


BEHIND = (3, 4)  # behind the target, and not next to its friend


def behind(make_battle, attacker=KNIGHT, moved=3, friend=True, **extra):
    pieces = [piece(*BEHIND, attacker, 0, moved=moved)]
    pieces.append(piece(*TARGET, FOOT, 1, facing=0, **extra))
    if friend:
        pieces.append(piece(*BESIDE, FOOT, 1, facing=0))
    return make_battle(pieces, game=GAME)


def test_a_formed_footsoldier_braces_against_a_charge_from_behind(make_battle, at):
    odds = GAME.odds(behind(make_battle), at(*BEHIND), at(*TARGET))
    assert (odds.flank, odds.atk) == (False, 2)
    assert odds.struck == pytest.approx(0.85 * 4)


def test_a_knight_that_did_not_charge_cannot_flank_a_formed_footsoldier_either(make_battle, at):
    odds = GAME.odds(behind(make_battle, moved=0), at(*BEHIND), at(*TARGET))
    assert (odds.flank, odds.atk, odds.struck) == (False, 2, 0.0)


def test_other_kinds_still_flank_a_formed_footsoldier(make_battle, at):
    odds = GAME.odds(behind(make_battle, attacker=FOOT), at(*BEHIND), at(*TARGET))
    assert (odds.flank, odds.atk) == (True, 12)


def test_a_knight_flanks_a_formed_footsoldier_attacked_this_round(make_battle, at):
    odds = GAME.odds(behind(make_battle, engaged=True), at(*BEHIND), at(*TARGET))
    assert (odds.flank, odds.atk, odds.struck) == (True, 15, 0.0)  # (2 + 1 + 2) x 3


def test_a_knight_flanks_a_lone_footsoldier(make_battle, at):
    odds = GAME.odds(behind(make_battle, friend=False), at(*BEHIND), at(*TARGET))
    assert (odds.flank, odds.atk, odds.struck) == (True, 12, 0.0)


def test_a_second_enemy_beside_the_wall_does_not_let_a_knight_flank_it(make_battle, at):
    position = make_battle(
        [
            piece(*RIDER, KNIGHT, 0, moved=3),
            piece(*TARGET, FOOT, 1, facing=0),
            piece(*BESIDE, FOOT, 1, facing=0),
            piece(5, 3, FOOT, 0),
        ],
        game=GAME,
    )
    odds = GAME.odds(position, at(*RIDER), at(*TARGET))
    assert (odds.flank, odds.atk) == (False, 2)
    assert odds.struck > 0.0


def test_a_knight_that_did_not_charge_is_not_struck(make_battle, at):
    position = make_battle(
        [
            piece(*RIDER, KNIGHT, 0, moved=2),
            piece(*TARGET, FOOT, 1, facing=0),
            piece(*BESIDE, FOOT, 1, facing=0),
        ],
        game=GAME,
    )
    _, events = attack_events(position, at)
    assert [e.attacker_kind for e in events if isinstance(e, Attacked)] == [KNIGHT]


def test_a_bracing_footsoldier_strikes_first_and_then_the_knight_attacks_without_its_bonus(
    make_battle, at
):
    position = battle(make_battle)
    _, events = attack_events(position, at)
    strike, *rest = events
    assert isinstance(strike, Attacked)
    assert (strike.attacker_kind, strike.target_kind, strike.braced) == (FOOT, KNIGHT, True)
    assert strike.atk == 4
    assert strike.flank is False
    attack = next(e for e in rest if isinstance(e, Attacked))
    assert (attack.attacker_kind, attack.atk, attack.braced) == (KNIGHT, 2, False)


def test_a_strike_that_kills_the_knight_ends_its_attack_and_its_turn(make_battle, at):
    for seed in range(40):
        pieces = [
            piece(*RIDER, KNIGHT, 0, moved=3, hp=1),
            piece(*TARGET, FOOT, 1, facing=0),
            piece(*BESIDE, FOOT, 1, facing=0),
            piece(0, 8, FOOT, 0),
        ]
        position = make_battle(pieces, game=GAME, seed=seed)
        after, events = attack_events(position, at)
        if isinstance(events[-1], Died):
            break
    else:
        pytest.fail("no strike hit")
    assert [type(e) for e in events] == [Attacked, Died]
    assert after.state.units.kind[at(*RIDER)] == -1
    assert after.state.units.hp[at(*TARGET)] == GAME.units[FOOT].hp
    assert not after.state.units.engaged[at(*TARGET)]
    assert not after.over
    assert GAME.active(after) == at(*TARGET)


def test_the_strike_engages_the_knight_and_the_knights_attack_engages_the_wall(make_battle, at):
    for seed in range(40):
        position = make_battle(
            [
                piece(*RIDER, KNIGHT, 0, moved=3),
                piece(*TARGET, FOOT, 1, facing=0),
                piece(*BESIDE, FOOT, 1, facing=0),
            ],
            game=GAME,
            seed=seed,
        )
        after, events = attack_events(position, at)
        strike = events[0]
        if strike.hit:
            break
    units = after.state.units
    assert units.hp[at(*RIDER)] == GAME.units[KNIGHT].hp - 4
    assert units.engaged[at(*RIDER)]
    assert units.engaged[at(*TARGET)]


def test_the_strike_is_told_apart_in_text(make_battle, at):
    after, events = attack_events(battle(make_battle), at)
    assert "braces against a charge" in event_text(GAME, after.state, events[0])


def test_greedy_counts_the_strike_against_an_attack(make_battle, at):
    position = battle(make_battle)
    player = GreedyPlayer()
    braced, _ = player._best_attack(GAME, position, at(*RIDER), at(*RIDER), [at(*TARGET)])
    alone = battle(make_battle, friend=False)
    plain, _ = player._best_attack(GAME, alone, at(*RIDER), at(*RIDER), [at(*TARGET)])
    odds = GAME.odds(position, at(*RIDER), at(*TARGET))
    expected = odds.chance * player.attack * odds.atk - player.attack * odds.struck
    assert braced == pytest.approx(expected)
    assert braced < plain


def test_the_code_players_damage_is_net_of_the_strike(make_battle, at):
    position = battle(make_battle)
    odds = GAME.odds(position, at(*RIDER), at(*TARGET))
    damage, _ = _attack(GAME, position, at(*RIDER), at(*RIDER), at(*TARGET))
    assert damage == pytest.approx(odds.chance * odds.atk - odds.struck)


def test_whole_games_play_by_the_rules():
    check_game(GAME, RandomPlayer(), range(2))
    check_game(GAME, GreedyPlayer(), range(2))


def test_knights_charge_and_footsoldiers_brace_by_default():
    rules = Tactics()
    knight, foot = rules.units[KNIGHT], rules.units[FOOT]
    assert (knight.atk, knight.move, knight.double_move, knight.charge) == (2, 3, False, 2)
    assert foot.braces
    assert foot.formation


def test_in_the_earlier_rules_no_kind_charges_or_braces():
    assert not any(stats.charge or stats.braces for stats in earlier_rules().units)


def test_a_charge_cannot_be_negative():
    with pytest.raises(ValueError, match="charge"):
        UnitStats("x", hp=1, atk=1, defense=0, move=1, reach=1, charge=-1)


def test_bracing_must_be_a_flag():
    with pytest.raises(TypeError, match="braces"):
        UnitStats("x", hp=1, atk=1, defense=0, move=1, reach=1, braces=1)  # ty: ignore[invalid-argument-type]  (the check)
