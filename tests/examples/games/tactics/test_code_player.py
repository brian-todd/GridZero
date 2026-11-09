import dataclasses
import json

import numpy as np
import pytest

from examples.games.tactics import (
    HAND_PICKED,
    MOVE_FEATURES,
    VALUE_FEATURES,
    CodeEvaluator,
    GreedyPlayer,
    Hold,
    RandomPlayer,
    Tactics,
    Verb,
    move_features,
    small_rules,
    value_features,
)
from examples.games.tactics.code_player import _by_name
from gridzero.agents import AlphaZeroPlayer
from gridzero.core import steps
from gridzero.game import Action, series, tally
from tactics_demo import (
    ARCHER,
    CAPTAIN,
    ENGINEER,
    FOOT,
    KNIGHT,
    SMALL,
    mirrored,
    piece,
    play_random,
)

GAME = small_rules()


def value_of(position, game=GAME):
    return dict(zip(VALUE_FEATURES, value_features(game, [position])[0], strict=True))


def moves_of(position, game=GAME):
    """Return each legal action and its features by name."""
    legal = game.legal(position)
    rows = move_features(game, position, legal)
    actions = [Action(int(v), int(c)) for v, c in zip(*legal.nonzero(), strict=True)]
    return {
        action: {name: rows[i, j] for j, name in enumerate(MOVE_FEATURES)}
        for i, action in enumerate(actions)
    }


def test_hit_points_count_as_shares_of_a_unit_each_side_its_own_way(make_battle):
    position = make_battle(
        [piece(1, 1, FOOT, 0, hp=2), piece(2, 1, KNIGHT, 0), piece(9, 9, KNIGHT, 1, hp=1)],
        game=GAME,
    )
    f = value_of(position)
    assert f["hp_footsoldier"] == pytest.approx(2 / GAME.units[FOOT].hp)
    assert f["hp_knight"] == pytest.approx(1 - 1 / GAME.units[KNIGHT].hp)
    assert f["hp_captain"] == f["hp_archer"] == f["hp_engineer"] == 0


@pytest.mark.parametrize(("owner", "sign"), [(0, 1), (1, -1)])
def test_owner_and_clock_are_signed_by_who_owns_the_point(make_battle, owner, sign):
    position = make_battle(
        [piece(1, 1, FOOT, 0), piece(9, 9, FOOT, 1)], game=GAME, hold=Hold(owner, 3)
    )
    f = value_of(position)
    assert f["owner"] == sign
    assert f["clock"] == pytest.approx(sign * 3 / GAME.hold_rounds)


def test_the_deadline_counts_only_while_nobody_has_captured(make_battle):
    pieces = [piece(1, 1, FOOT, 0), piece(9, 9, FOOT, 1)]
    free = make_battle(pieces, game=GAME, round=4)
    assert value_of(free)["deadline"] == pytest.approx(4 / GAME.capture_by)
    assert value_of(free)["owner"] == value_of(free)["clock"] == 0
    assert value_of(make_battle(pieces, game=GAME, round=4, hold=Hold(1, 1)))["deadline"] == 0


def test_walks_near_and_on_point_are_read_around_walls(make_battle):
    # Team 0's footsoldier stands on the point; team 1's walks 3 around the wall, not 2 through it.
    position = make_battle(
        [piece(7, 7, FOOT, 0), piece(9, 7, FOOT, 1)],
        game=GAME,
        point=(7, 7),
        walls=[(8, 7)],
    )
    f = value_of(position)
    assert f["on_point"] == 1
    assert f["walk_nearest"] == pytest.approx(3 / 15)
    assert f["walk_mean"] == pytest.approx(3 / 15)
    assert f["near_point"] == 1  # a walk of 2 reaches only team 0's own tile


def test_walks_stay_between_minus_1_and_1_when_a_side_cannot_reach_the_point(make_battle):
    walled = [(7, 7), (8, 7), (9, 6), (9, 5), (8, 5), (7, 6)]  # all six around (8, 6)
    position = make_battle(
        [piece(1, 1, FOOT, 0), piece(13, 13, FOOT, 1)], game=GAME, point=(8, 6), walls=walled
    )
    f = value_of(position)
    assert f["walk_nearest"] == 0  # neither reaches it: both count as n_cells away
    far = make_battle(
        [piece(8, 6, FOOT, 0), piece(13, 13, FOOT, 1)], game=GAME, point=(8, 6), walls=walled
    )
    assert value_of(far)["walk_nearest"] == 1
    assert value_of(mirrored(GAME, far))["walk_nearest"] == -1


def test_threat_sums_the_atk_that_can_reach_each_enemy(make_battle):
    # A knight strikes from 3 + 1 = 4 tiles, a footsoldier from 2 + 1 = 3: 4 apart.
    position = make_battle([piece(2, 2, KNIGHT, 0), piece(6, 2, FOOT, 1)], game=GAME)
    assert value_of(position)["threat"] == GAME.units[KNIGHT].atk


def test_to_act_counts_turns_left_this_round(make_battle):
    position = make_battle(
        [
            piece(1, 1, FOOT, 0),
            piece(2, 1, FOOT, 0, done=True),
            piece(3, 1, FOOT, 0),
            piece(9, 9, FOOT, 1),
        ],
        game=GAME,
    )
    assert value_of(position)["to_act"] == 1


def test_the_start_reads_nothing_and_a_placement_only_what_it_changes():
    start = GAME.start(0)
    assert value_features(GAME, [start]).tolist() == [[0.0] * len(VALUE_FEATURES)]
    legal = GAME.legal(start)
    verb, cell = np.argwhere(legal)[0]
    placed = GAME.step(start, Action(int(verb), int(cell)))
    f = value_of(placed)
    assert f["to_place"] == -1  # player 0 has one fewer left to place
    assert f["hp_footsoldier"] == 1
    assert all(f[name] == 0 for name in VALUE_FEATURES[6:])


def test_value_features_turn_over_with_the_board_but_for_the_deadline():
    rng = np.random.default_rng(3)
    checked = 0
    for position, _ in play_random(GAME, 4, rng):
        if position.over or position.deploying or rng.random() > 0.05:
            continue
        original = value_of(position)
        turned = value_of(mirrored(GAME, position))
        for name in VALUE_FEATURES:
            if name == "deadline":
                assert turned[name] == original[name]
            else:
                assert turned[name] == pytest.approx(-original[name]), name
        checked += 1
    assert checked > 0


def test_moves_come_in_the_order_of_flatnonzero_and_each_has_one_verb():
    rng = np.random.default_rng(5)
    verbs = [name for name in MOVE_FEATURES if name.startswith("is_")]
    for position, _ in play_random(GAME, 5, rng):
        if position.over or rng.random() > 0.1:
            continue
        legal = GAME.legal(position)
        rows = move_features(GAME, position, legal)
        assert len(rows) == np.count_nonzero(legal)
        flags = rows[:, [MOVE_FEATURES.index(name) for name in verbs]]
        kinds = rows[
            :,
            [
                MOVE_FEATURES.index(name)
                for name in MOVE_FEATURES
                if name[:6] == "place_" and "ahead" not in name
            ],
        ]
        flat = np.flatnonzero(legal)
        verb = flat // GAME.space.n_cells
        if position.deploying:
            assert kinds.sum(axis=1).tolist() == [1.0] * len(rows)
            assert kinds.argmax(axis=1).tolist() == (verb - Verb.PLACE).tolist()
            assert (flags == 0).all()
        else:
            assert flags.sum(axis=1).tolist() == [1.0] * len(rows)
            assert (rows[:, MOVE_FEATURES.index("is_end")] == (verb == Verb.END)).all()
            assert (rows[:, MOVE_FEATURES.index("is_attack")] == (verb == Verb.ATTACK)).all()
            assert (kinds == 0).all()


def test_an_attack_reads_as_the_game_odds_it(make_battle, at):
    position = make_battle(
        [piece(5, 5, FOOT, 0), piece(6, 5, ARCHER, 1, hp=1), piece(5, 6, CAPTAIN, 1)], game=GAME
    )
    moves = moves_of(position)
    a, weak = at(5, 5, GAME), at(6, 5, GAME)
    odds = GAME.odds(position, a, weak)
    f = moves[Action(Verb.ATTACK, weak)]
    assert f["damage"] == pytest.approx(odds.chance * 1)  # only 1 hit point left to take
    assert f["kill"] == pytest.approx(
        odds.chance * (GAME.units[ARCHER].hp + GAME.units[ARCHER].atk)
    )


def test_an_attack_for_exactly_the_hit_points_left_kills(make_battle, at):
    target = at(6, 5, GAME)
    pieces = [piece(5, 5, FOOT, 0)]
    odds = GAME.odds(
        make_battle([*pieces, piece(6, 5, KNIGHT, 1)], game=GAME), at(5, 5, GAME), target
    )
    position = make_battle([*pieces, piece(6, 5, KNIGHT, 1, hp=odds.atk)], game=GAME)
    assert moves_of(position)[Action(Verb.ATTACK, target)]["kill"] > 0
    position = make_battle([*pieces, piece(6, 5, KNIGHT, 1, hp=odds.atk + 1)], game=GAME)
    assert moves_of(position)[Action(Verb.ATTACK, target)]["kill"] == 0


@pytest.mark.parametrize("facing", range(6))
def test_an_attack_that_cannot_kill_reads_its_damage_only(make_battle, at, facing):
    position = make_battle(
        [piece(5, 5, ARCHER, 0), piece(6, 5, CAPTAIN, 1, facing=facing)], game=GAME
    )
    a, target = at(5, 5, GAME), at(6, 5, GAME)
    odds = GAME.odds(position, a, target)
    assert odds.atk < GAME.units[CAPTAIN].hp  # at most 2 x 3
    f = moves_of(position)[Action(Verb.ATTACK, target)]
    assert f["damage"] == pytest.approx(odds.chance * odds.atk)
    assert f["kill"] == 0


def test_a_move_reads_the_best_attack_from_its_tile_and_only_if_it_can_still_attack(
    make_battle, at
):
    pieces = [piece(5, 5, FOOT, 0), piece(8, 5, ARCHER, 1, hp=1)]
    tile = at(7, 5, GAME)
    position = make_battle(pieces, game=GAME)
    f = moves_of(position)[Action(Verb.MOVE, tile)]
    odds = GAME.odds(position, at(5, 5, GAME), at(8, 5, GAME), at=tile)
    assert f["damage_after"] == pytest.approx(odds.chance)
    assert f["kill_after"] > 0
    assert moves_of(position)[Action(Verb.MOVE, at(6, 5, GAME))]["damage_after"] == 0
    pieces = [piece(5, 5, FOOT, 0, attacked=True), piece(8, 5, ARCHER, 1, hp=1)]
    assert moves_of(make_battle(pieces, game=GAME))[Action(Verb.MOVE, tile)]["damage_after"] == 0


def test_a_double_mover_moving_past_its_move_reads_no_attack_after(make_battle, at):
    units = list(GAME.units)
    units[KNIGHT] = dataclasses.replace(units[KNIGHT], double_move=True)
    game = dataclasses.replace(GAME, units=tuple(units))
    position = make_battle([piece(2, 5, KNIGHT, 0), piece(9, 5, FOOT, 1)], game=game)
    moves = moves_of(position, game)
    # 6 tiles, past its move of 3: adjacent to the enemy but no attack.
    assert moves[Action(Verb.MOVE, at(8, 5, game))]["damage_after"] == 0


def test_moves_toward_the_point_count_only_if_the_unit_has_reason_to_go(make_battle, at):
    pieces = [piece(3, 7, FOOT, 0), piece(13, 13, FOOT, 1)]
    tile = at(5, 7, GAME)
    open_point = make_battle(pieces, game=GAME, point=(7, 7))
    assert moves_of(open_point)[Action(Verb.MOVE, tile)]["closer_point"] == 2
    owned = make_battle(pieces, game=GAME, point=(7, 7), hold=Hold(0, 1))
    assert moves_of(owned)[Action(Verb.MOVE, tile)]["closer_point"] == 0
    theirs = make_battle(pieces, game=GAME, point=(7, 7), hold=Hold(1, 1))
    assert moves_of(theirs)[Action(Verb.MOVE, tile)]["closer_point"] == 2


def test_closer_enemy_means_closer_to_the_units_own_range(make_battle, at):
    # An archer stands 3 from its target already: walking up loses the range, as does walking off.
    position = make_battle([piece(5, 5, ARCHER, 0), piece(8, 5, FOOT, 1)], game=GAME)
    moves = moves_of(position)
    assert moves[Action(Verb.MOVE, at(6, 5, GAME))]["closer_enemy"] == -1
    assert moves[Action(Verb.MOVE, at(4, 5, GAME))]["closer_enemy"] == -1
    melee = make_battle([piece(5, 5, FOOT, 0), piece(9, 5, FOOT, 1)], game=GAME)
    assert moves_of(melee)[Action(Verb.MOVE, at(6, 5, GAME))]["closer_enemy"] == 1


def test_exposure_is_the_atk_a_tile_adds_over_the_one_it_leaves(make_battle, at):
    # The enemy footsoldier strikes from 3 tiles: 4 is out of its reach, 3 in it.
    position = make_battle([piece(3, 5, FOOT, 0), piece(7, 5, FOOT, 1)], game=GAME)
    moves = moves_of(position)
    assert moves[Action(Verb.MOVE, at(4, 5, GAME))]["exposure"] == GAME.units[FOOT].atk
    assert moves[Action(Verb.MOVE, at(2, 5, GAME))]["exposure"] == 0


def test_a_tile_out_of_reach_reads_the_atk_it_leaves_behind(make_battle, at):
    position = make_battle([piece(4, 5, FOOT, 0), piece(7, 5, FOOT, 1)], game=GAME)
    moves = moves_of(position)
    assert moves[Action(Verb.MOVE, at(5, 5, GAME))]["exposure"] == 0  # in reach before and after
    assert moves[Action(Verb.MOVE, at(3, 5, GAME))]["exposure"] == -GAME.units[FOOT].atk


def test_formation_counts_formed_friends_up_to_the_cap_and_aura_who_gives_it(make_battle, at):
    position = make_battle(
        [
            piece(5, 5, FOOT, 0),
            piece(6, 6, FOOT, 0),
            piece(7, 5, FOOT, 0),
            piece(6, 4, FOOT, 0),
            piece(1, 13, CAPTAIN, 0),
            piece(13, 13, FOOT, 1),
        ],
        game=GAME,
    )
    f = moves_of(position)[Action(Verb.MOVE, at(6, 5, GAME))]
    assert f["formation"] == 2  # three beside it, but no more than the cap
    assert f["aura"] == 0  # the captain is far away
    captain = make_battle(
        [
            piece(5, 5, CAPTAIN, 0),
            piece(7, 5, FOOT, 0),
            piece(8, 5, FOOT, 0),
            piece(13, 13, FOOT, 1),
        ],
        game=GAME,
    )
    assert moves_of(captain)[Action(Verb.MOVE, at(6, 5, GAME))]["aura"] == 2
    led = make_battle(
        [piece(5, 5, FOOT, 0), piece(7, 5, CAPTAIN, 0), piece(13, 13, FOOT, 1)], game=GAME
    )
    assert moves_of(led)[Action(Verb.MOVE, at(6, 5, GAME))]["aura"] == 1


def test_only_the_turn_toward_the_nearest_enemy_faces_it(make_battle, at):
    position = make_battle(
        [piece(5, 5, FOOT, 0, facing=3), piece(8, 5, FOOT, 1), piece(13, 13, FOOT, 1)], game=GAME
    )
    moves = moves_of(position)
    facing = [action for action, f in moves.items() if f["faces_enemy"] == 1]
    assert len(facing) == 1
    assert Verb.FACE <= facing[0].verb < Verb.END
    assert GAME.odds(position, at(8, 5, GAME), at(5, 5, GAME)).flank  # from behind before it turns
    after = GAME.step(position, facing[0])
    assert not GAME.odds(after, at(8, 5, GAME), at(5, 5, GAME)).flank


def test_a_deferred_attack_reads_the_point_and_how_far_the_enemy_is(make_battle, at):
    position = make_battle(
        [piece(5, 5, ARCHER, 0), piece(7, 5, FOOT, 0), piece(9, 5, FOOT, 1)],
        game=GAME,
        point=(7, 5),
    )
    moves = moves_of(position)
    on_point = moves[Action(Verb.DEFER, at(7, 5, GAME))]
    assert on_point["defer_on_point"] == 1
    assert on_point["defer_away"] == 2
    other = moves[Action(Verb.DEFER, at(6, 5, GAME))]
    assert other["defer_on_point"] == 0
    assert other["defer_away"] == 3


def test_clearing_rubble_on_the_enemys_way_costs_its_side_a_step(make_battle, at):
    # The enemy walks 8 along the row to the point, 9 around the rubble; the engineer walks 4.
    position = make_battle(
        [piece(5, 5, ENGINEER, 0), piece(9, 5, FOOT, 1)], game=GAME, point=(1, 5), rubble=[(6, 5)]
    )
    assert moves_of(position)[Action(Verb.BREAK, at(6, 5, GAME))]["work_walks"] == -1


def test_building_on_the_enemys_way_gains_a_step_unless_it_delays_a_friend_too(make_battle, at):
    tile = at(6, 5, GAME)
    alone = [piece(5, 5, ENGINEER, 0), piece(9, 5, FOOT, 1)]
    position = make_battle(alone, game=GAME, point=(1, 5))
    assert moves_of(position)[Action(Verb.BUILD, tile)]["work_walks"] == 1
    both = [*alone, piece(8, 5, FOOT, 0)]
    position = make_battle(both, game=GAME, point=(1, 5))
    assert moves_of(position)[Action(Verb.BUILD, tile)]["work_walks"] == 0


def test_work_that_leaves_tiles_blocked_reads_no_change_in_walks(make_battle, at):
    tile = at(6, 5, GAME)
    pieces = [piece(5, 5, ENGINEER, 0), piece(9, 5, FOOT, 1)]
    wall = make_battle(pieces, game=GAME, point=(1, 5), walls=[(6, 5)])
    assert moves_of(wall)[Action(Verb.BREAK, tile)]["work_walks"] == 0  # wall to rubble
    rubble = make_battle(pieces, game=GAME, point=(1, 5), rubble=[(6, 5)])
    assert moves_of(rubble)[Action(Verb.BUILD, tile)]["work_walks"] == 0  # rubble to wall


def test_work_reads_nothing_without_a_point(make_battle, at):
    position = make_battle([piece(5, 5, ENGINEER, 0), piece(9, 5, FOOT, 1)], game=GAME)
    assert moves_of(position)[Action(Verb.BUILD, at(6, 5, GAME))]["work_walks"] == 0


def test_placements_read_the_kind_and_how_far_ahead_apart_for_melee_and_ranged():
    position = GAME.start(0)
    moves = moves_of(position)
    for action, f in moves.items():
        kind = action.verb - Verb.PLACE
        ranged = GAME.units[kind].reach > 1
        assert f["place_ahead_ranged" if ranged else "place_ahead_melee"] >= 0
        assert f["place_ahead_melee" if ranged else "place_ahead_ranged"] == 0
    ahead = [f["place_ahead_melee"] for f in moves.values()]
    assert min(ahead) == 0
    assert max(ahead) > 0


def test_the_tile_nearest_the_point_reads_as_furthest_ahead():
    position = GAME.start(0)
    state = position.state
    walks = steps(state.space, state.point, blocked=~state.passable)
    moves = moves_of(position)
    placed = [a for a in moves if a.verb == Verb.PLACE + FOOT]
    farthest = max(walks[a.cell] for a in placed)
    for action in placed:
        assert moves[action]["place_ahead_melee"] == farthest - walks[action.cell]


def test_other_kinds_of_unit_are_refused():
    with pytest.raises(ValueError, match="for the kinds"):
        value_features(SMALL, [SMALL.start(0)])
    with pytest.raises(ValueError, match="for the kinds"):
        move_features(SMALL, SMALL.start(0), SMALL.legal(SMALL.start(0)))


def test_a_weight_naming_no_feature_is_refused():
    with pytest.raises(ValueError, match="no such features"):
        _by_name(VALUE_FEATURES, hp_foot=1.0)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"value": [0.0] * len(VALUE_FEATURES)}, TypeError),
        ({"prior": [0.0] * len(MOVE_FEATURES)}, TypeError),
        ({"value": (0.0,)}, ValueError),
        ({"prior": (0.0,)}, ValueError),
        ({"value": (float("nan"), *[0.0] * (len(VALUE_FEATURES) - 1))}, ValueError),
        ({"prior": ("a", *[0.0] * (len(MOVE_FEATURES) - 1))}, TypeError),
        ({"intercept": float("inf")}, ValueError),
    ],
)
def test_bad_constants_are_refused(change, error):
    with pytest.raises(error):
        dataclasses.replace(HAND_PICKED, **change)


def test_the_evaluator_scores_each_position_as_its_features_say_alone_or_in_a_batch():
    rng = np.random.default_rng(7)
    positions = [p for p, _ in play_random(GAME, 7, rng) if not p.over][::20]
    legal = [GAME.legal(p) for p in positions]
    logits, values = HAND_PICKED(GAME, positions, legal)
    assert len(logits) == len(positions)
    assert ((values >= 0) & (values <= 1)).all()
    for i, (position, mask) in enumerate(zip(positions, legal, strict=True)):
        one_logits, one_values = HAND_PICKED(GAME, [position], [mask])
        assert one_values[0] == values[i]
        assert one_logits[0].tolist() == logits[i].tolist()
        assert logits[i].shape == (np.count_nonzero(mask),)
        assert np.isfinite(logits[i]).all()


def test_the_intercept_alone_is_the_lean_of_the_seats():
    lean = CodeEvaluator((0.0,) * len(VALUE_FEATURES), 1.0, (0.0,) * len(MOVE_FEATURES))
    start = GAME.start(0)
    logits, values = lean(GAME, [start], [GAME.legal(start)])
    assert values[0] == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert (logits[0] == 0).all()


def test_the_hand_picked_values_lean_the_right_way(make_battle):
    def value(*pieces, **kw):
        position = make_battle(list(pieces), game=GAME, point=(7, 7), **kw)
        return HAND_PICKED(GAME, [position], [GAME.legal(position)])[1][0]

    even = value(piece(1, 1, FOOT, 0), piece(13, 13, FOOT, 1))
    assert value(piece(1, 1, CAPTAIN, 0), piece(13, 13, FOOT, 1)) > even
    assert value(piece(1, 1, FOOT, 0), piece(13, 13, CAPTAIN, 1)) < even
    assert value(piece(1, 1, FOOT, 0), piece(13, 13, FOOT, 1), hold=Hold(0, 3)) > even


def test_the_search_on_the_hand_picked_evaluator_beats_random():
    player = AlphaZeroPlayer(HAND_PICKED, simulations=8)
    records = series(GAME, [("code", player), ("random", RandomPlayer())], 2, 0)
    assert tally(records).wins["code"] == len(records)


def test_the_full_board_works_too():
    game = Tactics()
    player = AlphaZeroPlayer(HAND_PICKED, simulations=2)
    position = game.start(1)
    rng = np.random.default_rng(1)
    for _ in range(25):  # deployment and into the battle
        action = player(game, game.player(position), position, game.legal(position), rng)
        position = game.step(position, action)
    assert position.round >= 1


def test_it_plays_the_rivals_it_meets_without_error():
    player = AlphaZeroPlayer(HAND_PICKED, simulations=2)
    records = series(GAME, [("code", player), ("greedy", GreedyPlayer())], 1, 0)
    assert len(records) == 2
    assert tally(records).draws == 0


def test_saved_constants_load_the_same(tmp_path):
    path = tmp_path / "fit.json"
    HAND_PICKED.save(path)
    assert CodeEvaluator.load(path) == HAND_PICKED


def test_saved_constants_can_be_read_by_name(tmp_path):
    path = tmp_path / "fit.json"
    HAND_PICKED.save(path)
    data = json.loads(path.read_text())
    assert list(data["value"]) == list(VALUE_FEATURES)
    assert list(data["prior"]) == list(MOVE_FEATURES)
    assert data["value"]["clock"] == HAND_PICKED.value[VALUE_FEATURES.index("clock")]
    assert data["prior"]["damage"] == HAND_PICKED.prior[MOVE_FEATURES.index("damage")]
    assert data["intercept"] == HAND_PICKED.intercept


def test_constants_of_other_features_refuse_to_load(tmp_path):
    path = tmp_path / "fit.json"
    HAND_PICKED.save(path)
    data = json.loads(path.read_text())
    del data["prior"]["damage"]
    path.write_text(json.dumps(data))
    with pytest.raises(KeyError, match="damage"):
        CodeEvaluator.load(path)
