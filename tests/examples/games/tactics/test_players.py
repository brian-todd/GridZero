import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    Fielded,
    GameOver,
    GreedyPlayer,
    Hold,
    Placed,
    RandomPlayer,
    SitterPlayer,
    Tactics,
    Verb,
    summarize,
)
from examples.games.tactics.board import home_corners
from gridzero.core import steps
from gridzero.game import play, replay, series
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, SMALL, TINY, WORKS, piece, play_random

GREEDY = GreedyPlayer()


def act(player, game, position, seed=0):
    return player(
        game, game.player(position), position, game.legal(position), np.random.default_rng(seed)
    )


@pytest.mark.parametrize("player", [RandomPlayer(), GreedyPlayer()], ids=["random", "greedy"])
def test_players_pick_legal_actions_everywhere(player):
    rng = np.random.default_rng(1)
    for position, _ in play_random(TINY, 1, rng):
        if position.over or rng.random() > 0.3:
            continue
        verb, cell = act(player, TINY, position, int(rng.integers(100)))
        assert TINY.legal(position)[verb, cell]


def test_the_same_generator_gives_the_same_random_action():
    position = TINY.start(1)
    assert act(RandomPlayer(), TINY, position, 5) == act(RandomPlayer(), TINY, position, 5)


def test_a_random_player_ending_always_ends(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    assert act(RandomPlayer(end_share=1.0), SMALL, position) == (Verb.END, at(4, 4))


def test_player_parameters_are_checked():
    with pytest.raises(ValueError, match="end_share"):
        RandomPlayer(end_share=1.5)
    with pytest.raises(ValueError, match="risk"):
        GreedyPlayer(risk=float("inf"))


def test_greedy_deploys_melee_first_nearest_the_enemy():
    position = TINY.start(1)
    verb, cell = act(GREEDY, TINY, position)
    assert verb == Verb.PLACE + KNIGHT
    space = position.state.space
    dist = steps(space, home_corners(space)[1])
    home = TINY.legal(position)[Verb.PLACE + KNIGHT]
    assert dist[cell] == dist[home].min()


def test_greedy_takes_the_kill_over_mere_damage(make_battle, at):
    pieces = [piece(4, 4, FOOT, 0, moved=1), piece(5, 4, ARCHER, 1, hp=1), piece(3, 4, CAPTAIN, 1)]
    assert act(GREEDY, SMALL, make_battle(pieces)) == (Verb.ATTACK, at(5, 4))


def test_greedy_moves_first_to_flank_when_that_attacks_better(make_battle, at):
    pieces = [piece(4, 4, FOOT, 0), piece(5, 4, FOOT, 1, facing=3)]  # facing the attacker
    verb, cell = act(GREEDY, SMALL, make_battle(pieces))
    assert verb == Verb.MOVE
    assert SMALL.odds(make_battle(pieces), at(4, 4), at(5, 4), at=cell).flank


def test_greedy_closes_in_on_a_distant_enemy(make_battle, at):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    verb, cell = act(GREEDY, SMALL, position)
    assert verb == Verb.MOVE
    assert steps(position.state.space, at(7, 4))[cell] == 4


def test_greedy_walks_around_a_wall_between_it_and_the_enemy(make_battle, at):
    # A wall across the board but for its two ends: 2 hops apart, a long walk around.
    walls = [(q, 4) for q in range(1, 8)]
    position = make_battle([piece(4, 3, FOOT, 0), piece(4, 5, FOOT, 1)], walls=walls)
    verb, cell = act(GREEDY, SMALL, position)
    assert verb == Verb.MOVE
    walk = steps(position.state.space, at(4, 5), blocked=~position.state.passable)
    assert walk[cell] == walk[at(4, 3)] - 2


def test_greedy_archers_defer_when_nothing_is_in_reach(make_battle, at):
    position = make_battle([piece(1, 1, ARCHER, 0, moved=2), piece(7, 7, FOOT, 1)])
    verb, cell = act(GREEDY, SMALL, position)
    assert verb == Verb.DEFER
    assert SMALL.legal(position)[Verb.DEFER, cell]


def test_greedy_turns_to_the_nearest_enemy_then_ends(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0, moved=2, facing=3), piece(8, 4, FOOT, 1)])
    assert act(GREEDY, SMALL, position) == (Verb.FACE + 0, at(4, 4))
    turned = SMALL.step(position, (Verb.FACE + 0, at(4, 4)))
    assert act(GREEDY, SMALL, turned) == (Verb.END, at(4, 4))


def test_greedy_fields_its_roster_and_by_default_a_mix():
    game = Tactics()
    picked = GreedyPlayer(roster=(2, 2, 2, 1, 0))
    record = play(game, [picked, GREEDY], game.start(1))
    placed = [e for e in record.events if isinstance(e, Placed)]
    for team, roster in ((0, (2, 2, 2, 1, 0)), (1, (3, 2, 1, 1, 1))):
        kinds = np.bincount([e.kind for e in placed if e.team == team], minlength=5)
        assert tuple(kinds) == roster


def test_greedy_past_a_roster_that_does_not_fill_the_team_places_any_legal_kind():
    final = play(TINY, [GreedyPlayer(roster=(1, 0, 0, 0)), GREEDY], TINY.start(1)).final
    assert final.over


def test_greedy_rejects_a_roster_that_is_not_a_tuple_of_counts(not_an_integer):
    with pytest.raises(TypeError, match="roster"):
        GreedyPlayer(roster=[1, 2])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="roster"):
        GreedyPlayer(roster=(1, not_an_integer))
    with pytest.raises(ValueError, match="roster"):
        GreedyPlayer(roster=(1, -1))


def test_greedy_walks_to_a_point_nobody_owns_when_nothing_else_differs(make_battle, at):
    pieces = [piece(4, 1, FOOT, 0), piece(0, 0, FOOT, 1, done=True)]
    position = make_battle(pieces, point=(4, 7))
    player = GreedyPlayer(advance=0.0, risk=0.0)
    verb, cell = act(player, SMALL, position)
    assert verb == Verb.MOVE
    assert steps(position.state.space, at(4, 7))[cell] == 4


def test_a_greedy_with_no_weight_on_the_point_ignores_it(make_battle, at):
    pieces = [piece(4, 1, FOOT, 0), piece(0, 0, FOOT, 1, done=True)]
    position = make_battle(pieces, point=(4, 7))
    cells = np.flatnonzero(SMALL.legal(position)[Verb.MOVE])
    scores = GreedyPlayer(advance=0.0, risk=0.0, point=0.0)._tile_scores(
        SMALL, position, at(4, 1), cells
    )
    assert (scores == 0).all()


@pytest.mark.parametrize(
    ("hold", "drawn"), [(Hold(0, 1), False), (Hold(1, 1), True), (Hold(), True)]
)
def test_greedy_wants_the_point_unless_its_side_owns_it(make_battle, at, hold, drawn):
    pieces = [piece(4, 1, FOOT, 0), piece(0, 0, FOOT, 1, done=True)]
    position = make_battle(pieces, point=(4, 7), hold=hold)
    near, far = at(4, 3), at(4, 0)
    player = GreedyPlayer(advance=0.0, risk=0.0)
    scores = player._tile_scores(SMALL, position, at(4, 1), np.array([near, far]))
    assert (scores[0] > scores[1]) == drawn


def test_greedy_goes_back_for_a_point_it_owns_when_an_enemy_stands_on_it(make_battle, at):
    pieces = [piece(4, 1, FOOT, 0), piece(4, 7, FOOT, 1, done=True)]
    position = make_battle(pieces, point=(4, 7), hold=Hold(0, 4))
    near, far = at(4, 3), at(4, 0)
    player = GreedyPlayer(advance=0.0, risk=0.0)
    scores = player._tile_scores(SMALL, position, at(4, 1), np.array([near, far]))
    assert scores[0] > scores[1]


def test_a_friend_on_the_point_lets_others_fight_and_a_unit_on_it_stays(make_battle, at):
    pieces = [piece(4, 6, FOOT, 0), piece(4, 7, FOOT, 0), piece(0, 0, FOOT, 1, done=True)]
    position = make_battle(pieces, point=(4, 7))
    player = GreedyPlayer(advance=0.0, risk=0.0)
    other = player._tile_scores(SMALL, position, at(4, 6), np.array([at(3, 6), at(4, 5)]))
    assert other[0] == other[1]
    standing = player._tile_scores(SMALL, position, at(4, 7), np.array([at(4, 7), at(4, 8)]))
    assert standing[0] > standing[1]


def test_a_greedy_archer_watches_the_point_when_an_enemy_could_walk_onto_it(make_battle, at):
    pieces = [piece(4, 2, ARCHER, 0, moved=2), piece(4, 6, FOOT, 1)]
    position = make_battle(pieces, point=(4, 4))
    assert act(GREEDY, SMALL, position) == (Verb.DEFER, at(4, 4))


def test_a_greedy_archer_watches_the_nearest_tile_when_no_enemy_is_near_the_point(make_battle, at):
    pieces = [piece(4, 2, ARCHER, 0, moved=2), piece(0, 8, FOOT, 1)]
    position = make_battle(pieces, point=(4, 4))
    verb, cell = act(GREEDY, SMALL, position)
    assert verb == Verb.DEFER
    assert cell != at(4, 4)


def test_a_greedy_archer_does_not_watch_a_point_someone_stands_on(make_battle, at):
    pieces = [
        piece(4, 2, ARCHER, 0, moved=2),
        piece(4, 4, FOOT, 0, done=True),
        piece(4, 6, FOOT, 1),
    ]
    verb, cell = act(GREEDY, SMALL, make_battle(pieces, point=(4, 4)))
    assert verb == Verb.DEFER
    assert cell != at(4, 4)


def test_a_sitter_never_moves_and_strikes_what_is_in_reach(make_battle, at):
    alone = make_battle([piece(4, 4, FOOT, 0), piece(8, 8, FOOT, 1)])
    assert act(SitterPlayer(), SMALL, alone) == (Verb.END, at(4, 4))
    near = make_battle([piece(4, 4, FOOT, 0), piece(4, 5, FOOT, 1)])
    assert act(SitterPlayer(), SMALL, near) == (Verb.ATTACK, at(4, 5))


def test_sitting_loses_to_a_greedy_that_walks_to_the_point():
    game = dataclasses.replace(TINY, max_rounds=12, capture_by=6)
    records = series(game, [("greedy", GREEDY), ("sitter", SitterPlayer())], 2, seed=20)
    assert summarize(game, records).tally.wins["greedy"] == 4


def test_two_sitters_end_at_the_deadline_with_team_one_losing():
    record = play(TINY, [SitterPlayer(), SitterPlayer()], TINY.start(2))
    assert record.final.winner == 0
    assert record.final.round == TINY.capture_by
    assert record.events[-1] == GameOver(0, "deadline")


def test_a_sitter_rejects_anything_but_a_greedy_player():
    with pytest.raises(TypeError, match="greedy"):
        SitterPlayer(greedy=RandomPlayer())  # ty: ignore[invalid-argument-type]


def test_greedy_games_end_with_short_turns():
    record = play(TINY, [GREEDY, GREEDY], TINY.start(4))
    assert record.final.over
    battle = record.final.ply - 2 * TINY.team_size
    turns = sum(1 for e in record.events if type(e).__name__ == "Ended")
    assert battle <= 6 * max(turns, 1)  # attack, move, defer, face, end at most


def test_greedy_beats_random():
    records = series(TINY, [("greedy", GREEDY), ("random", RandomPlayer())], 3, seed=10)
    assert summarize(TINY, records).tally.wins["greedy"] >= 5


@pytest.mark.parametrize(("risk", "moves"), [(10.0, False), (0.0, True)])
def test_greedy_weighs_the_risk_of_stepping_into_reach(make_battle, at, risk, moves):
    pieces = [piece(1, 4, FOOT, 0, attacked=True), piece(5, 4, FOOT, 1)]  # 4 apart: out of reach
    verb, _ = act(GreedyPlayer(risk=risk), SMALL, make_battle(pieces))
    assert (verb == Verb.MOVE) == moves


def test_greedy_archers_keep_to_their_range(make_battle, at):
    pieces = [piece(4, 4, ARCHER, 0, attacked=True), piece(6, 4, FOOT, 1)]
    position = make_battle(pieces)
    verb, cell = act(GreedyPlayer(risk=0.0), SMALL, position)
    assert verb == Verb.MOVE
    assert steps(position.state.space, at(6, 4))[cell] == 3


def test_greedy_deploys_knights_footsoldiers_the_captain_then_archers():
    position = TINY.start(1)
    kinds = []
    while position.deploying:
        team = TINY.player(position)
        verb, cell = act(GREEDY, TINY, position)
        if team == 0:
            kinds.append(verb - Verb.PLACE)
        position = TINY.step(position, (verb, cell))
    assert kinds == [KNIGHT, FOOT, FOOT, CAPTAIN, ARCHER]


def test_fielded_deploys_the_greedy_roster_then_only_the_other_player_plays():
    decided = []

    def play(game, player, position, legal, rng):
        decided.append(position.deploying)
        return RandomPlayer()(game, player, position, legal, rng)

    fielded = Fielded(play, GreedyPlayer(roster=(0, 2, 1, 1, 1)))
    record = series(WORKS, [("a", fielded), ("b", fielded)], 1)[0]
    assert decided
    assert not any(decided)
    units = next(p for p in replay(WORKS, record) if not p.deploying).state.units
    assert np.bincount(units.kind[units.kind >= 0], minlength=5).tolist() == [0, 4, 2, 2, 2]


def test_fielded_refuses_a_deployer_that_is_not_greedy():
    with pytest.raises(TypeError, match="deploy"):
        Fielded(RandomPlayer(), RandomPlayer())  # ty: ignore[invalid-argument-type]  (the check)
