import numpy as np
import pytest

from examples.games.tactics import (
    GreedyPlayer,
    RandomPlayer,
    Verb,
    summarize,
)
from examples.games.tactics.board import home_corners
from gridzero.core import steps
from gridzero.game import play, series
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, SMALL, TINY, piece, play_random

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
