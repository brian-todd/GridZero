"""Tests of the Voronoi players: greedy's moves against exact values, and against uniform."""

import numpy as np
import pytest

from examples.games.voronoi import PLACE, Greedy, Voronoi
from gridzero.agents import UniformPlayer
from gridzero.agents.exact import optimal, solve
from gridzero.game import ladder, play, replay


@pytest.mark.parametrize("sea", [0.0, 0.25])
def test_greedys_last_stone_is_a_perfect_move(sea):
    # With two players, the last stone's best lead is its best result.
    game = Voronoi(frequency=1, stones=2, sea=sea)
    for seed in range(3):
        start = game.start(seed)
        values = solve(game, start)
        for game_seed in range(4):
            record = play(game, [UniformPlayer()] * 2, start.__class__(start.state, game_seed))
            last = list(replay(game, record))[-2]
            rng = np.random.default_rng(game_seed)
            verb, cell = Greedy()(game, game.player(last), last, game.legal(last), rng)
            assert optimal(game, last, values)[verb, cell]


def test_greedy_places_a_legal_stone_and_its_ties_follow_its_generator():
    game = Voronoi(frequency=2, players=3)
    position = game.start(0)
    legal = game.legal(position)
    picks = {
        Greedy()(game, game.player(position), position, legal, np.random.default_rng(s))
        for s in range(8)
    }
    assert all(verb == PLACE and legal[verb, cell] for verb, cell in picks)
    assert len(picks) > 1  # on an empty board, many cells tie


@pytest.mark.parametrize("players", [2, 3])
def test_greedy_beats_uniform(players):
    game = Voronoi(frequency=2, players=players)
    won, lost, _ = ladder(game, Greedy(), {"uniform": UniformPlayer()}, 4)["uniform"]
    assert won > 3 * lost
