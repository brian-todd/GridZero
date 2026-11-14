"""MuZero on Tactics: what it is shown, what it learns from, and a generation of training."""

import copy
import dataclasses

import numpy as np
import torch

from agents_demo import check_revealable
from examples.games.tactics import GreedyPlayer, RandomPlayer, Tactics
from game_demo import check_game
from gridzero.agents.muzero import MuZeroPlayer, trajectories
from gridzero.agents.muzero_net import NetModel, WorldNet
from gridzero.agents.muzero_train import Fit, MuZero, fit, train
from gridzero.game import play, series

# The default rules on a small board, with a short game.
GAME = dataclasses.replace(
    Tactics(), shape=(9, 9), home_depth=3, team_size=5, capture_by=4, max_rounds=6
)


def test_tactics_is_revealable():
    check_revealable(GAME, check_game(GAME, RandomPlayer(), range(1)))


def test_what_is_revealed_is_the_observation_of_the_position_itself():
    position = GAME.start(3)
    assert GAME.revealed == GAME.planes
    for team in (0, 1):
        np.testing.assert_array_equal(GAME.reveal(position, team), GAME.observe(position, team))


def test_greedy_games_make_trajectories_a_network_learns_from():
    torch.manual_seed(0)
    records = series(GAME, [("a", GreedyPlayer()), ("b", GreedyPlayer())], 2, 5)
    data = trajectories(GAME, records)
    assert len(data) == sum(len(r.actions) for r in records)
    assert data.obs.shape[1:] == (len(GAME.planes), GAME.space.n_cells)
    assert set(np.unique(data.outcome).tolist()) == {-1.0, 1.0}
    net = WorldNet(GAME, channels=8, blocks=1)
    settings = Fit(unroll=2, batch_size=16)
    first = fit(copy.deepcopy(net), data, settings, seconds=60, steps=1)
    trained = fit(net, data, settings, seconds=60, steps=60)
    assert trained.policy[0] < first.policy[0]
    assert trained.belief[0] < first.belief[0]


def test_a_muzero_player_plays_tactics_to_the_end():
    torch.manual_seed(0)
    player = MuZeroPlayer(NetModel(WorldNet(GAME, channels=8, blocks=1).eval()), simulations=4)
    record = play(GAME, [player, player], GAME.start(1))
    assert record.final.over


def test_a_generation_of_muzero_self_play_trains_on_tactics():
    torch.manual_seed(0)
    net = WorldNet(GAME, channels=8, blocks=1)
    settings = MuZero(
        games=2, simulations=4, considered=4, buffer=2000, steps=2, fit=Fit(unroll=2, batch_size=8)
    )
    (generation,) = train(net, GAME, settings, seconds=120, generations=1, seed=7)
    assert generation.decisions == sum(p.kept for p in generation.played) > 0
    assert np.isfinite(generation.losses.value).all()
