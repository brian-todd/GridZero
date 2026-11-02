"""PPO (`agents.ppo`) on Tactics, against greedy; its tests on a generic game are in agents."""

import dataclasses

import numpy as np
import torch

from examples.games.tactics import GreedyPlayer
from gridzero.agents.net import PolicyNet
from gridzero.agents.ppo import PPO, self_play
from tactics_demo import TINY


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(TINY, channels=8, blocks=1)


def capped(played):
    """Games that ended with both teams on the board: at the round cap."""
    return sum(
        len(set(p.final.state.units.team[p.final.state.occupied].tolist())) == 2 for p in played
    )


def test_self_play_against_greedy_reports_and_trains():
    model = net()
    before = [p.clone() for p in model.parameters()]
    settings = PPO(games=4, opponent=GreedyPlayer(), against_opponent=0.5, epochs=2)
    first, second = self_play(model, TINY, settings, iterations=2, seed=3)
    assert (first.opponent_games, second.iteration) == (2, 2)
    assert all(1 <= p.final.round <= TINY.max_rounds for p in first.played)
    assert np.isfinite([first.policy_loss, first.value_loss, first.entropy, first.kl]).all()
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))


def test_games_at_the_round_cap_show_in_their_final_positions():
    game = dataclasses.replace(TINY, max_rounds=1)  # no game ends in one round
    (it,) = self_play(net(), game, PPO(games=2, epochs=1), iterations=1, seed=0)
    assert capped(it.played) == 2
    assert it.draws == 2  # TINY draws at the cap


def test_greedy_beats_an_untrained_network_outright():
    game = dataclasses.replace(TINY, max_rounds=30)
    settings = PPO(games=2, opponent=GreedyPlayer(), against_opponent=1.0, epochs=1)
    (it,) = self_play(net(), game, settings, iterations=1, seed=0)
    assert it.opponent_losses == 2
    assert capped(it.played) == 0
    assert it.draws == 0
