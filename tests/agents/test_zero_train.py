import numpy as np
import torch

from agents_demo import Patches
from gridzero.agents import AlphaZeroPlayer, self_play
from gridzero.agents.net import NetEvaluator, PolicyNet
from gridzero.agents.zero_train import Zero, generate, train

GAME = Patches()
FIELDS = ("obs", "legal", "start", "outcome", "game", "ply")


QUICK = Zero(games=2, actors=1, simulations=4, considered=4, keep=1.0, steps=3, batch_size=8)
"""Settings for a generation that takes a few seconds."""


def test_actors_play_a_games_self_play_as_one_process_would():
    torch.manual_seed(0)
    net = PolicyNet(GAME, channels=8, blocks=1)
    player = AlphaZeroPlayer(NetEvaluator(net), simulations=4, considered=4, noise=True)
    experience, played, costs = generate(GAME, player, [3, 4, 5], actors=2, keep=0.5)
    expected, expected_played = self_play(GAME, player, [3, 4, 5], keep=0.5)
    assert [p[:-1] for p in played] == [p[:-1] for p in expected_played]
    assert [p.final.key() for p in played] == [p.final.key() for p in expected_played]
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(experience, name), getattr(expected, name))
    assert costs.positions == sum(a.positions for a in costs.actors) > 0
    assert experience.obs.shape[1:] == (len(GAME.planes), GAME.space.n_cells)


def test_a_generation_on_squares_plays_keeps_and_trains():
    torch.manual_seed(0)
    net = PolicyNet(GAME, channels=8, blocks=1)
    before = [p.detach().clone() for p in net.parameters()]
    [generation] = list(train(net, GAME, QUICK, generations=1, seed=2))
    assert generation.games == len(generation.played) == 2
    assert generation.kept == generation.searched == generation.buffer > 0
    assert generation.plies == np.mean([p.plies for p in generation.played])
    assert all(p.final.over for p in generation.played)
    assert np.isfinite([generation.policy_loss, generation.value_loss]).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))
