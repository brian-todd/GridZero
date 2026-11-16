import numpy as np
import pytest
import torch

from agents_demo import Patches
from gridzero.agents import AlphaZeroPlayer, self_play
from gridzero.agents.net import NetEvaluator, PolicyNet, outputs
from gridzero.agents.zero_train import Zero, generate, losses, train

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
    # The network's values differ in their last bits with the batch it runs them in, and the
    # search's value of a root averages them.
    np.testing.assert_allclose(experience.root, expected.root, rtol=1e-6, atol=1e-7)
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


def test_the_value_is_trained_toward_the_outcome_blended_with_the_search_s_value():
    torch.manual_seed(0)
    net = PolicyNet(GAME, channels=8, blocks=1)
    player = AlphaZeroPlayer(NetEvaluator(net), simulations=4, considered=4, noise=True)
    experience, _ = self_play(GAME, player, [1, 2], keep=1.0)
    device = torch.device("cpu")
    _, value, _ = outputs(net, experience.obs, experience.legal, device)
    for share in (0.0, 0.5, 1.0):
        _, got = losses(net, experience, device, root_value=share)
        wanted = (value - torch.from_numpy(experience.target(share))) ** 2
        torch.testing.assert_close(got, wanted)


def test_the_share_of_the_search_s_value_must_be_a_share():
    with pytest.raises(ValueError, match="root_value"):
        Zero(root_value=1.5)
    with pytest.raises(TypeError, match="root_value"):
        Zero(root_value=True)  # a bool is not a share
