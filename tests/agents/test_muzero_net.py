import numpy as np
import pytest
import torch

from agents_demo import Patches
from gridzero.agents.muzero_net import NetModel, WorldNet

GAME = Patches()
N_CELLS = GAME.space.n_cells


def small_net(seed=0):
    torch.manual_seed(seed)
    return WorldNet(GAME, channels=8, blocks=2)


def test_a_latent_state_has_a_vector_per_cell_scaled_into_zero_to_one():
    net = small_net()
    latent = net.represent(torch.rand(5, 3, N_CELLS))
    assert latent.shape == (5, 8, N_CELLS)
    flat = latent.flatten(1)
    torch.testing.assert_close(flat.min(dim=1).values, torch.zeros(5))
    torch.testing.assert_close(flat.max(dim=1).values, torch.ones(5))


def test_predictions_score_every_action_value_in_minus_one_to_one_and_reveal_every_plane():
    pred = small_net()(torch.rand(4, 3, N_CELLS))
    assert pred.logits.shape == (4, GAME.n_verbs, N_CELLS)
    assert pred.value.shape == (4,)
    assert ((pred.value >= -1) & (pred.value <= 1)).all()
    assert pred.belief.shape == (4, len(GAME.revealed), N_CELLS)


def test_the_dynamics_give_a_scaled_latent_state_that_depends_on_the_action():
    net = small_net()
    with torch.no_grad():
        for p in net.parameters():  # blocks start as the identity: wake them
            p.add_(0.1 * torch.randn_like(p))
    latent = net.represent(torch.rand(1, 3, N_CELLS)).repeat(3, 1, 1)
    action = torch.tensor([0, 5, N_CELLS + 5])  # cell 0 placed; cell 5 placed; cell 5 gambled
    after = net.dynamics(latent, action)
    assert after.shape == latent.shape
    flat = after.flatten(1)
    torch.testing.assert_close(flat.min(dim=1).values, torch.zeros(3))
    torch.testing.assert_close(flat.max(dim=1).values, torch.ones(3))
    assert not torch.allclose(after[0], after[1])  # another cell
    assert not torch.allclose(after[1], after[2])  # another verb on the same cell


def test_a_state_of_one_value_scales_to_zero_rather_than_dividing_by_zero():
    net = small_net()
    with torch.no_grad():
        for p in net.stem.parameters():
            p.zero_()
    latent = net.represent(torch.rand(2, 3, N_CELLS))
    assert torch.isfinite(latent).all()
    assert (latent == 0).all()


@pytest.mark.parametrize(("channels", "blocks"), [(0, 1), (1, -1), (1.5, 1)])
def test_the_network_refuses_bad_sizes(channels, blocks):
    with pytest.raises((TypeError, ValueError)):
        WorldNet(GAME, channels=channels, blocks=blocks)


def test_the_same_seed_makes_the_same_network():
    a, b = small_net(3), small_net(3)
    for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_an_action_is_a_one_on_its_cell_for_its_verb_and_a_plane_of_ones_for_its_verb():
    planes = small_net().action_planes(torch.tensor([5, N_CELLS + 0]))
    assert planes.shape == (2, 2 * GAME.n_verbs, N_CELLS)
    want = torch.zeros(2, 4, N_CELLS)
    want[0, 0, 5] = 1.0
    want[0, 2] = 1.0  # verb 0 everywhere
    want[1, 1, 0] = 1.0
    want[1, 3] = 1.0  # verb 1 everywhere
    torch.testing.assert_close(planes, want)


def test_the_value_stays_in_minus_one_to_one_however_large_its_inputs():
    net = small_net()
    with torch.no_grad():
        net.value_out.weight.mul_(1000.0)
        net.value_out.bias.fill_(50.0)
    value = net(torch.rand(6, 3, N_CELLS)).value
    assert ((value >= -1) & (value <= 1)).all()
    assert (value.abs() > 0.99).any()


def test_the_belief_reads_its_own_head():
    net = small_net()
    obs = torch.rand(2, 3, N_CELLS)
    before = net(obs)
    with torch.no_grad():
        net.belief_conv.own.weight.add_(1.0)
    after = net(obs)
    assert not torch.allclose(before.belief, after.belief)
    torch.testing.assert_close(before.logits, after.logits)


def test_the_net_as_a_model_gives_its_own_states_scores_and_values():
    net = small_net()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    model = NetModel(net)
    obs = [np.random.default_rng(i).random((3, N_CELLS), dtype=np.float32) for i in range(2)]
    latents, logits, values = model.initial(obs)
    with torch.no_grad():
        latent = net.represent(torch.from_numpy(np.stack(obs)))
        pred = net.predict(latent)
    np.testing.assert_allclose(logits, pred.logits.flatten(1).double().numpy(), rtol=1e-6)
    np.testing.assert_allclose(values, pred.value.double().numpy(), rtol=1e-6)
    actions = np.array([3, N_CELLS + 7])
    after, logits, values = model.recurrent(latents, actions)
    with torch.no_grad():
        want = net.dynamics(latent, torch.from_numpy(actions))
        pred = net.predict(want)
    torch.testing.assert_close(torch.stack(after), want)
    np.testing.assert_allclose(values, pred.value.double().numpy(), rtol=1e-6)
