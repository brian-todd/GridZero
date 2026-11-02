import numpy as np
import pytest
import torch

from agents_demo import Patches
from gridzero.agents.muzero_net import NetModel, StochasticNetModel, WorldNet, world_model

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


def chance_net(seed=0, codes=4):
    torch.manual_seed(seed)
    net = WorldNet(GAME, channels=8, blocks=2, codes=codes)
    with torch.no_grad():
        for p in net.parameters():  # blocks start as the identity: wake them
            p.add_(0.1 * torch.randn_like(p))
    return net


def test_a_net_without_codes_keeps_its_parameters_and_has_no_afterstates():
    torch.manual_seed(0)
    old = WorldNet(GAME, channels=8, blocks=2)
    torch.manual_seed(0)
    zero = WorldNet(GAME, channels=8, blocks=2, codes=0)
    assert old.state_dict().keys() == zero.state_dict().keys()
    latent = old.represent(torch.rand(1, 3, N_CELLS))
    with pytest.raises(ValueError, match="no afterstates"):
        old.afterstate(latent, torch.tensor([0]))
    with pytest.raises(ValueError, match="no afterstates"):
        old.encode(torch.rand(1, 3, N_CELLS), torch.rand(1, 3, N_CELLS))


def test_a_net_with_codes_steps_by_afterstate_then_chance_not_by_dynamics():
    net = chance_net()
    assert not any(k.startswith("dynamics_") for k in net.state_dict())
    with pytest.raises(ValueError, match="afterstate, then chance"):
        net.dynamics(net.represent(torch.rand(1, 3, N_CELLS)), torch.tensor([0]))


def test_afterstates_and_their_chances_are_scaled_states_codes_scores_and_values():
    net = chance_net()
    latent = net.represent(torch.rand(1, 3, N_CELLS)).repeat(2, 1, 1)
    after = net.afterstate(latent, torch.tensor([0, N_CELLS + 5]))
    assert after.shape == latent.shape
    flat = after.flatten(1)
    torch.testing.assert_close(flat.min(dim=1).values, torch.zeros(2))
    torch.testing.assert_close(flat.max(dim=1).values, torch.ones(2))
    assert not torch.allclose(after[0], after[1])  # the action matters
    pred = net.predict_afterstate(after)
    assert pred.sigma.shape == (2, 4)
    assert ((pred.value >= -1) & (pred.value <= 1)).all()
    assert not torch.allclose(pred.sigma[0], pred.sigma[1])


def test_the_chance_step_gives_a_scaled_state_that_depends_on_the_code():
    net = chance_net()
    after = net.afterstate(net.represent(torch.rand(1, 3, N_CELLS)), torch.tensor([3]))
    codes = torch.eye(4)
    nxt = net.chance(after.repeat(4, 1, 1), codes)
    assert nxt.shape == (4, 8, N_CELLS)
    torch.testing.assert_close(nxt.flatten(1).max(dim=1).values, torch.ones(4))
    for i in range(1, 4):
        assert not torch.allclose(nxt[0], nxt[i])


def test_the_encoder_scores_each_code_from_both_observations():
    net = chance_net()
    obs, then, other = (torch.rand(1, 3, N_CELLS) for _ in range(3))
    scores = net.encode(obs, then)
    assert scores.shape == (1, 4)
    assert not torch.allclose(scores, net.encode(obs, other))
    assert not torch.allclose(scores, net.encode(other, then))


def test_the_net_with_codes_as_a_model_gives_its_own_afterstates_and_states():
    net = chance_net()
    model = StochasticNetModel(net)
    obs = [np.random.default_rng(i).random((3, N_CELLS), dtype=np.float32) for i in range(2)]
    latents, _, _ = model.initial(obs)
    actions = np.array([3, N_CELLS + 7])
    after, probs, values = model.afterstates(latents, actions)
    with torch.no_grad():
        want = net.afterstate(torch.stack(latents), torch.from_numpy(actions))
        pred = net.predict_afterstate(want)
    torch.testing.assert_close(torch.stack(after), want)
    np.testing.assert_allclose(probs, torch.softmax(pred.sigma.double(), 1).numpy(), rtol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(values, pred.value.double().numpy(), rtol=1e-6)
    codes = np.array([2, 0])
    states, logits, values = model.chance(after, codes)
    with torch.no_grad():
        want = net.chance(torch.stack(after), torch.eye(4)[torch.from_numpy(codes)])
        pred = net.predict(want)
    torch.testing.assert_close(torch.stack(states), want)
    np.testing.assert_allclose(logits, pred.logits.flatten(1).double().numpy(), rtol=1e-6)
    np.testing.assert_allclose(values, pred.value.double().numpy(), rtol=1e-6)


def test_each_net_is_the_model_of_its_kind():
    assert isinstance(world_model(small_net()), NetModel)
    assert isinstance(world_model(chance_net()), StochasticNetModel)
    with pytest.raises(ValueError, match="StochasticNetModel"):
        NetModel(chance_net())
    with pytest.raises(ValueError, match="NetModel"):
        StochasticNetModel(small_net())


@pytest.mark.parametrize("codes", [-1, 1.5, True])
def test_the_net_refuses_bad_codes(codes):
    with pytest.raises((TypeError, ValueError), match="codes"):
        WorldNet(GAME, channels=8, blocks=1, codes=codes)
