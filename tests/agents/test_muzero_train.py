import copy
import time

import numpy as np
import pytest
import torch

from agents_demo import Lowest, Patches
from gridzero.agents import UniformPlayer
from gridzero.agents.muzero import MuZeroPlayer, Trajectories, self_play, trajectories
from gridzero.agents.muzero_net import NetModel, WorldNet
from gridzero.agents.muzero_train import (
    Fit,
    MuZero,
    Unrolled,
    _halved,
    _policies,
    fit,
    losses,
    train,
    weighted,
)
from gridzero.game import series

GAME = Patches()
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def data():
    return trajectories(GAME, series(GAME, [("u", UniformPlayer()), ("l", Lowest())], 6))


def small_net(seed=0):
    torch.manual_seed(seed)
    return WorldNet(GAME, channels=8, blocks=1)


def test_past_a_trajectorys_end_only_the_value_learns(data):
    ends = data.bounds[1:4] - 1
    rows = np.concatenate([ends - 1, ends])  # one decision left, then none
    policy, value, belief, live = losses(small_net(), data, rows, 2, DEVICE)
    assert policy.shape == value.shape == belief.shape == live.shape == (3, 6)
    want = np.array([[True] * 6, [True] * 3 + [False] * 3, [False] * 6])
    np.testing.assert_array_equal(live.numpy(), want)
    for x in (policy, belief):
        assert (x[~live] == 0).all()
        assert (x[live] > 0).all()
    assert (value > 0).all()  # toward the outcome, which stays


def test_the_policy_targets_of_rows_follow_one_another(data):
    rows = np.array([4, 0, 4])
    got = _policies(data, rows)
    want = np.concatenate([data.policy[data.start[r] : data.start[r + 1]] for r in rows])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("unroll", [0, 2])
def test_fitting_lowers_the_losses(data, unroll):
    net = small_net()
    rows = np.arange(len(data))
    before = [x.sum().item() for x in losses(net, data, rows, unroll, DEVICE)[:3]]
    fit(net, data, Fit(unroll=unroll, batch_size=32, lr=3e-3), seconds=60, steps=150)
    after = [x.sum().item() for x in losses(net, data, rows, unroll, DEVICE)[:3]]
    assert all(a < b for a, b in zip(after, before, strict=True)), (before, after)


def test_fitting_stops_at_its_step_limit_and_reports_each_unrolled_step(data):
    result = fit(small_net(), data, Fit(unroll=3, batch_size=8), seconds=60, steps=2)
    assert result.policy.shape == result.value.shape == result.belief.shape == (4,)
    assert np.isfinite(result.policy).all()


def test_fitting_stops_at_its_time_box(data):
    begin = time.perf_counter()
    fit(small_net(), data, Fit(batch_size=8), seconds=0.3)
    assert time.perf_counter() - begin < 2.0


def test_the_same_seed_fits_the_same_network(data):
    nets = [small_net(), small_net()]
    for net in nets:
        fit(net, data, Fit(unroll=1, batch_size=16), seconds=60, steps=5, seed=4)
    for x, y in zip(nets[0].state_dict().values(), nets[1].state_dict().values(), strict=True):
        np.testing.assert_array_equal(x.numpy(), y.numpy())


@pytest.mark.parametrize(
    "settings",
    [
        {"unroll": -1},
        {"batch_size": 0},
        {"lr": 0.0},
        {"weight_decay": -1.0},
        {"value_weight": -0.5},
        {"belief_weight": float("nan")},
        {"unroll": 1.5},
    ],
)
def test_fit_settings_refuse_bad_values(settings):
    with pytest.raises((TypeError, ValueError)):
        Fit(**settings)


def test_fit_refuses_a_time_box_or_step_limit_out_of_range(data):
    with pytest.raises(ValueError, match="seconds"):
        fit(small_net(), data, seconds=0)
    with pytest.raises(ValueError, match="steps"):
        fit(small_net(), data, seconds=1, steps=0)


def test_training_plays_learns_and_keeps_the_latest_trajectories():
    torch.manual_seed(0)
    net = WorldNet(GAME, channels=8, blocks=1)
    settings = MuZero(
        games=4, simulations=4, considered=4, buffer=40, steps=3, fit=Fit(unroll=1, batch_size=8)
    )
    generations = list(train(net, GAME, settings, seconds=120, generations=3, seed=7))
    assert [g.generation for g in generations] == [1, 2, 3]
    assert [p.seed for p in generations[1].played] == [11, 12, 13, 14]
    for g in generations:
        assert g.decisions == sum(p.kept for p in g.played)
        assert g.buffer <= 40
        assert np.isfinite(g.losses.policy).all()
    assert generations[-1].buffer > 40 - 12  # whole trajectories of at most 6 decisions


def test_training_stops_at_its_time_box():
    net = WorldNet(GAME, channels=8, blocks=1)
    settings = MuZero(games=2, simulations=2, considered=2, steps=1, fit=Fit(batch_size=4))
    begin = time.perf_counter()
    generations = list(train(net, GAME, settings, seconds=1.0))
    assert len(generations) >= 1
    assert time.perf_counter() - begin < 4.0


@pytest.mark.parametrize("settings", [{"games": 0}, {"buffer": 0}, {"steps": 1.5}, {"fit": 3}])
def test_loop_settings_refuse_bad_values(settings):
    with pytest.raises((TypeError, ValueError)):
        MuZero(**settings)


def test_past_a_trajectorys_end_the_dynamics_take_its_last_action_again(data):
    """A wrong action into the end was once fed: the decision before the row drawn."""
    torch.manual_seed(1)
    net = WorldNet(GAME, channels=8, blocks=1)
    with torch.no_grad():
        for p in net.parameters():  # blocks start as the identity: wake them
            p.add_(0.1 * torch.randn_like(p))
    row = int(data.bounds[2] - 1)  # the last decision of a trajectory
    _, value, _, live = losses(net, data, np.array([row]), 2, DEVICE)
    assert not live[1:, 0].any()
    with torch.no_grad():
        obs = torch.from_numpy(data.obs[[row]]).float() / 255
        last = torch.from_numpy(data.action[[row]])
        latent = net.dynamics(net.represent(obs), last)
        want_1 = (net.predict(latent).value - float(data.outcome[row])) ** 2
        latent = net.dynamics(latent, last)
        want_2 = (net.predict(latent).value - float(data.outcome[row])) ** 2
    torch.testing.assert_close(value[1:, 0].detach(), torch.cat([want_1, want_2]))


def test_an_optimizer_passed_in_keeps_its_statistics_from_call_to_call(data):
    net = small_net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(2):
        fit(net, data, Fit(batch_size=8), seconds=60, steps=3, optimizer=optimizer)
    first = next(iter(net.parameters()))
    assert int(optimizer.state[first]["step"]) == 6


def test_the_time_box_counts_the_callers_work_between_generations():
    net = WorldNet(GAME, channels=8, blocks=1)
    settings = MuZero(games=1, simulations=2, considered=2, steps=1, fit=Fit(batch_size=4))
    begin = time.perf_counter()
    count = 0
    for _ in train(net, GAME, settings, seconds=3.0):
        count += 1
        time.sleep(1.0)  # measuring between generations, as muzero_scratch does
    assert count == 2  # a third would have started at about 2.2 s and ended past 3 s
    assert time.perf_counter() - begin < 3.5


def test_the_gradient_through_each_step_of_the_dynamics_is_halved():
    x = torch.ones(3, requires_grad=True)
    y = _halved(x)
    torch.testing.assert_close(y, torch.ones(3))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full((3,), 0.5))


def test_unrolled_steps_weigh_one_over_the_unroll_of_the_first():
    one = torch.ones(4, 2)
    settings = Fit(unroll=3, value_weight=2.0, belief_weight=0.5)
    total = weighted(Unrolled(one, one, one, one.bool()), settings)
    each = 1 + 2 + 0.5  # a decision a step
    assert total.item() == pytest.approx(each * (1 + 3 * (1 / 3)))
    alone = weighted(Unrolled(one[:1], one[:1], one[:1], one[:1].bool()), Fit())
    assert alone.item() == pytest.approx(1 + 1 + 1)


def test_each_unrolled_step_is_scored_on_the_decision_it_reaches(data):
    net = small_net()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    row = int(data.bounds[1] - 3)  # two decisions follow it
    policy, value, belief, _ = losses(net, data, np.array([row]), 1, DEVICE)
    with torch.no_grad():
        obs = torch.from_numpy(data.obs[[row]]).float() / 255
        latent = net.dynamics(net.represent(obs), torch.from_numpy(data.action[[row]]))
        pred = net.predict(latent)
        revealed = torch.from_numpy(data.revealed[[row + 1]]).float() / 255
        want = torch.nn.functional.binary_cross_entropy_with_logits(pred.belief, revealed)
        n_bits = GAME.n_verbs * GAME.space.n_cells
        legal = np.unpackbits(data.legal[row + 1])[:n_bits].astype(bool)
        logp = torch.log_softmax(pred.logits.flatten(), dim=0)[torch.from_numpy(legal)]
        target = torch.from_numpy(data.policy[data.start[row + 1] : data.start[row + 2]])
    assert belief[1, 0].item() == pytest.approx(want.item(), rel=1e-5)
    outcome = float(data.outcome[row + 1])
    assert value[1, 0].item() == pytest.approx((pred.value.item() - outcome) ** 2, rel=1e-5)
    assert policy[1, 0].item() == pytest.approx(-(target * logp).sum().item(), rel=1e-5)


def test_fit_reports_the_policy_and_belief_over_live_steps_and_the_value_over_all(data):
    settings = Fit(unroll=2, batch_size=16)
    net = small_net()
    rows = np.random.default_rng(5).integers(len(data), size=16)
    policy, value, belief, live = (x.detach() for x in losses(net, data, rows, 2, DEVICE))
    reported = fit(net, data, settings, seconds=60, steps=1, seed=5)
    counts = live.sum(dim=1).double().numpy()
    np.testing.assert_allclose(
        reported.policy, policy.sum(dim=1).double().numpy() / counts, rtol=1e-5
    )
    np.testing.assert_allclose(
        reported.belief, belief.sum(dim=1).double().numpy() / counts, rtol=1e-5
    )
    np.testing.assert_allclose(reported.value, value.sum(dim=1).double().numpy() / 16, rtol=1e-5)


def test_training_is_self_play_then_fitting_with_one_optimizer_on_the_latest_trajectories():
    """Two generations of `train`, replayed by hand, give the same weights bit for bit."""
    settings = MuZero(
        games=3, simulations=4, considered=4, buffer=20, steps=4, fit=Fit(unroll=1, batch_size=8)
    )
    torch.manual_seed(3)
    net = WorldNet(GAME, channels=8, blocks=1)
    by_hand = copy.deepcopy(net)
    generations = list(train(net, GAME, settings, seconds=120, generations=2, seed=10))
    optimizer = torch.optim.AdamW(by_hand.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = None
    for g in range(2):
        player = MuZeroPlayer(NetModel(by_hand.eval()), 4, 4, noise=True)
        data, _ = self_play(GAME, player, range(10 + 3 * g, 13 + 3 * g))
        buffer = data if buffer is None else Trajectories.join([buffer, data])
        sizes = np.diff(buffer.bounds)
        first = next(t for t in range(len(sizes) + 1) if sizes[t:].sum() <= 20)  # most that fit
        buffer = buffer.since(first)
        assert generations[g].buffer == len(buffer)
        fit(by_hand, buffer, settings.fit, seconds=120, steps=4, seed=10 + g, optimizer=optimizer)
    for x, y in zip(net.state_dict().values(), by_hand.state_dict().values(), strict=True):
        np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_the_policy_learns_to_put_nothing_on_illegal_actions(data):
    """It is trained over every action, so that a search below the root can trust it."""
    net = small_net()
    n_bits = GAME.n_verbs * GAME.space.n_cells
    rows = np.arange(len(data))
    legal = torch.from_numpy(np.unpackbits(data.legal, axis=1)[:, :n_bits].astype(bool))

    def illegal_mass():
        with torch.no_grad():
            obs = torch.from_numpy(data.obs).float() / 255
            p = torch.softmax(net(obs).logits.flatten(1), dim=1)
        return float(p[~legal].sum() / len(rows))

    before = illegal_mass()
    fit(net, data, Fit(batch_size=32, lr=3e-3), seconds=60, steps=150)
    assert illegal_mass() < before / 3
