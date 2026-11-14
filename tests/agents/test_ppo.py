import dataclasses
import time

import numpy as np
import pytest
import torch

from agents_demo import Patches
from gridzero.agents import UniformPlayer
from gridzero.agents.net import PolicyNet
from gridzero.agents.net import outputs as net_outputs
from gridzero.agents.ppo import (
    PPO,
    _jobs,
    _learn,
    _play,
    gae,
    self_play,
)

GAME = Patches()
UNIFORM = UniformPlayer()


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(GAME, channels=8, blocks=1)


def play_one(model, seed, learner=None, gamma=1.0, lam=0.95):
    return _play(GAME, model, UNIFORM, seed, learner, gamma, lam)


def taken_logp(model, batch):
    """The log-probability of each decision's action now, and the values."""
    e = batch.experience
    with torch.no_grad():
        scores, value, _ = net_outputs(model, e.obs, e.legal, torch.device("cpu"))
    rows = np.arange(len(e))
    return torch.log_softmax(scores, dim=1)[rows, batch.action].numpy(), value.numpy()


def test_gae_waits_for_the_result_at_lambda_one():
    values = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    np.testing.assert_allclose(gae(values, 1.0, 1.0, 1.0), 1.0 - values, rtol=1e-6)


def test_gae_bootstraps_from_the_next_value_at_lambda_zero():
    values = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    expected = [-0.2 * 0.9 - 0.1, 0.5 * 0.9 + 0.2, -1.0 - 0.5]
    np.testing.assert_allclose(gae(values, -1.0, 0.9, 0.0), expected, rtol=1e-6)


def test_gae_between_by_hand():
    values = np.array([0.0, 0.5], dtype=np.float32)
    # delta = [0.8 * 0.5 - 0, 1 - 0.5]; A[1] = 0.5, A[0] = 0.4 + 0.8 * 0.5 * 0.5
    np.testing.assert_allclose(gae(values, 1.0, 0.8, 0.5), [0.6, 0.5], rtol=1e-6)


def test_a_played_game_keeps_each_decision_as_the_network_saw_it():
    model = net()
    batch, result = play_one(model, 3)
    e = batch.experience
    assert len(e) == result.played.plies  # every ply had a choice: place or gamble
    assert result.learner is None
    assert result.played.final.over
    assert result.played.plies == result.played.final.ply
    assert result.played.kept == result.played.searched == len(e)
    with torch.no_grad():
        logits, value = model(torch.from_numpy(e.obs.astype(np.float32) / 255))
    np.testing.assert_allclose(value.numpy(), batch.value, atol=0.02)  # obs stored as bytes
    np.testing.assert_allclose(batch.target, batch.advantage + batch.value, rtol=1e-6)
    masks = np.unpackbits(e.legal, axis=1)[:, : logits[0].numel()].astype(bool)
    assert masks[np.arange(len(e)), batch.action].all()
    assert (masks.sum(axis=1) > 1).all()  # forced moves teach nothing
    assert (batch.logp <= 0).all()
    np.testing.assert_array_equal(e.ply, np.arange(len(e)))


def test_each_decisions_policy_is_all_on_its_action():
    batch, _ = play_one(net(), 3)
    e = batch.experience
    legal = np.unpackbits(e.legal, axis=1).astype(bool)
    flat = np.nonzero(legal)[1]
    np.testing.assert_array_equal(np.diff(e.start), legal.sum(axis=1))
    np.testing.assert_array_equal(np.add.reduceat(e.policy, e.start[:-1]), 1.0)
    np.testing.assert_array_equal(flat[np.flatnonzero(e.policy == 1)], batch.action)


def test_each_players_advantages_follow_its_own_decisions_and_result():
    winners = set()
    for seed, learner in [(s, k) for s in range(4, 8) for k in (None, 0, 1)]:
        batch, result = play_one(net(1), seed, learner, lam=1.0)
        outcome = batch.experience.outcome
        winners.add(result.played.winner)
        if result.played.winner is None:
            np.testing.assert_array_equal(outcome, 0.0)
        # At lambda 1 the advantage is the result less the value at every decision.
        np.testing.assert_allclose(batch.advantage, outcome - batch.value, atol=1e-5)
    assert {0, 1} <= winners


def test_against_the_opponent_only_the_networks_side_is_kept():
    for learner in (0, 1):
        batch, result = play_one(net(), 5, learner)
        assert result.learner == learner
        # One player's decisions only: they share its result, on its own plies.
        winner = result.played.winner
        expected = 0.0 if winner is None else (1.0 if winner == learner else -1.0)
        np.testing.assert_array_equal(batch.experience.outcome, expected)
        np.testing.assert_array_equal(batch.experience.ply % 2, learner)
        assert (
            len(batch.experience) == result.played.kept == (result.played.plies + 1 - learner) // 2
        )


def test_without_an_opponent_every_game_is_self_play():
    weights = net().state_dict()
    jobs = _jobs(GAME, net(), weights, PPO(games=4, against_opponent=1.0), first=0, workers=1)
    assert [learner for job in jobs for _, learner, *_ in job.games] == [None] * 4


def test_the_first_step_starts_from_the_policy_that_played():
    model = net()
    batch, _ = play_one(model, 6)
    settings = PPO(epochs=1, batch_size=len(batch.experience), lr=1e-12)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    stats, passes = _learn(
        model, optimizer, batch, settings, torch.device("cpu"), np.random.default_rng(0)
    )
    assert passes == 1
    assert abs(stats[3]) < 1e-4  # KL from the policy that played
    assert stats[4] == 0.0  # nothing clipped


def test_a_step_makes_better_than_average_actions_likelier():
    model = net()
    batch, _ = play_one(model, 7)
    two = batch.take(np.array([0, 1]))
    two = dataclasses.replace(two, advantage=np.array([1.0, -1.0], dtype=np.float32))
    settings = PPO(epochs=1, lr=1e-3, value_weight=0.0, entropy_weight=0.0, target_kl=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    _learn(model, optimizer, two, settings, torch.device("cpu"), np.random.default_rng(0))
    logp, _ = taken_logp(model, two)
    assert logp[0] > two.logp[0]
    assert logp[1] < two.logp[1]


def test_self_play_reports_each_iteration_and_trains_the_network():
    model = net()
    before = [p.clone() for p in model.parameters()]
    settings = PPO(games=4, opponent=UNIFORM, against_opponent=0.5, epochs=2, batch_size=128)
    run = self_play(model, GAME, settings, iterations=2, seed=3)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
    first, second = list(run)
    assert (first.iteration, second.iteration) == (1, 2)
    assert first.opponent_games == 2
    assert 0 <= first.opponent_wins + first.opponent_losses <= 2
    assert first.decisions == sum(p.kept for p in first.played)
    assert [p.kept < p.plies for p in first.played] == [True, True, False, False]
    assert [p.seed for p in first.played] == [3, 4, 5, 6]
    assert first.draws == sum(p.winner is None for p in first.played)
    assert 1 <= first.passes <= 2
    assert np.isfinite([first.policy_loss, first.value_loss, first.entropy, first.kl]).all()
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))


def test_playing_in_this_process_leaves_torchs_threads_as_they_were():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)  # not 1, which play sets while it runs
    try:
        list(self_play(net(), GAME, PPO(games=2, epochs=1), iterations=1, seed=0, workers=1))
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)


def test_the_same_seed_trains_the_same_network():
    settings = PPO(games=2, epochs=1, batch_size=64)
    runs = []
    for _ in range(2):
        model = net()
        list(self_play(model, GAME, settings, iterations=2, seed=1))
        runs.append([p.detach().clone() for p in model.parameters()])
    for p, q in zip(*runs, strict=True):
        assert torch.equal(p, q)


@pytest.mark.slow
def test_self_play_in_processes_learns_the_same():
    settings = PPO(games=4, opponent=UNIFORM, against_opponent=0.5, epochs=1, batch_size=64)
    results = []
    for workers in (1, 2):
        model = net()
        (it,) = self_play(model, GAME, settings, iterations=1, seed=2, workers=workers)
        it = dataclasses.replace(it, play_seconds=0, learn_seconds=0)
        keys = [(p[:-1], p.final.key()) for p in it.played]
        results.append((dataclasses.replace(it, played=()), keys, model))
    assert results[0][:2] == results[1][:2]
    for p, q in zip(results[0][2].parameters(), results[1][2].parameters(), strict=True):
        assert torch.equal(p, q)


def test_observations_kept_are_the_teams_own():
    batch, _ = play_one(net(), 8, 0)
    first = GAME.observe(GAME.start(8), 0)
    np.testing.assert_allclose(batch.experience.obs[0] / 255, first, atol=0.5 / 255)
    second, _ = play_one(net(), 8, 1)
    assert second.experience.ply[0] == 1  # player 1 decides on odd plies
    assert second.experience.obs[0][0].sum() == 0  # none of its own stones yet


def test_ppo_settings_are_checked(not_an_integer):
    with pytest.raises(TypeError, match="games"):
        PPO(games=not_an_integer)
    with pytest.raises(ValueError, match="games"):
        PPO(games=0)
    with pytest.raises(ValueError, match="against_opponent"):
        PPO(against_opponent=1.5)
    with pytest.raises(TypeError, match="opponent"):
        PPO(opponent=3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="lam"):
        PPO(lam=-0.1)
    with pytest.raises(ValueError, match="clip"):
        PPO(clip=0.0)
    with pytest.raises(ValueError, match="lr"):
        PPO(lr=0.0)
    with pytest.raises(ValueError, match="entropy_weight"):
        PPO(entropy_weight=-1.0)
    with pytest.raises(ValueError, match="target_kl"):
        PPO(target_kl=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        PPO(batch_size=0)


def test_self_play_checks_its_counts(not_an_integer):
    with pytest.raises(TypeError, match="iterations"):
        self_play(net(), GAME, iterations=not_an_integer)
    with pytest.raises(ValueError, match="workers"):
        self_play(net(), GAME, workers=0)
    with pytest.raises(ValueError, match="seed"):
        self_play(net(), GAME, seed=-1)


def test_a_time_box_with_no_room_still_runs_the_first_iteration_and_no_more():
    settings = PPO(games=2, epochs=1, batch_size=128)
    run = self_play(net(), GAME, settings, iterations=50, seed=3, seconds=0.0)
    assert [i.iteration for i in run] == [1]


def test_a_roomy_time_box_runs_every_iteration():
    settings = PPO(games=2, epochs=1, batch_size=128)
    run = self_play(net(), GAME, settings, iterations=3, seed=3, seconds=3600.0)
    assert [i.iteration for i in run] == [1, 2, 3]


def test_no_iteration_starts_that_would_end_past_the_time_box_at_the_last_ones_pace():
    settings = PPO(games=2, epochs=1, batch_size=128)
    done = []
    for i in self_play(net(), GAME, settings, iterations=50, seed=3, seconds=1.0):
        done.append(i.iteration)
        time.sleep(0.6)  # the caller's work counts: 0.6 s are left, and the last took more
    assert done == [1]


def test_self_play_checks_its_time_box():
    with pytest.raises(ValueError, match="seconds"):
        self_play(net(), GAME, seconds=-1.0)
    with pytest.raises(TypeError, match="seconds"):
        self_play(net(), GAME, seconds="60")  # ty: ignore[invalid-argument-type]  # a string on purpose


def two_decisions(seed=7):
    model = net()
    batch, _ = play_one(model, seed)
    return model, batch.take(np.array([0, 1]))


def step(model, batch, **settings):
    settings = dataclasses.replace(PPO(epochs=1, lr=1e-3, target_kl=None), **settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    return _learn(model, optimizer, batch, settings, torch.device("cpu"), np.random.default_rng(0))


def test_ratios_past_the_clip_teach_nothing():
    model, two = two_decisions()
    now, _ = taken_logp(model, two)
    # Already 2.7 times likelier where better, 2.7 times less likely where worse.
    two = dataclasses.replace(
        two,
        logp=(now + np.array([-1.0, 1.0])).astype(np.float32),
        advantage=np.array([1.0, -1.0], dtype=np.float32),
    )
    before = [p.clone() for p in model.parameters()]
    stats, _ = step(model, two, value_weight=0.0, entropy_weight=0.0)
    assert stats[4] == 1.0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))


def test_advantages_count_only_relative_to_each_other():
    runs = []
    for shift in (0.0, 5.0):
        model, two = two_decisions()
        two = dataclasses.replace(two, advantage=np.array([1.0, -1.0], np.float32) + shift)
        step(model, two, value_weight=0.0, entropy_weight=0.0)
        runs.append([p.detach().clone() for p in model.parameters()])
    for p, q in zip(*runs, strict=True):
        torch.testing.assert_close(p, q)


def test_the_entropy_bonus_spreads_the_policy():
    model, two = two_decisions()
    two = dataclasses.replace(two, advantage=np.zeros(2, dtype=np.float32))
    before = step(model, two, lr=1e-12, value_weight=0.0, entropy_weight=1.0)[0][2]
    after = step(model, two, lr=1e-12, value_weight=0.0, entropy_weight=1.0)[0][2]
    step(model, two, lr=1e-2, value_weight=0.0, entropy_weight=1.0)
    spread = step(model, two, lr=1e-12, value_weight=0.0, entropy_weight=1.0)[0][2]
    assert before == pytest.approx(after)
    assert spread > before


def test_the_value_learns_its_target():
    model, two = two_decisions()
    experience = dataclasses.replace(two.experience, outcome=np.full(2, -1.0, dtype=np.float32))
    two = dataclasses.replace(
        two,
        experience=experience,
        advantage=np.zeros(2, dtype=np.float32),
        target=np.ones(2, dtype=np.float32),
    )
    _, before = taken_logp(model, two)
    step(model, two, lr=1e-2, value_weight=1.0, entropy_weight=0.0)
    _, after = taken_logp(model, two)
    assert (after > before).all()


def test_passes_stop_once_the_policy_moves_past_the_target_kl():
    model, two = two_decisions()
    two = dataclasses.replace(two, advantage=np.array([1.0, -1.0], dtype=np.float32))
    # KL is measured before each step: one batch's first pass sees the policy that played.
    assert step(model, two, epochs=3, lr=1e-2, target_kl=1e-12)[1] == 2
    assert step(model, two, epochs=3, lr=1e-2, target_kl=100.0)[1] == 3


def test_games_against_the_opponent_come_first_with_the_network_on_each_side_in_turn():
    weights = net().state_dict()
    settings = PPO(games=8, opponent=UNIFORM, against_opponent=0.5)
    jobs = _jobs(GAME, net(), weights, settings, first=20, workers=2)
    games = [g for job in jobs for g in job.games]
    assert [seed for seed, *_ in games] == list(range(20, 28))
    assert [learner for _, learner, *_ in games] == [0, 1, 0, 1, None, None, None, None]


def test_each_iteration_plays_new_games():
    settings = PPO(games=2, lr=1e-12, epochs=1)
    first, second = self_play(net(), GAME, settings, iterations=2, seed=0)
    alone = next(self_play(net(), GAME, settings, iterations=1, seed=2))
    assert [p.seed for p in first.played] == [0, 1]
    assert [p.seed for p in second.played] == [p.seed for p in alone.played] == [2, 3]
    assert [p.final.key() for p in second.played] == [p.final.key() for p in alone.played]


def test_the_network_runs_in_float32_on_the_cpu():
    model, two = two_decisions()
    e = two.experience
    scores, value, _ = net_outputs(model, e.obs, e.legal, torch.device("cpu"))
    with torch.no_grad():
        _, direct = model(torch.from_numpy(e.obs).float() / 255)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(value.detach(), direct, rtol=0, atol=0)


class ThreePatches(Patches):
    @property
    def n_players(self) -> int:
        return 3


def test_self_play_is_for_games_of_two_players():
    with pytest.raises(ValueError, match="two players, not 3"):
        self_play(net(), ThreePatches(), iterations=1)
