import dataclasses
import itertools
import multiprocessing

import numpy as np
import pytest
import torch

from examples.games.tactics import (
    Captured,
    GameOver,
    Tactics,
    TacticsPosition,
    small_rules,
)
from examples.games.tactics.experiments.zero_scratch import endings
from gridzero.agents.net import NetEvaluator, PolicyNet, as_input
from gridzero.agents.zero import AlphaZeroPlayer, self_play
from gridzero.agents.zero_train import (
    Zero,
    _add,
    _entropy,
    _Forward,
    _learn,
    _mean_losses,
    _room,
    _Server,
    _sizes,
    _views,
    generate,
    losses,
    train,
)
from tactics_demo import TINY

FIELDS = ("obs", "legal", "policy", "start", "outcome", "game", "ply")


def player(seed=0):
    torch.manual_seed(seed)
    net = PolicyNet(TINY, channels=8, blocks=2)
    return AlphaZeroPlayer(NetEvaluator(net), simulations=3, considered=3, noise=True)


def test_actors_play_the_games_as_one_process_would():
    # The server's batches mix games, which moves the network's last bits, and rescaling the
    # completed values magnifies them in the policies; the games themselves are the same.
    zero = player()
    experience, played, costs = generate(TINY, zero, [3, 4, 5], actors=2, keep=0.5)
    expected, expected_played = self_play(TINY, zero, [3, 4, 5], keep=0.5)
    assert [p[:-1] for p in played] == [p[:-1] for p in expected_played]
    assert [p.final.key() for p in played] == [p.final.key() for p in expected_played]
    for name in FIELDS:
        if name != "policy":
            np.testing.assert_array_equal(getattr(experience, name), getattr(expected, name))
    np.testing.assert_allclose(experience.policy, expected.policy, atol=1e-2)
    assert [s.games for s in costs.actors] == [2, 1]
    assert sum(s.positions for s in costs.actors) == costs.positions
    assert 0 < costs.batches <= sum(s.calls for s in costs.actors)
    assert all(s.evaluator_seconds <= s.seconds <= costs.seconds for s in costs.actors)
    assert costs.peak_mib == 0.0


def test_one_actor_suffices_and_none_are_needed_without_games():
    experience, played, costs = generate(TINY, player(), [6])
    assert len(played) == 1
    assert len(costs.actors) == 1
    assert len(experience) == played[0].kept
    none, nothing, idle = generate(TINY, player(), [], actors=3)
    assert (len(none), nothing, idle.actors) == (0, [], [])


def test_a_failing_actor_raises_its_error_here():
    with pytest.raises(RuntimeError, match="actor 0 failed"):
        generate(TINY, player(), [1], keep=2.0)


def test_generate_needs_a_network_evaluator(not_an_integer):
    def evaluator(game, positions, legal):
        return [np.zeros(np.count_nonzero(m)) for m in legal], np.full(len(positions), 0.5)

    with pytest.raises(TypeError, match="NetEvaluator"):
        generate(TINY, AlphaZeroPlayer(evaluator), [1])
    with pytest.raises(TypeError, match="actors"):
        generate(TINY, player(), [1], actors=not_an_integer)
    with pytest.raises(ValueError, match="actors"):
        generate(TINY, player(), [1], actors=0)


def quick_settings(**changes):
    settings = Zero(games=1, actors=1, simulations=2, considered=2, steps=3, batch_size=8)
    return dataclasses.replace(settings, **changes)


def test_losses_are_the_cross_entropy_of_the_policy_and_the_squared_error_of_the_value():
    experience, _ = self_play(TINY, player(), [16], keep=0.5)
    assert set(experience.outcome.tolist()) == {1.0, -1.0}
    net = player().evaluator.net
    policy, value = losses(net, experience, torch.device("cpu"))
    n_actions = TINY.n_verbs * TINY.shape[0] * TINY.shape[1]
    with torch.no_grad():
        logits, values = net(torch.from_numpy(experience.obs).float() / 255)
    for i in range(len(experience)):
        legal = np.flatnonzero(np.unpackbits(experience.legal[i])[:n_actions])
        log_policy = torch.log_softmax(logits[i].flatten()[legal].double(), dim=0).numpy()
        target = experience.policy[experience.start[i] : experience.start[i + 1]]
        assert policy[i].item() == pytest.approx(-(target * log_policy).sum(), rel=1e-5)
        expected = (values[i].item() - experience.outcome[i]) ** 2
        assert value[i].item() == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_a_generation_plays_keeps_and_trains():
    torch.manual_seed(0)
    net = PolicyNet(TINY, channels=8, blocks=2)
    before = [p.detach().clone() for p in net.parameters()]
    [generation] = list(train(net, TINY, quick_settings(keep=1.0), generations=1, seed=5))
    assert generation.generation == 1
    assert generation.games == 1
    assert generation.kept == generation.searched == generation.buffer > 0
    assert len(generation.played) == generation.games
    assert all(p.final.round == TINY.max_rounds for p in generation.played)  # to the cap
    assert generation.plies == generation.played[0].plies
    assert np.isfinite([generation.policy_loss, generation.value_loss]).all()
    assert generation.fresh_policy_loss >= generation.target_entropy > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_training_lowers_the_loss_on_the_buffer():
    experience, _ = self_play(TINY, player(), [17, 18], keep=1.0)
    torch.manual_seed(1)
    net = PolicyNet(TINY, channels=8, blocks=2)
    cpu = torch.device("cpu")
    settings = quick_settings(steps=400, lr=3e-3, batch_size=32)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.lr)
    entropy = _entropy(experience)
    policy, value = _mean_losses(net, experience, 64, cpu)
    _learn(net, optimizer, experience, settings, cpu, np.random.default_rng(0))
    trained_policy, trained_value = _mean_losses(net, experience, 64, cpu)
    # A tiny network fits noisy search targets slowly (over seeds, the gap above the entropy falls
    # to 0.54-0.71 of what it was) and outcomes fast; a wrong sign or misaligned target would raise
    # them.
    assert trained_policy - entropy < 0.85 * (policy - entropy)
    assert trained_value < 0.5 * value


def test_the_buffer_keeps_the_latest_decisions():
    old, _ = self_play(TINY, player(), [1], keep=0.5)
    new, _ = self_play(TINY, player(), [2], keep=0.5)
    assert _add(None, old, 10_000) is old
    buffer = _add(_add(None, old, 10_000), new, len(new) + 3)
    assert len(buffer) == len(new) + 3
    np.testing.assert_array_equal(buffer.game, [1] * 3 + [2] * len(new))
    np.testing.assert_array_equal(buffer.ply[:3], old.ply[-3:])
    np.testing.assert_array_equal(
        buffer.policy[: old.start[-1] - old.start[-4]], old.policy[old.start[-4] :]
    )
    assert len(_add(None, old, 2)) == 2


@pytest.mark.slow
def test_each_generation_plays_new_boards_with_noise_and_scores_its_own_decisions():
    # Without training steps the network stays as it is, so self-play with it replays each
    # generation's games.
    torch.manual_seed(0)
    net = PolicyNet(TINY, channels=8, blocks=2)
    settings = quick_settings(keep=0.5, steps=0)
    first, second = train(net, TINY, settings, generations=2, seed=20)
    assert second.buffer == first.kept + second.kept
    assert np.isnan(first.policy_loss)
    zero = AlphaZeroPlayer(NetEvaluator(net), simulations=2, considered=2, noise=True)
    for generation, seed in ((first, 20), (second, 21)):
        experience, [played] = self_play(TINY, zero, [seed], keep=0.5)
        assert (generation.kept, generation.searched) == (len(experience), played.searched)
        expected = _mean_losses(net, experience, settings.batch_size, torch.device("cpu"))
        fresh = (generation.fresh_policy_loss, generation.fresh_value_loss)
        assert fresh == pytest.approx(expected, rel=1e-5)


def test_a_generation_starts_only_if_one_as_long_as_the_last_fits_the_time_box():
    assert _room(10.0, 5.0, None)
    assert _room(10.0, 5.0, 15.0)
    assert not _room(10.0, 5.1, 15.0)
    assert not _room(9.0, 7.0, 15.0)  # an elapsed time alone would fit


@pytest.mark.slow
def test_the_time_box_stops_before_a_generation_that_would_pass_it():
    net = PolicyNet(TINY, channels=8, blocks=2)
    boxed = list(train(net, TINY, quick_settings(steps=0), generations=3, seconds=0.0))
    assert [g.generation for g in boxed] == [1]
    unboxed = list(train(net, TINY, quick_settings(steps=0), generations=2))
    assert [g.generation for g in unboxed] == [1, 2]


def test_settings_and_training_arguments_are_checked(not_an_integer):
    for name in ("games", "actors", "simulations", "considered", "buffer", "batch_size"):
        with pytest.raises(ValueError, match=name):
            Zero(**{name: 0})
    with pytest.raises(TypeError, match="steps"):
        Zero(steps=not_an_integer)
    with pytest.raises(ValueError, match="keep"):
        Zero(keep=1.5)
    with pytest.raises(ValueError, match="lr"):
        Zero(lr=0.0)
    net = PolicyNet(TINY, channels=8, blocks=2)
    with pytest.raises(ValueError, match="generations"):
        train(net, TINY, generations=0)
    with pytest.raises(ValueError, match="seconds"):
        train(net, TINY, seconds=-1.0)


def test_graph_sizes_grow_by_half_in_multiples_of_eight():
    for most in (1, 8, 9, 40, 264, 512):
        sizes = _sizes(most)
        assert sizes[0] == 8
        assert sizes[-1] >= most > (sizes[-2] if len(sizes) > 1 else 0)
        assert all(s % 8 == 0 for s in sizes)
        assert all(b > a and b <= 1.5 * a + 8 for a, b in itertools.pairwise(sizes))


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@gpu
def test_graphed_batches_judge_as_the_network_does():
    torch.manual_seed(0)
    net = PolicyNet(TINY, channels=8, blocks=2).cuda().eval()
    shape = (len(TINY.planes), TINY.shape[0] * TINY.shape[1])
    forward = _Forward(net, torch.device("cuda"), shape, most=40)
    assert sorted(forward.graphs) == [8, 16, 24, 40]
    rng = np.random.default_rng(0)
    for n in (16, 9, 1, 40, 41, 24):  # padded rows hold the last batch's observations
        obs = torch.from_numpy(rng.integers(0, 256, (n, *shape), dtype=np.uint8))
        scores, value = forward(obs)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            expected_scores, expected_value = net(as_input(obs.cuda()))
        assert scores.shape == expected_scores.shape
        assert scores.dtype == torch.float32
        np.testing.assert_allclose(scores.cpu(), expected_scores.float().cpu(), atol=2e-2)
        np.testing.assert_allclose(value.cpu(), expected_value.float().cpu(), atol=2e-2)


@gpu
def test_the_server_answers_with_the_networks_judgements_on_the_gpu():
    # The default rules' board and network, so that the GPU takes about 1.6 ms a batch and
    # answers read before it is done would be stale.
    torch.manual_seed(0)
    game = Tactics()
    net = PolicyNet(game).cuda().eval()
    server = _Server(game, net, [64], multiprocessing.get_context("spawn"))
    obs_view, scores_view, value_view = _views(game, server.slots[0])
    line = server.slots[0].line
    rng = np.random.default_rng(0)
    server.start()
    try:
        for n in (64, 40, 64, 7, 64):
            obs = rng.integers(0, 256, (n, *obs_view.shape[1:]), dtype=np.uint8)
            obs_view[:n] = obs
            line.send_bytes(n.to_bytes(4, "little"))
            line.recv_bytes()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                scores, value = net(as_input(torch.from_numpy(obs).cuda()))
            np.testing.assert_allclose(scores_view[:n], scores.float().cpu(), atol=2e-2)
            np.testing.assert_allclose(value_view[:n], value.float().cpu(), atol=2e-2)
    finally:
        server.stop()


@gpu
def test_the_server_judges_on_the_gpu():
    zero = player()
    zero.evaluator.net.cuda()
    experience, played, costs = generate(TINY, zero, [7, 8, 9], actors=2, keep=1.0)
    assert all(p.plies > 0 for p in played)
    assert len(experience) == sum(p.searched for p in played)
    assert costs.peak_mib > 0


def test_the_rules_zero_scratch_trains_on_make_a_game_and_a_network():
    game = small_rules()
    assert game.start(0).deploying
    assert PolicyNet(game, channels=8, blocks=1) is not None


def test_endings_count_reasons_seats_and_flips_in_the_last_round():
    state = TINY.start(0).state

    def final(winner, *events):
        return TacticsPosition(state, 0, over=True, winner=winner, events=events)

    report = endings(
        [
            final(0, GameOver(0, "killed")),
            final(1, Captured(1, 4, True), GameOver(1, "held")),  # a flip, but not at the cap
            final(1, Captured(1, 16, True), GameOver(1, "cap")),
            final(0, GameOver(0, "cap")),
        ]
    )
    assert report == {
        "reasons": {"killed": 1, "held": 1, "cap": 2},
        "last_round_flips": 1,
        "team_0": 2,
        "games": 4,
    }
