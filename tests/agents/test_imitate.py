from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from agents_demo import Lowest, Patches
from gridzero.agents import Experience
from gridzero.agents.imitate import _best, collect, imitate
from gridzero.agents.net import PolicyNet, outputs, unpack
from gridzero.agents.zero_train import losses
from gridzero.game import Action, play

GAME = Patches()


@dataclass
class Watcher:
    """Plays as `Lowest` and keeps what each decision with a choice looked like."""

    seen: list[tuple[int, int, np.ndarray, np.ndarray, Action]] = field(default_factory=list)

    def __call__(self, game, player, position, legal, rng):
        action = Lowest()(game, player, position, legal, rng)
        if legal.sum() > 1:
            player = game.player(position)
            self.seen.append((player, position.ply, game.observe(position, player), legal, action))
        return action


@pytest.fixture(scope="module")
def experience():
    return collect(GAME, Lowest(), 12, seed=10)


def test_collected_decisions_are_those_with_a_choice_in_the_players_frame():
    got = collect(GAME, Lowest(), 2, seed=3)
    watcher = Watcher()
    winners = [play(GAME, (watcher, watcher), GAME.start(seed)).final.winner for seed in (3, 4)]
    assert len(got) == len(watcher.seen)
    best = _best(got)
    n_cells = GAME.space.n_cells
    for i, (player, ply, obs, legal, action) in enumerate(watcher.seen):
        np.testing.assert_allclose(got.obs[i] / 255, obs, atol=0.5 / 255)
        mine = GAME.frame(player).into(legal).ravel()
        mask = np.unpackbits(got.legal[i])[: legal.size].astype(bool)
        np.testing.assert_array_equal(mask, mine)
        verb, cell = GAME.frame(player).action_into(action)
        assert best[i] == verb * n_cells + cell
        policy = got.policy[got.start[i] : got.start[i + 1]]
        assert policy.size == mine.sum()
        assert sorted(policy.tolist()) == [0.0] * (policy.size - 1) + [1.0]
        assert got.ply[i] == ply
    assert set(got.game.tolist()) == {3, 4}
    for seed, winner in zip((3, 4), winners, strict=True):
        players = np.array([s[0] for s, g in zip(watcher.seen, got.game, strict=True) if g == seed])
        expected = 0.0 if winner is None else np.where(players == winner, 1.0, -1.0)
        np.testing.assert_array_equal(got.outcome[got.game == seed], expected)


def test_every_collected_action_is_legal_and_there_was_a_choice(experience):
    masks = np.unpackbits(experience.legal, axis=1).astype(bool)
    assert masks[np.arange(len(experience)), _best(experience)].all()
    assert (masks.sum(axis=1) > 1).all()
    assert experience.obs.dtype == np.uint8
    assert set(np.unique(experience.outcome)) <= {-1.0, 0.0, 1.0}


def test_the_best_action_of_a_spread_policy_is_its_largest_first_on_ties():
    legal = np.zeros((2, 16), dtype=bool)
    legal[0, [1, 4, 9]] = legal[1, [0, 2]] = True
    batch = Experience(
        obs=np.zeros((2, 1, 8), dtype=np.uint8),
        legal=np.packbits(legal, axis=1),
        policy=np.array([0.2, 0.5, 0.3, 0.5, 0.5], dtype=np.float32),
        start=np.array([0, 3, 5]),
        outcome=np.zeros(2, dtype=np.float32),
        game=np.zeros(2, dtype=np.int64),
        ply=np.zeros(2, dtype=np.int64),
    )
    np.testing.assert_array_equal(_best(batch), [4, 0])


@pytest.mark.slow
def test_collecting_in_parallel_gives_the_same_decisions_in_order():
    alone = collect(GAME, Lowest(), 3, seed=1)
    together = collect(GAME, Lowest(), 3, seed=1, workers=2)
    for name in ("obs", "legal", "policy", "start", "outcome", "game", "ply"):
        np.testing.assert_array_equal(getattr(alone, name), getattr(together, name))


def test_collect_checks_its_counts(not_an_integer):
    with pytest.raises(ValueError, match="games"):
        collect(GAME, Lowest(), 0)
    with pytest.raises(TypeError, match="games"):
        collect(GAME, Lowest(), not_an_integer)
    with pytest.raises(ValueError, match="seed"):
        collect(GAME, Lowest(), 1, seed=-1)
    with pytest.raises(ValueError, match="workers"):
        collect(GAME, Lowest(), 1, workers=0)


def test_unpacking_bits_inverts_numpy_packbits(rng):
    bits = rng.random((4, 37)) < 0.5
    packed = torch.from_numpy(np.packbits(bits, axis=1))
    np.testing.assert_array_equal(unpack(packed, 37).numpy(), bits)


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(GAME, channels=8, blocks=1)


def test_training_learns_the_teachers_actions(experience):
    epochs = list(imitate(net(), experience, epochs=30, batch_size=32, lr=1e-2, holdout=0.25))
    assert [e.epoch for e in epochs] == list(range(1, 31))
    assert epochs[-1].policy_loss < 0.5 * epochs[0].policy_loss
    assert epochs[-1].holdout_accuracy > 0.8


def test_games_differ_by_their_first_stone(experience):
    first = experience.take(experience.ply == 0)
    assert len(first) == 12
    assert len({tuple(np.flatnonzero(experience.policy[s : s + 24])) for s in first.start[:-1]}) > 1


def test_the_held_out_accuracy_is_the_share_of_the_teachers_actions_the_network_picks(
    experience,
):
    model = net()
    epoch = next(imitate(model, experience, epochs=1, holdout=0.25))
    held = experience.take(np.isin(experience.game, [19, 20, 21]))
    with torch.no_grad():
        scores, _, _ = outputs(model.eval(), held.obs, held.legal, torch.device("cpu"))
    expected = np.mean(scores.argmax(dim=1).numpy() == _best(held))
    assert 0 < expected < 1
    assert epoch.holdout_accuracy == pytest.approx(expected)


def test_the_held_out_losses_are_the_mean_losses_of_the_held_out_decisions(experience):
    model = net()
    epoch = next(imitate(model, experience, epochs=1, holdout=0.25))
    held = experience.take(np.isin(experience.game, [19, 20, 21]))
    with torch.no_grad():
        policy, value = losses(model.eval(), held, torch.device("cpu"))
    assert policy.mean().item() != pytest.approx(value.mean().item())  # a swap would show
    assert epoch.holdout_policy_loss == pytest.approx(policy.mean().item())
    assert epoch.holdout_value_loss == pytest.approx(value.mean().item())


def test_held_out_decisions_come_from_the_last_games(experience):
    # A sixth of twelve games is two: training never sees games 20 and 21.
    held_out = list(imitate(net(), experience, epochs=2, holdout=1 / 6))
    first_ten = experience.take(~np.isin(experience.game, [20, 21]))
    alone = list(imitate(net(), first_ten, epochs=2, holdout=0.0))
    for a, b in zip(held_out, alone, strict=True):
        assert (a.policy_loss, a.value_loss) == (b.policy_loss, b.value_loss)
        assert 0 <= a.holdout_accuracy <= 1
        assert np.isfinite(a.holdout_policy_loss)
    assert np.isnan(alone[-1].holdout_accuracy)


def test_any_share_above_zero_holds_out_a_game(experience):
    held_out = next(imitate(net(), experience, epochs=1, holdout=0.01))
    alone = next(imitate(net(), experience.take(experience.game != 21), epochs=1, holdout=0.0))
    assert held_out.policy_loss == alone.policy_loss
    assert np.isfinite(held_out.holdout_accuracy)


def test_training_runs_only_as_epochs_are_taken(experience):
    trained = net()
    before = [p.clone() for p in trained.parameters()]
    run = imitate(trained, experience, epochs=3)
    for p, q in zip(trained.parameters(), before, strict=True):
        assert torch.equal(p, q)
    next(run)
    assert any(not torch.equal(p, q) for p, q in zip(trained.parameters(), before, strict=True))


def test_the_same_seeds_train_the_same_network(experience):
    runs = [list(imitate(net(), experience, epochs=2, seed=4)) for _ in range(2)]
    assert runs[0] == runs[1]


def test_imitate_checks_its_parameters(experience, not_an_integer):
    with pytest.raises(TypeError, match="epochs"):
        imitate(net(), experience, epochs=not_an_integer)
    with pytest.raises(ValueError, match="batch_size"):
        imitate(net(), experience, batch_size=0)
    with pytest.raises(ValueError, match="lr"):
        imitate(net(), experience, lr=0.0)
    with pytest.raises(ValueError, match="value_weight"):
        imitate(net(), experience, value_weight=-1.0)
    with pytest.raises(ValueError, match="holdout"):
        imitate(net(), experience, holdout=1.0)
    with pytest.raises(ValueError, match="no decisions"):
        imitate(net(), experience.take(np.arange(0)))
