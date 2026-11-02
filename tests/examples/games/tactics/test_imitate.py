"""Imitation (`agents.imitate`) on Tactics: its tests on a generic game are in tests/agents."""

from dataclasses import dataclass, field

import numpy as np
import torch

from examples.games.tactics import Action, GreedyPlayer
from gridzero.agents.imitate import _best, collect, imitate
from gridzero.agents.net import PolicyNet
from gridzero.game import play
from tactics_demo import TINY


@dataclass
class Watcher:
    """Plays greedily and keeps what each decision with a choice looked like."""

    seen: list[tuple[int, np.ndarray, np.ndarray, Action]] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    def __call__(self, game, player, position, legal, rng):
        action = GreedyPlayer()(game, player, position, legal, rng)
        if legal.sum() > 1:
            team = game.player(position)
            self.seen.append((team, game.observe(position, team), legal, action))
            self.plies.append(position.ply)
        return action


def test_greedys_decisions_are_kept_in_the_teams_frame():
    got = collect(TINY, GreedyPlayer(), 2, seed=3)
    watcher = Watcher()
    winners = [play(TINY, (watcher, watcher), TINY.start(seed)).final.winner for seed in (3, 4)]
    assert len(got) == len(watcher.seen)
    np.testing.assert_array_equal(got.ply, watcher.plies)  # forced moves are skipped
    assert (np.diff(got.ply)[np.diff(got.game) == 0] > 1).any()
    n_cells = TINY.space.n_cells
    best = _best(got)
    for i, (team, obs, legal, action) in enumerate(watcher.seen):
        np.testing.assert_allclose(got.obs[i] / 255, obs, atol=0.5 / 255)
        mask = np.unpackbits(got.legal[i])[: legal.size].astype(bool)
        np.testing.assert_array_equal(mask, TINY.frame(team).into(legal).ravel())
        verb, cell = TINY.frame(team).action_into(action)
        assert best[i] == verb * n_cells + cell
    for seed, winner in zip((3, 4), winners, strict=True):
        teams = np.array(
            [t for (t, *_), g in zip(watcher.seen, got.game, strict=True) if g == seed]
        )
        expected = 0.0 if winner is None else np.where(teams == winner, 1.0, -1.0)
        np.testing.assert_array_equal(got.outcome[got.game == seed], expected)


def test_a_network_on_hexes_learns_greedys_actions():
    experience = collect(TINY, GreedyPlayer(), 6, seed=10)
    torch.manual_seed(0)
    net = PolicyNet(TINY, channels=8, blocks=1)
    epochs = list(imitate(net, experience, epochs=8, batch_size=64, lr=3e-3, holdout=0.0))
    assert epochs[-1].policy_loss < 0.7 * epochs[0].policy_loss
