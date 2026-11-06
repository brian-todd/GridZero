"""A pin on how games are played, so that a change meant only for speed can't change a game.

If a test here fails, either a speed-up changed play, which is a bug, or a rule or a player was
changed on purpose: then replace its digest with the one the failure prints.
"""

import hashlib

import numpy as np
import pytest

from examples.games.tactics import GreedyPlayer, SearchPlayer
from tactics_demo import WORKS, play_random


def add(digest, position, action):
    """Add an action and the board it was taken on to a digest."""
    state, units = position.state, position.state.units
    for values in (units.kind, units.team, units.hp, units.facing, state.wall, state.rubble):
        digest.update(values.tobytes())
    digest.update(np.array([*action, position.round, position.hold.clock]).tobytes())


def by_player(player, games, decisions=10_000):
    """Return a digest of ``player`` playing itself, every decision of each game."""
    digest = hashlib.sha256()
    rng = np.random.default_rng(0)
    for seed in range(games):
        position = WORKS.start(seed)
        for _ in range(decisions):
            if position.over:
                break
            legal = WORKS.legal(position)
            digest.update(legal.tobytes())
            action = player(WORKS, WORKS.player(position), position, legal, rng)
            add(digest, position, action)
            position = WORKS.step(position, action)
    return digest.hexdigest()[:16]


def by_chance(games):
    """Return a digest of random play, which reaches the actions players seldom take."""
    digest = hashlib.sha256()
    for seed in range(games):
        for position, action in play_random(WORKS, seed, np.random.default_rng(seed), 0.02):
            digest.update(WORKS.legal(position).tobytes())
            add(digest, position, action)
    return digest.hexdigest()[:16]


@pytest.mark.parametrize(
    ("play", "pinned"),
    [
        (lambda: by_chance(4), "e370826d44966fc3"),
        (lambda: by_player(GreedyPlayer(), 4), "2213b714f4b1fa8c"),
        (
            lambda: by_player(SearchPlayer(simulations=8, rollout_turns=4), 1, 80),
            "a96d6e363ad8c1f3",
        ),
    ],
    ids=["random", "greedy", "search"],
)
def test_games_are_played_as_they_were(play, pinned):
    assert play() == pinned
