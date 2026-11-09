"""A pin on how games are played, so that a change meant only for speed can't change a game.

If a test here fails, either a speed-up changed play, which is a bug, or a rule or a player was
changed on purpose: then replace its digest with the one the failure prints.
"""

import dataclasses
import hashlib

import numpy as np
import pytest

from examples.games.tactics import GreedyPlayer, SearchPlayer, earlier_rules
from tactics_demo import WORKS, play_random

EARLIER = dataclasses.replace(
    earlier_rules(), shape=(9, 9), home_depth=4, team_size=5, max_rounds=12, capture_by=8
)
"""`WORKS` under the earlier rules, which must still play as the defaults did before them."""


def add(digest, position, action):
    """Add an action and the board it was taken on to a digest."""
    state, units = position.state, position.state.units
    for values in (units.kind, units.team, units.hp, units.facing, state.wall, state.rubble):
        digest.update(values.tobytes())
    digest.update(np.array([*action, position.round, position.hold.clock]).tobytes())


def by_player(player, games, decisions=10_000, game=WORKS):
    """Return a digest of ``player`` playing itself, every decision of each game."""
    digest = hashlib.sha256()
    rng = np.random.default_rng(0)
    for seed in range(games):
        position = game.start(seed)
        for _ in range(decisions):
            if position.over:
                break
            legal = game.legal(position)
            digest.update(legal.tobytes())
            action = player(game, game.player(position), position, legal, rng)
            add(digest, position, action)
            position = game.step(position, action)
    return digest.hexdigest()[:16]


def by_chance(games, game=WORKS):
    """Return a digest of random play, which reaches the actions players seldom take."""
    digest = hashlib.sha256()
    for seed in range(games):
        for position, action in play_random(game, seed, np.random.default_rng(seed), 0.02):
            digest.update(game.legal(position).tobytes())
            add(digest, position, action)
    return digest.hexdigest()[:16]


@pytest.mark.parametrize(
    ("play", "pinned"),
    [
        (lambda: by_chance(4), "1417a127fc067015"),
        (lambda: by_player(GreedyPlayer(), 4), "25ca91e1d523ba6b"),
        (
            lambda: by_player(SearchPlayer(simulations=8, rollout_turns=4), 1, 80),
            "f43e3effe48b5c84",
        ),
    ],
    ids=["random", "greedy", "search"],
)
def test_games_are_played_as_they_were(play, pinned):
    assert play() == pinned


@pytest.mark.parametrize(
    ("play", "pinned"),
    [
        (lambda: by_chance(4, EARLIER), "e370826d44966fc3"),
        (lambda: by_player(GreedyPlayer(), 4, game=EARLIER), "2213b714f4b1fa8c"),
        (
            lambda: by_player(SearchPlayer(simulations=8, rollout_turns=4), 1, 80, EARLIER),
            "a96d6e363ad8c1f3",
        ),
    ],
    ids=["random", "greedy", "search"],
)
def test_the_earlier_rules_play_as_the_defaults_did_before_them(play, pinned):
    assert play() == pinned
