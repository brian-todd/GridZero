"""AlphaZero's training in arrays on Tactics under fog: each mover decides from its belief."""

import dataclasses

import numpy as np

from examples.games.tactics import GreedyPlayer, small_rules
from examples.games.tactics.arrays import TacticsArrays
from gridzero.agents.zero_arrays_train import _Games

GAME = small_rules()
"""Under fog, on a board large enough for the sides to lose sight of each other (on 9x9 they
never do)."""


def in_battle(seed, plies=30):
    """The position after greedy has played both sides for ``plies`` plies, or to the end."""
    position, player, rng = GAME.start(seed), GreedyPlayer(), np.random.default_rng(seed)
    for _ in range(plies):
        if position.over:
            break
        mover = GAME.player(position)
        shown = GAME.view(position, mover)
        position = GAME.step(position, player(GAME, mover, shown, GAME.legal(shown), rng))
    return position


def test_each_mover_is_shown_its_belief_as_the_game_shows_it():
    positions = [in_battle(seed) for seed in range(4)]
    positions = [p for p in positions if not p.over]
    rules = TacticsArrays(GAME)
    games = _Games(rules, lambda seed: rules.to_arrays(positions[seed % len(positions)]), 4, 0)
    _, movers, obs, legal, _ = games._shown(games.states)
    hidden = 0
    for i, mover in enumerate(np.asarray(movers).tolist()):
        position = positions[i % len(positions)]
        assert mover == GAME.player(position)
        shown = GAME.view(position, mover)
        hidden += not np.array_equal(shown.state.units.uid, position.state.units.uid)
        wanted = np.rint(GAME.observe(shown, mover) * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(obs)[i], wanted)
        assert np.asarray(legal)[i].sum() == GAME.legal(shown).sum()
    assert hidden, "some mover believed a unit elsewhere than it is"


def test_without_fog_each_mover_is_shown_the_position():
    game = dataclasses.replace(GAME, fog=False)
    rules = TacticsArrays(game)
    games = _Games(rules, lambda seed: rules.to_arrays(game.start(seed)), 2, 0)
    views, *_ = games._shown(games.states)
    for a, b in zip(views, games.states, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
