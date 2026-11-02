"""The checks every game's tests run (`game_demo.check_game`): they pass a sound game and catch
broken ones."""

import dataclasses

import numpy as np
import pytest

from game_demo import FLIP, Claims, ClaimsPosition, RandomClaimer, check_game


class Lying(Claims):
    """Says its flips drew nothing."""

    def step(self, position, action):
        return dataclasses.replace(super().step(position, action), drew=False)


class Lax(Claims):
    """Accepts claims of cells already owned, as no-ops that pass the turn."""

    def step(self, position, action):
        if position.state.owner[int(action[1])] >= 0 and not position.over:
            return dataclasses.replace(position, ply=position.ply + 1, drew=False, events=())
        return super().step(position, action)


class Restless(Claims):
    """Still offers actions once the game is over."""

    def legal(self, position):
        if position.over:
            return np.ones((2, position.state.n_cells), dtype=np.bool_)
        return super().legal(position)


class Fickle(Claims):
    """Flips with fresh entropy, not the position's stream."""

    def step(self, position: ClaimsPosition, action):
        after = super().step(position, action)
        if int(action[0]) != FLIP:
            return after
        fresh = dataclasses.replace(position, seed=int(np.random.default_rng().integers(2**62)))
        return super().step(fresh, action)


class Endless(Claims):
    """Never ends: each claim frees the cell claimed before it."""

    def step(self, position, action):
        after = super().step(position, action)
        owner = after.state.owner.copy()
        owner[(int(action[1]) + 1) % owner.size] = -1
        state = after.state.replace({type(after.state).owner: owner})
        return dataclasses.replace(after, state=state, over=False, winner=None)


def test_a_sound_game_passes():
    records = check_game(Claims(), RandomClaimer(), range(10))
    assert len(records) == 10


def test_a_game_that_hides_a_draw_fails():
    with pytest.raises(AssertionError, match="drew nothing"):
        check_game(Lying(), RandomClaimer(flip=1.0))


def test_a_game_that_accepts_an_illegal_action_fails():
    with pytest.raises(AssertionError, match="illegal action"):
        check_game(Lax(), RandomClaimer())


def test_a_game_that_offers_actions_after_the_end_fails():
    with pytest.raises(AssertionError):
        check_game(Restless(), RandomClaimer())


def test_a_game_that_draws_outside_its_stream_fails():
    with pytest.raises(AssertionError):
        check_game(Fickle(), RandomClaimer(flip=1.0))


def test_a_game_longer_than_the_limit_fails():
    with pytest.raises(AssertionError):
        check_game(Endless(), RandomClaimer(flip=0.0), max_plies=50)
