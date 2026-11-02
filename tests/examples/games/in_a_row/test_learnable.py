"""Tests of k in a row as a game a learner plays: observations, frames, and a network on it."""

import numpy as np
import pytest
import torch

from agents_demo import check_learnable
from examples.games.in_a_row import PLACE, InARow
from game_demo import check_game
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.agents.net import NetEvaluator, PolicyNet

CONNECT_FOUR = InARow()


@pytest.mark.parametrize(
    "game", [CONNECT_FOUR, InARow((3, 3), 3, gravity=False), InARow((4, 3), 3)]
)
def test_the_game_keeps_the_promises_of_a_learnable_game(game):
    check_learnable(game, check_game(game, UniformPlayer(), seeds=range(4)))


def test_each_player_sees_its_own_stones_first_and_whether_it_moved_first():
    position = CONNECT_FOUR.start(0)
    for c in (3, 3, 4):  # by column: player 0 at (3, 0) and (4, 0), player 1 at (3, 1)
        legal = np.flatnonzero(CONNECT_FOUR.legal(position)[PLACE])
        cell = next(int(i) for i in legal if CONNECT_FOUR.space.unravel(i)[0] == c)
        position = CONNECT_FOUR.step(position, (PLACE, cell))
    owner = position.state.owner
    zero, one = CONNECT_FOUR.observe(position, 0), CONNECT_FOUR.observe(position, 1)
    assert CONNECT_FOUR.planes == ("own", "other", "first")
    np.testing.assert_array_equal(zero[0], owner == 0)
    np.testing.assert_array_equal(zero[1], owner == 1)
    np.testing.assert_array_equal(one[0], owner == 1)
    np.testing.assert_array_equal(one[1], owner == 0)
    assert (zero[2] == 1).all()
    assert (one[2] == 0).all()
    assert zero[0].sum() == 2
    assert zero[1].sum() == 1


def test_both_players_see_the_board_as_it_is():
    for player in (0, 1):
        assert CONNECT_FOUR.frame(player).identity


def test_a_network_scores_the_landing_cells_of_a_board_that_is_not_square():
    torch.manual_seed(0)
    net = PolicyNet(CONNECT_FOUR, channels=8, blocks=1)
    position = CONNECT_FOUR.step(CONNECT_FOUR.start(0), (PLACE, 18))
    legal = CONNECT_FOUR.legal(position)
    logits, values = NetEvaluator(net)(CONNECT_FOUR, [position], [legal])
    assert logits[0].shape == (7,)
    assert np.isclose(np.exp(logits[0]).sum(), 1.0)
    assert 0.0 <= values[0] <= 1.0
    player = AlphaZeroPlayer(NetEvaluator(net), simulations=8)
    action = player(
        CONNECT_FOUR, CONNECT_FOUR.player(position), position, legal, np.random.default_rng(0)
    )
    assert legal[action.verb, action.cell]
