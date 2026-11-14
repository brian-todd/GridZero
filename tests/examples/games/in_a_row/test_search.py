"""Tests of the search on Connect Four and tic-tac-toe: what it must look at, and answer."""

import numpy as np
import pytest

from examples.games.in_a_row import PLACE, InARow
from gridzero.agents import AlphaZeroPlayer

GAME = InARow()
ROWS = GAME.shape[1]


class Pessimist:
    """Thinks little of red's turn, so what red tries first looks better than the position.

    Its prior favors the fourth column, and it keeps the boards it judged.
    """

    def __init__(self):
        self.judged = set()

    def __call__(self, game, positions, legal):
        logits = []
        values = []
        for position, mask in zip(positions, legal, strict=True):
            self.judged.add(position.state.owner.tobytes())
            logits.append(2.0 * (np.flatnonzero(mask[PLACE]) // ROWS == 3))
            values.append(0.4 if position.ply % 2 == 0 else 0.5)
        return logits, np.array(values)


def test_a_node_below_the_root_doesnt_stop_at_the_first_action_that_looks_better():
    # Red opened in the fourth and third columns. After blue's stone on the fourth, red's in the
    # second makes three in a row open at both ends: a forced win the prior doesn't favor.
    position = GAME.start(0)
    for cell in (3 * ROWS, 5 * ROWS, 2 * ROWS):
        position = GAME.step(position, (PLACE, cell))
    after_blue = GAME.step(position, (PLACE, 3 * ROWS + 1))
    forced_win = GAME.step(after_blue, (PLACE, 1 * ROWS))
    evaluator = Pessimist()
    AlphaZeroPlayer(evaluator, simulations=1024, considered=7).search(
        GAME, position, np.random.default_rng(0)
    )
    assert after_blue.state.owner.tobytes() in evaluator.judged
    assert forced_win.state.owner.tobytes() in evaluator.judged


TIC_TAC_TOE = InARow(shape=(3, 3), length=3, gravity=False)


def know_nothing(game, positions, legal):
    """A uniform prior and even values: whatever the search finds, it finds by looking."""
    return [np.zeros(int(mask.sum())) for mask in legal], np.full(len(positions), 0.5)


@pytest.mark.parametrize(
    ("moves", "block"),
    [
        ((0, 4, 1), 2),  # player 0 holds cells 0 and 1 of a line: player 1 must take 2
        ((0, 4, 8, 5), 3),  # player 1 holds cells 4 and 5 of a line: player 0 must take 3
    ],
)
def test_the_search_blocks_a_win_the_other_player_threatens(moves, block):
    # The mover can't win at once, and only the block stops the other player's next stone from
    # winning: the tree holds the other player's replies, each chosen for that player's own value.
    position = TIC_TAC_TOE.start(0)
    for cell in moves:
        position = TIC_TAC_TOE.step(position, (PLACE, cell))
    player = AlphaZeroPlayer(know_nothing, simulations=128)
    for seed in range(20):
        result = player.search(TIC_TAC_TOE, position, np.random.default_rng(seed))
        assert result.action == (PLACE, block)
