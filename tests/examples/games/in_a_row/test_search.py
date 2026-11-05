"""Tests of the search on Connect Four: what it must look at, at the depth it reaches."""

import numpy as np

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
