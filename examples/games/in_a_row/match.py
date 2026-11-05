"""What an evaluator makes of the opening of k in a row.

`opening` reads what an evaluator (such as an AlphaZero agent's network,
`agents.net.NetEvaluator`) makes of the empty board and of each first move: theory says the
first player wins Connect Four, only by opening in the center column (`FIRST_MOVES`). A
player's record against rivals is `game.ladder`'s, from `EVAL_SEED` in every experiment.
"""

from typing import NamedTuple

import numpy as np

from gridzero.agents import Evaluator

from .game import PLACE, InARow
from .state import InARowPosition

EVAL_SEED = 3_000_000
"""The first start of every series that scores players: games no training run plays."""

FIRST_MOVES = (0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0)
"""Player 0's value of Connect Four after each first column, under perfect play: the center
wins, the columns beside it draw, the rest lose (Allis 1988; Tromp's solver)."""


class Opening(NamedTuple):
    """What an evaluator makes of the empty board."""

    column: int
    """The column of the highest prior."""
    prior: tuple[float, ...]
    """The prior of each column, in order."""
    value: float
    """Player 0's value of the empty board, in [0, 1]."""
    after: tuple[float, ...]
    """Player 0's value after each first column, in order, in [0, 1] (`FIRST_MOVES` on
    Connect Four's rules)."""


def opening(game: InARow, evaluator: Evaluator[InARow, InARowPosition]) -> Opening:
    """Return what an evaluator makes of the empty board of a game with gravity, and after it.

    Parameters
    ----------
    game : InARow
        The rules, with gravity: each legal first move is the bottom of a column.
    evaluator : Evaluator
        Gives the prior and the value, such as `agents.net.NetEvaluator`.

    Returns
    -------
    Opening
        The column of the highest prior (the lowest of ties), the priors, the value, and the
        value after each first column.

    Raises
    ------
    ValueError
        If the game has no gravity.
    """
    if not game.gravity:
        raise ValueError("the opening is read by column, which needs gravity")
    start = game.start(0)
    legal = game.legal(start)
    after = [game.step(start, (PLACE, int(cell))) for cell in np.flatnonzero(legal[PLACE])]
    logits, values = evaluator(
        game, [start, *after], [legal, *(game.legal(position) for position in after)]
    )
    prior = np.exp(logits[0] - logits[0].max())
    prior /= prior.sum()
    columns = game.space.unravel(np.flatnonzero(legal[PLACE]))[:, 0]
    return Opening(
        int(columns[np.argmax(prior)]),
        tuple(float(p) for p in prior),
        float(values[0]),
        tuple(float(v) for v in values[1:]),
    )
