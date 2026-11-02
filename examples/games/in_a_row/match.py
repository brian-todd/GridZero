"""What an evaluator makes of the opening of k in a row.

`opening` reads what an evaluator (such as an AlphaZero agent's network,
`agents.net.NetEvaluator`) makes of the empty board: theory says the first player wins Connect
Four, only by opening in the center column. A player's record against rivals is
`game.ladder`'s, from `EVAL_SEED` in every experiment.
"""

from typing import NamedTuple

import numpy as np

from gridzero.agents import Evaluator

from .game import PLACE, InARow
from .state import InARowPosition

EVAL_SEED = 3_000_000
"""The first start of every series that scores players: games no training run plays."""


class Opening(NamedTuple):
    """What an evaluator makes of the empty board."""

    column: int
    """The column of the highest prior."""
    prior: tuple[float, ...]
    """The prior of each column, in order."""
    value: float
    """Player 0's value of the empty board, in [0, 1]."""


def opening(game: InARow, evaluator: Evaluator[InARow, InARowPosition]) -> Opening:
    """Return what an evaluator makes of the empty board of a game with gravity.

    Parameters
    ----------
    game : InARow
        The rules, with gravity: each legal first move is the bottom of a column.
    evaluator : Evaluator
        Gives the prior and the value, such as `agents.net.NetEvaluator`.

    Returns
    -------
    Opening
        The column of the highest prior (the lowest of ties), the priors and the value.

    Raises
    ------
    ValueError
        If the game has no gravity.
    """
    if not game.gravity:
        raise ValueError("the opening is read by column, which needs gravity")
    start = game.start(0)
    legal = game.legal(start)
    logits, values = evaluator(game, [start], [legal])
    prior = np.exp(logits[0] - logits[0].max())
    prior /= prior.sum()
    columns = game.space.unravel(np.flatnonzero(legal[PLACE]))[:, 0]
    return Opening(
        int(columns[np.argmax(prior)]),
        tuple(float(p) for p in prior),
        float(values[0]),
    )
