"""The arithmetic of code players: features and weights to scores, the same in any batch.

A code player is the AlphaZero search on an evaluator of plain code, named features and constants
(`in_a_row.code_player`, `tactics.code_player`). What they share is how a row of features becomes
a score, and a score a value.
"""

import numpy as np
from numpy.typing import NDArray


def weighted(
    features: NDArray[np.float64], weights: tuple[float, ...], start: float
) -> NDArray[np.float64]:
    """Return ``start`` plus the weighted features of each row, the same in any batch.

    Parameters
    ----------
    features : ndarray of float64, shape (n, len(weights))
        The features of each row.
    weights : tuple of float
        A weight per feature, added in order: a matrix product's sum depends on the batch.
    start : float
        What every sum starts from.

    Returns
    -------
    ndarray of float64, shape (n,)
        The sums.

    Raises
    ------
    ValueError
        If a sum overflows.
    """
    score = np.full(len(features), float(start))
    with np.errstate(over="ignore", invalid="ignore"):  # checked below
        for j, weight in enumerate(weights):
            score += weight * features[:, j]
    if not np.isfinite(score).all():
        raise ValueError("a position's score overflows: the weights are too large")
    return score


def sigmoid(score: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the sigmoid of scores, without overflow at any size.

    Parameters
    ----------
    score : ndarray of float64, any shape
        Scores, in logits.

    Returns
    -------
    ndarray of float64, shape as given
        Values in [0, 1].
    """
    return 0.5 + 0.5 * np.tanh(0.5 * score)
