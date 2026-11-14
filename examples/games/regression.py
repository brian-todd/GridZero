"""The regressions behind code players' constants, and how they are tested.

A code player's value is a logistic regression of a teacher's value, and its prior a conditional
logit of the teacher's moves, tested on games kept out of the fit (`in_a_row.fit`, `tactics.fit`).
What a shipped player needs of this is only its constants, so `fitting.py`, which it does import,
stays free of what is here.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

HELD_OUT = 5
"""One game in this many is kept out of a fit."""
PRIOR_L2 = 1e-3
"""A prior's ridge penalty per position."""
VALUE_C = 10.0
"""A value's inverse ridge penalty."""


def held_out(games: NDArray[np.int64], seed: int) -> NDArray[np.bool_]:
    """Return which rows' games are kept out of a fit: whole games, one in `HELD_OUT`.

    Parameters
    ----------
    games : ndarray of int64, shape (n,)
        The game of each row.
    seed : int
        Sets which games.

    Returns
    -------
    ndarray of bool, shape (n,)
        True on the rows kept out.
    """
    numbers = np.unique(games)
    rng = np.random.default_rng(seed)
    out = rng.choice(numbers, size=max(1, len(numbers) // HELD_OUT), replace=False)
    return np.isin(games, out)


def correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the correlation of two series, shape (n,), 0 if there is none to measure.

    Parameters
    ----------
    a, b : ndarray of float64, shape (n,)
        The series.

    Returns
    -------
    float
        Their correlation; 0 if fewer than two rows, or either is constant.
    """
    if len(a) < 2 or a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_logistic(features: NDArray[np.float64], target: NDArray[np.float64]) -> LogisticRegression:
    """Return a logistic regression of soft targets: a row of each class, weighted by the target.

    Parameters
    ----------
    features : ndarray of float64, shape (n, n_features)
        The features.
    target : ndarray of float64, shape (n,)
        The targets, in [0, 1].

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The model.
    """
    return LogisticRegression(C=VALUE_C, max_iter=2000).fit(
        np.concatenate([features, features]),
        np.concatenate([np.ones(len(target)), np.zeros(len(target))]),
        sample_weight=np.concatenate([target, 1.0 - target]),
    )


def fit_choices(
    features: NDArray[np.float64], target: NDArray[np.float64], first: NDArray[np.int64]
) -> tuple[float, ...]:
    """Return the weights of a conditional logit: the softmax of each position's moves.

    Parameters
    ----------
    features : ndarray of float64, shape (m, n_features)
        The features of every position's moves, each position's together.
    target : ndarray of float64, shape (m,)
        Each move's share of its position, in [0, 1], summing to 1 over a position.
    first : ndarray of int64, shape (n,)
        Where each position's moves start, ascending, the first at 0.

    Returns
    -------
    tuple of float
        A weight per feature, in logits.
    """
    n_positions = len(first)
    groups = np.repeat(np.arange(n_positions), np.diff(np.r_[first, len(features)]))

    def loss(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = features @ theta
        normal = np.logaddexp.reduceat(logits, first)
        probability = np.exp(logits - normal[groups])
        value = -(target * (logits - normal[groups])).sum() / n_positions
        grad = ((probability - target)[:, None] * features).sum(axis=0) / n_positions
        return value + PRIOR_L2 * float(theta @ theta), grad + 2.0 * PRIOR_L2 * theta

    found = minimize(loss, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B")
    return tuple(float(w) for w in found.x)
