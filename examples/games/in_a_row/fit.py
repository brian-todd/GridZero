"""Fit a code player's constants to what a teacher judged (`lessons.py`).

The value is a logistic regression of the teacher's value on `value_features`, and the prior a
conditional logit of the teacher's improved policy on `move_features`: both with the teacher's
soft targets, and both tested on games kept out of the fit.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .code_player import MOVE_FEATURES, FittedEvaluator, move_features, value_features
from .game import InARow
from .lessons import Lessons

MIN_PLY = 4
"""The value is fit from this ply: earlier, agent A's value is flat (it fails the opening)."""
HELD_OUT = 5
"""One game in this many is kept out of the fit."""
PRIOR_L2 = 1e-3
"""The prior's ridge penalty per position."""
VALUE_C = 10.0
"""The value's inverse ridge penalty."""


class FitScore(NamedTuple):
    """How a fit does on games kept out of it, and how many positions it saw."""

    trained: int
    """Positions in the fit."""
    held_out: int
    """Positions kept out."""
    value_correlation: float
    """Correlation of its value with the teacher's, from `MIN_PLY`; 0 if either is constant."""
    agreement: float
    """How often its prior's favorite is the teacher's."""
    regret: float
    """How much value its prior's favorite loses by the teacher's q, on average."""


def fit(game: InARow, lessons: Lessons, seed: int) -> tuple[FittedEvaluator, FitScore]:
    """Fit an evaluator to lessons, and score it on games kept out.

    Parameters
    ----------
    game : InARow
        The rules: four in a row with gravity.
    lessons : Lessons
        What to fit.
    seed : int
        Sets which games are kept out.

    Returns
    -------
    evaluator : FittedEvaluator
        The fit.
    score : FitScore
        How it does on the games kept out.
    """
    test = held_out(lessons.game, seed)
    weights, intercept, correlation = _fit_value(game, lessons, test)
    prior, moves = _fit_prior(game, lessons, test)
    evaluator = FittedEvaluator(weights, intercept, prior)
    agreement, regret = _prior_score(moves, lessons, test, prior)
    return evaluator, FitScore(int((~test).sum()), int(test.sum()), correlation, agreement, regret)


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


def _fit_value(
    game: InARow, lessons: Lessons, test: NDArray[np.bool_]
) -> tuple[tuple[float, ...], float, float]:
    """Return the value's weights, its intercept, and its correlation on the rows of ``test``."""
    late = lessons.ply >= MIN_PLY
    features = value_features(game, lessons.owner[late], lessons.ply[late])
    target = lessons.value[late]
    train = ~test[late]
    model = fit_logistic(features[train], target[train])
    predicted = model.predict_proba(features[~train])[:, 1]
    score = correlation(predicted, target[~train])
    return tuple(float(w) for w in model.coef_[0]), float(model.intercept_[0]), score


def _moves(
    game: InARow, lessons: Lessons
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Return every position's move features in one array, shape (m, n_features), by position.

    Also returns where each position's start, shape (n,), and each move's column, shape (m,).
    """
    parts, columns = [], []
    starts = np.zeros(len(lessons), dtype=np.int64)
    at = 0
    for i in range(len(lessons)):
        cells, features = move_features(game, lessons.owner[i], int(lessons.ply[i]))
        starts[i] = at
        at += len(cells)
        parts.append(features)
        columns.append(game.space.unravel(cells)[:, 0])
    return np.concatenate(parts), starts, np.concatenate(columns)


def _fit_prior(
    game: InARow, lessons: Lessons, test: NDArray[np.bool_]
) -> tuple[tuple[float, ...], tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]]:
    """Return the prior's weights, and the moves of all lessons (see `_moves`)."""
    moves = _moves(game, lessons)
    features, starts, columns = moves
    position = np.zeros(len(features), dtype=np.int64)
    position[starts[1:]] = 1
    position = np.cumsum(position)
    target = lessons.policy[position, columns]
    train = ~test[position]
    x, y = features[train], target[train]
    rows = position[train]
    # The same positions kept, renumbered 0..: each one's moves stay contiguous.
    first = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    n_positions = len(first)

    def loss(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = x @ theta
        normal = np.logaddexp.reduceat(logits, first)
        groups = np.repeat(np.arange(n_positions), np.diff(np.r_[first, len(x)]))
        probability = np.exp(logits - normal[groups])
        value = -(y * (logits - normal[groups])).sum() / n_positions
        grad = ((probability - y)[:, None] * x).sum(axis=0) / n_positions
        return value + PRIOR_L2 * float(theta @ theta), grad + 2.0 * PRIOR_L2 * theta

    found = minimize(loss, np.zeros(len(MOVE_FEATURES)), jac=True, method="L-BFGS-B")
    return tuple(float(w) for w in found.x), moves


def _prior_score(
    moves: tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
    lessons: Lessons,
    test: NDArray[np.bool_],
    prior: tuple[float, ...],
) -> tuple[float, float]:
    """Return how often the prior picks the teacher's favorite on ``test``, and its regret."""
    features, starts, columns = moves
    ends = np.r_[starts[1:], len(features)]
    logits = features @ np.asarray(prior)
    same, lost, counted = 0, 0.0, 0
    for i in np.flatnonzero(test):
        chosen = int(columns[starts[i] + np.argmax(logits[starts[i] : ends[i]])])
        same += chosen == int(np.argmax(lessons.policy[i]))
        q = lessons.q[i] if lessons.ply[i] % 2 == 0 else 1.0 - lessons.q[i]  # the mover's
        if not np.isnan(q[chosen]):
            lost += float(np.nanmax(q) - q[chosen])
            counted += 1
    return same / max(1, int(test.sum())), lost / max(1, counted)
