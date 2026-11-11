"""Fit a code player's constants to a search player's games (`lessons.py`).

The value is a logistic regression of the teacher's value on `value_features`, at the positions it
searched. It searches no deployment, so a game's deployment positions take the value of its first
search: what the teacher came to think of how it began. The prior is a conditional logit of the
action the teacher took at every decision with a choice: it plans whole turns, and a decision's
action is the only label that says what to do at each step of one.

Both are scored on whole games kept out of the fit, against what simple rules do on the same games:
the mean value, material alone, the verb alone, `HAND_PICKED` and `GreedyPlayer`.

The constants that play differ from a fit in two ways, both decided by how they played: the
weights `OVERRULED`, and the value read `LOOKAHEAD` unit turns ahead (`playing`).
"""

import dataclasses
import math
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from examples.games.fitting import sigmoid, weighted
from examples.games.regression import correlation, fit_choices, fit_logistic, held_out

from .code_player import (
    HAND_PICKED,
    MOVE_FEATURES,
    VALUE_FEATURES,
    CodeEvaluator,
    move_features,
    value_features,
)
from .game import Tactics
from .lessons import Lessons, positions_of
from .players import GreedyPlayer

_MATERIAL = np.array([j for j, name in enumerate(VALUE_FEATURES) if name.startswith("hp_")])
"""The columns of the value's baseline: units by kind."""
_VERBS = np.array(
    [
        j
        for j, name in enumerate(MOVE_FEATURES)
        if name.startswith("is_") or (name.startswith("place_") and "ahead" not in name)
    ]
)
"""The columns of the prior's baseline: what kind of action it is, and nothing about where."""
_LOGIT_EDGE = 1e-6
"""How close to 0 or 1 a value counts, in a score of who won."""
OVERRULED = ("hp_knight",)
"""The value's weights taken from `HAND_PICKED` in the constants that play. The teacher's value is
flat in a knight's lead, though the side ahead goes on to win more often, and a fit copies that: on
the prior of a fit, this one weight is the difference between 35% and 51% against greedy."""
LOOKAHEAD = 4
"""The unit turns the constants that play look ahead (`CodeEvaluator`'s ``lookahead``). The fit is
of positions as they stand, and what a static value lacks is what the next few turns do. Under fog
on 15x15, against greedy over 192 games, greedy deploying: at 128 simulations the same weights
scored 111-81 as the position stands, 131-61 read 3 turns ahead and 137-55 read 4; at 64
simulations 102-90 read 2 turns ahead and 138-54 read 4, in about half a second a unit's turn.
Refitting the weights to the positions ahead changed nothing (133-59)."""


class Cases(NamedTuple):
    """What fits read of games: ``g`` games, ``d`` decisions, ``m`` moves, ``v`` judged positions.

    Each decision's moves lie together, in the order of ``np.flatnonzero(legal)``.
    """

    games: int
    """How many games, ``g``."""
    game: NDArray[np.int64]
    """Each decision's game, shape (d,)."""
    moves: NDArray[np.float64]
    """Every legal move's features, shape (m, len(MOVE_FEATURES))."""
    start: NDArray[np.int64]
    """Where each decision's moves start, and their end, shape (d + 1,)."""
    chosen: NDArray[np.int64]
    """Each decision's teacher's move, as a row of ``moves``, shape (d,)."""
    greedy: NDArray[np.int64]
    """What `GreedyPlayer` would have done there, as a row of ``moves``, shape (d,)."""
    value_game: NDArray[np.int64]
    """Each judged position's game, shape (v,)."""
    values: NDArray[np.float64]
    """Each judged position's features, shape (v, len(VALUE_FEATURES))."""
    target: NDArray[np.float64]
    """The teacher's value of each for team 0, in [0, 1], shape (v,)."""
    searched: NDArray[np.bool_]
    """Whether the teacher searched it, rather than it being a deployment, shape (v,)."""
    won: NDArray[np.float64]
    """1 where team 0 went on to win, shape (v,)."""


class ValueScore(NamedTuple):
    """How a value does on positions the teacher searched, in games kept out of the fit."""

    correlation: float
    """Its correlation with the teacher's value; 0 if either is constant."""
    error: float
    """Its root mean squared difference from the teacher's value."""
    winner_loss: float
    """Its log loss on who went on to win, in nats a position. A coin is 0.693."""


class PriorScore(NamedTuple):
    """How a prior does at decisions with a choice, in games kept out of the fit."""

    agreement: float
    """How often its favorite is the teacher's action, ties counted by their share."""
    loss: float
    """Its log loss on the teacher's action, in nats a decision; nan without probabilities."""


class FitScore(NamedTuple):
    """How a fit and its baselines do, by name."""

    games: int
    """Games in the fit."""
    games_out: int
    """Games kept out."""
    value: dict[str, ValueScore]
    """``mean``, ``material``, ``hand``, ``fit``, and ``teacher``: its own value's winner loss."""
    prior: dict[str, PriorScore]
    """``uniform``, ``verbs``, ``hand``, ``greedy``, and ``fit``."""


def cases_of(game: Tactics, lessons: Lessons, *, workers: int) -> Cases:
    """Replay games, and compute what fits read of them.

    Parameters
    ----------
    game : Tactics
        The rules they were played by.
    lessons : Lessons
        The games.
    workers : int
        How many processes.

    Returns
    -------
    Cases
    --------
        Their cases, game by game.
    """
    context = multiprocessing.get_context("spawn")  # as `game.series`
    one_game = [lessons.only(i) for i in range(len(lessons))]
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        parts = list(pool.map(game_cases, [game] * len(lessons), one_game, chunksize=8))
    return join_cases(parts)


def game_cases(game: Tactics, lessons: Lessons) -> Cases:
    """Replay one game, and compute what fits read of it.

    Parameters
    ----------
    game : Tactics
        The rules it was played by.
    lessons : Lessons
        The game alone.

    Returns
    -------
    Cases
    --------
        Its cases, as game 0 of 1.
    """
    rng = np.random.default_rng(int(lessons.seed[0]))
    greedy = GreedyPlayer()
    searched = {int(at): s for s, at in enumerate(lessons.searched)}
    moves, sizes, chosen, followed = [], [], [], []
    judged, target, was_searched = [], [], []
    first_value = float(lessons.value[0]) if lessons.value.size else None
    for d, (position, action) in enumerate(positions_of(game, lessons, 0)):
        legal = game.legal(position)
        flat = np.flatnonzero(legal)
        n_cells = legal.shape[1]
        moves.append(move_features(game, position, legal))
        sizes.append(flat.size)
        chosen.append(int(np.searchsorted(flat, action.verb * n_cells + action.cell)))
        rival = greedy(game, game.player(position), position, legal, rng)
        followed.append(int(np.searchsorted(flat, rival.verb * n_cells + rival.cell)))
        if d in searched:
            judged.append(position)
            target.append(float(lessons.value[searched[d]]))
            was_searched.append(True)
        elif position.deploying and first_value is not None:
            judged.append(position)
            target.append(first_value)
            was_searched.append(False)
    start = np.cumsum([0, *sizes], dtype=np.int64)
    return Cases(
        1,
        np.zeros(len(sizes), dtype=np.int64),
        np.concatenate(moves),
        start,
        start[:-1] + np.array(chosen, dtype=np.int64),
        start[:-1] + np.array(followed, dtype=np.int64),
        np.zeros(len(judged), dtype=np.int64),
        value_features(game, judged),
        np.array(target, dtype=np.float64),
        np.array(was_searched, dtype=np.bool_),
        np.full(len(judged), float(lessons.winner[0] == 0)),
    )


def join_cases(parts: Sequence[Cases]) -> Cases:
    """Return the cases of several games as one, numbered in order.

    Parameters
    ----------
    parts : sequence of Cases
        At least one.

    Returns
    -------
    Cases
    --------
        All of them.
    """
    games = np.cumsum([0, *(part.games for part in parts[:-1])], dtype=np.int64)
    rows = np.cumsum([0, *(part.moves.shape[0] for part in parts[:-1])], dtype=np.int64)
    return Cases(
        sum(part.games for part in parts),
        np.concatenate([p.game + g for p, g in zip(parts, games, strict=True)]),
        np.concatenate([part.moves for part in parts]),
        np.concatenate(
            [parts[0].start[:1], *(p.start[1:] + r for p, r in zip(parts, rows, strict=True))]
        ),
        np.concatenate([p.chosen + r for p, r in zip(parts, rows, strict=True)]),
        np.concatenate([p.greedy + r for p, r in zip(parts, rows, strict=True)]),
        np.concatenate([p.value_game + g for p, g in zip(parts, games, strict=True)]),
        np.concatenate([part.values for part in parts]),
        np.concatenate([part.target for part in parts]),
        np.concatenate([part.searched for part in parts]),
        np.concatenate([part.won for part in parts]),
    )


def fit(cases: Cases, seed: int) -> tuple[CodeEvaluator, FitScore]:
    """Fit an evaluator to cases, and score it and its baselines on games kept out.

    Parameters
    ----------
    cases : Cases
        What to fit.
    seed : int
        Sets which games are kept out.

    Returns
    -------
    evaluator : CodeEvaluator
        The fit.
    score : FitScore
        How it and its baselines do on the games kept out.

    Raises
    ------
    ValueError
        If there are fewer than two games: one to fit to, one to keep out.
    """
    if cases.games < 2:
        raise ValueError(f"a fit needs at least 2 games, got {cases.games}")
    out = held_out(np.arange(cases.games, dtype=np.int64), seed)
    value, intercept = fit_value(cases, ~out, np.arange(len(VALUE_FEATURES)))
    prior = fit_prior(cases, ~out, np.arange(len(MOVE_FEATURES)))
    evaluator = CodeEvaluator(value, intercept, prior)
    return evaluator, FitScore(
        int((~out).sum()),
        int(out.sum()),
        _value_scores(cases, out, evaluator),
        _prior_scores(cases, out, evaluator),
    )


def overruled(fitted: CodeEvaluator) -> CodeEvaluator:
    """Return a fit with the `OVERRULED` weights taken from `HAND_PICKED`.

    Parameters
    ----------
    fitted : CodeEvaluator
        A fit.

    Returns
    -------
    CodeEvaluator
        The same but for the weights of `OVERRULED`.
    """
    value = tuple(
        HAND_PICKED.value[j] if name in OVERRULED else fitted.value[j]
        for j, name in enumerate(VALUE_FEATURES)
    )
    return dataclasses.replace(fitted, value=value)


def playing(fitted: CodeEvaluator) -> CodeEvaluator:
    """Return a fit's constants for playing: `overruled`, and looking `LOOKAHEAD` turns ahead.

    Parameters
    ----------
    fitted : CodeEvaluator
        A fit.

    Returns
    -------
    CodeEvaluator
        The fit's constants but for the weights of `OVERRULED`, its value read `LOOKAHEAD` unit
        turns ahead.
    """
    return dataclasses.replace(overruled(fitted), lookahead=LOOKAHEAD)


def fit_value(
    cases: Cases, games: NDArray[np.bool_], columns: NDArray[np.intp]
) -> tuple[tuple[float, ...], float]:
    """Return a value's weights on some features and its intercept, fit to some games.

    Parameters
    ----------
    cases : Cases
        What to fit.
    games : ndarray of bool, shape (g,)
        The games to fit to.
    columns : ndarray of intp, shape (k,)
        The features to use; the others get a weight of 0.

    Returns
    -------
    weights : tuple of float
        A weight per name in `VALUE_FEATURES`.
    intercept : float
        The value's score when every feature is 0, in logits.
    """
    rows = games[cases.value_game]
    model = fit_logistic(cases.values[rows][:, columns], cases.target[rows])
    weights = np.zeros(len(VALUE_FEATURES))
    weights[columns] = model.coef_[0]
    return tuple(float(w) for w in weights), float(model.intercept_[0])


def fit_prior(
    cases: Cases, games: NDArray[np.bool_], columns: NDArray[np.intp]
) -> tuple[float, ...]:
    """Return a prior's weights on some features, fit to some games' decisions with a choice.

    Parameters
    ----------
    cases : Cases
        What to fit.
    games : ndarray of bool, shape (g,)
        The games to fit to.
    columns : ndarray of intp, shape (k,)
        The features to use; the others get a weight of 0.

    Returns
    -------
    tuple of float
        A weight per name in `MOVE_FEATURES`.
    """
    sizes = np.diff(cases.start)
    used = games[cases.game] & (sizes > 1)
    rows = np.repeat(used, sizes)
    target = np.zeros(cases.moves.shape[0])
    target[cases.chosen[used]] = 1.0
    first = np.cumsum(np.r_[0, sizes[used][:-1]], dtype=np.int64)
    found = fit_choices(cases.moves[rows][:, columns], target[rows], first)
    weights = np.zeros(len(MOVE_FEATURES))
    weights[columns] = found
    return tuple(float(w) for w in weights)


def _value_scores(
    cases: Cases, out: NDArray[np.bool_], evaluator: CodeEvaluator
) -> dict[str, ValueScore]:
    """Return how the fit's value and its baselines do on the searched positions of games out."""
    rows = out[cases.value_game] & cases.searched
    mean = cases.target[~out[cases.value_game] & cases.searched].mean()
    material, at_zero = fit_value(cases, ~out, _MATERIAL)
    target, won = cases.target[rows], cases.won[rows]
    x = cases.values[rows]

    def score(predicted: NDArray[np.float64]) -> ValueScore:
        return value_score(predicted, target, won)

    return {
        "mean": score(np.full(len(target), float(mean))),
        "material": score(sigmoid(weighted(x, material, at_zero))),
        "hand": score(sigmoid(weighted(x, HAND_PICKED.value, HAND_PICKED.intercept))),
        "fit": score(sigmoid(weighted(x, evaluator.value, evaluator.intercept))),
        "teacher": score(target),
    }


def value_score(
    predicted: NDArray[np.float64], target: NDArray[np.float64], won: NDArray[np.float64]
) -> ValueScore:
    """Return how values do against the teacher's and against who won.

    Parameters
    ----------
    predicted : ndarray of float64, shape (n,)
        The values, in [0, 1].
    target : ndarray of float64, shape (n,)
        The teacher's values.
    won : ndarray of float64, shape (n,)
        1 where team 0 went on to win, else 0.

    Returns
    -------
    ValueScore
        The score; nan where there are no positions to measure.
    """
    if not len(target):
        return ValueScore(0.0, math.nan, math.nan)
    p = np.clip(predicted, _LOGIT_EDGE, 1.0 - _LOGIT_EDGE)
    loss = -(won * np.log(p) + (1.0 - won) * np.log(1.0 - p)).mean()
    error = float(np.sqrt(((predicted - target) ** 2).mean()))
    return ValueScore(correlation(predicted, target), error, float(loss))


def _prior_scores(
    cases: Cases, out: NDArray[np.bool_], evaluator: CodeEvaluator
) -> dict[str, PriorScore]:
    """Return how the fit's prior and its baselines do on the decisions of games out."""
    decisions = out[cases.game] & (np.diff(cases.start) > 1)
    verbs = fit_prior(cases, ~out, _VERBS)
    chosen, greedy = cases.chosen[decisions], cases.greedy[decisions]
    return {
        "uniform": prior_score(cases, (0.0,) * len(MOVE_FEATURES), decisions),
        "verbs": prior_score(cases, verbs, decisions),
        "hand": prior_score(cases, HAND_PICKED.prior, decisions),
        "greedy": PriorScore(float((chosen == greedy).mean()) if chosen.size else 0.0, math.nan),
        "fit": prior_score(cases, evaluator.prior, decisions),
    }


def prior_score(
    cases: Cases, weights: tuple[float, ...], decisions: NDArray[np.bool_]
) -> PriorScore:
    """Return how a prior's weights do at some decisions.

    Parameters
    ----------
    cases : Cases
        The decisions and their moves.
    weights : tuple of float
        A weight per name in `MOVE_FEATURES`.
    decisions : ndarray of bool, shape (d,)
        The decisions to score.

    Returns
    -------
    PriorScore
        The score; agreement 0 and a loss of nan with no decisions to score.
    """
    if not decisions.any():
        return PriorScore(0.0, math.nan)
    logits = cases.moves @ np.asarray(weights)
    first = cases.start[:-1]
    best = np.maximum.reduceat(logits, first)
    tied = np.add.reduceat(logits == np.repeat(best, np.diff(cases.start)), first)
    picked = logits[cases.chosen]
    agreement = ((picked == best) / tied)[decisions].mean()
    loss = (np.logaddexp.reduceat(logits, first) - picked)[decisions].mean()
    return PriorScore(float(agreement), float(loss))
