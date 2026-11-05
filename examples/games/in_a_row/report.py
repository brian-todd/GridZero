"""A report on a code player: where it and its teacher disagree, and who is right.

`analyze` measures, on the lessons a fit came from: how each fit does on each round's positions;
whether a search steers into its own value's errors; how far the teacher's labels can be
trusted, against a deeper search and against exact values; which candidate features a fit lacks;
and the positions of the widest disagreements. `page` draws that as one HTML file, with no
outside dependency: boards are inline SVG from the lattice's own cell polygons.
"""

import html
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import Evaluator
from gridzero.agents.exact import solve

from .code_player import (
    MOVE_FEATURES,
    VALUE_FEATURES,
    FittedEvaluator,
    clean_threats,
    fitted_values,
    threat_cells,
    value_features,
)
from .fit import correlation, fit_logistic, held_out
from .game import PLACE, InARow
from .lessons import Lessons, judge, position_of
from .state import InARowPosition

PLAYED_AT = 64
"""The code player's simulations a decision, where the report judges its moves."""
TAUGHT_AT = 256
"""The simulations of the search that judged the lessons."""
REFEREE_AT = 2048
"""The simulations of the deeper search that referees the teacher's own labels."""
SAMPLE = 2000
"""Positions a round per fit in the steering test."""
REFEREE_SAMPLE = 300
"""Positions a round the deeper search judges."""
TRUTH_FROM = 26
"""Exact values are solved from this ply on: from about here, seconds a position at most."""
TRUTH_SAMPLE = 400
"""Positions a round solved exactly."""
TRUTH_IN = 8
"""Positions a worker solves at a time."""
PLY_BUCKETS = ((4, 13), (13, 21), (21, 29), (29, 43))
"""The ranges of plies the tables and galleries are split into: low included, high not."""
MOST_FROM_A_GAME = 2
"""A gallery's cards from any one game, at most: neighbors in a game are near repeats."""
PLAY_PORT = 8000
"""The port the play page serves on by default, where the report's cards link to."""


Candidate = Callable[[InARow, NDArray[np.int8], NDArray[np.int64]], NDArray[np.float64]]
"""A feature a fit lacks: owners, shape (n, n_cells), and plies, shape (n,), to shape (n,)."""


def _threat_grid(
    game: InARow, owner: NDArray[np.int8]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return each player's threat cells, each of shape (n, columns, rows)."""
    threats = threat_cells(game, owner)
    grid = threats.reshape(len(owner), 2, *game.shape)
    return grid[:, 0], grid[:, 1]


def stacked(game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return threats directly above one's own.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        Player 0's count minus player 1's.
    """
    first, second = _threat_grid(game, owner)
    return (
        (first[:, :, 1:] & first[:, :, :-1]).sum(axis=(1, 2))
        - (second[:, :, 1:] & second[:, :, :-1]).sum(axis=(1, 2))
    ).astype(np.float64)


def parity_holder(
    game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return who alone holds a clean threat of its parity.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        1 if only the first player does, -1 if only the second, else 0.
    """
    first, second = _threat_grid(game, owner)
    holds_first = clean_threats(first, second)[:, :, ::2].any(axis=(1, 2))
    holds_second = clean_threats(second, first)[:, :, 1::2].any(axis=(1, 2))
    return holds_first.astype(np.float64) - holds_second.astype(np.float64)


def late_threats(
    game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the lead in threats, weighed by how full the board is: a lead late counts more.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        Player 0's lead in threats times the share of the board filled.
    """
    first, second = _threat_grid(game, owner)
    lead = first.sum(axis=(1, 2)) - second.sum(axis=(1, 2))
    return lead * ply / game.space.n_cells


def one_ready(game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return who has one win ready while the player to move has none: a forced reply.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        1 if that is player 0, -1 if player 1, else 0.
    """
    n = len(owner)
    first, second = _threat_grid(game, owner)
    heights = (owner.reshape(n, *game.shape) >= 0).sum(axis=2)
    i, column = np.nonzero(heights < game.shape[1])
    ready = np.zeros((n, *game.shape), dtype=bool)
    ready[i, column, heights[i, column]] = True
    counts = (first & ready).sum(axis=(1, 2)), (second & ready).sum(axis=(1, 2))
    first_to_move = ply % 2 == 0
    mover = np.where(first_to_move, counts[0], counts[1])
    other = np.where(first_to_move, counts[1], counts[0])
    toward_other = np.where(first_to_move, -1.0, 1.0)
    return np.where((other == 1) & (mover == 0), toward_other, 0.0)


def over_ready(
    game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return threats just above a playable cell.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        Player 0's count minus player 1's.
    """
    n = len(owner)
    first, second = _threat_grid(game, owner)
    heights = (owner.reshape(n, *game.shape) >= 0).sum(axis=2)
    i, column = np.nonzero(heights + 1 < game.shape[1])
    above = np.zeros((n, *game.shape), dtype=bool)
    above[i, column, heights[i, column] + 1] = True
    return ((first & above).sum(axis=(1, 2)) - (second & above).sum(axis=(1, 2))).astype(np.float64)


CANDIDATES: dict[str, tuple[Candidate, str]] = {
    "stacked": (stacked, "threats directly above one's own: two wins in one column"),
    "parity_holder": (parity_holder, "only one player holds a clean threat of its parity"),
    "late_threats": (late_threats, "lead in threats, weighed by how full the board is"),
    "one_ready": (one_ready, "the other has one win ready, the mover none: a forced reply"),
    "over_ready": (over_ready, "threats just above a playable cell: no one wants to play under"),
}
"""Features a fit doesn't have, measured but not played: name, function, what it counts."""


class Ranked(NamedTuple):
    """How a fit's value does on some positions, against the teacher's."""

    correlation: float
    """Correlation of its value with the teacher's."""
    mse: float
    """Mean squared difference."""
    n: int
    """Positions."""


class Steering(NamedTuple):
    """Whether a search overrates the moves it picks more than moves in general."""

    label: str
    """Whose search, on whose positions."""
    n: int
    """Positions."""
    chosen: float
    """Its q of the move it picks, minus the teacher's q of that move, for the mover."""
    every: float
    """The same, averaged over every move."""
    extra: float
    """``chosen - every``: how much picking a move adds to overrating it."""
    error: float
    """The standard error of ``extra``."""


class LabelCheck(NamedTuple):
    """How a round's labels and its fit hold up against a deeper search."""

    source: int
    """The round."""
    n: int
    """Positions."""
    teacher: float
    """Correlation of the teacher's labels with the deeper search."""
    fit: float
    """Correlation of the final fit's value with the deeper search."""
    moved: float
    """Mean absolute change of the label when searched deeper."""


class TruthCheck(NamedTuple):
    """How a round's labels and its fit hold up against exact values."""

    source: int
    """The round."""
    n: int
    """Positions solved."""
    decisive: float
    """The share of them won by a player under perfect play."""
    contested: float
    """The share of them where some move loses value: the only ones a move can get wrong."""
    teacher: float
    """Correlation of the teacher's labels with exact values."""
    fit: float
    """Correlation of the final fit's value with exact values."""
    teacher_side: float
    """How often the teacher's label is on the winner's side of 1/2, of decisive positions."""
    fit_side: float
    """How often the fit's is."""
    teacher_move: float
    """How often the teacher's move, at `TAUGHT_AT`, keeps the exact value, where contested."""
    fit_move: float
    """How often the fit's move, at `TAUGHT_AT`, does."""
    fit_move_played: float
    """How often the fit's move, at `PLAYED_AT`, does."""


class ByPly(NamedTuple):
    """Where in a game the fit's value and the teacher's part."""

    low: int
    """First ply."""
    high: int
    """Past the last ply."""
    n: int
    """Positions."""
    mean_absolute: float
    """Mean absolute difference, fit minus teacher."""
    mean_signed: float
    """Mean difference, toward player 0."""
    regret: float
    """What the code player's move loses by the teacher's q, for the mover."""


class CandidateScore(NamedTuple):
    """What a candidate feature would add to the value's fit, on games kept out."""

    name: str
    """Its name."""
    meaning: str
    """What it counts."""
    present: float
    """The share of positions where it isn't 0."""
    weight: float
    """Its fitted weight, in logits."""
    gain: float
    """The correlation gained on every kept-out position."""
    gain_last: float
    """The correlation gained on the last round's kept-out positions."""


class ValueCard(NamedTuple):
    """A position where the fit's value and the teacher's disagree."""

    owner: NDArray[np.int8]
    """Who owns each cell, shape (n_cells,)."""
    ply: int
    """Its ply."""
    source: int
    """The round that reached it."""
    fit: float
    """The fit's value for player 0."""
    teacher: float
    """The teacher's label."""
    referee: float
    """A deeper search's value."""
    truth: float
    """The exact value; NaN if not solved."""
    reasons: tuple[tuple[str, float], ...]
    """The features that moved the fit's value most, with their logits toward player 0."""
    tags: tuple[str, ...]
    """The candidate features that hold here."""


class MoveCard(NamedTuple):
    """A position where the code player's move loses most by the teacher's reckoning."""

    owner: NDArray[np.int8]
    """Who owns each cell, shape (n_cells,)."""
    ply: int
    """Its ply."""
    source: int
    """The round that reached it."""
    played: int
    """The code player's column."""
    best: int
    """The teacher's best column."""
    q: NDArray[np.float64]
    """The teacher's q by column, for the mover, shape (n_columns,); NaN if not legal."""
    referee: NDArray[np.float64]
    """A deeper search's q by column, for the mover, shape (n_columns,)."""
    truth: NDArray[np.float64]
    """Exact values by column, for the mover, shape (n_columns,); NaN if not solved."""
    regret: float
    """What the move loses by the teacher's q."""


@dataclass(frozen=True, eq=False)
class Report:
    """What `analyze` measured; `page` draws it."""

    referee_at: int
    """The simulations of the deeper search."""
    truth_from: int
    """The ply from which positions are solved exactly."""
    fits: tuple[FittedEvaluator, ...]
    """Every round's fit, in order."""
    positions: tuple[int, ...]
    """The positions of each round."""
    plies: tuple[float, ...]
    """The mean ply of each round."""
    by_source: tuple[tuple[Ranked, ...], ...]
    """Each fit, in order, on each round's positions, in order."""
    steering: tuple[Steering, ...]
    """The steering test."""
    labels: tuple[LabelCheck, ...]
    """The labels against a deeper search, by round."""
    truth: tuple[TruthCheck, ...]
    """The labels and the fit against exact values, by round."""
    by_ply: tuple[ByPly, ...]
    """The final fit by ply."""
    candidates: tuple[CandidateScore, ...]
    """What candidate features would add, the most useful first."""
    held_out_correlation: tuple[float, float]
    """The final constants' features refit without candidates, kept out: all, last round."""
    value_cards: tuple[ValueCard, ...]
    """The widest value disagreements."""
    move_cards: tuple[MoveCard, ...]
    """The costliest moves."""


def _mover(q: NDArray[np.float64], ply: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return q for the player to move, from player 0's, shape as ``q``, by plies of shape (n,)."""
    return np.where((ply % 2 == 0)[:, None], q, 1.0 - q)


def _sample(rows: NDArray[np.bool_], n: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Return up to ``n`` of the true rows, at random, ascending."""
    found = np.flatnonzero(rows)
    return np.sort(rng.choice(found, size=min(n, len(found)), replace=False))


def _judge_rows(
    game: InARow,
    evaluator: Evaluator[InARow, InARowPosition],
    lessons: Lessons,
    rows: NDArray[np.int64],
    simulations: int,
    seed: int,
    workers: int,
) -> Lessons:
    """Return what a search makes of some of the lessons' positions."""
    return judge(
        game,
        evaluator,
        lessons.owner[rows],
        lessons.ply[rows],
        lessons.game[rows],
        simulations=simulations,
        seed=seed,
        workers=workers,
    )


def steering(label: str, searched: Lessons, taught_q: NDArray[np.float64]) -> Steering:
    """Measure how much a search overrates the moves it picks, against a teacher's q.

    Parameters
    ----------
    label : str
        Whose search, on whose positions.
    searched : Lessons
        What the search made of the positions.
    taught_q : ndarray of float64, shape (n, n_columns)
        The teacher's q of the same positions, for player 0.

    Returns
    -------
    Steering
        The test. Positions where the picked move wasn't visited by both are left out.
    """
    difference = _mover(searched.q, searched.ply) - _mover(taught_q, searched.ply)
    both = np.isfinite(difference)
    row = np.arange(len(searched))
    keep = both[row, searched.move] & (both.sum(axis=1) > 1)
    chosen = difference[row[keep], searched.move[keep]]
    every = np.where(both, difference, 0.0)[keep].sum(axis=1) / both[keep].sum(axis=1)
    extra = chosen - every
    error = float(extra.std() / np.sqrt(len(extra))) if len(extra) else 0.0
    return Steering(
        label,
        int(keep.sum()),
        float(chosen.mean()) if len(extra) else 0.0,
        float(every.mean()) if len(extra) else 0.0,
        float(extra.mean()) if len(extra) else 0.0,
        error,
    )


def exact_values(
    game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64], workers: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return positions' exact values under perfect play, with `agents.exact`.

    Parameters
    ----------
    game : InARow
        The rules, with gravity.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell of each position, none over.
    ply : ndarray of int64, shape (n,)
        Each position's ply.
    workers : int, default 1
        How many processes solve.

    Returns
    -------
    values : ndarray of float64, shape (n,)
        Player 0's value of each: 0, 0.5 or 1.
    by_column : ndarray of float64, shape (n, n_columns)
        Player 0's value after a stone in each column; NaN where a column is full.
    """
    jobs = [
        _Solve(game, owner[i : i + TRUTH_IN], ply[i : i + TRUTH_IN])
        for i in range(0, len(ply), TRUTH_IN)
    ]
    if not jobs:
        return np.zeros(0), np.zeros((0, game.shape[0]))
    if workers == 1:
        parts = [job() for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")  # as `game.series`
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=context) as pool:
            parts = list(pool.map(_Solve.__call__, jobs))
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
    )


@dataclass(frozen=True)
class _Solve:
    """Some positions to solve, here or in another process."""

    game: InARow
    owner: NDArray[np.int8]
    ply: NDArray[np.int64]

    def __call__(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the values, shape (n,), and the values by column, shape (n, n_columns)."""
        game, n = self.game, len(self.ply)
        values = np.zeros(n)
        by_column = np.full((n, game.shape[0]), np.nan)
        for i in range(n):
            position = position_of(game, self.owner[i], int(self.ply[i]))
            solved = solve(game, position)
            values[i] = solved[position.key()]
            for cell in np.flatnonzero(game.legal(position)[PLACE]):
                after = game.step(position, (PLACE, int(cell)))
                by_column[i, game.space.unravel(cell)[0]] = solved[after.key()]
        return values, by_column


def _contested(by_column: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return which positions have a move that loses value, by exact values, shape (n,)."""
    if len(by_column) == 0:
        return np.zeros(0, dtype=bool)
    return np.nanmax(by_column, axis=1) > np.nanmin(by_column, axis=1)


def _keeps_value(
    by_column: NDArray[np.float64], ply: NDArray[np.int64], move: NDArray[np.int64]
) -> float:
    """Return how often a move keeps the exact value, where some move would lose it."""
    rows = _contested(by_column)
    if not rows.any():
        return 0.0
    mover = _mover(by_column[rows], ply[rows])
    picked = mover[np.arange(len(mover)), move[rows]]
    return float((picked == np.nanmax(mover, axis=1)).mean())


def _on_side(value: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """Return how often a value is on the winner's side of 1/2, of decisive positions."""
    decisive = truth != 0.5
    if not decisive.any():
        return 0.0
    return float(((value[decisive] > 0.5) == (truth[decisive] == 1.0)).mean())


def _pick(score: NDArray[np.float64], lessons: Lessons, game: InARow, shown: int) -> list[int]:
    """Return the rows of a gallery: the highest scores per ply bucket, without near repeats.

    A board and its mirror count once, and any game gives at most `MOST_FROM_A_GAME`.
    """
    picked: list[int] = []
    for low, high in PLY_BUCKETS:
        rows = np.flatnonzero((lessons.ply >= low) & (lessons.ply < high) & np.isfinite(score))
        seen: set[bytes] = set()
        from_game: dict[int, int] = {}
        taken = 0
        for row in rows[np.argsort(-score[rows], kind="stable")]:
            board = lessons.owner[row].reshape(game.shape)
            key = min(board.tobytes(), board[::-1].tobytes())
            number = int(lessons.game[row])
            if key in seen or from_game.get(number, 0) >= MOST_FROM_A_GAME:
                continue
            seen.add(key)
            from_game[number] = from_game.get(number, 0) + 1
            picked.append(int(row))
            taken += 1
            if taken == shown:
                break
    return picked


def analyze(
    game: InARow,
    teacher: Evaluator[InARow, InARowPosition],
    fits: Sequence[FittedEvaluator],
    lessons: Lessons,
    *,
    shown: int,
    seed: int,
    referee_at: int = REFEREE_AT,
    truth_from: int = TRUTH_FROM,
    workers: int = 1,
) -> Report:
    """Measure a code player against its teacher, on the lessons it came from.

    Parameters
    ----------
    game : InARow
        The rules: four in a row with gravity.
    teacher : Evaluator
        The teacher.
    fits : sequence of FittedEvaluator
        Each round's fit, in order: the last is the one the galleries judge.
    lessons : Lessons
        Every round's lessons.
    shown : int
        Cards per ply bucket in each gallery.
    seed : int
        Sets the samples, the games kept out and the searches' ties.
    referee_at : int, default `REFEREE_AT`
        The simulations of the deeper search.
    truth_from : int, default `TRUTH_FROM`
        The ply exact values are solved from.
    workers : int, default 1
        How many processes search and solve.

    Returns
    -------
    Report
        What `page` draws.
    """
    rng = np.random.default_rng(seed)
    final = fits[-1]
    rounds = sorted(int(s) for s in np.unique(lessons.source))
    late = lessons.ply >= PLY_BUCKETS[0][0]

    by_source = []
    for fit in fits:
        predicted = fitted_values(game, fit, lessons.owner, lessons.ply)
        row = []
        for source in rounds:
            rows = late & (lessons.source == source)
            row.append(
                Ranked(
                    correlation(predicted[rows], lessons.value[rows]),
                    float(np.mean((predicted[rows] - lessons.value[rows]) ** 2)),
                    int(rows.sum()),
                )
            )
        by_source.append(tuple(row))

    tests = []
    for k, fit in enumerate(fits):
        for source in rounds:
            rows = _sample(lessons.source == source, SAMPLE, rng)
            searched = _judge_rows(game, fit, lessons, rows, PLAYED_AT, seed, workers)
            mark = " (the positions it chose)" if source == k + 1 else ""
            tests.append(steering(f"fit {k} on round {source}{mark}", searched, lessons.q[rows]))
    rows = _sample(lessons.source == rounds[-1], SAMPLE, rng)
    shallow = _judge_rows(game, teacher, lessons, rows, PLAYED_AT, seed, workers)
    tests.append(
        steering(
            f"teacher at {PLAYED_AT} on round {rounds[-1]} (a baseline: no fit)",
            shallow,
            lessons.q[rows],
        )
    )

    labels, truths = [], []
    for source in rounds:
        rows = _sample(late & (lessons.source == source), REFEREE_SAMPLE, rng)
        deeper = _judge_rows(game, teacher, lessons, rows, referee_at, seed, workers)
        predicted = fitted_values(game, final, lessons.owner[rows], lessons.ply[rows])
        labels.append(
            LabelCheck(
                source,
                len(rows),
                correlation(lessons.value[rows], deeper.value),
                correlation(predicted, deeper.value),
                float(np.abs(lessons.value[rows] - deeper.value).mean()) if len(rows) else 0.0,
            )
        )
        rows = _sample((lessons.ply >= truth_from) & (lessons.source == source), TRUTH_SAMPLE, rng)
        exact, by_column = exact_values(game, lessons.owner[rows], lessons.ply[rows], workers)
        predicted = fitted_values(game, final, lessons.owner[rows], lessons.ply[rows])
        same_budget = _judge_rows(game, final, lessons, rows, TAUGHT_AT, seed, workers)
        played = _judge_rows(game, final, lessons, rows, PLAYED_AT, seed, workers)
        truths.append(
            TruthCheck(
                source,
                len(rows),
                float((exact != 0.5).mean()) if len(rows) else 0.0,
                float(_contested(by_column).mean()) if len(rows) else 0.0,
                correlation(lessons.value[rows], exact),
                correlation(predicted, exact),
                _on_side(lessons.value[rows], exact),
                _on_side(predicted, exact),
                _keeps_value(by_column, lessons.ply[rows], lessons.move[rows]),
                _keeps_value(by_column, lessons.ply[rows], same_budget.move),
                _keeps_value(by_column, lessons.ply[rows], played.move),
            )
        )

    every_row = np.arange(len(lessons))
    played = _judge_rows(game, final, lessons, every_row, PLAYED_AT, seed, workers)
    taught = _mover(lessons.q, lessons.ply)
    picked_q = taught[every_row, played.move]
    visited = np.isfinite(taught).any(axis=1)
    best_q = np.full(len(lessons), np.nan)
    best_q[visited] = np.nanmax(taught[visited], axis=1)
    regret = best_q - picked_q  # NaN where the teacher didn't visit the move

    predicted = fitted_values(game, final, lessons.owner, lessons.ply)
    residual = predicted - lessons.value
    by_ply = []
    for low, high in PLY_BUCKETS:
        rows = (lessons.ply >= low) & (lessons.ply < high)
        finite = rows & np.isfinite(regret)
        by_ply.append(
            ByPly(
                low,
                high,
                int(rows.sum()),
                float(np.abs(residual[rows]).mean()) if rows.any() else 0.0,
                float(residual[rows].mean()) if rows.any() else 0.0,
                float(regret[finite].mean()) if finite.any() else 0.0,
            )
        )

    candidates, base = _score_candidates(game, lessons, seed)
    value_rows = _pick(np.abs(residual), lessons, game, shown)
    value_cards = _value_cards(
        game, teacher, final, lessons, value_rows, seed, referee_at, truth_from, workers
    )
    move_rows = _pick(regret, lessons, game, shown)
    move_cards = _move_cards(
        game,
        teacher,
        lessons,
        move_rows,
        played.move,
        regret,
        seed,
        referee_at,
        truth_from,
        workers,
    )
    return Report(
        referee_at,
        truth_from,
        tuple(fits),
        tuple(int((lessons.source == s).sum()) for s in rounds),
        tuple(float(lessons.ply[lessons.source == s].mean()) for s in rounds),
        tuple(by_source),
        tuple(tests),
        tuple(labels),
        tuple(truths),
        tuple(by_ply),
        candidates,
        base,
        tuple(value_cards),
        tuple(move_cards),
    )


def _score_candidates(
    game: InARow, lessons: Lessons, seed: int
) -> tuple[tuple[CandidateScore, ...], tuple[float, float]]:
    """Return what each candidate adds to a refit of the value, and the refit's own correlation."""
    late = lessons.ply >= PLY_BUCKETS[0][0]
    owner, ply, target = lessons.owner[late], lessons.ply[late], lessons.value[late]
    features = value_features(game, owner, ply)
    test = held_out(lessons.game[late], seed)
    last = lessons.source[late] == lessons.source.max()

    def score(x: NDArray[np.float64]) -> tuple[float, float, NDArray[np.float64]]:
        model = fit_logistic(x[~test], target[~test])
        predicted = model.predict_proba(x)[:, 1]
        return (
            correlation(predicted[test], target[test]),
            correlation(predicted[test & last], target[test & last]),
            model.coef_[0],
        )

    base_all, base_last, _ = score(features)
    found = []
    for name, (function, meaning) in CANDIDATES.items():
        extra = function(game, owner, ply)
        with_all, with_last, coefficients = score(np.column_stack([features, extra]))
        found.append(
            CandidateScore(
                name,
                meaning,
                float((extra != 0.0).mean()),
                float(coefficients[-1]),
                with_all - base_all,
                with_last - base_last,
            )
        )
    found.sort(key=lambda c: -c.gain)
    return tuple(found), (base_all, base_last)


def _value_cards(
    game: InARow,
    teacher: Evaluator[InARow, InARowPosition],
    final: FittedEvaluator,
    lessons: Lessons,
    rows: list[int],
    seed: int,
    referee_at: int,
    truth_from: int,
    workers: int,
) -> list[ValueCard]:
    """Return a value gallery's cards, with a referee on each and the truth where it's cheap."""
    if not rows:
        return []
    chosen = np.array(rows)
    deeper = _judge_rows(game, teacher, lessons, chosen, referee_at, seed, workers)
    exact = np.full(len(rows), np.nan)
    solvable = lessons.ply[chosen] >= truth_from
    exact[solvable] = exact_values(
        game, lessons.owner[chosen[solvable]], lessons.ply[chosen[solvable]], workers
    )[0]
    features = value_features(game, lessons.owner[chosen], lessons.ply[chosen])
    amounts = features * np.asarray(final.value)
    cards = []
    predicted = fitted_values(game, final, lessons.owner[chosen], lessons.ply[chosen])
    for i, row in enumerate(rows):
        order = np.argsort(-np.abs(amounts[i]), kind="stable")[:4]
        reasons = tuple(
            (VALUE_FEATURES[j], float(amounts[i, j])) for j in order if amounts[i, j] != 0.0
        )
        one_owner, one_ply = lessons.owner[row][None], lessons.ply[row][None]
        tags = tuple(
            name
            for name, (function, _) in CANDIDATES.items()
            if function(game, one_owner, one_ply)[0] != 0.0
        )
        cards.append(
            ValueCard(
                lessons.owner[row],
                int(lessons.ply[row]),
                int(lessons.source[row]),
                float(predicted[i]),
                float(lessons.value[row]),
                float(deeper.value[i]),
                float(exact[i]),
                reasons,
                tags,
            )
        )
    return cards


def _move_cards(
    game: InARow,
    teacher: Evaluator[InARow, InARowPosition],
    lessons: Lessons,
    rows: list[int],
    played: NDArray[np.int64],
    regret: NDArray[np.float64],
    seed: int,
    referee_at: int,
    truth_from: int,
    workers: int,
) -> list[MoveCard]:
    """Return a move gallery's cards, with a referee on each and the truth where it's cheap."""
    if not rows:
        return []
    chosen = np.array(rows)
    ply = lessons.ply[chosen]
    deeper = _judge_rows(game, teacher, lessons, chosen, referee_at, seed, workers)
    taught = _mover(lessons.q[chosen], ply)
    refereed = _mover(deeper.q, ply)
    exact = np.full((len(rows), game.shape[0]), np.nan)
    solvable = ply >= truth_from
    exact[solvable] = _mover(
        exact_values(game, lessons.owner[chosen[solvable]], ply[solvable], workers)[1],
        ply[solvable],
    )
    return [
        MoveCard(
            lessons.owner[row],
            int(lessons.ply[row]),
            int(lessons.source[row]),
            int(played[row]),
            int(np.nanargmax(taught[i])),
            taught[i],
            refereed[i],
            exact[i],
            float(regret[row]),
        )
        for i, row in enumerate(rows)
    ]


def board_svg(
    game: InARow,
    owner: NDArray[np.int8],
    marks: Sequence[tuple[int, str]] = (),
    scale: float = 22.0,
) -> str:
    """Draw a board as inline SVG, from the lattice's cell polygons, row 0 at the bottom.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n_cells,)
        Who owns each cell.
    marks : sequence of (int, str), default no marks
        Cells to ring, each with the CSS class of its ring.
    scale : float, default 22.0
        The pixels of a unit of the plane.

    Returns
    -------
    str
        An ``<svg>`` element, classed by the page's stylesheet.
    """
    shapes = board_shapes(game, scale)
    parts = [
        f'<svg class="board" viewBox="0 0 {shapes.width:.0f} {shapes.height:.0f}" '
        f'width="{shapes.width:.0f}" height="{shapes.height:.0f}" role="img" aria-label="board">'
    ]
    for polygon in shapes.polygons:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
        parts.append(f'<polygon class="cell" points="{points}"/>')
    for cell in np.flatnonzero(owner >= 0):
        stone = "first" if owner[cell] == 0 else "second"
        parts.append(
            f'<circle class="{stone}" cx="{shapes.centers[cell, 0]:.1f}" '
            f'cy="{shapes.centers[cell, 1]:.1f}" r="{shapes.radius:.1f}"/>'
        )
    for cell, kind in marks:
        parts.append(
            f'<circle class="{kind}" cx="{shapes.centers[cell, 0]:.1f}" '
            f'cy="{shapes.centers[cell, 1]:.1f}" r="{shapes.radius:.1f}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


class BoardShapes(NamedTuple):
    """A board in a picture's coordinates: pixels, y down, from the lattice's cell polygons."""

    width: float
    """The picture's width."""
    height: float
    """The picture's height."""
    polygons: NDArray[np.float64]
    """Each cell's corners, shape (n_cells, n_corners, 2)."""
    centers: NDArray[np.float64]
    """Each cell's center, shape (n_cells, 2)."""
    radius: float
    """A stone's radius."""


def board_shapes(game: InARow, scale: float) -> BoardShapes:
    """Return a board's shapes in a picture's coordinates, with row 0 at the bottom.

    Parameters
    ----------
    game : InARow
        The rules.
    scale : float
        The pixels of a unit of the plane.

    Returns
    -------
    BoardShapes
        The picture's size, and what to draw.
    """
    corners = game.space.corners()  # y up, as on the plane
    low = corners.min(axis=(0, 1))
    high = corners.max(axis=(0, 1))
    width, height = (high - low) * scale
    polygons = np.stack(
        [(corners[:, :, 0] - low[0]) * scale, (high[1] - corners[:, :, 1]) * scale], axis=2
    )
    return BoardShapes(float(width), float(height), polygons, polygons.mean(axis=1), 0.38 * scale)


def board_key(owner: NDArray[np.int8]) -> str:
    """Return a board as text, one character per cell in id order: ``.``, ``0`` or ``1``.

    Parameters
    ----------
    owner : ndarray of int8, shape (n_cells,)
        Who owns each cell, -1 for no one.

    Returns
    -------
    str
        The text, of length n_cells.
    """
    return "".join(".01"[int(who) + 1] for who in owner)


def _play_link(owner: NDArray[np.int8]) -> str:
    """Return a link to play on from a board, on the play page at its default port."""
    return f'<a href="http://localhost:{PLAY_PORT}/#board={board_key(owner)}">Play from here</a>'


STYLE = """
:root { --bg:#f7f8fa; --panel:#ffffff; --ink:#1f2328; --soft:#5b6470; --line:#d9dde3;
  --cell:#e9ecf0; --first:#c8452f; --second:#2b6cb0; --good:#2f8f4e; --bad:#b3261e; --bar:#8a94a3; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme:dark;
  --bg:#14171b; --panel:#1c2026; --ink:#e7e9ec; --soft:#a0a8b3; --line:#343a42; --cell:#2a2f36;
  --first:#e8735c; --second:#6aa9e9; --good:#5cc27f; --bad:#ef7b73; --bar:#6d7682; } }
:root[data-theme="dark"] { color-scheme:dark;
  --bg:#14171b; --panel:#1c2026; --ink:#e7e9ec; --soft:#a0a8b3; --line:#343a42; --cell:#2a2f36;
  --first:#e8735c; --second:#6aa9e9; --good:#5cc27f; --bad:#ef7b73; --bar:#6d7682; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding-block:24px 64px;
  padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size:26px; margin:0 0 4px; text-wrap:balance; }
h2 { font-size:19px; margin:36px 0 6px; text-wrap:balance; }
.reading { background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:4px 14px; margin:14px 0; } .reading li { margin:6px 0; max-width:76ch; }
.reading ul { padding-left:18px; }
p { margin:6px 0 12px; max-width:76ch; } .soft { color:var(--soft); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; margin:8px 0 4px; }
th,td { padding:4px 12px 4px 0; border-bottom:1px solid var(--line); text-align:right;
  white-space:nowrap; }
th:first-child,td:first-child { text-align:left; } th { font-weight:600; color:var(--soft); }
td:has(> .soft), td:has(> .bar) { text-align:left; white-space:normal; }
td:has(> .soft) { min-width:220px; }
.card table { width:100%; margin:6px 0 0; font-size:11px; }
.card th, .card td { padding:1px 2px; }
.bar { display:inline-block; height:9px; background:var(--bar); vertical-align:middle; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px;
  font-size:13px; line-height:1.4; }
.card b { font-weight:600; } .card ul { margin:6px 0 0; padding:0; list-style:none; }
.board { display:block; margin:6px 0; max-width:100%; height:auto; }
.cell { fill:var(--cell); stroke:var(--line); stroke-width:1; }
circle.first { fill:var(--first); } circle.second { fill:var(--second); }
circle.played { fill:none; stroke:var(--bad); stroke-width:2.5; }
circle.best { fill:none; stroke:var(--good); stroke-width:2.5; stroke-dasharray:3 2; }
.tag { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:0 4px;
  margin:2px 2px 0 0; color:var(--soft); }
.key i { display:inline-block; width:10px; height:10px; border-radius:50%; margin:0 4px 0 10px; }
"""


def _number(value: float, digits: int = 3) -> str:
    """Return a number as text, with a dash for NaN."""
    return "&ndash;" if np.isnan(value) else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return an HTML table of cells already escaped."""
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def _landing(game: InARow, owner: NDArray[np.int8], column: int) -> int | None:
    """Return the cell a stone in a column lands on, or None if the column is full."""
    height = int((owner.reshape(game.shape)[column] >= 0).sum())
    return column * game.shape[1] + height if height < game.shape[1] else None


def _to_move(ply: int) -> str:
    """Return who moves at a ply, as text."""
    return "first (red) to move" if ply % 2 == 0 else "second (blue) to move"


def _value_card(game: InARow, card: ValueCard) -> str:
    """Return a value gallery's card."""
    reasons = "".join(
        f"<li>{html.escape(name)} {amount:+.2f}</li>" for name, amount in card.reasons
    )
    tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in card.tags)
    return (
        f'<div class="card">Ply {card.ply}, round {card.source}, {_to_move(card.ply)}'
        f"{board_svg(game, card.owner)}"
        f"Fit <b>{card.fit:.2f}</b> &middot; teacher <b>{card.teacher:.2f}</b><br>"
        f"Deeper search {card.referee:.2f} &middot; exact {_number(card.truth, 1)}"
        f'<ul class="soft">{reasons}</ul>{tags}<br>{_play_link(card.owner)}</div>'
    )


def _move_card(game: InARow, card: MoveCard) -> str:
    """Return a move gallery's card."""
    marks = []
    for column, kind in ((card.best, "best"), (card.played, "played")):
        cell = _landing(game, card.owner, column)
        if cell is not None:
            marks.append((cell, kind))
    rows = [
        ("teacher", card.q),
        ("deeper", card.referee),
        ("exact", card.truth),
    ]
    table = "".join(
        f"<tr><td>{label}</td>" + "".join(f"<td>{_number(v, 2)}</td>" for v in values) + "</tr>"
        for label, values in rows
        if np.isfinite(values).any()
    )
    head = "".join(f"<th>{c + 1}</th>" for c in range(game.shape[0]))
    return (
        f'<div class="card">Ply {card.ply}, round {card.source}, {_to_move(card.ply)}'
        f"{board_svg(game, card.owner, marks)}"
        f"Played column <b>{card.played + 1}</b>, teacher's best <b>{card.best + 1}</b>, "
        f"costing <b>{card.regret:.2f}</b>"
        f'<div class="scroll"><table><tr><th></th>{head}</tr>{table}'
        f"</table></div>{_play_link(card.owner)}</div>"
    )


def _constants(fit: FittedEvaluator) -> str:
    """Return the final fit's constants, as a table with bars."""
    rows = []
    biggest = max(abs(w) for w in (*fit.value, *fit.prior, fit.intercept)) or 1.0
    for names, weights, kind in (
        (VALUE_FEATURES, fit.value, "value"),
        (("intercept",), (fit.intercept,), "value"),
        (MOVE_FEATURES, fit.prior, "prior"),
    ):
        for name, weight in zip(names, weights, strict=True):
            width = abs(weight) / biggest * 120.0
            bar = f'<span class="bar" style="width:{width:.0f}px"></span>'
            rows.append((html.escape(name), kind, f"{weight:+.3f}", bar))
    return _table(("Feature", "In", "Weight", ""), rows)


def page(game: InARow, report: Report) -> str:
    """Return the report as one HTML file.

    Parameters
    ----------
    game : InARow
        The rules.
    report : Report
        What `analyze` measured.

    Returns
    -------
    str
        The page: self-contained, in the reader's light or dark colors.
    """
    rounds = range(len(report.positions))
    final = report.fits[-1]
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f"<title>Connect Four Code Player</title><style>{STYLE}</style></head><body><main>",
        "<h1>Connect Four code player</h1>",
        f'<p class="soft">Connect Four. {sum(report.positions)} positions judged by the teacher, '
        f"{len(report.fits)} rounds of fits, "
        f"{len(final.value) + len(final.prior) + 1} constants in the last.</p>",
        '<p class="key soft">Stones:<i style="background:var(--first)"></i>first player'
        '<i style="background:var(--second)"></i>second. Values are the first player\'s, 1 a '
        "win; q in the move gallery is for the player to move. A card's link opens its board on "
        "the play page, while that is running (<code>in_a_row.play</code>).</p>",
        "<h2>Each fit on each round's positions</h2>",
        "<p>Correlation of a fit's value with the teacher's, from ply "
        f"{PLY_BUCKETS[0][0]}. Round k + 1's games are fit k's own: compare a fit on the round "
        "it chose with the same round under other fits.</p>",
    ]
    parts.append(
        _table(
            ("Fit", *(f"round {s}" for s in rounds)),
            [
                (
                    f"fit {k}",
                    *(f"{cell.correlation:.3f} (MSE {cell.mse:.3f})" for cell in row),
                )
                for k, row in enumerate(report.by_source)
            ],
        )
    )
    parts.append(
        '<p class="soft">Positions by round: '
        + ", ".join(
            f"{n} (mean ply {ply:.1f})"
            for n, ply in zip(report.positions, report.plies, strict=True)
        )
        + ".</p>"
    )
    parts += [
        "<h2>Does a search steer into its own errors?</h2>",
        "<p>A search at "
        f"{PLAYED_AT} simulations, on sampled positions. <b>Chosen</b> is how much it rates the "
        f"move it picks above the teacher's q ({TAUGHT_AT} simulations), for the mover; "
        "<b>every</b> the same over all moves. <b>Extra</b> is what picking adds: any noisy "
        "estimate picks its own overestimates a little, so read it against the baseline.</p>",
        _table(
            ("Search", "n", "chosen", "every", "extra", "s.e."),
            [
                (
                    html.escape(t.label),
                    str(t.n),
                    f"{t.chosen:+.3f}",
                    f"{t.every:+.3f}",
                    f"{t.extra:+.3f}",
                    f"{t.error:.3f}",
                )
                for t in report.steering
            ],
        ),
        "<h2>Can the labels be trusted?</h2>",
        f"<p>The teacher's labels ({TAUGHT_AT} simulations) and the final fit's value against the "
        f"teacher searching {report.referee_at} simulations, on {REFEREE_SAMPLE} positions a "
        "round. A lower correlation for a round says its labels are noisier, whichever fit reads "
        "them.</p>",
        _table(
            ("Round", "n", "teacher", "fit", "label moves by"),
            [
                (
                    str(c.source),
                    str(c.n),
                    f"{c.teacher:.3f}",
                    f"{c.fit:.3f}",
                    f"{c.moved:.3f}",
                )
                for c in report.labels
            ],
        ),
        "<p>Against <b>exact values</b> (<code>agents.exact</code>), from ply "
        f"{report.truth_from}. <b>Side</b> is how often a value is on the winner's side of 1/2, of "
        "positions someone wins. "
        "<b>Contested</b> positions are those where some move loses value; <b>keeps</b> is how "
        f"often a move loses nothing there, at {TAUGHT_AT} simulations for both, and at "
        f"{PLAYED_AT} for the fit as played.</p>",
        _table(
            (
                "Round",
                "n",
                "decisive",
                "contested",
                "corr. teacher",
                "corr. fit",
                "side teacher",
                "side fit",
                "keeps: teacher",
                "fit",
                f"fit at {PLAYED_AT}",
            ),
            [
                (
                    str(t.source),
                    str(t.n),
                    f"{t.decisive:.2f}",
                    f"{t.contested:.2f}",
                    f"{t.teacher:.3f}",
                    f"{t.fit:.3f}",
                    f"{t.teacher_side:.3f}",
                    f"{t.fit_side:.3f}",
                    f"{t.teacher_move:.3f}",
                    f"{t.fit_move:.3f}",
                    f"{t.fit_move_played:.3f}",
                )
                for t in report.truth
            ],
        ),
        "<h2>Where in a game they part</h2>",
        _table(
            ("Plies", "n", "mean |fit - teacher|", "mean fit - teacher", "code player's regret"),
            [
                (
                    f"{b.low}&ndash;{b.high - 1}",
                    str(b.n),
                    f"{b.mean_absolute:.3f}",
                    f"{b.mean_signed:+.3f}",
                    f"{b.regret:.3f}",
                )
                for b in report.by_ply
            ],
        ),
        "<h2>What a fit lacks</h2>",
        "<p>Each candidate added alone to a refit of the value, against the teacher's labels, on "
        f"games kept out (correlation without it {report.held_out_correlation[0]:.3f} on all "
        f"rounds, {report.held_out_correlation[1]:.3f} on the last). Measured only: none of these "
        "is played. A gain says a feature explains the teacher, not that it makes a player "
        "stronger, and the teacher can be wrong.</p>",
        _table(
            ("Candidate", "what it counts", "present", "weight", "gain", "gain, last round"),
            [
                (
                    html.escape(c.name),
                    f'<span class="soft">{html.escape(c.meaning)}</span>',
                    f"{c.present:.2f}",
                    f"{c.weight:+.2f}",
                    f"{c.gain:+.4f}",
                    f"{c.gain_last:+.4f}",
                )
                for c in report.candidates
            ],
        ),
        "<h2>The final constants</h2>",
        _constants(report.fits[-1]),
        "<h2>Widest value disagreements</h2>",
        "<p>By ply bucket, a board and its mirror once. Under each: the features moving the "
        "fit's value most, in logits toward the first player, and the candidates that hold. "
        "Where the deeper search and the exact value side with the fit, the teacher is the one "
        "wrong.</p>",
        '<div class="cards">',
        *(_value_card(game, card) for card in report.value_cards),
        "</div>",
        "<h2>Costliest moves</h2>",
        f"<p>The code player at {PLAYED_AT} simulations, by what its move loses on the teacher's "
        "q. "
        "Solid red ring: its move. Dashed green: the teacher's best. The teacher's q of a move "
        "its search dropped early is noisy, so a cost here can be a false alarm: read the deeper "
        "and exact rows before believing one.</p>",
        '<div class="cards">',
        *(_move_card(game, card) for card in report.move_cards),
        "</div>",
        "</main></body></html>",
    ]
    return "\n".join(parts)
