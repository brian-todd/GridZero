"""Lessons: positions of k in a row and what a trained agent's search makes of them.

A code player learns from lessons (`fit.py`). They come from the games a teacher plays, and from
those a code player plays against it, whose mistakes lead to positions the teacher alone never
reaches. A position of k in a row is whole in its owners and its ply, so that is what is kept.
"""

import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import AlphaZeroPlayer, Evaluator
from gridzero.game import Record, replay

from .game import PLACE, InARow
from .state import InARowPosition, InARowState

JUDGED_IN = 64
"""Positions a worker judges at a time."""


@dataclass(frozen=True, eq=False)
class Lessons:
    """Positions and what a teacher made of them, one row per position.

    Parameters
    ----------
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.
    game : ndarray of int64, shape (n,)
        The game each was first seen in: positions of one game go on one side of a split.
    source : ndarray of int64, shape (n,)
        The round of lessons each came from: whose games reached it (`experiments.code_fit`).
    value : ndarray of float64, shape (n,)
        Player 0's value of the root by the teacher's search.
    q : ndarray of float64, shape (n, n_columns)
        Player 0's mean value after a stone in each column; NaN where unvisited or full.
    policy : ndarray of float64, shape (n, n_columns)
        The search's improved policy over columns.
    move : ndarray of int64, shape (n,)
        The column the search chose.
    """

    owner: NDArray[np.int8]
    ply: NDArray[np.int64]
    game: NDArray[np.int64]
    source: NDArray[np.int64]
    value: NDArray[np.float64]
    q: NDArray[np.float64]
    policy: NDArray[np.float64]
    move: NDArray[np.int64]

    def __len__(self) -> int:
        """Return how many positions."""
        return len(self.ply)

    def __add__(self, other: Lessons) -> Lessons:
        """Return these lessons, then another's whose positions are new, its games renumbered."""
        seen = {row.tobytes() for row in self.owner}
        keep = np.array([row.tobytes() not in seen for row in other.owner], dtype=bool)
        offset = int(self.game.max(initial=-1)) + 1
        return Lessons(
            np.concatenate([self.owner, other.owner[keep]]),
            np.concatenate([self.ply, other.ply[keep]]),
            np.concatenate([self.game, other.game[keep] + offset]),
            np.concatenate([self.source, other.source[keep]]),
            np.concatenate([self.value, other.value[keep]]),
            np.concatenate([self.q, other.q[keep]]),
            np.concatenate([self.policy, other.policy[keep]]),
            np.concatenate([self.move, other.move[keep]]),
        )

    def save(self, path: str | Path) -> None:
        """Write the lessons as a compressed ``.npz``.

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        arrays = {field.name: getattr(self, field.name) for field in fields(self)}
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path: str | Path) -> Lessons:
        """Read lessons written by `save`.

        Parameters
        ----------
        path : str or Path
            The file to read.

        Returns
        -------
        Lessons
            The lessons.
        """
        with np.load(path) as data:
            return Lessons(*(data[field.name] for field in fields(Lessons)))


def position_of(game: InARow, owner: NDArray[np.int8], ply: int) -> InARowPosition:
    """Return the position of given owners and ply, not over.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n_cells,)
        Who owns each cell.
    ply : int
        The ply.

    Returns
    -------
    InARowPosition
        The position.
    """
    return InARowPosition(InARowState(game.space, {InARowState.owner: owner}), 0, ply=ply)


def distinct_positions(
    game: InARow, records: Iterable[Record[InARowPosition]]
) -> tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.int64]]:
    """Return the positions of games, not over, each once.

    Parameters
    ----------
    game : InARow
        The rules.
    records : iterable of Record
        The games.

    Returns
    -------
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.
    games : ndarray of int64, shape (n,)
        The number of the game each was first seen in.
    """
    seen: dict[bytes, tuple[NDArray[np.int8], int, int]] = {}
    for number, record in enumerate(records):
        for position in replay(game, record):
            key = position.state.owner.tobytes()
            if not position.over and key not in seen:
                seen[key] = (position.state.owner, position.ply, number)
    rows = list(seen.values())
    if not rows:
        empty = np.empty(0, dtype=np.int64)
        return np.empty((0, game.space.n_cells), dtype=np.int8), empty, empty.copy()
    return (
        np.stack([row[0] for row in rows]),
        np.array([row[1] for row in rows], dtype=np.int64),
        np.array([row[2] for row in rows], dtype=np.int64),
    )


def judge(
    game: InARow,
    teacher: Evaluator[InARow, InARowPosition],
    owner: NDArray[np.int8],
    ply: NDArray[np.int64],
    games: NDArray[np.int64],
    *,
    simulations: int,
    seed: int,
    source: int = 0,
    workers: int = 1,
) -> Lessons:
    """Return what a teacher's search makes of positions.

    Parameters
    ----------
    game : InARow
        The rules, with gravity: a stone is read by its column.
    teacher : Evaluator
        The teacher.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.
    games : ndarray of int64, shape (n,)
        The game each was first seen in.
    simulations : int
        The search's simulations per position.
    seed : int
        Sets the search's ties: the lessons don't depend on ``workers``.
    source : int, default 0
        The round of lessons these are.
    workers : int, default 1
        How many processes judge.

    Returns
    -------
    Lessons
        The lessons, in the order given.
    """
    agent = AlphaZeroPlayer(teacher, simulations=simulations, considered=game.shape[0])
    jobs = [
        _Judge(game, agent, owner[i : i + JUDGED_IN], ply[i : i + JUDGED_IN], seed + i)
        for i in range(0, len(ply), JUDGED_IN)
    ]
    if not jobs:
        n_columns = game.shape[0]
        parts = [
            (
                np.zeros(0),
                np.zeros((0, n_columns)),
                np.zeros((0, n_columns)),
                np.zeros(0, dtype=np.int64),
            )
        ]
    elif workers == 1:
        parts = [job() for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")  # as `game.series`
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=context) as pool:
            parts = list(pool.map(_Judge.__call__, jobs))
    return Lessons(
        owner,
        ply,
        games,
        np.full(len(ply), source, dtype=np.int64),
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        np.concatenate([part[3] for part in parts]),
    )


@dataclass(frozen=True)
class _Judge:
    """Some positions to judge, here or in another process."""

    game: InARow
    agent: AlphaZeroPlayer[InARow, InARowPosition]
    owner: NDArray[np.int8]
    ply: NDArray[np.int64]
    seed: int

    def __call__(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return the values, q and policy by column, shape (n, n_columns), and the moves."""
        game, n = self.game, len(self.ply)
        rng = np.random.default_rng(self.seed)
        values = np.zeros(n)
        qs = np.full((n, game.shape[0]), np.nan)
        policies = np.zeros((n, game.shape[0]))
        moves = np.zeros(n, dtype=np.int64)
        for i in range(n):
            position = position_of(game, self.owner[i], int(self.ply[i]))
            result = self.agent.search(game, position, rng)
            cells = np.flatnonzero(game.legal(position)[PLACE])
            columns = game.space.unravel(cells)[:, 0]
            values[i] = result.value
            qs[i, columns] = result.q[PLACE][cells]
            policies[i, columns] = result.policy[PLACE][cells]
            moves[i] = game.space.unravel(result.action.cell)[0]
        return values, qs, policies, moves
