"""Lessons: a search player's games of Tactics and what it made of them, for code players to fit.

A game is kept as its seed and its actions, since steps are pure: `positions_of` replays it, so
features can change without playing again. Each search keeps its value and its plans' visits and
values, which replaying can't give back. Games take long, so `teach` saves as it goes and stops at
a time limit, keeping what it finished.
"""

import multiprocessing
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gridzero.game import Action, Record, play

from .game import Tactics
from .search import Judgment, SearchPlayer
from .state import TacticsPosition

SAVED_EVERY = 24
"""Games finished between saves."""


@dataclass(frozen=True, eq=False)
class Lessons:
    """Games of a search player, with ``g`` games, ``d`` decisions, ``s`` searches and ``p`` plans.

    Parameters
    ----------
    seed : ndarray of int64, shape (g,)
        Each game's seed: ``game.start(seed)``.
    winner : ndarray of int8, shape (g,)
        The team that won each.
    rounds : ndarray of int16, shape (g,)
        The round each ended in.
    game_start : ndarray of int64, shape (g + 1,)
        Where each game's decisions start in ``verb`` and ``cell``, and their end.
    verb, cell : ndarray of int8 and int32, shape (d,)
        Each decision's action.
    searched : ndarray of int64, shape (s,)
        The decision each search was made at, an index into ``verb``, ascending.
    value : ndarray of float64, shape (s,)
        Team 0's value of each search's root, in [0, 1].
    plan_start : ndarray of int64, shape (s + 1,)
        Where each search's plans start in the arrays below, and their end.
    plan_tile : ndarray of int32, shape (p,)
        The tile each plan moves to.
    plan_attack_first : ndarray of bool, shape (p,)
        Whether it attacks before moving.
    plan_work_verb, plan_work_cell : ndarray of int8 and int32, shape (p,)
        The work it does in place of attacking, -1 for none.
    plan_visits : ndarray of int32, shape (p,)
        The simulations through it.
    plan_value : ndarray of float64, shape (p,)
        Team 0's mean value through it, in [0, 1].
    """

    seed: NDArray[np.int64]
    winner: NDArray[np.int8]
    rounds: NDArray[np.int16]
    game_start: NDArray[np.int64]
    verb: NDArray[np.int8]
    cell: NDArray[np.int32]
    searched: NDArray[np.int64]
    value: NDArray[np.float64]
    plan_start: NDArray[np.int64]
    plan_tile: NDArray[np.int32]
    plan_attack_first: NDArray[np.bool_]
    plan_work_verb: NDArray[np.int8]
    plan_work_cell: NDArray[np.int32]
    plan_visits: NDArray[np.int32]
    plan_value: NDArray[np.float64]

    def __len__(self) -> int:
        """Return how many games."""
        return int(self.seed.size)

    @staticmethod
    def join(parts: Sequence[Lessons]) -> Lessons:
        """Return the games of several lessons, in order of their seeds.

        Parameters
        ----------
        parts : sequence of Lessons
            At least one.

        Returns
        -------
        Lessons
            Every game, ascending by seed, each with its own decisions, searches and plans.

        Raises
        ------
        ValueError
            If a seed repeats, or parts' seeds interleave.
        """
        parts = sorted(parts, key=lambda part: int(part.seed[0]))
        decisions = np.cumsum([0] + [part.verb.size for part in parts[:-1]])
        plans = np.cumsum([0] + [part.plan_tile.size for part in parts[:-1]])
        game_start = [parts[0].game_start[:1]]
        game_start += [p.game_start[1:] + d for p, d in zip(parts, decisions, strict=True)]
        plan_start = [parts[0].plan_start[:1]]
        plan_start += [p.plan_start[1:] + n for p, n in zip(parts, plans, strict=True)]
        seed = np.concatenate([part.seed for part in parts])
        if (np.diff(seed) <= 0).any():
            raise ValueError("lessons to join must have seeds of their own, not interleaved")
        return Lessons(
            seed,
            np.concatenate([part.winner for part in parts]),
            np.concatenate([part.rounds for part in parts]),
            np.concatenate(game_start),
            np.concatenate([part.verb for part in parts]),
            np.concatenate([part.cell for part in parts]),
            np.concatenate([p.searched + d for p, d in zip(parts, decisions, strict=True)]),
            np.concatenate([part.value for part in parts]),
            np.concatenate(plan_start),
            np.concatenate([part.plan_tile for part in parts]),
            np.concatenate([part.plan_attack_first for part in parts]),
            np.concatenate([part.plan_work_verb for part in parts]),
            np.concatenate([part.plan_work_cell for part in parts]),
            np.concatenate([part.plan_visits for part in parts]),
            np.concatenate([part.plan_value for part in parts]),
        )

    def save(self, path: str | Path) -> None:
        """Write the lessons as a compressed ``.npz``, replacing the file only once it is whole.

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        path = Path(path)
        partial = path.with_name(path.name + ".partial")
        with partial.open("wb") as file:
            np.savez_compressed(file, **{f.name: getattr(self, f.name) for f in fields(self)})
        os.replace(partial, path)

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
            return Lessons(*(data[f.name] for f in fields(Lessons)))


def positions_of(
    game: Tactics, lessons: Lessons, index: int
) -> Iterator[tuple[TacticsPosition, Action]]:
    """Replay a game of lessons.

    Parameters
    ----------
    game : Tactics
        The rules it was played by.
    lessons : Lessons
        The lessons.
    index : int
        Which game.

    Yields
    ------
    position : TacticsPosition
        The position at each decision, in order.
    action : Action
        What the teacher did there.
    """
    position = game.start(int(lessons.seed[index]))
    first, last = lessons.game_start[index], lessons.game_start[index + 1]
    for verb, cell in zip(lessons.verb[first:last], lessons.cell[first:last], strict=True):
        action = Action(int(verb), int(cell))
        yield position, action
        position = game.step(position, action)


@dataclass(eq=False)
class _Noting:
    """A search player that notes what it judged at each decision, for `lessons_of`."""

    teacher: SearchPlayer
    judgments: list[Judgment | None] = field(default_factory=list)

    def __call__(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the teacher's action (see `game.Player`)."""
        action, judgment = self.teacher.decide(game, player, position, legal, rng)
        self.judgments.append(judgment)
        return action


def lessons_of(game: Tactics, teacher: SearchPlayer, seed: int) -> Lessons:
    """Play one game of a search player against itself, and keep what it did and judged.

    Parameters
    ----------
    game : Tactics
        The rules.
    teacher : SearchPlayer
        The player, on both sides.
    seed : int
        The game's seed, which sets its board, its dice and the player's draws.

    Returns
    -------
    Lessons
        The game.
    """
    noting = _Noting(teacher)
    record: Record[TacticsPosition] = play(game, [noting, noting], game.start(seed))
    found = [(i, j) for i, j in enumerate(noting.judgments) if j is not None]
    judged = [j for _, j in found]
    sizes = [len(j.plans) for j in judged]
    plans = [plan for j in judged for plan in j.plans]
    winner = record.final.winner
    return Lessons(
        np.array([seed], dtype=np.int64),
        np.array([-1 if winner is None else winner], dtype=np.int8),
        np.array([record.final.round], dtype=np.int16),
        np.array([0, len(record.actions)], dtype=np.int64),
        np.array([a.verb for a in record.actions], dtype=np.int8),
        np.array([a.cell for a in record.actions], dtype=np.int32),
        np.array([i for i, _ in found], dtype=np.int64),
        np.array([j.value for j in judged], dtype=np.float64),
        np.cumsum([0, *sizes], dtype=np.int64),
        np.array([p.tile for p in plans], dtype=np.int32),
        np.array([p.attack_first for p in plans], dtype=np.bool_),
        np.array([-1 if p.work is None else p.work.verb for p in plans], dtype=np.int8),
        np.array([-1 if p.work is None else p.work.cell for p in plans], dtype=np.int32),
        np.array([n for j in judged for n in j.visits], dtype=np.int32),
        np.array([v for j in judged for v in j.values], dtype=np.float64),
    )


def teach(
    game: Tactics,
    teacher: SearchPlayer,
    seeds: Sequence[int],
    path: str | Path,
    *,
    seconds: float,
    workers: int,
) -> Lessons:
    """Play games of a search player against itself, saving as they finish, within a time limit.

    Parameters
    ----------
    game : Tactics
        The rules.
    teacher : SearchPlayer
        The player.
    seeds : sequence of int
        A seed per game: at least one, ascending.
    path : str or Path
        The file to save to, every `SAVED_EVERY` games and at the end (`Lessons.save`).
    seconds : float
        No game starts after this long, and games under way finish.
    workers : int
        How many games to play at once, each in a process of its own.

    Returns
    -------
    Lessons
        Every game finished, ascending by seed.

    Raises
    ------
    RuntimeError
        If the time ran out before any game finished.
    """
    context = multiprocessing.get_context("spawn")  # as `game.series`
    finished: list[Lessons] = []
    deadline = time.perf_counter() + seconds
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        pending: set[Future[Lessons]] = {
            pool.submit(lessons_of, game, teacher, seed) for seed in seeds
        }
        while pending:
            left = deadline - time.perf_counter()
            if left <= 0:
                pool.shutdown(wait=False, cancel_futures=True)
                pending = {future for future in pending if not future.cancelled()}
            done, pending = wait(
                pending, timeout=max(left, 0.0) or None, return_when=FIRST_COMPLETED
            )
            finished += [future.result() for future in done]
            if len(finished) // SAVED_EVERY > (len(finished) - len(done)) // SAVED_EVERY:
                Lessons.join(finished).save(path)
    if not finished:
        raise RuntimeError(f"no game finished in {seconds} seconds")
    lessons = Lessons.join(finished)
    lessons.save(path)
    return lessons
