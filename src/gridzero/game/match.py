"""Playing games between players: one game, a series, records, replays and tallies.

`play` plays one game to its end and keeps its record: the first position, every action, the
final position and every event. `replay` steps through a record's actions again. `series`
plays players against each other from several starts, each start once per rotation of the
seats, so no player gains from a seat or a start, and with the same dice or not. `tally` counts
wins, `Tally.score` gives one player's wins, losses and draws, `ladder` scores a player against
several rivals, and `fairness` scores how fair the seats were.
"""

import dataclasses
import math
import multiprocessing
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from gridzero.core import State
from gridzero.core._arrays import check_int

from .position import Action, Position
from .protocols import Game, Player

PLAYER_STREAM = 2
"""First entry of the spawn key of the players' streams: ``(2, ply)`` of the position's seed."""
SERIES_STREAM = 3
"""First entry of the spawn key of a series' own dice: ``(3, r)`` of a start's seed."""


# Declared covariant, so that a game's records are records of positions (`tally` takes any).
# The syntax of `class Record[P: ...]` can't declare variance, and from Python 3.13 a frozen
# dataclass gets `__replace__(*, start: P, final: P)`, from which type checkers infer
# invariance.
P_co = TypeVar("P_co", bound=Position[State, object], covariant=True)


@dataclass(frozen=True, eq=False)
class Record(Generic[P_co]):  # noqa: UP046 (variance can't be declared otherwise, above)
    """One game played to its end.

    Parameters
    ----------
    names : tuple of str
        The players' names, by player number.
    start : P
        The first position.
    actions : tuple of Action
        Every decision, in order: `replay` steps through them again.
    final : P
        The last position: over, with its winner.
    events : tuple
        Every step's events, in order, in the game's own types.
    """

    names: tuple[str, ...]
    start: P_co
    actions: tuple[Action, ...]
    final: P_co
    events: tuple[object, ...]

    @property
    def winner(self) -> str | None:
        """The name of the player who won, or None for a draw."""
        player = self.final.winner
        return None if player is None else self.names[player]


def play[P: Position[State, object], V](
    game: Game[P, V],
    # Any: the game's own type can't be tied to Game[P, V] in Python, so a player for
    # another game of the same view type passes the checker.
    players: Sequence[Player[Any, V]],
    start: P,
    *,
    names: Sequence[str] | None = None,
) -> Record[P]:
    """Play one game to its end.

    Each decision goes to the player whose turn it is, with what the game shows that player
    (``game.view``), the legal actions, and a generator of its own from the position's seed
    keyed by the ply, ``(2, ply)``, so the same players and start replay the same game.

    Parameters
    ----------
    game : Game
        The rules.
    players : sequence of Player
        One per player of the game, by player number.
    start : P
        The position to play from, such as ``game.start(seed)``.
    names : sequence of str, optional
        The players' names, by player number, for the record. By default "player 0",
        "player 1" and so on.

    Returns
    -------
    Record
        The first position, the actions, the final position and the events.

    Raises
    ------
    ValueError
        If there aren't as many players or names as the game has players, or ``start`` is over.
    Exception
        Whatever the game's ``step`` raises for an action a player returns, such as ValueError
        for an illegal one, with a note naming the player and the ply.
    """
    n = game.n_players
    if len(players) != n:
        raise ValueError(f"the game has {n} players, got {len(players)} players")
    names = tuple(f"player {i}" for i in range(n)) if names is None else tuple(names)
    if len(names) != n:
        raise ValueError(f"the game has {n} players, got {len(names)} names")
    if start.over:
        raise ValueError("the start is over: there is nothing to play")
    position = start
    actions: list[Action] = []
    events: list[object] = []
    while not position.over:
        p = game.player(position)
        seed = np.random.SeedSequence(position.seed, spawn_key=(PLAYER_STREAM, position.ply))
        action = players[p](
            game, p, game.view(position, p), game.legal(position), np.random.default_rng(seed)
        )
        try:
            position = game.step(position, action)
        except Exception as error:
            error.add_note(f"{names[p]} (player {p}) chose {action!r} at ply {position.ply}")
            raise
        actions.append(Action(int(action[0]), int(action[1])))
        events.extend(position.events)
    return Record(names, start, tuple(actions), position, tuple(events))


def replay[P: Position[State, object], V](game: Game[P, V], record: Record[P]) -> Iterator[P]:
    """Step through a record's actions again.

    Steps are pure, so the positions are the game's: the last one's key equals the record's
    final position's key.

    Parameters
    ----------
    game : Game
        The rules the game was played by.
    record : Record
        The game.

    Yields
    ------
    P
        The record's start, then the position after each action.

    Raises
    ------
    Exception
        Whatever the game's ``step`` raises, if the record isn't a game of these rules.
    """
    position = record.start
    yield position
    for action in record.actions:
        position = game.step(position, action)
        yield position


def series[P: Position[State, object], V](
    game: Game[P, V],
    players: Sequence[tuple[str, Player[Any, V]]],  # Any: as in `play`
    starts: int,
    seed: int = 0,
    *,
    same_dice: bool = True,
    workers: int = 1,
) -> list[Record[P]]:
    """Play players against each other, from each start once per rotation of the seats.

    Start ``k`` is ``game.start(seed + k)``, played ``n`` times for a game of ``n`` players: in
    its game ``r``, seat ``s`` goes to ``players[(s + r) % n]``, so every player sits in every
    seat once, in the same order round the table. With two players that is each start twice,
    with the sides swapped. Players may share a name, such as copies of one player, and a
    `tally` then counts them as one.

    Games are independent, so with ``workers`` above 1 they run in that many processes at once,
    with the same records as one at a time. The rules and players then travel to each process
    by pickling, so they must pickle, and a script that runs series in parallel must guard its
    entry point with ``if __name__ == "__main__":`` (processes are started fresh, not forked).

    Parameters
    ----------
    game : Game
        The rules.
    players : sequence of (str, Player)
        The players and their names, one per seat of the game: ``players[s]`` sits in seat
        ``s`` in each start's first game.
    starts : int
        How many starts, at least 1: ``starts * n`` games.
    seed : int, default 0
        The seed of the first start, non-negative.
    same_dice : bool, default True
        Whether the games of a start share its seed, and so their dice and the players'
        generators. If not, each game keeps its start but with a seed of its own,
        ``dataclasses.replace(start, seed=own)``, made from the start's seed and its rotation.
        Shared dice make the players' results less noisy, but tie each start's luck to the
        seats, so counting wins by seat needs dice of their own.
    workers : int, default 1
        How many games to play at once, at least 1, each in a process of its own.

    Returns
    -------
    list of Record
        One per game, in order: a start's games together, by rotation.

    Raises
    ------
    TypeError, ValueError
        If ``starts`` or ``workers`` is not an integer of at least 1, ``seed`` is not a
        non-negative integer, ``same_dice`` is not a bool, or there isn't one player per seat.
    """
    starts = check_int(starts, "starts", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    workers = check_int(workers, "workers", minimum=1)
    if not isinstance(same_dice, bool):
        raise TypeError(f"same_dice must be a bool, got {type(same_dice).__name__}")
    n = game.n_players
    if len(players) != n:
        raise ValueError(f"the game has {n} players, got {len(players)} players")
    jobs: list[_Job[P, V]] = []
    for k in range(starts):
        for r in range(n):
            seated = [players[(s + r) % n] for s in range(n)]
            start = game.start(seed + k)
            if not same_dice:
                spawn = np.random.SeedSequence(seed + k, spawn_key=(SERIES_STREAM, r))
                own = int(spawn.generate_state(1, np.uint64)[0]) >> 1  # a non-negative int64
                start = dataclasses.replace(start, seed=own)
            jobs.append(
                _Job(game, tuple(p for _, p in seated), start, tuple(name for name, _ in seated))
            )
    if workers == 1:
        return [job() for job in jobs]
    context = multiprocessing.get_context("spawn")  # forking a process with threads is unsafe
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=context) as pool:
        return list(pool.map(_Job.__call__, jobs))


@dataclass(frozen=True)
class _Job[P: Position[State, object], V]:
    """One game of a series, to play here or in another process."""

    game: Game[P, V]
    players: tuple[Player[Any, V], ...]  # Any: as in `play`
    start: P
    names: tuple[str, ...]

    def __call__(self) -> Record[P]:
        """Play the game to its end."""
        return play(self.game, self.players, self.start, names=self.names)


class Score(NamedTuple):
    """One player's games: those it won, those another player won, and the draws."""

    won: int
    lost: int
    drawn: int


@dataclass(frozen=True, eq=False)
class Tally:
    """Who won a set of games.

    Parameters
    ----------
    games : int
        How many games.
    wins : dict of str to int
        Wins by player name. Every player who played is a key.
    draws : int
        Games with no winner.
    seat_wins : ndarray of int64, shape (n_players,)
        Wins by player number, whoever played it: a seat that wins more has an edge.
    plies : float
        The mean number of decisions per game.
    """

    games: int
    wins: dict[str, int]
    draws: int
    seat_wins: NDArray[np.int64]
    plies: float

    def score(self, name: str) -> Score:
        """Return one player's wins, losses and draws.

        Parameters
        ----------
        name : str
            A player who played.

        Returns
        -------
        Score
            Its wins, the games another player won, and the draws.

        Raises
        ------
        KeyError
            If no player of that name played.
        """
        won = self.wins[name]
        return Score(won, self.games - self.draws - won, self.draws)


def tally(records: Sequence[Record[Position[State, object]]]) -> Tally:
    """Count who won a set of games.

    Parameters
    ----------
    records : sequence of Record
        The games, at least one, all with the same number of players.

    Returns
    -------
    Tally
        Wins by name and by seat, draws, and the mean length.

    Raises
    ------
    ValueError
        If ``records`` is empty, or its games have different numbers of players.
    """
    if not records:
        raise ValueError("tally needs at least one game")
    seats = {len(record.names) for record in records}
    if len(seats) != 1:
        raise ValueError(f"the games have different numbers of players: {sorted(seats)}")
    wins = {name: 0 for record in records for name in record.names}
    seat_wins = np.zeros(seats.pop(), dtype=np.int64)
    for record in records:
        if record.final.winner is not None:
            seat_wins[record.final.winner] += 1
            wins[record.names[record.final.winner]] += 1
    return Tally(
        games=len(records),
        wins=wins,
        draws=sum(record.final.winner is None for record in records),
        seat_wins=seat_wins,
        plies=float(np.mean([record.final.ply for record in records])),
    )


def ladder[P: Position[State, object], V](
    game: Game[P, V],
    player: Player[Any, V],  # Any: as in `play`
    rivals: Mapping[str, Player[Any, V]],
    starts: int,
    seed: int = 0,
    *,
    same_dice: bool = True,
    workers: int = 1,
) -> dict[str, Score]:
    """Score a player against each of several rivals, over a `series` each.

    Each rival takes every seat but the player's, from the same starts as every other rival,
    so the scores compare: a game the rival's copies win is a loss. The player is named
    ``"player"`` in the records.

    Parameters
    ----------
    game : Game
        The rules.
    player : Player
        The player scored.
    rivals : mapping of str to Player
        Its rivals, by name, none named ``"player"``.
    starts : int
        Starts against each rival, at least 1: each once per rotation of the seats.
    seed : int, default 0
        The seed of the first start, non-negative.
    same_dice : bool, default True
        Whether the games of a start share their dice (see `series`).
    workers : int, default 1
        How many games to play at once (see `series`).

    Returns
    -------
    dict of str to Score
        The player's score against each rival, in the order of ``rivals``.

    Raises
    ------
    TypeError, ValueError
        If ``starts`` is not an integer of at least 1, a rival is named ``"player"``, or
        `series` refuses its arguments.
    """
    starts = check_int(starts, "starts", minimum=1)
    if "player" in rivals:
        raise ValueError('a rival is named "player", the name of the player scored')
    others = game.n_players - 1
    return {
        name: tally(
            series(
                game,
                [("player", player)] + [(name, rival)] * others,
                starts,
                seed,
                same_dice=same_dice,
                workers=workers,
            )
        ).score("player")
        for name, rival in rivals.items()
    }


@dataclass(frozen=True, eq=False)
class Fairness:
    """How fair the seats of a game were over a set of games.

    Parameters
    ----------
    decided : int
        Games with a winner.
    seat_edges : ndarray of float64, shape (n_players,)
        Each seat's share of the decided games minus its fair share ``1 / n_players``: positive
        for a seat that wins more. They add up to 0, and are all 0 when no game was decided.
        With two seats, ``seat_edges[0]`` is from -0.5 to 0.5, and ``seat_edges[1]`` its
        negative.
    draws : float
        The share of games drawn, from 0 to 1.
    """

    decided: int
    seat_edges: NDArray[np.float64]
    draws: float

    def excess_edge(self) -> float:
        """Return how far the largest seat edge lies beyond the noise of fair seats, at least 0.

        Over ``d`` decided games of ``n`` fair seats, a seat's share has a standard deviation of
        ``sqrt((n - 1) / d) / n``, and the noise is twice that: a seat's edge stays within it
        about 95% of the time. With two seats it is ``1 / sqrt(d)``.

        Returns
        -------
        float
            ``max(0, max(|seat_edges|) - noise)``, and 0 with no decided game.
        """
        if not self.decided:
            return 0.0
        n = self.seat_edges.size
        noise = 2 * math.sqrt((n - 1) / self.decided) / n
        return max(0.0, float(np.abs(self.seat_edges).max()) - noise)


def fairness(tally: Tally) -> Fairness:
    """Score how fair the seats were: each seat's edge over the decided games, and the draws.

    Parameters
    ----------
    tally : Tally
        Games, from `tally`.

    Returns
    -------
    Fairness
        The scores.
    """
    decided = int(tally.seat_wins.sum())
    n = tally.seat_wins.size
    edges = tally.seat_wins / decided - 1 / n if decided else np.zeros(n)
    return Fairness(decided=decided, seat_edges=edges, draws=tally.draws / tally.games)
