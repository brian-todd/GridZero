"""Take a seat: a person plays Connect Four against any player, on a page served locally.

`Table` holds the games: it steps them, has the players reply, explains each reply (`explain`),
and keeps every finished game with its note in one JSON file. `serve` puts it behind the standard
library's HTTP server, on this machine only, for the page in ``play.html``, which draws the board
from the lattice's own cell polygons (`report.board_shapes`). Not imported by the package: `report`
needs scikit-learn, which is slow to import in every worker of a series.
"""

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import AlphaZeroPlayer, Evaluator, UniformPlayer
from gridzero.agents.zero import SearchResult
from gridzero.core import check_int
from gridzero.game import Player

from .code_player import (
    MOVE_FEATURES,
    SEARCH_ALONE,
    SEARCH_C_SCALE,
    VALUE_FEATURES,
    FittedEvaluator,
    LineEvaluator,
    move_features,
    value_features,
)
from .game import PLACE, InARow
from .lessons import position_of
from .players import ThreatPlayer
from .report import STYLE, board_key, board_shapes
from .state import InARowPosition

VALUE_TEXT = {
    "open_one": "Lines with one stone and no rival stone, red's minus blue's",
    "open_two": "Lines with two stones and no rival stone, red's minus blue's",
    "first_threats_odd": "Cells that complete a line for red, on odd rows",
    "first_threats_even": "Cells that complete a line for red, on even rows",
    "second_threats_odd": "Cells that complete a line for blue, on odd rows",
    "second_threats_even": "Cells that complete a line for blue, on even rows",
    "win_now": "A win ready for the player to move (+1 for red, -1 for blue)",
    "lost_now": "Two wins against the player to move, none of its own (+1 for red, -1 for blue)",
    "center": "Stones in the center column, red's minus blue's",
    "beside_center": "Stones beside the center column, red's minus blue's",
    "to_move": "Who moves next (+1 for red, -1 for blue)",
    "first_clean_odd": "Red's threats on odd rows with no blue threat below them",
    "second_clean_even": "Blue's threats on even rows with no red threat below them",
}
"""What each feature of a position counts, for the page. Rows count from 1 at the bottom."""

MOVE_TEXT = {
    "center": "Nearness to the center column (minus its distance)",
    "own_empty": "Empty lines through the stone",
    "own_one": "Lines it would extend from one of its own stones",
    "own_two": "Lines it would extend from two of its own stones",
    "win": "Lines it would complete",
    "cut_one": "Rival lines of one stone it would cut",
    "cut_two": "Rival lines of two stones it would cut",
    "block": "Rival wins it would block",
    "gives_win": "Whether it opens a rival win on the cell above",
    "under_own_win": "Whether it lets the rival take its own win on the cell above",
    "own_parity": "Whether its row is of the player's parity (odd for red, even for blue)",
}
"""What each feature of a stone counts, for the page, for the player placing it."""

SHAPE_SCALE = 60.0
"""The pixels of a unit of the plane, in the page's picture of the board."""

JsonObject = dict[str, Any]  # Any: what JSON holds, whose fields differ by name
"""A record or an answer, ready for JSON."""


@dataclass(frozen=True)
class Rung:
    """A player a person can sit down against.

    Parameters
    ----------
    name : str
        What it is called on the page and in the records.
    about : str
        What it knows, in a sentence.
    player : Player
        The player.
    """

    name: str
    about: str
    player: Player[InARow, InARowPosition]


def rungs(
    fit: FittedEvaluator, teacher: Evaluator[InARow, InARowPosition] | None = None
) -> tuple[Rung, ...]:
    """Return the players on the page, roughly weakest first, as a round robin ranked them.

    Neighbors can be close: ``fit 64`` and ``teacher 64``, or ``teacher 64`` and ``teacher 256``,
    are within what 100 games a pair can tell.

    Every search player plays without noise, its best as it sees it: `Table.new` turns noise on
    for a game. The round robin was played with noise, as all scores are.

    Parameters
    ----------
    fit : FittedEvaluator
        The code player's constants.
    teacher : Evaluator, optional
        The network it was fit to, for rungs above the code player's.

    Returns
    -------
    tuple of Rung
        The rungs.
    """

    def search(
        evaluator: Evaluator[InARow, InARowPosition], simulations: int
    ) -> Player[InARow, InARowPosition]:
        return AlphaZeroPlayer(
            evaluator, simulations=simulations, considered=7, c_scale=SEARCH_C_SCALE
        )

    knows_fit = "Named features with constants fit to a trained agent's judgment"
    knows_teacher = "The trained network the code player was fit to"
    ladder = [
        Rung("uniform", "Picks any legal column at random.", UniformPlayer()),
        Rung(
            "search 64",
            "A search of 64 simulations that knows only the rules.",
            search(SEARCH_ALONE, 64),
        ),
        Rung(
            "threat",
            "A script: wins when it can, blocks, never hands over a win. It looks no further.",
            ThreatPlayer(),
        ),
        Rung(
            "lines 64",
            "Open lines and the center column, weighed by hand, with a search of 64 simulations.",
            search(LineEvaluator(), 64),
        ),
        Rung("fit 16", f"{knows_fit}, with a search of 16 simulations.", search(fit, 16)),
        Rung("fit 64", f"{knows_fit}, with a search of 64 simulations.", search(fit, 64)),
    ]

    def network(simulations: int) -> list[Rung]:
        if teacher is None:
            return []
        about = f"{knows_teacher}, with a search of {simulations} simulations."
        return [Rung(f"teacher {simulations}", about, search(teacher, simulations))]

    ladder += [
        *network(64),
        *network(256),
        Rung("fit 256", f"{knows_fit}, with a search of 256 simulations.", search(fit, 256)),
        Rung("fit 1024", f"{knows_fit}, with a search of 1024 simulations.", search(fit, 1024)),
        *network(1024),
    ]
    return tuple(ladder)


@dataclass(frozen=True)
class Reason:
    """What one feature adds to a move's lead over the other move.

    Parameters
    ----------
    feature : str
        The feature's name.
    text : str
        What it counts.
    chosen : float
        The feature, for the move played.
    rival : float
        The feature, for the other move.
    logits : float
        What the difference adds to the move played, in logits: positive favors it.
    """

    feature: str
    text: str
    chosen: float
    rival: float
    logits: float


@dataclass(frozen=True)
class Searched:
    """What a search made of one column.

    Parameters
    ----------
    column : int
        The column, from 0.
    visits : int
        Simulations through it.
    value : float or None
        The mean value for the player moving: 1 a win, 0 a loss, 1/2 a draw. None if unvisited.
    """

    column: int
    visits: int
    value: float | None


@dataclass(frozen=True)
class Why:
    """Why a search played a column, in the search's own numbers and its evaluator's.

    The search decides; the reasons only say how its evaluator reads the two moves. Where they
    read the other way, the search has seen something the features haven't.

    Parameters
    ----------
    column : int
        The column played.
    expects : float
        What the search found the column played worth to the player moving: its mean value, or
        the position's, if it never visited the column. The position's own mean isn't enough:
        it averages every column tried, however bad.
    columns : tuple of Searched
        Each legal column, ascending.
    glance : int
        The legal column the evaluator's prior likes most, before any search.
    rival : int or None
        The search's next choice: the other column most visited, then the best valued, then the
        lowest. None if no other is legal.
    stone : tuple of Reason
        The features of a stone: the column played over the rival, largest first, without those
        that add nothing. Empty for an evaluator with no named features.
    board : tuple of Reason
        The features of the board after each, the same way. Empty also if either ends the game,
        where the value isn't asked.
    named : bool
        Whether the evaluator has named features at all.
    """

    column: int
    expects: float
    columns: tuple[Searched, ...]
    glance: int
    rival: int | None
    stone: tuple[Reason, ...]
    board: tuple[Reason, ...]
    named: bool


def _reasons(
    names: Sequence[str],
    texts: Mapping[str, str],
    weights: tuple[float, ...],
    chosen: NDArray[np.float64],
    rival: NDArray[np.float64],
    sign: float,
) -> tuple[Reason, ...]:
    """Return what each feature adds, largest first; features of shape (len(names),)."""
    logits = sign * np.asarray(weights) * (chosen - rival)
    order: list[int] = np.argsort(-np.abs(logits), kind="stable").tolist()
    return tuple(
        Reason(names[j], texts[names[j]], float(chosen[j]), float(rival[j]), float(logits[j]) + 0.0)
        for j in order
        if logits[j] != 0.0
    )


def explain(
    game: InARow,
    evaluator: Evaluator[InARow, InARowPosition],
    position: InARowPosition,
    result: SearchResult,
) -> Why:
    """Explain a search's decision.

    Parameters
    ----------
    game : InARow
        The rules.
    evaluator : Evaluator
        What the search judged positions by. Only a `FittedEvaluator` gives reasons.
    position : InARowPosition
        The position searched, not over.
    result : SearchResult
        What the search found there.

    Returns
    -------
    Why
        The explanation. The stone's reasons add up to the prior's logit of the column played
        minus the rival's, and the board's to the difference of the values' logits after each,
        both for the player moving.
    """
    mover = game.player(position)
    rows = game.shape[1]
    legal = game.legal(position)
    cells = np.flatnonzero(legal[PLACE])
    logits = np.asarray(evaluator(game, [position], [legal])[0][0], dtype=np.float64)
    visits = result.visits[PLACE, cells]
    q = result.q[PLACE, cells]  # player 0's; NaN where unvisited
    mine = q if mover == 0 else 1.0 - q
    columns = tuple(
        Searched(
            int(cell // rows),
            int(visits[i]),
            None if np.isnan(mine[i]) else float(mine[i]),
        )
        for i, cell in enumerate(cells)
    )
    played = int(np.flatnonzero(cells == result.action.cell)[0])
    root = result.value if mover == 0 else 1.0 - result.value
    others = [i for i in range(len(cells)) if i != played]
    # Most visited, then best valued (an unvisited column has no value), then the lowest column.
    other = (
        min(
            others,
            key=lambda i: (-visits[i], np.inf if np.isnan(mine[i]) else -mine[i], cells[i]),
        )
        if others
        else None
    )
    stone: tuple[Reason, ...] = ()
    board: tuple[Reason, ...] = ()
    if isinstance(evaluator, FittedEvaluator) and other is not None:
        _, features = move_features(game, position.state.owner, position.ply)
        stone = _reasons(
            MOVE_FEATURES, MOVE_TEXT, evaluator.prior, features[played], features[other], 1.0
        )
        after = [game.step(position, (PLACE, int(cells[i]))) for i in (played, other)]
        if not (after[0].over or after[1].over):
            owner = np.stack([a.state.owner for a in after])
            ply = np.array([a.ply for a in after], dtype=np.int64)
            read = value_features(game, owner, ply)
            board = _reasons(
                VALUE_FEATURES,
                VALUE_TEXT,
                evaluator.value,
                read[0],
                read[1],
                1.0 if mover == 0 else -1.0,
            )
    return Why(
        int(cells[played] // rows),
        root if (value := columns[played].value) is None else value,
        columns,
        int(cells[int(np.argmax(logits))] // rows),
        None if other is None else int(cells[other] // rows),
        stone,
        board,
        isinstance(evaluator, FittedEvaluator),
    )


def start_of(game: InARow, board: str) -> InARowPosition:
    """Return the position a board's text (`report.board_key`) holds, with the ply it implies.

    Parameters
    ----------
    game : InARow
        The rules: with gravity.
    board : str
        One of ``.``, ``0`` and ``1`` per cell, in id order.

    Returns
    -------
    InARowPosition
        The position, not over.

    Raises
    ------
    ValueError
        If the text isn't a board one of these games reaches while it is still going: of the
        wrong length or characters, with the wrong count of stones for either player, a stone in
        the air, a line completed, or no free cell.
    """
    n_cells = game.space.n_cells
    if len(board) != n_cells or set(board) - set(".01"):
        raise ValueError(f"a board needs one of . 0 1 for each of its {n_cells} cells")
    owner = np.array([".01".index(c) - 1 for c in board], dtype=np.int8)
    first, second = int((owner == 0).sum()), int((owner == 1).sum())
    if first - second not in (0, 1):
        raise ValueError(f"no game has {first} first stones and {second} second ones")
    grid = owner.reshape(game.shape)  # (columns, rows)
    if ((grid[:, 1:] >= 0) & (grid[:, :-1] < 0)).any():
        raise ValueError("a stone is in the air")
    if (owner[game.lines][:, :, None] == np.array([0, 1])).all(axis=1).any():
        raise ValueError("a line is already complete")
    if first + second == n_cells:
        raise ValueError("the board is full")
    return position_of(game, owner, first + second)


@dataclass(frozen=True)
class Move:
    """One stone of a game.

    Parameters
    ----------
    ply : int
        The ply it was played at.
    player : int
        The player placing it, 0 or 1.
    by : str
        "person", or the name of the rung.
    cell : int
        Its flat cell id.
    column : int
        Its column, from 0.
    seconds : float
        The time taken to decide.
    why : Why or None
        The search's explanation, for a search player.
    """

    ply: int
    player: int
    by: str
    cell: int
    column: int
    seconds: float
    why: Why | None


@dataclass
class _Sitting:
    """A game in progress, or finished and open to a note."""

    number: int
    rung: Rung
    player: Player[InARow, InARowPosition]
    noise: bool
    seat: int
    start: str
    played_at: str
    rng: np.random.Generator
    position: InARowPosition
    offered: float
    moves: list[Move] = field(default_factory=list)
    note: JsonObject | None = None


class Table:
    """The games of one person, and their records in one JSON file.

    A record holds the rung, what it was built from, whether it played with noise, the person's
    seat, the start, every move with its time and the search's explanation, the result, and the
    note. A game is written when it ends, and rewritten when it gets a note. A game left unfinished
    isn't kept.

    Parameters
    ----------
    game : InARow
        The rules: four in a row with gravity.
    ladder : sequence of Rung
        The players a person can choose, with no two of one name.
    path : str or Path
        The JSON file of records: read if it exists, and made with its folder if not.
    seed : int
        Sets every player's draws, per game.
    players : str, default ""
        What the rungs were built from, kept in every record: a rung's name says only its kind and
        budget. `setup` counts only the games against these same players.

    Raises
    ------
    ValueError
        If two rungs share a name, or the file doesn't hold a list of records.
    """

    def __init__(
        self,
        game: InARow,
        ladder: Sequence[Rung],
        path: str | Path,
        seed: int,
        players: str = "",
    ) -> None:
        self.game = game
        self.ladder = {rung.name: rung for rung in ladder}
        if len(self.ladder) != len(ladder):
            raise ValueError("two rungs share a name")
        self.path = Path(path)
        self.seed = seed
        self.players = players
        self.records: list[JsonObject] = []
        if self.path.exists():
            kept = json.loads(self.path.read_text())
            if not isinstance(kept, list):
                raise ValueError(f"{self.path} doesn't hold a list of records")
            self.records = kept
        self._sittings: dict[int, _Sitting] = {}
        self._next = 1 + max((r["game"] for r in self.records), default=-1)
        self._lock = threading.Lock()

    def setup(self) -> JsonObject:
        """Return what the page starts from: the board's shapes, and the rungs and their records.

        Returns
        -------
        dict
            JSON-ready: ``columns``, ``rows``, ``width``, ``height``, ``radius``, ``cells``
            (each ``points`` and ``x``, ``y``, by cell id), and ``rungs`` (each ``name``,
            ``about``, the person's ``won``, ``lost`` and ``drawn``, and ``rating``, the mean
            of its ratings or None), weakest first.
        """
        shapes = board_shapes(self.game, SHAPE_SCALE)
        return {
            "columns": self.game.shape[0],
            "rows": self.game.shape[1],
            "width": shapes.width,
            "height": shapes.height,
            "radius": shapes.radius,
            "cells": [
                {
                    "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in shapes.polygons[cell]),
                    "x": float(shapes.centers[cell, 0]),
                    "y": float(shapes.centers[cell, 1]),
                }
                for cell in range(self.game.space.n_cells)
            ],
            "rungs": [self._standing(rung) for rung in self.ladder.values()],
        }

    def _standing(self, rung: Rung) -> JsonObject:
        """Return a rung on the page: what it is, and how the person has done against it."""
        with self._lock:
            games = [
                r
                for r in self.records
                if r["rung"] == rung.name and r.get("players") == self.players
            ]
        ratings = [
            r["note"]["rating"] for r in games if r["note"] and r["note"]["rating"] is not None
        ]
        return {
            "name": rung.name,
            "about": rung.about,
            "won": sum(r["result"] == "won" for r in games),
            "lost": sum(r["result"] == "lost" for r in games),
            "drawn": sum(r["result"] == "drawn" for r in games),
            "rating": sum(ratings) / len(ratings) if ratings else None,
        }

    def new(self, rung: str, seat: int, board: str = "", noise: bool = False) -> JsonObject:
        """Sit down to a new game.

        Parameters
        ----------
        rung : str
            The name of the player to face.
        seat : int
            The person's seat: 0 moves first, as red.
        board : str, default ""
            The board to start from (`start_of`), or the game's own start.
        noise : bool, default False
            Whether a search player draws Gumbel noise at each decision. On, it varies its play as
            in scored games, and can blunder at any budget. Off, it plays its best as it sees it,
            and answers the same moves the same way. Other players ignore it, and record False.

        Returns
        -------
        dict
            The game (`state`).

        Raises
        ------
        TypeError, ValueError
            If the seat isn't an integer, there is no such rung, the seat isn't 0 or 1, or the
            board isn't one a game reaches.
        """
        if rung not in self.ladder:
            raise ValueError(f"no rung is named {rung!r}")
        if check_int(seat, "seat", 0) > 1:
            raise ValueError(f"seat must be 0 or 1, got {seat}")
        with self._lock:
            number = self._next
            self._next += 1
            position = start_of(self.game, board) if board else self.game.start(number)
            player = self.ladder[rung].player
            searches = isinstance(player, AlphaZeroPlayer)
            self._sittings[number] = _Sitting(
                number,
                self.ladder[rung],
                replace(player, noise=noise) if isinstance(player, AlphaZeroPlayer) else player,
                noise and searches,
                seat,
                board_key(position.state.owner),
                datetime.now(UTC).isoformat(timespec="seconds"),
                np.random.default_rng([self.seed, number]),
                position,
                time.monotonic(),
            )
            return self._state(self._sittings[number])

    def state(self, number: int) -> JsonObject:
        """Return a game as the page reads it.

        Parameters
        ----------
        number : int
            The game's number, from `new`.

        Returns
        -------
        dict
            JSON-ready: its record (``game``, ``rung``, ``players``, ``noise``, ``seat``, ``start``,
            ``played_at``, ``moves``, ``winner``, ``result``, ``note``) and ``about``, ``over``,
            ``to_move`` (None once over), ``legal`` (the free cells a stone can land on) and
            ``line`` (the cells of the winning lines).

        Raises
        ------
        TypeError, ValueError
            If the number has the wrong type, or no game of this run has it.
        """
        with self._lock:
            return self._state(self._sitting(number))

    def move(self, number: int, column: int) -> JsonObject:
        """Place the person's stone.

        Parameters
        ----------
        number : int
            The game's number.
        column : int
            The column, from 0.

        Returns
        -------
        dict
            The game (`state`).

        Raises
        ------
        TypeError, ValueError
            If an argument has the wrong type, the game has ended, it isn't the person's turn,
            or the column is off the board or full.
        """
        with self._lock:
            sitting = self._sitting(number)
            if sitting.position.over or self.game.player(sitting.position) != sitting.seat:
                raise ValueError("it isn't your move")
            check_int(column, "column", 0)
            cells = np.flatnonzero(self.game.legal(sitting.position)[PLACE])
            landing = cells[cells // self.game.shape[1] == column]
            if len(landing) == 0:
                raise ValueError(f"no stone can go in column {column}")
            seconds = time.monotonic() - sitting.offered
            self._place(sitting, "person", int(landing[0]), seconds, None)
            return self._state(sitting)

    def reply(self, number: int) -> JsonObject:
        """Have the player at the table move.

        Parameters
        ----------
        number : int
            The game's number.

        Returns
        -------
        dict
            The game (`state`).

        Raises
        ------
        TypeError, ValueError
            If the number has the wrong type or no game has it, the game has ended, or it is the
            person's turn.
        """
        with self._lock:
            sitting = self._sitting(number)
            position = sitting.position
            if position.over or self.game.player(position) == sitting.seat:
                raise ValueError("it isn't this player's move")
            begun = time.monotonic()
            player = sitting.player
            why = None
            if isinstance(player, AlphaZeroPlayer):
                result = player.search(self.game, position, sitting.rng)
                cell = result.action.cell
                why = explain(self.game, player.evaluator, position, result)
            else:
                mover = self.game.player(position)
                legal = self.game.legal(position)
                cell = player(self.game, mover, position, legal, sitting.rng).cell
            seconds = time.monotonic() - begun
            self._place(sitting, sitting.rung.name, int(cell), seconds, why)
            return self._state(sitting)

    def note(
        self, number: int, rating: int | None, interesting: Sequence[int], text: str
    ) -> JsonObject:
        """Keep a person's note on a finished game with its record, replacing any before.

        Parameters
        ----------
        number : int
            The game's number.
        rating : int or None
            How much the person enjoyed it, from 1 to 5, or None for no rating.
        interesting : sequence of int
            The plies of the moves the person found interesting.
        text : str
            Anything else.

        Returns
        -------
        dict
            The game (`state`).

        Raises
        ------
        TypeError, ValueError
            If an argument has the wrong type, the game isn't over, the rating is outside 1 to
            5, or a ply isn't one of the game's.
        """
        if rating is not None and check_int(rating, "rating", 1) > 5:
            raise ValueError(f"rating must be at most 5, got {rating}")
        with self._lock:
            sitting = self._sitting(number)
            if not sitting.position.over:
                raise ValueError("the game isn't over")
            plies = {move.ply for move in sitting.moves}
            for ply in interesting:
                check_int(ply, "a ply")
                if ply not in plies:
                    raise ValueError(f"the game has no move at ply {ply}")
            sitting.note = {"rating": rating, "interesting": list(interesting), "text": text}
            for i, record in enumerate(self.records):
                if record["game"] == sitting.number:
                    self.records[i] = self._record(sitting)
            self._save()
            return self._state(sitting)

    def _sitting(self, number: int) -> _Sitting:
        """Return the game of a number, or raise as `state` says."""
        check_int(number, "a game's number")
        if number not in self._sittings:
            raise ValueError(f"no game of this run has the number {number}")
        return self._sittings[number]

    def _place(
        self, sitting: _Sitting, by: str, cell: int, seconds: float, why: Why | None
    ) -> None:
        """Step a game by a stone, and write its record if that ends it."""
        position = sitting.position
        sitting.moves.append(
            Move(
                position.ply,
                self.game.player(position),
                by,
                cell,
                cell // self.game.shape[1],
                round(seconds, 3),
                why,
            )
        )
        sitting.position = self.game.step(position, (PLACE, cell))
        sitting.offered = time.monotonic()
        if sitting.position.over:
            self.records.append(self._record(sitting))
            self._save()

    def _record(self, sitting: _Sitting) -> JsonObject:
        """Return what is kept of a game, JSON-ready."""
        over = sitting.position.over
        winner = sitting.position.winner
        result = None
        if over:
            result = "drawn" if winner is None else "won" if winner == sitting.seat else "lost"
        return {
            "game": sitting.number,
            "rung": sitting.rung.name,
            "players": self.players,
            "noise": sitting.noise,
            "seat": sitting.seat,
            "start": sitting.start,
            "played_at": sitting.played_at,
            "moves": [asdict(move) for move in sitting.moves],
            "winner": winner,
            "result": result,
            "note": sitting.note,
        }

    def _state(self, sitting: _Sitting) -> JsonObject:
        """Return a game as the page reads it (`state`)."""
        position = sitting.position
        over = position.over
        owner = position.state.owner
        line: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        if position.winner is not None:
            won = (owner[self.game.lines] == position.winner).all(axis=1)
            line = np.unique(self.game.lines[won])
        return {
            **self._record(sitting),
            "about": sitting.rung.about,
            "over": over,
            "to_move": None if over else self.game.player(position),
            "legal": [] if over else np.flatnonzero(self.game.legal(position)[PLACE]).tolist(),
            "line": line.tolist(),
        }

    def _save(self) -> None:
        """Write the records whole, so that a crash leaves the last complete file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = self.path.with_name(f"{self.path.name}.tmp")
        written.write_text(json.dumps(self.records, indent=1, allow_nan=False))
        os.replace(written, self.path)


def page() -> str:
    """Return the page, with the reports' styles in it.

    Returns
    -------
    str
        One HTML document, which needs nothing but the routes of `serve`.
    """
    return (Path(__file__).parent / "play.html").read_text().replace("/* STYLE */", STYLE)


def _field[T](data: JsonObject, name: str, kind: type[T]) -> T:
    """Return a field of a request's JSON, raising ValueError or TypeError if it can't be read."""
    if name not in data:
        raise ValueError(f"the request has no {name!r}")
    value = data[name]
    if not isinstance(value, kind):
        raise TypeError(f"{name!r} must be {kind.__name__}, got {type(value).__name__}")
    return value


def _rating(data: JsonObject) -> int | None:
    """Return the rating of a request's note: None if it has none."""
    return None if data.get("rating") is None else _field(data, "rating", int)


def serve(table: Table, port: int) -> ThreadingHTTPServer:
    """Return a server of the page and its routes, on this machine only, not yet serving.

    ``GET /`` is the page and ``GET /api/setup`` is `Table.setup`. ``POST`` to ``/api/new``,
    ``/api/state``, ``/api/move``, ``/api/reply`` and ``/api/note`` calls those of `Table`, with
    JSON fields of those names and ``game`` for the number; each answers with the game. A wrong
    request is answered 400, with ``error`` in JSON saying why.

    Only requests addressed to this machine are served, and a write must say it is JSON: no other
    site can make a browser send that without asking first, which nothing here answers, or reach
    the routes through a name of its own.

    Parameters
    ----------
    table : Table
        The games.
    port : int
        The port, or 0 for any free one (``server.server_port``).

    Returns
    -------
    ThreadingHTTPServer
        The server: call ``serve_forever``, and ``server_close`` after.
    """
    routes: dict[str, Callable[[JsonObject], JsonObject]] = {
        "/api/new": lambda d: table.new(
            _field(d, "rung", str),
            _field(d, "seat", int),
            _field({"board": "", **d}, "board", str),
            _field({"noise": False, **d}, "noise", bool),
        ),
        "/api/state": lambda d: table.state(_field(d, "game", int)),
        "/api/move": lambda d: table.move(_field(d, "game", int), _field(d, "column", int)),
        "/api/reply": lambda d: table.reply(_field(d, "game", int)),
        "/api/note": lambda d: table.note(
            _field(d, "game", int),
            _rating(d),
            _field(d, "interesting", list),
            _field(d, "text", str),
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        """Answers the page's requests."""

        def _send(self, status: int, kind: str, body: bytes) -> None:
            """Send an answer whole."""
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, answer: JsonObject) -> None:
            """Send an answer in JSON."""
            body = json.dumps(answer, allow_nan=False).encode()
            self._send(status, "application/json", body)

        def _addressed_here(self) -> bool:
            """Return whether the request names this machine, having refused it if not."""
            allowed = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}
            if self.headers.get("Host") in allowed:
                return True
            self._send_json(403, {"error": "this page serves this machine only"})
            return False

        def do_GET(self) -> None:
            """Answer the page and its setup."""
            if not self._addressed_here():
                return
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", page().encode())
            elif self.path == "/api/setup":
                self._send_json(200, table.setup())
            else:
                self._send_json(404, {"error": f"nothing is at {self.path}"})

        def do_POST(self) -> None:
            """Answer a route of `Table`."""
            try:
                length = max(0, int(self.headers.get("Content-Length") or 0))
            except ValueError:
                self._send_json(400, {"error": "Content-Length must be an integer"})
                return
            # The body is read first: closing with it unread could lose the answer on its way.
            body = self.rfile.read(length)
            if not self._addressed_here():
                return
            route = routes.get(self.path)
            if route is None:
                self._send_json(404, {"error": f"nothing is at {self.path}"})
                return
            if self.headers.get("Content-Type") != "application/json":
                self._send_json(415, {"error": "a request must be JSON"})
                return
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise TypeError("a request must be a JSON object")
                answer = route(data)
            except (TypeError, ValueError) as error:  # JSON's errors are ValueErrors
                self._send_json(400, {"error": str(error)})
                return
            self._send_json(200, answer)

        def log_message(self, format: str, *args: object) -> None:
            """Print nothing per request: the terminal is left to the experiment."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server
