"""Code players of k in a row: evaluators written as code, for the AlphaZero search.

A code player is `agents.AlphaZeroPlayer` on an evaluator that is plain code: named features
of a position, and constants. It needs no network, so it can ship, and it still searches. The
evaluators here judge positions in batches, with array arithmetic only. `LineEvaluator`'s
constants are picked by hand; `FittedEvaluator`'s are fit (`fit.py`) to what a trained agent
judged (`lessons.py`).
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from examples.games.fitting import sigmoid, weighted
from gridzero.core import check_real

from .game import PLACE, InARow
from .state import InARowPosition


def open_lines(game: InARow, positions: Sequence[InARowPosition]) -> NDArray[np.int64]:
    """Count each player's open lines by their stones.

    A line is open for a player when the other has no stone on it.

    Parameters
    ----------
    game : InARow
        The rules.
    positions : sequence of InARowPosition
        The positions.

    Returns
    -------
    ndarray of int64, shape (n_positions, 2, length + 1)
        ``counts[i, p, c]``: the lines of position ``i`` holding ``c`` stones of player ``p``
        and none of the other's. Lines with no stone at all count for both players at ``c = 0``.
    """
    owner = np.stack([position.state.owner for position in positions])  # (n, n_cells) int8
    on_line = owner[:, game.lines]  # (n, n_lines, length)
    stones = np.stack([(on_line == 0).sum(axis=2), (on_line == 1).sum(axis=2)], axis=1)
    open_ = stones[:, ::-1] == 0  # open for player p: no stone of 1 - p
    length = game.length
    counts = np.zeros((len(positions), 2, length + 1), dtype=np.int64)
    for c in range(length + 1):  # length + 1 passes over (n, 2, n_lines): no Python per line
        counts[:, :, c] = ((stones == c) & open_).sum(axis=2)
    return counts


@cache
def _distance_to_center(game: InARow) -> NDArray[np.float64]:
    """Return each cell's distance to the board's middle column, shape (n_cells,), read-only."""
    columns = game.space.unravel(np.arange(game.space.n_cells))[:, 0]
    distance = np.abs(columns - (game.shape[0] - 1) / 2).astype(np.float64)
    distance.flags.writeable = False
    return distance


@dataclass(frozen=True)
class LineEvaluator:
    """Judges positions by their open lines, and prefers moves near the middle column.

    The value of a position for player 0 is ``sigmoid(sum_c weights[c - 1] * (L0[c] - L1[c]))``
    over ``c = 1 .. length - 1``, where ``Lp[c]`` counts the lines open for player ``p`` with
    ``c`` of its stones (`open_lines`). The prior's logit of a stone is ``-center`` times its
    distance to the middle column. All weights 0 and ``center`` 0 give the search alone: a flat
    prior and every position even, so only the rules' own ends inform it.

    Parameters
    ----------
    weights : tuple of float, default (0.15, 0.45, 1.35)
        The weight of an open line with 1, 2, ... ``length - 1`` stones, in logits. The
        default, for four in a row, was picked by hand, not fitted.
    center : float, default 0.5
        How much the prior prefers the middle column, in logits per column; at least 0.

    Raises
    ------
    TypeError, ValueError
        If a weight is not a finite real, or ``center`` not a finite real of at least 0.
    """

    weights: tuple[float, ...] = (0.15, 0.45, 1.35)
    center: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.weights, tuple):
            raise TypeError(f"weights must be a tuple, got {type(self.weights).__name__}")
        for weight in self.weights:
            check_real(weight, "a weight")
        check_real(self.center, "center", 0.0)

    def __call__(
        self,
        game: InARow,
        positions: Sequence[InARowPosition],
        legal: Sequence[NDArray[np.bool_]],
        /,
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the prior's logits and player 0's value of each position (see `Evaluator`).

        Raises
        ------
        ValueError
            If there is not one weight per count of stones short of a line
            (``game.length - 1``), or a position's score overflows.
        """
        if len(self.weights) != game.length - 1:
            raise ValueError(
                f"{game.length}-in-a-row needs {game.length - 1} weights, got {len(self.weights)}"
            )
        counts = open_lines(game, positions)[:, :, 1 : game.length]  # (n, 2, length - 1)
        lead = (counts[:, 0] - counts[:, 1]).astype(np.float64)  # (n, length - 1)
        score = weighted(lead, self.weights, 0.0)
        values = sigmoid(score)
        distance = _distance_to_center(game)
        logits = [-self.center * distance[np.flatnonzero(mask[PLACE])] for mask in legal]
        return logits, values


SEARCH_ALONE = LineEvaluator((0.0, 0.0, 0.0), center=0.0)
"""The search with no knowledge of four in a row but its rules: the baseline of code players."""

SEARCH_C_SCALE = 0.1
"""The value weight (`agents.AlphaZeroPlayer`'s ``c_scale``) of code players' searches, and of those
they are laddered against: they were fit and measured at 0.1, the default until 2026-11-09."""


VALUE_FEATURES = (
    "open_one",
    "open_two",
    "first_threats_odd",
    "first_threats_even",
    "second_threats_odd",
    "second_threats_even",
    "win_now",
    "lost_now",
    "center",
    "beside_center",
    "to_move",
    "first_clean_odd",
    "second_clean_even",
)
"""The features of a position, in the order of `value_features`, each player 0's way: lines
open with one and two stones, player 0's minus player 1's; the free cells that would complete
a line for the first player (0) and the second (1), on odd and even rows counted from 1 at the
bottom; 1 toward the player to move if it can win now; 1 against the player to move if the
other has two wins ready and it has none; stones in the center column and the columns beside
it, player 0's minus player 1's; 1 toward the player to move; and the threats that count most
in theory: the first player's on odd rows and the second's on even rows, each with no threat of
the other's lower in its column, which would be played first."""

MOVE_FEATURES = (
    "center",
    "own_empty",
    "own_one",
    "own_two",
    "win",
    "cut_one",
    "cut_two",
    "block",
    "gives_win",
    "under_own_win",
    "own_parity",
)
"""The features of a stone, in the order of `move_features`, for the player placing it: minus
its distance to the middle column; lines through it open for the player with no, one, two and
three of its stones (three wins); lines through it open for the other with one, two and three
of its stones (three blocks a win); whether the cell above wins for the other, or for the
player (so the other can take it); and whether its row is of the player's parity (odd rows,
counted from 1, for the first player)."""


class _Board(NamedTuple):
    """What features read of a board, computed once per game."""

    columns: NDArray[np.int64]
    """Each cell's column, shape (n_cells,)."""
    rows: NDArray[np.int64]
    """Each cell's row, from 0 at the bottom, shape (n_cells,)."""
    center: NDArray[np.int64]
    """The cells' columns nearest the middle, shape (n_center,)."""
    beside: NDArray[np.int64]
    """The columns next nearest, shape (n_beside,)."""
    above: NDArray[np.int64]
    """The cell above each, shape (n_cells,); ``n_cells`` above the top row."""
    through: NDArray[np.bool_]
    """Whether each line passes through each cell, shape (n_cells, n_lines)."""


@cache
def _board(game: InARow) -> _Board:
    """Return what features read of the game's board, after checking they apply to it."""
    if game.length != 4 or not game.gravity:
        raise ValueError("these features are for four in a row with gravity")
    n_cells = game.space.n_cells
    where = game.space.unravel(np.arange(n_cells))
    columns, rows = where[:, 0], where[:, 1]
    middle = np.abs(np.arange(game.shape[0]) - (game.shape[0] - 1) / 2)
    above = np.where(rows + 1 < game.shape[1], np.arange(n_cells) + 1, n_cells)
    through = np.zeros((n_cells, len(game.lines)), dtype=bool)
    through[game.lines, np.arange(len(game.lines))[:, None]] = True
    return _Board(
        columns,
        rows,
        np.flatnonzero(middle == middle.min()),
        np.flatnonzero(middle == middle.min() + 1),
        above,
        through,
    )


def _threats(
    game: InARow, on_line: NDArray[np.int8], mine: NDArray[np.int64], theirs: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Return which free cells complete a line for a player.

    Parameters
    ----------
    game : InARow
        The rules.
    on_line : ndarray of int8, shape (n, n_lines, length)
        The owners along each line.
    mine, theirs : ndarray of int64, shape (n, n_lines)
        The stones of the player and of the other on each line.

    Returns
    -------
    ndarray of bool, shape (n, n_cells)
        Whether each cell completes a line of the player's.
    """
    found = np.zeros((len(on_line), game.space.n_cells), dtype=bool)
    i, line = np.nonzero((mine == game.length - 1) & (theirs == 0))
    free = np.argmax(on_line[i, line] == -1, axis=1)
    found[i, game.lines[line, free]] = True
    return found


def threat_cells(game: InARow, owner: NDArray[np.int8]) -> NDArray[np.bool_]:
    """Return which free cells complete a line for each player.

    Parameters
    ----------
    game : InARow
        The rules.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell of each position.

    Returns
    -------
    ndarray of bool, shape (n, 2, n_cells)
        Whether each cell completes a line for each player.
    """
    on_line = owner[:, game.lines]
    first, second = (on_line == 0).sum(axis=2), (on_line == 1).sum(axis=2)
    return np.stack(
        [_threats(game, on_line, first, second), _threats(game, on_line, second, first)], axis=1
    )


def clean_threats(mine: NDArray[np.bool_], other: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return a player's threats with no threat of the other's lower in their column.

    Parameters
    ----------
    mine, other : ndarray of bool, shape (n, columns, rows)
        The threat cells of the player and of the other.

    Returns
    -------
    ndarray of bool, shape (n, columns, rows)
        Those of ``mine`` that hold: a rival threat below is reached first.
    """
    lower = np.cumsum(other, axis=2)[:, :, :-1] > 0
    return mine & ~np.concatenate([np.zeros_like(other[:, :, :1]), lower], axis=2)


def value_features(
    game: InARow, owner: NDArray[np.int8], ply: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the features of positions (`VALUE_FEATURES`), each player 0's way.

    Parameters
    ----------
    game : InARow
        The rules: four in a row with gravity.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell of each position.
    ply : ndarray of int64, shape (n,)
        Each position's ply: player 0 moves on even plies.

    Returns
    -------
    ndarray of float64, shape (n, len(VALUE_FEATURES))
        The features.

    Raises
    ------
    ValueError
        If the game isn't four in a row with gravity.
    """
    board = _board(game)
    n = len(owner)
    on_line = owner[:, game.lines]  # (n, n_lines, 4)
    first, second = (on_line == 0).sum(axis=2), (on_line == 1).sum(axis=2)
    threats = (_threats(game, on_line, first, second), _threats(game, on_line, second, first))
    odd = board.rows % 2 == 0  # odd rows, counted from 1
    grid = owner.reshape(n, *game.shape)  # (n, columns, rows)
    heights = (grid >= 0).sum(axis=2)  # (n, columns)
    playable = np.zeros((n, game.space.n_cells), dtype=bool)
    i, column = np.nonzero(heights < game.shape[1])
    playable[i, column * game.shape[1] + heights[i, column]] = True
    ready = (threats[0] & playable).sum(axis=1), (threats[1] & playable).sum(axis=1)
    first_to_move = ply % 2 == 0
    mover_ready = np.where(first_to_move, ready[0], ready[1])
    other_ready = np.where(first_to_move, ready[1], ready[0])
    toward_mover = np.where(first_to_move, 1.0, -1.0)
    features = np.zeros((n, len(VALUE_FEATURES)))
    features[:, 0] = ((first == 1) & (second == 0)).sum(axis=1) - (
        (second == 1) & (first == 0)
    ).sum(axis=1)
    features[:, 1] = ((first == 2) & (second == 0)).sum(axis=1) - (
        (second == 2) & (first == 0)
    ).sum(axis=1)
    features[:, 2] = (threats[0] & odd).sum(axis=1)
    features[:, 3] = (threats[0] & ~odd).sum(axis=1)
    features[:, 4] = (threats[1] & odd).sum(axis=1)
    features[:, 5] = (threats[1] & ~odd).sum(axis=1)
    features[:, 6] = np.where(mover_ready > 0, toward_mover, 0.0)
    features[:, 7] = np.where((other_ready >= 2) & (mover_ready == 0), -toward_mover, 0.0)
    for j, columns in ((8, board.center), (9, board.beside)):
        in_them = grid[:, columns]  # (n, n_columns, rows)
        features[:, j] = (in_them == 0).sum(axis=(1, 2)) - (in_them == 1).sum(axis=(1, 2))
    features[:, 10] = toward_mover
    first_grid = threats[0].reshape(n, *game.shape)
    second_grid = threats[1].reshape(n, *game.shape)
    features[:, 11] = clean_threats(first_grid, second_grid)[:, :, ::2].sum(axis=(1, 2))
    features[:, 12] = clean_threats(second_grid, first_grid)[:, :, 1::2].sum(axis=(1, 2))
    return features


def move_features(
    game: InARow, owner: NDArray[np.int8], ply: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the legal stones of a position and their features (`MOVE_FEATURES`).

    Parameters
    ----------
    game : InARow
        The rules: four in a row with gravity.
    owner : ndarray of int8, shape (n_cells,)
        Who owns each cell.
    ply : int
        The position's ply: player 0 moves on even plies.

    Returns
    -------
    cells : ndarray of int64, shape (n_legal,)
        The legal stones, ascending: the order of ``np.flatnonzero(game.legal(position)[PLACE])``.
    features : ndarray of float64, shape (n_legal, len(MOVE_FEATURES))
        Their features.

    Raises
    ------
    ValueError
        If the game isn't four in a row with gravity.
    """
    board = _board(game)
    me = ply % 2
    on_line = owner[game.lines][None]  # (1, n_lines, 4)
    mine, theirs = (on_line == me).sum(axis=2), (on_line == 1 - me).sum(axis=2)
    heights = (owner.reshape(game.shape) >= 0).sum(axis=1)
    columns = np.flatnonzero(heights < game.shape[1])
    cells = columns * game.shape[1] + heights[columns]
    through = board.through[cells]  # (n_legal, n_lines)
    features = np.zeros((len(cells), len(MOVE_FEATURES)))
    features[:, 0] = -_distance_to_center(game)[cells]
    for j, stones in enumerate(range(4)):  # own_empty, own_one, own_two, win
        features[:, 1 + j] = (through & ((mine[0] == stones) & (theirs[0] == 0))).sum(axis=1)
    for j, stones in enumerate(range(1, 4)):  # cut_one, cut_two, block
        features[:, 5 + j] = (through & ((theirs[0] == stones) & (mine[0] == 0))).sum(axis=1)
    above = board.above[cells]  # n_cells past the top row: the padding below is False
    own = np.append(_threats(game, on_line, mine, theirs)[0], False)
    other = np.append(_threats(game, on_line, theirs, mine)[0], False)
    features[:, 8] = other[above]
    features[:, 9] = own[above]
    features[:, 10] = board.rows[cells] % 2 == me
    return cells, features


@dataclass(frozen=True)
class FittedEvaluator:
    """Judges positions by named features and fitted constants.

    The value of a position for player 0 is ``sigmoid(intercept + sum_j value[j] * f[j])`` over
    `value_features`. The prior's logit of a stone is ``sum_j prior[j] * g[j]`` over
    `move_features`.

    Parameters
    ----------
    value : tuple of float
        A weight per name in `VALUE_FEATURES`, in logits.
    intercept : float
        The value's score when every feature is 0, in logits.
    prior : tuple of float
        A weight per name in `MOVE_FEATURES`, in logits.

    Raises
    ------
    TypeError, ValueError
        If a constant is not a finite real, or a tuple has the wrong length.
    """

    value: tuple[float, ...]
    intercept: float
    prior: tuple[float, ...]

    def __post_init__(self) -> None:
        for weights, names, name in (
            (self.value, VALUE_FEATURES, "value"),
            (self.prior, MOVE_FEATURES, "prior"),
        ):
            if not isinstance(weights, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(weights).__name__}")
            if len(weights) != len(names):
                raise ValueError(f"{name} needs {len(names)} weights, got {len(weights)}")
            for weight in weights:
                check_real(weight, f"a weight of {name}")
        check_real(self.intercept, "intercept")

    def __call__(
        self,
        game: InARow,
        positions: Sequence[InARowPosition],
        legal: Sequence[NDArray[np.bool_]],
        /,
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the prior's logits and player 0's value of each position (see `Evaluator`).

        Raises
        ------
        ValueError
            If the game isn't four in a row with gravity, or a score overflows.
        """
        owner = np.stack([position.state.owner for position in positions])
        ply = np.array([position.ply for position in positions], dtype=np.int64)
        values = fitted_values(game, self, owner, ply)
        logits = [
            weighted(move_features(game, owner[i], int(ply[i]))[1], self.prior, 0.0)
            for i in range(len(positions))
        ]
        return logits, values

    def save(self, path: str | Path) -> None:
        """Write the constants as JSON, by name, so they can be read and ported.

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        fields = asdict(self)
        Path(path).write_text(
            json.dumps(
                {
                    "value": dict(zip(VALUE_FEATURES, fields["value"], strict=True)),
                    "intercept": fields["intercept"],
                    "prior": dict(zip(MOVE_FEATURES, fields["prior"], strict=True)),
                },
                indent=2,
            )
        )

    @staticmethod
    def load(path: str | Path) -> FittedEvaluator:
        """Read constants written by `save`.

        Parameters
        ----------
        path : str or Path
            The file to read.

        Returns
        -------
        FittedEvaluator
            The evaluator.

        Raises
        ------
        KeyError
            If a feature's weight is missing: the file is of another set of features.
        """
        data = json.loads(Path(path).read_text())
        return FittedEvaluator(
            tuple(float(data["value"][name]) for name in VALUE_FEATURES),
            float(data["intercept"]),
            tuple(float(data["prior"][name]) for name in MOVE_FEATURES),
        )


def fitted_values(
    game: InARow, fit: FittedEvaluator, owner: NDArray[np.int8], ply: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return a fit's values of positions, without a search.

    Parameters
    ----------
    game : InARow
        The rules.
    fit : FittedEvaluator
        The constants.
    owner : ndarray of int8, shape (n, n_cells)
        Who owns each cell.
    ply : ndarray of int64, shape (n,)
        Each position's ply.

    Returns
    -------
    ndarray of float64, shape (n,)
        Player 0's value of each.

    Raises
    ------
    ValueError
        If a score overflows.
    """
    return sigmoid(weighted(value_features(game, owner, ply), fit.value, fit.intercept))
