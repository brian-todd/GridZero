"""Exact values of small deterministic games of two players, by searching every position.

`solve` gives player 0's value under perfect play of every position reachable from a start,
and `optimal` the actions that keep the best value for the player to move: the yardstick on
games small enough to search whole (tic-tac-toe has 5,478 positions), against which a learner's
decisions or a search's can be scored exactly. Both read a game only through `game.Game`
(`legal`, `step`, `player`) and `Position.key`, and draw nothing.

Values have no distance to the end: every move of a lost position has the same value, so a
player choosing among `optimal`'s actions plays a lost game aimlessly.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridzero.core import State
from gridzero.game import Game, Position


def solve[P: Position[State, object]](
    game: Game[P, Any],  # Any: the view, which a solver never reads
    start: P | None = None,
) -> dict[bytes, float]:
    """Return player 0's value under perfect play of every position reachable from ``start``.

    Each position's value is its result once over (1 if player 0 won, 0 if it lost, 0.5 for a
    draw); before that, the best value of its children for the player to move. The search
    visits each position once, keyed by `Position.key`, and goes as deep as the longest game
    in Python's recursion, so it suits only small games.

    Parameters
    ----------
    game : Game
        The rules, of a game of two players whose steps draw nothing.
    start : Position, optional
        Where to start; ``game.start(0)`` by default.

    Returns
    -------
    dict of bytes to float
        Player 0's value, 0, 0.5 or 1, by `Position.key`.

    Raises
    ------
    ValueError
        If the game doesn't have two players, or a step drew from its stream (``drew``): its
        outcome is chance, which a value under perfect play doesn't cover.
    """
    if game.n_players != 2:
        raise ValueError(f"solve plays games of two players, not {game.n_players}")
    values: dict[bytes, float] = {}
    _value(game, game.start(0) if start is None else start, values)
    return values


def optimal[P: Position[State, object]](
    game: Game[P, Any],  # Any: as in `solve`
    position: P,
    values: dict[bytes, float],
) -> NDArray[np.bool_]:
    """Return the legal actions that keep the best value for the player to move.

    Parameters
    ----------
    game : Game
        The rules.
    position : Position
        A position not over, reachable where ``values`` was solved from.
    values : dict of bytes to float
        From `solve`.

    Returns
    -------
    ndarray of bool, shape (n_verbs, n_cells)
        True on the legal actions whose child has the best value for the player to move.

    Raises
    ------
    ValueError
        If the game is over (`game.Game.player`).
    KeyError
        If a child of ``position`` was not solved.
    """
    mover = game.player(position)
    legal = game.legal(position)
    actions = np.argwhere(legal)
    child = np.array([values[game.step(position, (v, c)).key()] for v, c in actions.tolist()])
    if mover == 1:
        child = 1.0 - child
    best = np.zeros_like(legal)
    best[tuple(actions[child == child.max()].T)] = True
    return best


def _value[P: Position[State, object]](
    game: Game[P, Any], position: P, values: dict[bytes, float]
) -> float:
    """Return player 0's value of ``position``, solving and keeping every position below it."""
    key = position.key()
    known = values.get(key)
    if known is not None:
        return known
    if position.over:
        value = 0.5 if position.winner is None else float(position.winner == 0)
    else:
        children = []
        for verb, cell in np.argwhere(game.legal(position)).tolist():
            after = game.step(position, (verb, cell))
            if after.drew:
                raise ValueError(
                    f"the step ({verb}, {cell}) at ply {position.ply} drew from its stream: "
                    "solve needs a game without chance"
                )
            children.append(_value(game, after, values))
        value = max(children) if game.player(position) == 0 else min(children)
    values[key] = value
    return value
