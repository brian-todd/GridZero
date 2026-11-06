"""What a game of Tactics is fought over: every rule about the point, as small pure functions.

One tile is the point. At the end of each round a unit standing on it takes it for its side, and
the side keeps it until an enemy stands there at a round's end. Each round's end adds 1 to the
owner's clock, the capture included, and a flip restarts the clock at 1 for the new owner.

A game ends at a round's end once the clock reaches ``hold_rounds``, once nobody has captured by
``capture_by`` (team 1 loses: it places last and sees everything, so it must take the first step),
or at ``max_rounds``, which the point's owner wins. Between them they leave no draws.

To try another objective, change this file and `state.Hold`, and the planes of `Tactics.observe`
that read them. `tactics.rules` then measures it against any earlier version.
"""

import numpy as np
from numpy.typing import NDArray

from .events import Ending
from .state import Hold

DEADLINE_LOSER = 1
"""The team that loses if nobody captures in time. It makes the seats different, so a player has
to be told which one it is in (the plane ``"loses the deadline"`` of `Tactics.observe`)."""


def after_round(hold: Hold, point: NDArray[np.bool_], team: NDArray[np.int8]) -> Hold:
    """Return who holds the point once a round has ended, and for how long.

    Parameters
    ----------
    hold : Hold
        Who held it during the round.
    point : ndarray of bool, shape (n_cells,)
        The point's tile.
    team : ndarray of int8, shape (n_cells,)
        The team of the unit on each tile, -1 on an empty one.

    Returns
    -------
    Hold
        The unit on the point's side, or the owner so far if it stands empty, with its clock one
        higher, or 1 after a flip. Unchanged while nobody has ever captured.
    """
    standing = team[point]
    standing = standing[standing >= 0]
    owner = int(standing[0]) if standing.size else hold.owner
    if owner < 0:
        return hold
    return Hold(owner, hold.clock + 1 if owner == hold.owner else 1)


def ending(
    hold: Hold, round_: int, *, hold_rounds: int, capture_by: int, max_rounds: int
) -> tuple[int, Ending] | None:
    """Return who wins, and why, if a game ends with the round that just ended.

    Parameters
    ----------
    hold : Hold
        Who holds the point after the round (`after_round`).
    round_ : int
        The round that just ended, from 1.
    hold_rounds : int
        Rounds in a row that win.
    capture_by : int
        The last round in which the first capture is in time.
    max_rounds : int
        The last round, at least ``capture_by``.

    Returns
    -------
    tuple of (int, str), or None
        The winning team and the reason, or None if the game goes on.
    """
    if hold.owner < 0:
        return (1 - DEADLINE_LOSER, "deadline") if round_ >= capture_by else None
    if hold.clock >= hold_rounds:
        return hold.owner, "held"
    if round_ >= max_rounds:
        return hold.owner, "cap"
    return None


def wanted(
    point: NDArray[np.bool_], hold: Hold, team: NDArray[np.int8], side: int, cell: int
) -> NDArray[np.int64]:
    """Return the tiles a unit has reason to walk to for the objective.

    Parameters
    ----------
    point : ndarray of bool, shape (n_cells,)
        The point's tile.
    hold : Hold
        Who holds the point now.
    team : ndarray of int8, shape (n_cells,)
        The team of the unit on each tile, -1 on an empty one.
    side : int
        The unit's team.
    cell : int
        Where the unit stands.

    Returns
    -------
    ndarray of int64, shape (k,)
        The point's tile, unless a friend stands there, or its side owns it and no enemy stands
        there. A unit standing on the point itself still wants it, so that it stays.
    """
    tiles = np.flatnonzero(point).astype(np.int64)
    standing = team[tiles]
    covered = (standing == side) | ((hold.owner == side) & (standing < 0))
    return tiles[(tiles == cell) | ~covered]
