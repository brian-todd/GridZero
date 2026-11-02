"""k in a row: a family of two-player games of lines, the second game.

Players take turns placing stones on a square board; the first with ``length`` in a straight
line wins. ``InARow()`` is Connect Four's rules (7 columns of 6, four in a row, stones falling
to the lowest free cell), and ``InARow((3, 3), 3, gravity=False)`` is tic-tac-toe. The game is
built on `game` and `agents` as they stood, to find what they lacked.

Game
----
InARow, PLACE
    The rules, as an immutable value, a `game.Game` and an `agents.Learnable` (``planes``,
    ``observe``, ``frame``), and their one verb.

State
-----
InARowState, InARowPosition, Placed
    Who owns each cell, a position, and what a step did.

Players
-------
ThreatPlayer
    Wins when it can, blocks, and never hands over a win: the scripted yardstick.

Strength
--------
opening, Opening, EVAL_SEED
    What an evaluator makes of the empty board (a player's record against rivals is
    `game.ladder`'s), and the games players are scored on.

Text
----
board_text, STONES
    The board as text.
"""

from examples.games.in_a_row.game import PLACE, InARow
from examples.games.in_a_row.match import EVAL_SEED, Opening, opening
from examples.games.in_a_row.players import ThreatPlayer
from examples.games.in_a_row.state import (
    InARowPosition,
    InARowState,
    Placed,
)
from examples.games.in_a_row.text import STONES, board_text

__all__ = [
    "EVAL_SEED",
    "PLACE",
    "STONES",
    "InARow",
    "InARowPosition",
    "InARowState",
    "Opening",
    "Placed",
    "ThreatPlayer",
    "board_text",
    "opening",
]
