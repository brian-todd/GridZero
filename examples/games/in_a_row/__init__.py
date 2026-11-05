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

Code players
------------
LineEvaluator, SEARCH_ALONE, open_lines
    Evaluators written as code for `agents.AlphaZeroPlayer`: open lines and the middle
    column, the search with no knowledge, and the lines each player keeps open.
FittedEvaluator, value_features, move_features, VALUE_FEATURES, MOVE_FEATURES
    An evaluator of named features and fitted constants, and those features.
threat_cells, clean_threats
    Who could complete a line where, and which of those threats no rival threat comes before.
Lessons, distinct_positions, judge, position_of
    Positions a teacher's search has judged. The fit of constants to them is in `fit.py`, not
    imported here: every worker of a series imports this package, and scikit-learn is slow.
    Nor are `report.py`, which needs it, and `play.py`, the page a person plays on, built on it.

Strength
--------
opening, Opening, EVAL_SEED, FIRST_MOVES
    What an evaluator makes of the empty board and of each first move (a player's record
    against rivals is `game.ladder`'s), the games players are scored on, and perfect play's
    value of each first column.

Text
----
board_text, STONES
    The board as text.
"""

from examples.games.in_a_row.code_player import (
    MOVE_FEATURES,
    SEARCH_ALONE,
    VALUE_FEATURES,
    FittedEvaluator,
    LineEvaluator,
    clean_threats,
    fitted_values,
    move_features,
    open_lines,
    threat_cells,
    value_features,
)
from examples.games.in_a_row.game import PLACE, InARow
from examples.games.in_a_row.lessons import Lessons, distinct_positions, judge, position_of
from examples.games.in_a_row.match import EVAL_SEED, FIRST_MOVES, Opening, opening
from examples.games.in_a_row.players import ThreatPlayer
from examples.games.in_a_row.state import (
    InARowPosition,
    InARowState,
    Placed,
)
from examples.games.in_a_row.text import STONES, board_text

__all__ = [
    "EVAL_SEED",
    "FIRST_MOVES",
    "MOVE_FEATURES",
    "PLACE",
    "SEARCH_ALONE",
    "STONES",
    "VALUE_FEATURES",
    "FittedEvaluator",
    "InARow",
    "InARowPosition",
    "InARowState",
    "Lessons",
    "LineEvaluator",
    "Opening",
    "Placed",
    "ThreatPlayer",
    "board_text",
    "clean_threats",
    "distinct_positions",
    "fitted_values",
    "judge",
    "move_features",
    "open_lines",
    "opening",
    "position_of",
    "threat_cells",
    "value_features",
]
