"""Tactics: a two-player skirmish on a hex board, the first game.

Each side fills a team of 8 places under caps per kind as it deploys them in its corner of a
25x25 hex board with walls and rubble, then units take turns in an order rolled once, moving,
attacking and turning, until one side has none left or has held the point at the center for 4
rounds in a row. Positioning is the heart of it: flanking triples Atk, knights charge, captains
lift those near them, and footsoldiers hold together against the charge.

Game
----
Tactics
    The rules, as an immutable value and a `game.Game`: ``start``, ``legal``, ``step``,
    ``player``, ``view``, ``odds``.
TacticsPosition
    A complete position, a `game.Position`: the state, the seed, the ply, the round, the result,
    whether its step drew dice, and its events.
Action, Verb
    One decision, ``(verb, cell)``, and what verbs there are.
Odds
    What an attack would be: effective Atk and Def, flanking, and the chance of a hit.

Rules and board
---------------
UnitStats, DEFAULT_UNITS, DEFAULT_CAPS, default_roster
    Kinds of unit as data, the default kinds and their caps, and a team that fits them.
roll_needed, hits, hit_chance
    The d20 rule.
make_board, board_space, homes
    Boards symmetric for both sides under the half turn (`LatticeSpace.opposite`), their
    lattice, and their homes.

State
-----
TacticsState, Unit, Hold
    The board's fields and the units', per cell, and who holds the point.
Units, units
    The units on a board as a list, ordered by uid.

Events
------
Event, Ending, Placed, Moved, Swapped, Turned, Deferred, Built, Broke, Attacked, Died, Ended,
RoundStarted, Captured, GameOver
    What a step did: each position carries the events of the step that made it.

Players and matches
-------------------
RandomPlayer, GreedyPlayer, Fielded, SitterPlayer
    Who picks an action from the legal ones.
summarize, Summary, balance, Balance, EVAL_SEED, small_rules, earlier_rules
    What games show, how fair and lively they were, the boards players are scored on, and rules
    for quick games on a 15x15 board.
    `game.play` and `game.series` play them: `Tactics` is a `game.Game`.
Tactics.observe, Tactics.planes, Tactics.frame
    What a team sees: planes per cell in its own frame, for networks (`agents.frame.Learnable`).
event_text, summary_text
    The events and what a set of games showed, as text, for terminals and logs.
SearchPlayer, Plan, Judgment, material, standing
    A player that searches plans for whole unit turns by open-loop Monte Carlo tree search,
    judged by the greedy player.
Lessons, lessons_of, positions_of, teach
    A search player's games kept to learn from, as seeds and actions, with what it judged.
CodeEvaluator, HAND_PICKED, SEARCH_C_SCALE, value_features, move_features, VALUE_FEATURES,
MOVE_FEATURES, look_ahead
    A code player: judges positions by named features and constants, for `agents.AlphaZeroPlayer`,
    the value weight it searches with, and the position greedy play reaches a few unit turns on,
    where its value may be read.

`pictures.draw` draws a position on matplotlib's axes, with `render.cells`. It is not imported
here, so importing the game doesn't import matplotlib.
"""

from gridzero.game import Action

from .board import board_space, homes, make_board
from .code_player import (
    HAND_PICKED,
    MOVE_FEATURES,
    SEARCH_C_SCALE,
    VALUE_FEATURES,
    CodeEvaluator,
    look_ahead,
    move_features,
    value_features,
)
from .events import (
    Attacked,
    Broke,
    Built,
    Captured,
    Deferred,
    Died,
    Ended,
    Ending,
    Event,
    GameOver,
    Moved,
    Placed,
    RoundStarted,
    Swapped,
    Turned,
)
from .game import Odds, Tactics
from .lessons import Lessons, lessons_of, positions_of, teach
from .match import EVAL_SEED, Balance, Summary, balance, earlier_rules, small_rules, summarize
from .players import Fielded, GreedyPlayer, RandomPlayer, SitterPlayer
from .rules import (
    DEFAULT_CAPS,
    DEFAULT_UNITS,
    UnitStats,
    default_roster,
    hit_chance,
    hits,
    roll_needed,
)
from .search import Judgment, Plan, SearchPlayer, material, standing
from .state import Hold, TacticsPosition, TacticsState, Unit, Units, Verb, units
from .text import event_text, summary_text

__all__ = [
    "DEFAULT_CAPS",
    "DEFAULT_UNITS",
    "EVAL_SEED",
    "HAND_PICKED",
    "MOVE_FEATURES",
    "SEARCH_C_SCALE",
    "VALUE_FEATURES",
    "Action",
    "Attacked",
    "Balance",
    "Broke",
    "Built",
    "Captured",
    "CodeEvaluator",
    "Deferred",
    "Died",
    "Ended",
    "Ending",
    "Event",
    "Fielded",
    "GameOver",
    "GreedyPlayer",
    "Hold",
    "Judgment",
    "Lessons",
    "Moved",
    "Odds",
    "Placed",
    "Plan",
    "RandomPlayer",
    "RoundStarted",
    "SearchPlayer",
    "SitterPlayer",
    "Summary",
    "Swapped",
    "Tactics",
    "TacticsPosition",
    "TacticsState",
    "Turned",
    "Unit",
    "UnitStats",
    "Units",
    "Verb",
    "balance",
    "board_space",
    "default_roster",
    "earlier_rules",
    "event_text",
    "hit_chance",
    "hits",
    "homes",
    "lessons_of",
    "look_ahead",
    "make_board",
    "material",
    "move_features",
    "positions_of",
    "roll_needed",
    "small_rules",
    "standing",
    "summarize",
    "summary_text",
    "teach",
    "units",
    "value_features",
]
