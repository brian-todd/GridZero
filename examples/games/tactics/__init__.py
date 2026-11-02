"""Tactics: a two-player skirmish on a hex board, the first game.

Each side deploys 6 footsoldiers, 3 archers, 2 knights and a captain in its corner of a 24x24
hex board with walls and rubble, then units take turns in a shuffled order, moving, attacking
and turning, until one side has none left. Positioning is the heart of it: flanking triples
Atk, captains lift those near them, and footsoldiers hold together.

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
UnitStats, DEFAULT_UNITS, DEFAULT_COUNTS
    Kinds of unit as data, and the first rules' four kinds.
roll_needed, hits, hit_chance
    The d20 rule.
make_board, board_space, homes
    Boards symmetric for both sides under the half turn (`LatticeSpace.opposite`), their
    lattice, and their homes.

State
-----
TacticsState, Unit
    The board's fields and the units', per cell.
Units, units
    The units on a board as a list, ordered by uid.

Events
------
Event, Placed, Moved, Swapped, Turned, Deferred, Attacked, Died, Ended, RoundStarted, GameOver
    What a step did: each position carries the events of the step that made it.

Players and matches
-------------------
RandomPlayer, GreedyPlayer
    Who picks an action from the legal ones.
summarize, Summary, balance, Balance, EVAL_SEED
    What games show, how fair and lively they were, and the boards players are scored on.
    `game.play` and `game.series` play them: `Tactics` is a `game.Game`.
Tactics.observe, Tactics.planes, Tactics.frame
    What a team sees: planes per cell in its own frame, for networks (`agents.frame.Learnable`).
event_text, summary_text
    The events and what a set of games showed, as text, for terminals and logs.
SearchPlayer, Plan, material
    A player that searches plans for whole unit turns by open-loop Monte Carlo tree search,
    judged by the greedy player.

`pictures.draw` draws a position on matplotlib's axes, with `render.cells`. It is not imported
here, so importing the game doesn't import matplotlib.
"""

from gridzero.game import Action

from .board import board_space, homes, make_board
from .events import (
    Attacked,
    Deferred,
    Died,
    Ended,
    Event,
    GameOver,
    Moved,
    Placed,
    RoundStarted,
    Swapped,
    Turned,
)
from .game import Odds, Tactics
from .match import EVAL_SEED, Balance, Summary, balance, summarize
from .players import GreedyPlayer, RandomPlayer
from .rules import DEFAULT_COUNTS, DEFAULT_UNITS, UnitStats, hit_chance, hits, roll_needed
from .search import Plan, SearchPlayer, material
from .state import TacticsPosition, TacticsState, Unit, Units, Verb, units
from .text import event_text, summary_text

__all__ = [
    "DEFAULT_COUNTS",
    "DEFAULT_UNITS",
    "EVAL_SEED",
    "Action",
    "Attacked",
    "Balance",
    "Deferred",
    "Died",
    "Ended",
    "Event",
    "GameOver",
    "GreedyPlayer",
    "Moved",
    "Odds",
    "Placed",
    "Plan",
    "RandomPlayer",
    "RoundStarted",
    "SearchPlayer",
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
    "event_text",
    "hit_chance",
    "hits",
    "homes",
    "make_board",
    "material",
    "roll_needed",
    "summarize",
    "summary_text",
    "units",
]
