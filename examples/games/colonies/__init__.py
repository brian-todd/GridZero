"""Colonies: orders on a living planet, the fourth game, the first to run `core`'s loop.

Each round every player gives one order in turn, gather toward a land cell or hold, and then
all the orders resolve together as a few steps of a `core.Simulation`: people march, fight
where players meet, and grow toward what the land supports. The most people after the last
round wins, unless all but one are out before. Players see only the cells near their own
people (fog of war). It tests `game` where the other games don't: hidden information, moves
made at once, players knocked out, and a simulation inside a step.

Game
----
Colonies, GATHER, HOLD, MAX_PLAYERS, MIN_PEOPLE, TIE, board
    The rules, as an immutable value and a `game.Game`, their verbs, the most players, the
    people a player needs to see from a cell and to stay in, how close two totals must be to
    be equal, and the board of a frequency.

State
-----
ColoniesState, ColoniesPosition, ColoniesView, Ordered, Resolved, Out, Event
    The land, its capacity and each player's people, a position, what a player is shown, and
    what a step did.

Players
-------
Settler, Raider
    Grows and spreads, never attacking; attacks the enemies it outnumbers, else settles.

Strength
--------
EVAL_SEED
    The games players are scored on (by `game.ladder`).

Pictures
--------
colors, view_colors, PLAYERS, FOG
    A color per cell, for `render.globe`, of the board or of what a player sees, each player's
    color, and how dark fog is.
"""

from examples.games.colonies.game import (
    GATHER,
    HOLD,
    MAX_PLAYERS,
    MIN_PEOPLE,
    TIE,
    Colonies,
    ColoniesPosition,
    ColoniesState,
    ColoniesView,
    Event,
    Ordered,
    Out,
    Resolved,
    board,
)
from examples.games.colonies.match import EVAL_SEED
from examples.games.colonies.picture import FOG, PLAYERS, colors, view_colors
from examples.games.colonies.players import Raider, Settler

__all__ = [
    "EVAL_SEED",
    "FOG",
    "GATHER",
    "HOLD",
    "MAX_PLAYERS",
    "MIN_PEOPLE",
    "PLAYERS",
    "TIE",
    "Colonies",
    "ColoniesPosition",
    "ColoniesState",
    "ColoniesView",
    "Event",
    "Ordered",
    "Out",
    "Raider",
    "Resolved",
    "Settler",
    "board",
    "colors",
    "view_colors",
]
