"""Games: players taking turns on a board, beside `core`'s simulations.

A simulation evolves a state under its rules; a game takes one decision at a time from its
players. A game is an immutable value of its rules (`Game`) with pure steps between `Position`s,
which hold a `core.State` and what isn't per cell. Players are shown a view of each position
and pick an `Action`, a verb and a cell, from a mask of legal ones. `game` imports `core`, and
`core` never imports it.

Game
----
Game
    Protocol of a game's rules: ``n_players``, ``n_verbs``, ``start``, ``player``, ``view``,
    ``legal`` and ``step``.
Position
    A complete position: the state, the seed, the ply, the result, whether its step drew from
    its stream, and its step's events. Games subclass it as ``*Position``.
Action, check_action
    One decision, ``(verb, cell)``, and the checks every game's ``step`` opens with.
winner
    The player with the highest score, or None for a draw: how a game decides by scores.
Player
    Protocol of a player: ``(game, player, view, legal, rng) -> Action``.

Playing
-------
play, Record, replay
    One game to its end, its record (start, actions, final position, events), and the
    positions again from its actions.
series
    Players from several starts, each start once per rotation of the seats, in processes.
tally, Tally, Score
    Wins by name and by seat, draws and lengths, and one player's wins, losses and draws.
ladder
    A player's score against each of several rivals, a series each.
fairness, Fairness
    How fair the seats were: each seat's edge over the decided games, and the draws.

Boards
------
symmetric, highest, joined, pockets, zones
    Making a board fair and playable: values the same under maps that swap the seats, the
    cells of the largest values, whether cells are joined and which are cut off, and each
    seat's zone of cells around a seed, such as its home.
"""

from gridzero.game.boards import highest, joined, pockets, symmetric, zones
from gridzero.game.match import (
    Fairness,
    Record,
    Score,
    Tally,
    fairness,
    ladder,
    play,
    replay,
    series,
    tally,
)
from gridzero.game.position import Action, Position, check_action, winner
from gridzero.game.protocols import Game, Player

__all__ = [
    "Action",
    "Fairness",
    "Game",
    "Player",
    "Position",
    "Record",
    "Score",
    "Tally",
    "check_action",
    "fairness",
    "highest",
    "joined",
    "ladder",
    "play",
    "pockets",
    "replay",
    "series",
    "symmetric",
    "tally",
    "winner",
    "zones",
]
