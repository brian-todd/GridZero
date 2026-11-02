"""What a step of Tactics did: placements, moves, rolls, hits, deaths and the rounds.

Each `TacticsPosition` carries the events of the step that made it, in the order they happened, so a
game's record is the concatenation of its positions' events. Units are named by uid, which
stays with them as they move, and attacks and deaths carry kinds and teams too, so metrics
need no lookup.
"""

from typing import NamedTuple


class Placed(NamedTuple):
    """A unit was deployed."""

    uid: int
    kind: int
    team: int
    cell: int


class Moved(NamedTuple):
    """A unit walked a route: the cells it stood on, from where it started.

    A move interrupted by a deferred attack is recorded in pieces, one up to each tile where
    archers fired, so that events keep their order in time.
    """

    uid: int
    route: tuple[int, ...]


class Swapped(NamedTuple):
    """The acting unit and a friend traded places: ``uid`` left ``cell`` for ``other_cell``."""

    uid: int
    other: int
    cell: int
    other_cell: int


class Turned(NamedTuple):
    """A unit turned to a direction with the face action."""

    uid: int
    facing: int


class Deferred(NamedTuple):
    """An archer deferred its attack onto a tile."""

    uid: int
    tile: int


class Attacked(NamedTuple):
    """One attack and its roll."""

    attacker: int
    """The attacker's uid."""
    attacker_kind: int
    attacker_team: int
    target: int
    """The target's uid."""
    target_kind: int
    cell: int
    """Where the target stood."""
    roll: int
    needed: int
    atk: int
    defense: int
    flank: bool
    hit: bool
    damage: int
    """The hit points the target lost: at most what it had."""
    deferred: bool
    """Whether it was a deferred attack, fired by a unit entering a watched tile."""


class Died(NamedTuple):
    """A unit was brought to 0 hit points and left the board."""

    uid: int
    kind: int
    team: int
    cell: int


class Ended(NamedTuple):
    """A unit ended its turn."""

    uid: int


class RoundStarted(NamedTuple):
    """A round began, with its order of turns shuffled."""

    round: int


class GameOver(NamedTuple):
    """The game ended: a side has no units, or the last round passed (a draw, ``winner=None``)."""

    winner: int | None


type Event = (
    Placed | Moved | Swapped | Turned | Deferred | Attacked | Died | Ended | RoundStarted | GameOver
)
"""Anything a step records."""
