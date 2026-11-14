"""What a step of Tactics did: placements, moves, work, rolls, hits, deaths and the rounds.

Each `TacticsPosition` carries the events of the step that made it, in the order they happened, so a
game's record is the concatenation of its positions' events. Units are named by uid, which
stays with them as they move, and attacks and deaths carry kinds and teams too, so metrics
need no lookup.
"""

from typing import Literal, NamedTuple

type Ending = Literal["killed", "held", "deadline", "cap"]
"""Why a game ended: a side had no units, a side held the point long enough, nobody captured the
point in time, or the last round passed."""


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


class Built(NamedTuple):
    """An engineer raised a tile one step of the ladder: open to rubble, rubble to wall."""

    uid: int
    tile: int
    terrain: Literal["rubble", "wall"]
    """What the tile is now."""


class Broke(NamedTuple):
    """An engineer lowered a tile one step of the ladder: wall to rubble, rubble to open."""

    uid: int
    tile: int
    terrain: Literal["rubble", "open"]
    """What the tile is now."""


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
    braced: bool = False
    """Whether it was the strike of a unit bracing against a charge (`rules.UnitStats.braces`)."""


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
    """A round began, the first with its order of turns rolled (`Tactics.shuffle_each_round`)."""

    round: int


class Captured(NamedTuple):
    """At the end of a round a side took the point."""

    team: int
    round: int
    flip: bool
    """Whether it took the point from the enemy, rather than being the first to hold it."""


class GameOver(NamedTuple):
    """The game ended, and why (see `objective`). A game has no draws."""

    winner: int
    reason: Ending


type Event = (
    Placed
    | Moved
    | Swapped
    | Turned
    | Deferred
    | Built
    | Broke
    | Attacked
    | Died
    | Ended
    | RoundStarted
    | Captured
    | GameOver
)
"""Anything a step records."""
