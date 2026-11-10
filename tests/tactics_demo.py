"""Shared pieces of the Tactics tests: small games, units for hand-made battles, random play and
mirroring. The fixtures that build battles are in `tests/examples/games/tactics/conftest.py`."""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from examples.games.tactics import (
    DEFAULT_UNITS,
    Hold,
    Tactics,
    TacticsPosition,
    TacticsState,
    Verb,
)
from gridzero.core import Field

FOOT, ARCHER, KNIGHT, CAPTAIN, ENGINEER = range(5)
"""The kinds of the default rules."""

FIRST_UNITS = (
    dataclasses.replace(DEFAULT_UNITS[FOOT], atk=2, braces=False),
    DEFAULT_UNITS[ARCHER],
    dataclasses.replace(
        DEFAULT_UNITS[KNIGHT], atk=3, move=4, double_move=True, charge=0, extra_places=0
    ),
    DEFAULT_UNITS[CAPTAIN],
)
"""The units of the first rules (2026-10-24), which the tests' hand-worked numbers assume, with
`SMALL` and `TINY` flanking x2, with no bonus on overwatch, turns shuffled every round, units
engaged until the round ends, the point held for 5 rounds and no fog, as they had, so that tuning
the defaults leaves the tests of mechanics alone."""

SMALL = Tactics(
    units=FIRST_UNITS,
    caps=(6, 3, 2, 1),
    team_size=12,
    flank_factor=2,
    overwatch_bonus=0,
    shuffle_each_round=True,
    engaged_until_turn=False,
    hold_rounds=5,
    shape=(9, 9),
    home_depth=4,
    fog=False,
)
"""The first rules on a 9x9 board, for hand-made positions: homes of 16 tiles."""

TINY = Tactics(
    units=FIRST_UNITS,
    flank_factor=2,
    overwatch_bonus=0,
    shuffle_each_round=True,
    engaged_until_turn=False,
    hold_rounds=5,
    shape=(9, 9),
    caps=(2, 1, 1, 1),
    team_size=5,
    home_depth=3,
    max_rounds=6,
    capture_by=6,
    fog=False,
)
"""Five units a side on a 9x9 board, for whole games."""

WORKS = Tactics(shape=(9, 9), home_depth=4, team_size=5, max_rounds=12, capture_by=8, fog=False)
"""The default kinds on a 9x9 board, with an engineer, for hand-made positions and whole games,
without fog, as the first rules were: hand-made positions say nothing of what was last seen
where, and `test_fog` tests fog."""


@dataclass
class Piece:
    """A unit for a hand-made battle, at axial ``(q, r)``. Other fields default as on deployment."""

    q: int
    r: int
    kind: int
    team: int
    extra: dict[str, object] = field(default_factory=dict)


def piece(q, r, kind, team, **extra) -> Piece:
    """Return a unit at ``(q, r)``; ``extra`` sets other fields of `Unit` (facing, hp, ...)."""
    return Piece(q, r, kind, team, extra)


def play_random(
    game: Tactics, seed: int, rng: np.random.Generator, end_share: float = 0.1
) -> Iterator[tuple[TacticsPosition, tuple[int, int]]]:
    """Play a game with random legal actions, yielding each position and the action taken.

    The verb END is taken with probability ``end_share`` when other actions are legal, so that
    units act before passing.
    """
    position = game.start(seed)
    while not position.over:
        legal = np.argwhere(game.legal(position))
        others = legal[legal[:, 0] != Verb.END]
        choices = others if others.size and rng.random() >= end_share else legal
        verb, cell = choices[rng.integers(len(choices))]
        action = (int(verb), int(cell))
        yield position, action
        position = game.step(position, action)
    yield position, (-1, -1)


def mirrored(game: Tactics, position: TacticsPosition) -> TacticsPosition:
    """Return a position turned half a turn on its board, with the teams swapped.

    Cells map through `LatticeSpace.opposite`, directions turn by 3, teams 0 and 1 trade places,
    the tiles archers watch and those units were last seen on map with the cells, and the point's
    owner trades places too.
    """
    state = position.state
    m = state.space.opposite()
    u = state.units
    values: dict[Field[Any], Any] = {f: state[f][m] for f in state}
    values[TacticsState.home] = np.where(state.home[m] >= 0, 1 - state.home[m], -1)
    present = u.kind[m] >= 0
    values[TacticsState.units.team] = np.where(present, 1 - u.team[m], -1)
    values[TacticsState.units.facing] = np.where(present, (u.facing[m] + 3) % 6, 0)
    for tiles in (TacticsState.units.watch, TacticsState.units.spotted):  # fields of cell ids
        cells = state[tiles][m]
        values[tiles] = np.where(cells >= 0, m[np.maximum(cells, 0)], -1)
    state = TacticsState(state.space, values)
    owner, clock = position.hold
    hold = Hold(1 - owner if owner >= 0 else owner, clock)
    return TacticsPosition(state, position.seed, ply=position.ply, round=position.round, hold=hold)
