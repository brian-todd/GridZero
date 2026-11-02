"""Shared pieces of the Tactics tests: small games, units for hand-made battles, random play and
mirroring. The fixtures that build battles are in `tests/examples/games/tactics/conftest.py`."""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from examples.games.tactics import (
    DEFAULT_UNITS,
    Tactics,
    TacticsPosition,
    TacticsState,
    Verb,
)
from gridzero.core import Field

FOOT, ARCHER, KNIGHT, CAPTAIN = range(4)
"""The kinds of the default rules."""

FIRST_UNITS = (
    dataclasses.replace(DEFAULT_UNITS[FOOT], atk=2),
    DEFAULT_UNITS[ARCHER],
    dataclasses.replace(DEFAULT_UNITS[KNIGHT], move=4),
    DEFAULT_UNITS[CAPTAIN],
)
"""The units of the first rules (2026-10-24), which the tests' hand-worked numbers assume, with
`SMALL` and `TINY` flanking x2 and drawing at the round cap as they did, so that tuning the
defaults (T6, T7) leaves the tests of mechanics alone."""

SMALL = Tactics(units=FIRST_UNITS, flank_factor=2, shape=(9, 9), home_depth=4, decide_at_cap=False)
"""The first rules on a 9x9 board, for hand-made positions: homes of 16 tiles."""

TINY = Tactics(
    units=FIRST_UNITS,
    flank_factor=2,
    shape=(9, 9),
    counts=(2, 1, 1, 1),
    home_depth=3,
    max_rounds=6,
    decide_at_cap=False,
)
"""Five units a side on a 9x9 board, for whole games."""


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
    and the tiles archers watch map with the cells.
    """
    state = position.state
    m = state.space.opposite()
    u = state.units
    values: dict[Field[Any], Any] = {f: state[f][m] for f in state}
    values[TacticsState.home] = np.where(state.home[m] >= 0, 1 - state.home[m], -1)
    present = u.kind[m] >= 0
    values[TacticsState.units.team] = np.where(present, 1 - u.team[m], -1)
    values[TacticsState.units.facing] = np.where(present, (u.facing[m] + 3) % 6, 0)
    watch = u.watch[m]
    values[TacticsState.units.watch] = np.where(watch >= 0, m[np.maximum(watch, 0)], -1)
    state = TacticsState(state.space, values)
    return TacticsPosition(state, position.seed, ply=position.ply, round=position.round)
