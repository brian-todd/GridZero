"""Fixtures for Tactics tests: cells by coordinates, and battles built by hand."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from examples.games.tactics import (
    TacticsPosition,
    TacticsState,
    board_space,
    homes,
)
from gridzero.core import Field
from tactics_demo import SMALL

type BattleFactory = Callable[..., TacticsPosition]
type CellFinder = Callable[..., int]


@pytest.fixture
def at() -> CellFinder:
    """Return ``at(q, r, game=SMALL)``: the cell id at axial coordinates on a game's board."""

    def find(q, r, game=SMALL) -> int:
        return int(board_space(game.shape).ravel([q, r]))

    return find


@pytest.fixture
def make_battle() -> BattleFactory:
    """Return a factory of positions in battle, built by hand.

    ``make_battle(pieces, game=SMALL, walls=(), rubble=(), seed=0, round=1)`` puts the pieces
    on an open board with the given walls and rubble, as axial ``(q, r)`` pairs. Units take
    turns in the order listed (their initiative), unless a piece sets ``initiative``. Each
    gets its kind's full hp unless it sets ``hp``.
    """

    def factory(pieces, game=SMALL, walls=(), rubble=(), seed=0, round=1) -> TacticsPosition:
        space = board_space(game.shape)
        n = space.n_cells
        wall = np.zeros(n, dtype=bool)
        rough = np.zeros(n, dtype=bool)
        for q, r in walls:
            wall[space.ravel([q, r])] = True
        for q, r in rubble:
            rough[space.ravel([q, r])] = True
        values: dict[Field[Any], Any] = {
            TacticsState.wall: wall,
            TacticsState.rubble: rough,
            TacticsState.home: homes(space, game.home_depth),
        }
        arrays = {
            "uid": np.full(n, -1),
            "kind": np.full(n, -1),
            "team": np.full(n, -1),
            "hp": np.zeros(n, dtype=int),
            "initiative": np.zeros(n, dtype=int),
            "watch": np.full(n, -1),
        }
        extras: dict[str, np.ndarray] = {}
        for uid, p in enumerate(pieces):
            c = int(space.ravel([p.q, p.r]))
            arrays["uid"][c] = uid
            arrays["kind"][c] = p.kind
            arrays["team"][c] = p.team
            arrays["hp"][c] = p.extra.get("hp", game.units[p.kind].hp)
            arrays["initiative"][c] = p.extra.get("initiative", uid)
            for name, value in p.extra.items():
                if name in ("hp", "initiative"):
                    continue
                if name not in extras:
                    extras[name] = np.zeros(n, dtype=type(value))
                    if name == "watch":
                        extras[name][:] = -1
                extras[name][c] = value
        arrays.update(extras)
        for name, value in arrays.items():
            values[getattr(TacticsState.units, name)] = value
        return TacticsPosition(TacticsState(space, values), seed, round=round)

    return factory
