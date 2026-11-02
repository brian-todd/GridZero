"""Tests of the scripted players of k in a row."""

import numpy as np
import pytest

from examples.games.in_a_row import (
    InARow,
    InARowPosition,
    InARowState,
    ThreatPlayer,
)
from gridzero.agents import UniformPlayer
from gridzero.game import series, tally

GAME = InARow()


def board(rows):
    """Return a Connect Four position from rows drawn top down ("X", "O", "."), X to move."""
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    for height, marks in enumerate(reversed(rows)):
        for column, mark in enumerate(marks):
            if mark != ".":
                owner[GAME.space.ravel([column, height])] = "XO".index(mark)
    stones = GAME.space.grid_view(owner) >= 0
    assert (stones[:, :-1] >= stones[:, 1:]).all(), "a stone floats"
    assert (owner == 0).sum() == (owner == 1).sum(), "X must be to move"
    state = InARowState(GAME.space, {InARowState.owner: owner})
    return InARowPosition(state, 0, ply=int((owner >= 0).sum()))


def choose(position, seed=0):
    return ThreatPlayer()(
        GAME, GAME.player(position), position, GAME.legal(position), np.random.default_rng(seed)
    )


def column(action):
    return int(GAME.space.unravel(action.cell)[0])


@pytest.mark.parametrize("seed", range(5))
def test_it_completes_its_own_line_before_blocking(seed):
    position = board(["...O...", "X..O...", "X..O...", "X..XO.."])  # X wins at 0, O at 3
    assert column(choose(position, seed)) == 0


@pytest.mark.parametrize("seed", range(5))
def test_it_blocks_the_other_players_line(seed):
    position = board(["..XX...", ".XOOO.."])  # O wins on the bottom of column 5 next
    assert column(choose(position, seed)) == 5
    position = board(["...O...", "...O...", "XXXO..."])  # O wins on top of column 3
    assert column(choose(position, seed)) == 3


@pytest.mark.parametrize("seed", range(10))
def test_it_never_fills_the_cell_under_the_other_players_win(seed):
    # O has three on row 1, columns 1-3; column 4 is empty, so its row 1 waits on row 0.
    position = board(["XOOO...", "XXOX..."])
    assert column(choose(position, seed)) != 4


def test_it_beats_a_uniform_player_nearly_always():
    records = series(GAME, [("threat", ThreatPlayer()), ("uniform", UniformPlayer())], 20)
    assert tally(records).wins["threat"] >= 38
