"""Tests of the Voronoi game's colors."""

import numpy as np

from examples.games.voronoi import Greedy, Voronoi, colors
from examples.games.voronoi.picture import NOBODY, SEA, STONE
from gridzero.agents import UniformPlayer
from gridzero.game import play


def test_each_cell_is_colored_by_the_sea_nobody_or_its_holder():
    game = Voronoi(frequency=2, players=3, stones=2)
    state = play(game, [Greedy(), UniformPlayer(), Greedy()], game.start(1)).final.state
    out = colors(state)
    assert out.shape == (42, 4)
    assert out.dtype == np.float64
    assert ((out >= 0) & (out <= 1)).all()
    assert (out[~state.land] == SEA).all()
    nobody = state.land & (state.territory < 0)
    assert (out[nobody] == NOBODY).all()
    held = (state.territory >= 0) & (state.stone < 0)
    for p in range(3):  # one color per player, and none the sea's or nobody's
        mine = out[held & (state.territory == p)]
        assert len({tuple(c) for c in mine}) <= 1
    holders = {tuple(c) for c in out[held]}
    assert len(holders) == len(set(state.territory[held].tolist()))
    assert not holders & {SEA, NOBODY}
    for cell in np.flatnonzero(state.stone >= 0):  # a stone is its player's color, darker
        land = out[held & (state.territory == state.stone[cell])]
        if land.size:
            np.testing.assert_allclose(out[cell, :3], land[0, :3] * STONE)
