import pytest
from matplotlib.figure import Figure

from examples.games.tactics import Hold
from examples.games.tactics.pictures import TEAM_COLORS, draw
from tactics_demo import ARCHER, FOOT, SMALL, piece


def test_a_picture_has_a_hexagon_per_tile_and_labels_per_unit(make_battle, at):
    pieces = [piece(1, 1, FOOT, 0), piece(5, 5, ARCHER, 1, watch=at(3, 3))]
    ax = draw(SMALL, make_battle(pieces, walls=[(4, 4)]))
    tiles = ax.collections[0]
    assert len(tiles.get_paths()) == 81
    assert len(ax.texts) == 2 * 2  # a letter and hit points per unit
    assert len(ax.lines) == 2  # a facing tick per unit
    assert len(ax.collections) == 3  # tiles, units, one watched tile


@pytest.mark.parametrize("owner", [0, 1])
def test_the_point_is_drawn_in_its_owners_color(make_battle, at, owner):
    ax = draw(SMALL, make_battle([], point=(4, 4), hold=Hold(owner, 1)))
    face = ax.collections[0].get_facecolor()[at(4, 4)]
    assert tuple(face[:3]) == pytest.approx(TEAM_COLORS[owner])


def test_a_picture_draws_on_given_axes(make_battle):
    ax = Figure().add_subplot()
    assert draw(SMALL, make_battle([]), ax=ax) is ax
    assert len(ax.collections) == 1
