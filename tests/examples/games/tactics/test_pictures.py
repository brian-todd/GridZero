from matplotlib.figure import Figure

from examples.games.tactics.pictures import draw
from tactics_demo import ARCHER, FOOT, SMALL, piece


def test_a_picture_has_a_hexagon_per_tile_and_labels_per_unit(make_battle, at):
    pieces = [piece(1, 1, FOOT, 0), piece(5, 5, ARCHER, 1, watch=at(3, 3))]
    ax = draw(SMALL, make_battle(pieces, walls=[(4, 4)]))
    tiles = ax.collections[0]
    assert len(tiles.get_paths()) == 81
    assert len(ax.texts) == 2 * 2  # a letter and hit points per unit
    assert len(ax.lines) == 2  # a facing tick per unit
    assert len(ax.collections) == 3  # tiles, units, one watched tile


def test_a_picture_draws_on_given_axes(make_battle):
    ax = Figure().add_subplot()
    assert draw(SMALL, make_battle([]), ax=ax) is ax
    assert len(ax.collections) == 1
