import numpy as np
import pytest

from gridzero.core import LatticeSpace, State, Tiling
from gridzero.worlds import generate_terrain, terrain_colors

TILINGS = ["hypercubic", "hex", "triangle"]

LATTICE_CASES = [
    (tiling, periodic)
    for tiling in TILINGS
    for periodic in ((True, True), (True, False), (False, True), (False, False))
]


def wide_lattice(tiling: Tiling = "hypercubic", periodic=(True, False), scale=1):
    """A lattice twice as wide as it is high, or about that for hexagons and triangles."""
    rows = {"hypercubic": 16, "hex": 18, "triangle": 10}[tiling]
    return LatticeSpace((32 * scale, rows * scale), tiling=tiling, periodic=periodic)


@pytest.mark.parametrize("tiling", TILINGS)
@pytest.mark.parametrize("land", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_land_covers_the_requested_fraction_of_the_cells(tiling, land):
    space = wide_lattice(tiling)
    elevation = generate_terrain(space, seed=2, land=land)
    assert abs(np.mean(elevation > 0) - land) <= 0.5 / space.n_cells


@pytest.mark.parametrize(("tiling", "periodic"), LATTICE_CASES)
def test_elevation_reaches_minus_1_at_the_deepest_sea_and_1_at_the_highest_peak(tiling, periodic):
    space = wide_lattice(tiling, periodic)
    elevation = generate_terrain(space, seed=0)
    assert elevation.dtype == np.float64
    assert elevation.shape == (space.n_cells,)
    assert elevation.min() == -1
    assert elevation.max() == 1


@pytest.mark.parametrize("tiling", TILINGS)
def test_terrain_repeats_with_its_seed_and_changes_with_it(tiling):
    space = wide_lattice(tiling, (True, True))
    first = generate_terrain(space, seed=5)
    np.testing.assert_array_equal(generate_terrain(space, seed=5), first)
    assert not np.array_equal(generate_terrain(space, seed=6) > 0, first > 0)


@pytest.mark.parametrize("tiling", TILINGS)
def test_one_seed_puts_land_in_the_same_places_on_two_lattices(tiling):
    coarse, fine = wide_lattice(tiling), wide_lattice(tiling, scale=2)
    half = fine.basis.sum(axis=1) / 2  # from a cell's center to its corner, in lattice steps
    under = coarse.locate((fine.centers + half) / 2 - half)  # the coarse cell over each fine one
    land = generate_terrain(coarse, seed=0)[under] > 0
    assert np.mean((generate_terrain(fine, seed=0) > 0) == land) > 0.85  # 0.94-0.95


@pytest.mark.parametrize("tiling", TILINGS)
def test_terrain_continues_across_the_edges_a_lattice_wraps_around(tiling):
    def steps(periodic):
        """Mean step in height across the edges x = 0 and y = 0, and between other rows."""
        space = LatticeSpace((48, 48), tiling=tiling, periodic=periodic)
        height = space.grid_view(generate_terrain(space, seed=0, land=0.5))
        across = [
            np.abs(height[-1] - height[0]).mean(),
            np.abs(height[:, -1] - height[:, 0]).mean(),
        ]
        inside = [np.abs(np.diff(height, axis=0)).mean(), np.abs(np.diff(height, axis=1)).mean()]
        return np.array(across), np.array(inside)

    across, inside = steps(periodic=True)
    assert (across < 1.5 * inside).all()  # 0.89 and 1.01 times, with seed 0 on hexagons
    across, inside = steps(periodic=False)  # without wrapping, the edges are far apart
    assert (across > 3 * inside).all()  # 3.8 and 5.4 times


def test_terrain_colors_are_green_on_the_lowland_and_snow_on_the_highest_peak():
    colors = terrain_colors([0.05, 0.1, 0.2, 1.0, -0.5, -1.0, 0.0])
    assert colors.shape == (7, 4)
    np.testing.assert_array_equal(colors[:, 3], 1.0)
    red, green, blue = colors[:3, 0], colors[:3, 1], colors[:3, 2]
    assert (green > red).all()
    assert (green > blue).all()
    assert (colors[3, :3] > 0.9).all()
    assert (colors[4:, 2] > colors[4:, 0]).all()  # blue seas, up to sea level itself
    assert colors[5, 2] < colors[4, 2]  # deeper seas are darker


@pytest.mark.parametrize(
    ("elevation", "error", "match"),
    [
        ([0.0, 1.5], ValueError, r"must lie in \[-1, 1\]"),
        ([-1.5], ValueError, "must lie in"),
        ([np.nan], ValueError, "NaN"),
        (["x"], TypeError, "numbers"),
        (np.zeros((2, 3)), ValueError, "1-D"),
    ],
)
def test_terrain_colors_need_heights_in_range(elevation, error, match):
    with pytest.raises(error, match=match):
        terrain_colors(elevation)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"land": 1.5}, ValueError, "land"),
        ({"land": -0.1}, ValueError, "land"),
        ({"seed": -1}, ValueError, "seed"),
        ({"roughness": 0}, ValueError, "roughness"),
    ],
)
def test_generate_terrain_rejects_values_out_of_range(kwargs, error, match):
    with pytest.raises(error, match=match):
        generate_terrain(LatticeSpace((4, 4)), **kwargs)


def test_generate_terrain_needs_an_integer_seed(not_an_integer):
    with pytest.raises(TypeError, match="seed"):
        generate_terrain(LatticeSpace((4, 4)), seed=not_an_integer)


def test_generate_terrain_needs_a_real_land_fraction(not_a_real):
    with pytest.raises(TypeError, match="land"):
        generate_terrain(LatticeSpace((4, 4)), land=not_a_real)


@pytest.mark.parametrize(
    ("space", "error", "match"),
    [
        (LatticeSpace((4, 4, 4)), ValueError, "terrain needs a 2-D lattice, got one with 3 axes"),
        ("icosphere", TypeError, "terrain needs a 2-D LatticeSpace, .*got MeshSpace"),
        ("state", TypeError, "terrain needs a 2-D LatticeSpace, got State"),
    ],
)
def test_terrain_lies_on_2d_lattices(make_icosphere, space, error, match):
    state = State(LatticeSpace((4, 4)))  # a board's state, not its space
    space = {"icosphere": make_icosphere(1), "state": state}.get(space, space)
    with pytest.raises(error, match=match):
        generate_terrain(space)
