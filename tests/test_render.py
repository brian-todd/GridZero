import itertools

import matplotlib
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from PIL import Image, ImageSequence

from gridzero.core import LatticeSpace, random_points
from gridzero.core.geometry import lon_lat, sphere_points
from gridzero.render import (
    Browser,
    cells,
    colorize,
    flat_map,
    globe,
    lattice_text,
    mix,
    over,
    world_map,
)

PICTURES = {
    "globe": lambda space: globe(space, view=(40, 25), size=61),
    "globe-from-a-pole": lambda space: globe(space, view=(10, 90), size=61),
    "mollweide": lambda space: world_map(space, 120),
    "equirectangular": lambda space: world_map(space, 120, projection="equirectangular"),
}


def wrap(lon):
    """Return longitudes in degrees as `lon_lat` gives them, in [-180, 180)."""
    return (np.asarray(lon) + 180) % 360 - 180


@pytest.fixture(params=list(PICTURES))
def picture(request):
    """Return a factory for each kind of picture, in turn."""
    return PICTURES[request.param]


def test_each_pixel_shows_the_cell_with_the_nearest_center(make_icosphere, picture):
    planet = make_icosphere(3, radius=2.0)  # the radius of the sphere does not matter
    raster = picture(planet)
    on = raster.cells >= 0
    closeness = raster.points[on] @ planet.centers.T  # larger is nearer
    shown = np.take_along_axis(closeness, raster.cells[on][:, None], axis=1)[:, 0]
    np.testing.assert_allclose(shown, closeness.max(axis=1), rtol=0, atol=1e-12)


def test_pixels_off_the_sphere_show_nothing(make_icosphere, picture):
    raster = picture(make_icosphere(2))
    off = raster.cells < 0
    assert off.any() != (raster.projection == "equirectangular")  # only those maps fill it
    assert np.isnan(raster.points[off]).all()
    np.testing.assert_array_equal(raster.light[off], 0.0)
    np.testing.assert_allclose(np.linalg.norm(raster.points[~off], axis=-1), 1.0)


def test_rasters_are_read_only(make_icosphere, picture):
    raster = picture(make_icosphere(1))
    for array in (raster.cells, raster.points, raster.light):
        assert not array.flags.writeable


def test_a_globe_is_centered_on_its_view_with_north_up_and_east_right(make_icosphere):
    raster = globe(make_icosphere(2), view=(30, 20), size=61)
    middle = 30
    np.testing.assert_allclose(raster.points[middle, middle], sphere_points(30, 20), atol=1e-12)
    lon, lat = lon_lat(raster.points[[0, -1], middle])  # well inside the disk, top and bottom
    assert lat[0] > 20 > lat[1]
    lon, lat = lon_lat(raster.points[middle, [5, -6]])  # left and right
    assert lon[0] < 30 < lon[1]


def test_a_globe_shows_only_the_hemisphere_facing_the_viewer(make_icosphere):
    raster = globe(make_icosphere(2), view=(-70, -10), size=61)
    on = raster.cells >= 0
    assert (raster.points[on] @ sphere_points(-70, -10) >= 0).all()
    disk = np.hypot(*np.meshgrid(np.arange(61) - 30, np.arange(61) - 30)) < 29.5
    assert on[disk].all()


def test_a_globe_is_lit_from_the_upper_left(make_icosphere):
    raster = globe(make_icosphere(2), size=60)
    on = raster.cells >= 0
    assert raster.light[:30, :30][on[:30, :30]].mean() > raster.light[30:, 30:][on[30:, 30:]].mean()
    assert raster.light[on].min() >= 0.3
    assert raster.light.max() <= 1


def test_a_mollweide_map_gives_each_cell_pixels_in_proportion_to_its_area(make_icosphere):
    planet = make_icosphere(2)
    raster = world_map(planet, 800)
    pixels = np.bincount(raster.cells[raster.cells >= 0], minlength=planet.n_cells)
    np.testing.assert_allclose(pixels / pixels.sum(), planet.measures / (4 * np.pi), rtol=0.05)


def test_an_equirectangular_map_has_longitude_across_and_latitude_down(make_icosphere):
    raster = world_map(make_icosphere(1), 360, projection="equirectangular", center=90)
    assert raster.extent == (-90.0, 270.0, -90.0, 90.0)
    lon, lat = lon_lat(raster.points)
    np.testing.assert_allclose(lat[:, 0], np.linspace(89.5, -89.5, 180), atol=1e-9)
    np.testing.assert_allclose(lon[90], wrap(np.linspace(-89.5, 269.5, 360)), atol=1e-9)


def test_a_map_centered_elsewhere_is_the_same_map_rolled(make_icosphere):
    planet = make_icosphere(3)
    for projection in ("mollweide", "equirectangular"):
        centered = world_map(planet, 360, projection=projection)
        shifted = world_map(planet, 360, projection=projection, center=180)
        if projection == "equirectangular":  # 180 degrees is half the width
            np.testing.assert_array_equal(shifted.cells, np.roll(centered.cells, 180, axis=1))
        else:  # the middle column turns from longitude 0 to longitude 180
            lon, _ = lon_lat(shifted.points[90, 180])  # just right of the middle
            np.testing.assert_allclose(lon, wrap(180.5), atol=1e-4)


def test_an_image_paints_each_pixel_with_its_cells_color(make_icosphere, picture):
    planet = make_icosphere(2)
    raster = picture(planet)
    colors = np.random.default_rng(0).uniform(size=(planet.n_cells, 4))
    on = raster.cells >= 0
    image = raster.image(colors, shade=False)
    np.testing.assert_array_equal(image[on], colors[raster.cells[on]])
    np.testing.assert_array_equal(image[~on], 0.0)
    shaded = raster.image(colors)
    np.testing.assert_allclose(shaded[..., :3], image[..., :3] * raster.light[..., None])
    np.testing.assert_array_equal(shaded[..., 3], image[..., 3])


def test_rgb_colors_are_opaque(make_icosphere):
    planet = make_icosphere(1)
    raster = globe(planet, size=20)
    rgb = np.full((planet.n_cells, 3), 0.5)
    np.testing.assert_array_equal(
        raster.image(rgb), raster.image(np.column_stack([rgb, np.ones(planet.n_cells)]))
    )


def test_borders_darken_the_pixels_where_cells_meet(make_icosphere, picture):
    planet = make_icosphere(2)
    raster = picture(planet)
    image = raster.image(np.ones((planet.n_cells, 3)), shade=False, borders=True)
    cells = raster.cells
    border = np.zeros(cells.shape, dtype=bool)
    for here, there in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        a, b = cells[here], cells[there]
        border[here] |= (a != b) & (a >= 0) & (b >= 0)  # not where the sphere ends
    on = cells >= 0
    np.testing.assert_array_equal((image[..., 0] < 1)[on], border[on])
    assert border.any()


def test_show_draws_the_image_and_hides_the_axes_of_globes(make_icosphere):
    planet = make_icosphere(1)
    colors = np.ones((planet.n_cells, 3))
    figure = Figure()
    for raster, axis_on in (
        (globe(planet, size=10), False),
        (world_map(planet, 20), False),
        (world_map(planet, 20, projection="equirectangular"), True),
    ):
        ax = figure.add_subplot()
        image = raster.show(ax, colors)
        np.testing.assert_array_equal(image.get_array(), raster.image(colors))
        assert image.get_extent() == list(raster.extent)
        assert ax.axison == axis_on


# Flat maps


FLATS = [
    (tiling, periodic)
    for tiling in ("hypercubic", "hex", "triangle")
    for periodic in ((True, True), (True, False), (False, True), (False, False))
]


def small_flat(tiling, periodic):
    """A small 2-D lattice to draw. Triangles wrap only along even axes."""
    return LatticeSpace(
        (6, 4) if tiling == "triangle" else (6, 5), tiling=tiling, periodic=periodic
    )


def pixel_centers(raster):
    """The point of the plane each pixel looks at, from the extent: shape (height, width, 2)."""
    left, right, bottom, top = raster.extent
    height, width = raster.cells.shape
    x = left + (np.arange(width) + 0.5) / width * (right - left)
    y = top - (np.arange(height) + 0.5) / height * (top - bottom)
    return np.stack(np.meshgrid(x, y), axis=-1)


def distances_to_centers(grid, points):
    """Distance from each point to the nearest copy of each cell's center: (n_points, n_cells).

    Copies lie one period away along each axis that wraps around.
    """
    periods = grid.basis * np.array(grid.shape)  # one per column
    best = np.full((len(points), grid.n_cells), np.inf)
    for turns in itertools.product(*[(-1, 0, 1) if wraps else (0,) for wraps in grid.periodic]):
        copies = grid.centers + periods @ np.array(turns)
        best = np.minimum(best, np.linalg.norm(points[:, None] - copies[None], axis=-1))
    return best


@pytest.mark.parametrize(("tiling", "periodic"), FLATS)
def test_each_pixel_of_a_flat_map_shows_the_cell_whose_center_is_nearest(tiling, periodic):
    grid = small_flat(tiling, periodic)
    raster = flat_map(grid, 90)
    on = raster.cells >= 0
    points = pixel_centers(raster)
    np.testing.assert_array_equal(raster.points[on], points[on])
    distance = distances_to_centers(grid, points[on])
    shown = np.take_along_axis(distance, raster.cells[on][:, None], axis=1)[:, 0]
    np.testing.assert_allclose(shown, distance.min(axis=1), rtol=0, atol=1e-12)
    for array in (raster.cells, raster.points, raster.light):
        assert not array.flags.writeable


@pytest.mark.parametrize(("tiling", "periodic"), FLATS)
def test_pixels_beyond_a_flat_grid_show_nothing(tiling, periodic):
    grid = small_flat(tiling, periodic)
    raster = flat_map(grid, 90)
    off = raster.cells < 0
    # Squares fill the map. Hexagons leave gaps along edges that do not wrap, and triangles
    # along the ends of their rows, where those do not wrap.
    gaps = {"hypercubic": False, "hex": not all(periodic), "triangle": not periodic[0]}
    assert off.any() == gaps[tiling]
    distance = distances_to_centers(grid, pixel_centers(raster)[off])
    assert (distance >= 0.5 - 1e-12).all()  # in no cell, or on its border
    assert np.isnan(raster.points[off]).all()
    np.testing.assert_array_equal(raster.light, np.where(off, 0.0, 1.0))


@pytest.mark.parametrize(("tiling", "periodic"), FLATS)
def test_a_flat_map_shows_each_cell_once_and_whole(tiling, periodic):
    # Along a wrapped axis the map spans one period, and along a fixed one every cell whole,
    # so each cell gets the pixels of its area, and no more.
    grid = small_flat(tiling, periodic)
    raster = flat_map(grid, 240)
    left, right, bottom, top = raster.extent
    pixel = (right - left) / raster.cells.shape[1] * (top - bottom) / raster.cells.shape[0]
    pixels = np.bincount(raster.cells[raster.cells >= 0], minlength=grid.n_cells)
    np.testing.assert_allclose(pixels * pixel, grid.measures, rtol=0.05)


def test_a_square_lattice_map_is_its_grid_scaled_up_with_y_up():
    grid = LatticeSpace((4, 3), periodic=(True, False))
    raster = flat_map(grid, 8)
    assert raster.extent == (-0.5, 3.5, -0.5, 2.5)
    ids = np.arange(12).reshape(4, 3)  # ids[x, y], row-major
    np.testing.assert_array_equal(raster.cells, np.kron(ids.T[::-1], np.ones((2, 2), dtype=int)))


@pytest.mark.parametrize(
    ("periodic", "extent"),
    [
        ((True, True), (-0.5, 5.5, -(3**0.5) / 4, 4.5 * 3**0.5 / 2)),  # one period each way
        ((False, False), (-0.5, 7.5, -(3**-0.5), 4 * 3**0.5 / 2 + 3**-0.5)),  # the rhombus
    ],
)
def test_a_hex_map_spans_one_period_where_it_wraps_and_every_cell_where_not(periodic, extent):
    raster = flat_map(LatticeSpace((6, 5), tiling="hex", periodic=periodic), 60)
    np.testing.assert_allclose(raster.extent, extent)
    assert raster.projection == "flat"


@pytest.mark.parametrize(
    ("periodic", "extent"),
    [
        ((True, True), (-(3**0.5) / 4, 5.5 * 3**0.5 / 2, -0.75, 5.25)),  # one period each way
        ((False, False), (-(3**0.5) / 2, 6 * 3**0.5 / 2, -0.75, 5.25)),  # every corner
    ],
)
def test_a_triangle_map_spans_one_period_where_it_wraps_and_every_cell_where_not(periodic, extent):
    # Rows are 3/2 high and straight, so they end at the same height whether they wrap or not.
    raster = flat_map(LatticeSpace((6, 4), tiling="triangle", periodic=periodic), 60)
    np.testing.assert_allclose(raster.extent, extent)


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_a_flat_map_paints_shows_and_borders_its_cells(tiling):
    grid = LatticeSpace((6, 5), tiling=tiling, periodic=False)
    raster = flat_map(grid, 60)
    colors = np.random.default_rng(0).uniform(size=(grid.n_cells, 4))
    on = raster.cells >= 0
    image = raster.image(colors)  # maps are evenly lit, so shading changes nothing
    np.testing.assert_array_equal(image[on], colors[raster.cells[on]])
    np.testing.assert_array_equal(image[~on], 0.0)
    bordered = raster.image(np.ones((grid.n_cells, 3)), borders=True)[..., 0] < 1
    cells = raster.cells
    expected = np.zeros(cells.shape, dtype=bool)
    expected[:, :-1] |= (cells[:, :-1] != cells[:, 1:]) & (cells[:, :-1] >= 0) & (cells[:, 1:] >= 0)
    expected[:-1] |= (cells[:-1] != cells[1:]) & (cells[:-1] >= 0) & (cells[1:] >= 0)
    np.testing.assert_array_equal(bordered[on], expected[on])
    ax = Figure().add_subplot()
    shown = raster.show(ax, colors)
    assert shown.get_extent() == list(raster.extent)
    assert ax.axison
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x", "y")


@pytest.mark.parametrize(
    ("tiling", "periodic"),
    # A hex lattice that wraps along y alone has a slanted period, which no rectangle shows
    # exactly: about 1 point in 1,100 lies in slivers of cells that flat_map leaves out.
    [case for case in FLATS if case != ("hex", (False, True))],
)
def test_random_points_lie_where_a_flat_map_draws_the_lattice(rng, tiling, periodic):
    shape = (32, 16) if tiling == "hypercubic" else (32, 18)
    grid = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    left, right, bottom, top = flat_map(grid, 64).extent
    x, y = random_points(grid, rng, 2000).T
    assert ((left <= x) & (x < right) & (bottom <= y) & (y < top)).all()


# Colors


def test_colorize_maps_the_value_range_onto_the_colormap():
    cmap = matplotlib.colormaps["magma"]
    colors = colorize([2.0, 4.0, 3.0], "magma")
    np.testing.assert_allclose(colors, cmap([0.0, 1.0, 0.5]))
    colors = colorize([2.0, 4.0, 3.0], cmap, vmin=0, vmax=8)
    np.testing.assert_allclose(colors, cmap([0.25, 0.5, 0.375]))


def test_colorize_leaves_cells_without_a_value_transparent():
    colors = colorize([np.nan, 1.0, 2.0])
    np.testing.assert_array_equal(colors[0], 0.0)
    assert colors[1:, 3].min() == 1
    np.testing.assert_array_equal(colorize([np.nan, np.nan]), 0.0)


@pytest.mark.parametrize(
    ("values", "error"), [(["a", "b"], TypeError), ([[1.0, 2.0]], ValueError)], ids=["text", "2d"]
)
def test_colorize_needs_one_number_per_cell(values, error):
    with pytest.raises(error, match="values"):
        colorize(values)


def test_mix_averages_the_palette_by_weight():
    palette = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.5]]
    weights = [[1.0, 0.0, 3.0, 0.0], [0.0, 2.0, 1.0, 0.0]]
    colors = mix(weights, palette)
    np.testing.assert_allclose(
        colors, [[1, 0, 0, 1], [0, 0, 1, 0.5], [0.75, 0, 0.25, 0.875], [0, 0, 0, 0]]
    )


def test_mix_takes_any_matplotlib_colors():
    np.testing.assert_allclose(mix([[1.0], [1.0]], ["red", "blue"]), [[0.5, 0, 0.5, 1]])


@pytest.mark.parametrize(
    "weights", [[[1.0, -1.0], [0.0, 0.0]], [[1.0, 1.0]], [[np.inf, 0.0], [0.0, 1.0]]]
)
def test_mix_rejects_negative_infinite_or_misaligned_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        mix(weights, ["red", "blue"])


def test_over_lets_the_lower_colors_show_through_the_clear_parts():
    top = [[1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0.5], [1, 0, 0, 0.5], [0, 0, 0, 0]]
    bottom = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]]
    np.testing.assert_allclose(
        over(top, bottom),
        [[1, 0, 0, 1], [0, 0, 1, 1], [0.5, 0, 0.5, 1], [1, 0, 0, 0.5], [0, 0, 0, 0]],
    )


# Bad input


def test_pictures_need_a_spherical_mesh(make_square_mesh):
    for draw in (globe, world_map):
        with pytest.raises(TypeError, match="spherical MeshSpace"):
            draw(LatticeSpace((4, 4)))  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="metric 'euclidean'"):
            draw(make_square_mesh())


@pytest.mark.parametrize(
    ("view", "error"),
    [((0, 91), ValueError), ((np.nan, 0), ValueError), ((0,), TypeError), (5, TypeError)],
)
def test_a_globe_needs_a_view_on_the_sphere(make_icosphere, view, error):
    with pytest.raises(error, match="view"):
        globe(make_icosphere(1), view=view)


def test_flat_maps_need_a_2d_lattice_and_a_pixel(make_icosphere):
    with pytest.raises(TypeError, match="LatticeSpace"):
        flat_map(make_icosphere(1))
    for grid in (LatticeSpace((4, 4, 4)), LatticeSpace((8,))):
        with pytest.raises(ValueError, match="2-D"):
            flat_map(grid)
    with pytest.raises(ValueError, match="width"):
        flat_map(LatticeSpace((4, 4)), 0)


def test_picture_sizes_must_be_integers(make_icosphere, not_an_integer):
    with pytest.raises(TypeError, match="size"):
        globe(make_icosphere(1), size=not_an_integer)
    with pytest.raises(TypeError, match="width"):
        world_map(make_icosphere(1), not_an_integer)
    with pytest.raises(TypeError, match="width"):
        flat_map(LatticeSpace((4, 4)), not_an_integer)


def test_a_map_needs_a_known_projection_and_two_pixels(make_icosphere):
    with pytest.raises(ValueError, match="projection"):
        world_map(make_icosphere(1), projection="mercator")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="width"):
        world_map(make_icosphere(1), 1)


@pytest.mark.parametrize(
    ("colors", "match"),
    [(np.ones((11, 3)), "shape"), (np.ones((12, 2)), "shape"), (np.full((12, 3), 1.5), "0, 1")],
)
def test_images_need_a_color_in_range_for_every_cell(make_icosphere, colors, match):
    raster = globe(make_icosphere(1), size=8)
    with pytest.raises(ValueError, match=match):
        raster.image(colors)


@pytest.mark.parametrize("tiling", ["hypercubic", "hex", "triangle"])
def test_cells_draws_each_cells_polygon_in_its_color(tiling):
    space = LatticeSpace((6, 4), tiling=tiling, periodic=False)
    colors = np.linspace(0, 1, space.n_cells)[:, None] * np.ones(3)
    ax = cells(space, colors)
    polygons = ax.collections[0]
    assert len(polygons.get_paths()) == space.n_cells
    k = space.corners().shape[1]
    for path, corners in zip(polygons.get_paths(), space.corners(), strict=True):
        np.testing.assert_allclose(np.asarray(path.vertices)[:k], corners)
    np.testing.assert_allclose(np.asarray(polygons.get_facecolor())[:, :3], colors)
    x0, x1 = ax.get_xlim()
    assert x0 <= space.corners()[..., 0].min()
    assert x1 >= space.corners()[..., 0].max()
    assert ax.get_aspect() == 1.0


def test_cells_draws_on_given_axes_and_checks_its_colors():
    space = LatticeSpace((3, 3), tiling="hex", periodic=False)
    ax = Figure().add_subplot()
    assert cells(space, np.zeros((9, 4)), ax=ax) is ax
    with pytest.raises(ValueError, match="colors must have shape"):
        cells(space, np.zeros((8, 3)))


def test_lattice_text_draws_hex_rows_top_down_shifted_half_a_cell_each():
    space = LatticeSpace((3, 2), tiling="hex", periodic=False)
    glyphs = np.array(list("abcdef"))  # cell (q, r) is q * 2 + r
    assert lattice_text(space, glyphs) == "\n".join(
        ["  1  b d f", "  0 a c e", "q   0 1 2"]  # no tens: no column reaches 10
    )


def test_lattice_text_draws_squares_without_a_shift_and_labels_two_digit_columns():
    space = LatticeSpace((12, 1), periodic=False)
    text = lattice_text(space, np.array(list("abcdefghijkl"))).splitlines()
    assert text[0] == "  0 a b c d e f g h i j k l"
    assert text[1] == "i                       1 1"
    assert text[2] == "    0 1 2 3 4 5 6 7 8 9 0 1"


def test_lattice_text_draws_a_window_around_a_cell():
    space = LatticeSpace((5, 5), periodic=False)
    glyphs = np.full(25, ".")
    glyphs[space.ravel([2, 2])] = "X"
    text = lattice_text(space, glyphs, around=int(space.ravel([2, 2])), radius=1).splitlines()
    assert text[:3] == ["  3 . . .", "  2 . X .", "  1 . . ."]
    assert text[3:] == ["i   1 2 3"]


def test_lattice_text_needs_a_2d_lattice_one_character_per_cell_and_a_cell(not_an_integer):
    space = LatticeSpace((3, 3), periodic=False)
    with pytest.raises(ValueError, match="2-D"):
        lattice_text(LatticeSpace((3,)), np.full(3, "."))
    with pytest.raises(ValueError, match="one character"):
        lattice_text(space, np.full(9, ".."))
    with pytest.raises(ValueError, match="9"):
        lattice_text(space, np.full(8, "."))
    with pytest.raises(IndexError):
        lattice_text(space, np.full(9, "."), around=9)
    with pytest.raises(TypeError):
        lattice_text(space, np.full(9, "."), radius=not_an_integer)


def browser(frames=5):
    """A browser of a figure whose title names the frame, and the frames it drew, in order."""
    figure = Figure()
    ax = figure.add_subplot()
    drawn = []

    def draw(frame):
        drawn.append(frame)
        ax.set_title(f"frame {frame}")

    return Browser(figure, frames, draw), drawn


def press(browser, key):
    event = KeyEvent("key_press_event", browser.figure.canvas, key)
    browser.figure.canvas.callbacks.process("key_press_event", event)


def tick(browser):
    for func, args, kwargs in browser.timer.callbacks:
        func(*args, **kwargs)


def test_a_browser_draws_its_first_frame_when_made():
    shown, drawn = browser()
    assert drawn == [0]
    assert shown.frame == 0
    assert shown.figure.axes[0].get_title() == "frame 0"


def test_going_to_a_frame_draws_it_and_moves_the_slider():
    shown, drawn = browser()
    shown.go(np.int64(3))
    assert drawn == [0, 3]
    assert shown.frame == 3
    assert shown.slider.val == 3


def test_moving_the_slider_draws_its_frame():
    shown, drawn = browser()
    shown.slider.set_val(2)
    assert drawn == [0, 2]
    assert shown.frame == 2


def test_arrow_keys_step_a_frame_and_stop_at_either_end():
    shown, drawn = browser(3)
    for key in ("left", "right", "right", "right", "left"):
        press(shown, key)
    assert drawn == [0, 1, 2, 1]
    assert shown.frame == 1


def test_home_and_end_go_to_the_first_and_the_last_frame():
    shown, _ = browser()
    press(shown, "end")
    assert shown.frame == 4
    press(shown, "home")
    assert shown.frame == 0


def test_other_keys_change_nothing():
    shown, drawn = browser()
    press(shown, "x")
    assert drawn == [0]
    assert not shown.playing


def test_space_plays_from_the_frame_shown_and_stops_at_the_last():
    shown, drawn = browser(4)
    shown.go(1)
    press(shown, " ")
    assert shown.playing
    tick(shown)
    assert shown.frame == 2
    assert shown.playing
    tick(shown)
    assert shown.frame == 3
    assert not shown.playing
    assert drawn == [0, 1, 2, 3]


def test_space_while_playing_pauses():
    shown, _ = browser()
    press(shown, " ")
    tick(shown)
    press(shown, " ")
    assert not shown.playing
    assert shown.frame == 1


def test_playing_from_the_last_frame_starts_again_from_the_first():
    shown, _ = browser(3)
    shown.go(2)
    shown.play()
    assert shown.playing
    assert shown.frame == 0
    tick(shown)
    assert shown.frame == 1


def test_save_writes_every_frame_in_order_and_shows_the_first_again(tmp_path):
    shown, drawn = browser(3)
    shown.go(2)
    path = tmp_path / "frames.gif"
    shown.save(path, fps=np.int64(5))
    assert shown.frame == 0
    with Image.open(path) as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 3
        assert gif.info["duration"] == 200
    assert drawn[2:-1] == [0, 0, 1, 2]  # FuncAnimation draws the first frame to begin with


def test_a_browser_labels_its_slider():
    figure = Figure()
    shown = Browser(figure, 2, lambda frame: None, label="round")
    assert shown.slider.label.get_text() == "round"


def test_a_browser_refuses_bad_frames_and_intervals(not_an_integer):
    figure = Figure()
    with pytest.raises(ValueError, match="frames"):
        Browser(figure, 1, lambda frame: None)
    with pytest.raises(TypeError):
        Browser(figure, not_an_integer, lambda frame: None)
    with pytest.raises(ValueError, match="interval"):
        Browser(figure, 2, lambda frame: None, interval=0)
    with pytest.raises(TypeError):
        Browser(figure, 2, lambda frame: None, interval=not_an_integer)


def test_going_to_a_frame_out_of_range_raises(not_an_integer):
    shown, drawn = browser(3)
    with pytest.raises(ValueError, match="less than 3"):
        shown.go(3)
    with pytest.raises(ValueError, match="frame"):
        shown.go(-1)
    with pytest.raises(TypeError):
        shown.go(not_an_integer)
    with pytest.raises(ValueError, match="fps"):
        shown.save("never.gif", fps=0)
    assert drawn == [0]


def test_redraw_draws_the_frame_shown_again():
    shown, drawn = browser()
    shown.go(2)
    shown.redraw()
    assert drawn == [0, 2, 2]
    assert shown.frame == 2
