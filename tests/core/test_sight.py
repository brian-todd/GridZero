"""Lines between cells and lines of sight on 2-D lattices, against an exact oracle.

The oracle clips each segment against every cell's polygon with fractions, in coordinates
where every corner is an integer: scaled lattice coordinates on squares and hexes, and columns
by quarters of a plane unit on triangles. Those are affine images of the plane, which keep what
a line crosses or touches, and in what order.
"""

import functools
import itertools
from fractions import Fraction

import numpy as np
import pytest

from gridzero.core import (
    LatticeSpace,
    Tiling,
    icosphere,
    line,
    sight,
    visible,
    within,
)
from gridzero.core.sight import _ends

pytestmark = pytest.mark.usefixtures("each_impl")

BOARDS: dict[str, tuple[tuple[int, ...], Tiling, bool | tuple[bool, ...]]] = {
    "square": ((6, 5), "hypercubic", False),
    "square-wrapped": ((6, 5), "hypercubic", True),
    "hex": ((6, 5), "hex", False),
    "hex-wrapped": ((6, 5), "hex", (True, False)),
    "triangle": ((7, 4), "triangle", False),
    "triangle-wrapped": ((8, 4), "triangle", True),
}

HEX_CORNERS = [(1, -2), (2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1)]  # in thirds, CCW


def polygon(space, i, j):
    """The center and corners, counterclockwise, of the cell at (i, j), in integers.

    Twice the lattice coordinates on squares, three times on hexes, and on triangles columns
    by quarters of a plane unit.
    """
    if space.tiling == "hypercubic":
        x, y = 2 * i, 2 * j
        return (x, y), [(x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x - 1, y - 1)]
    if space.tiling == "hex":
        x, y = 3 * i, 3 * j
        return (x, y), [(x + a, y + b) for a, b in HEX_CORNERS]
    middle = 6 * j
    if (i + j) % 2 == 0:  # pointing up
        return (i, middle - 1), [(i + 1, middle - 3), (i, middle + 3), (i - 1, middle - 3)]
    return (i, middle + 1), [(i, middle - 3), (i + 1, middle + 3), (i - 1, middle + 3)]


def cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def sub(u, v):
    return (u[0] - v[0], u[1] - v[1])


def clip(p, d, corners):
    """The stretch [t0, t1] of p + t d, 0 <= t <= 1, inside a closed convex polygon, or None."""
    t0, t1 = Fraction(0), Fraction(1)
    for v, w in zip(corners, corners[1:] + corners[:1], strict=True):
        e = sub(w, v)
        a, b = cross(e, sub(p, v)), cross(e, d)
        if b == 0:
            if a < 0:
                return None
        elif b > 0:
            t0 = max(t0, Fraction(-a, b))
        else:
            t1 = min(t1, Fraction(-a, b))
    return (t0, t1) if t0 <= t1 else None


def strictly_inside(x, corners):
    edges = zip(corners, corners[1:] + corners[:1], strict=True)
    return all(cross(sub(w, v), sub(x, v)) > 0 for v, w in edges)


def nearest_copy(space, a, b):
    """The lattice coordinates of the copy of b nearest a, or None if two are as near."""
    wrapped = [k for k in range(2) if space.periodic[k]]
    copies = []
    for turns in itertools.product((-1, 0, 1), repeat=len(wrapped)):
        coords = space.coords[b].copy()
        for k, n in zip(wrapped, turns, strict=True):
            coords[k] += n * space.shape[k]
        step = coords @ space.basis.T - space.coords[a] @ space.basis.T
        if space.tiling == "triangle":
            step = step + space.centers[b] - space.coords[b] @ space.basis.T
            step = step - (space.centers[a] - space.coords[a] @ space.basis.T)
        copies.append((float(np.linalg.norm(step)), tuple(coords)))
    copies.sort()
    if len(copies) > 1 and copies[1][0] - copies[0][0] < 1e-9:
        return None
    return copies[0][1]


@functools.cache
def oracle(name, a, b):
    """Each cell near the line from a to b: (id, crossed, side, t0, t1), or None on a tie."""
    space = board(name)
    low, high = min(a, b), max(a, b)
    end = nearest_copy(space, low, high)
    if end is None:
        return None
    p, _ = polygon(space, *space.coords[low])
    q, _ = polygon(space, *end)
    d = sub(q, p)
    length = float(np.hypot(*d))
    found = []
    reach = [
        range(min(s, e) - 3, max(s, e) + 4) for s, e in zip(space.coords[low], end, strict=True)
    ]
    (width, height), (wrap_x, wrap_y) = space.shape, space.periodic
    for i, j in itertools.product(*reach):
        x, y = i % width if wrap_x else i, j % height if wrap_y else j
        if not (0 <= x < width and 0 <= y < height):
            continue
        center, corners = polygon(space, i, j)
        if abs(cross(d, sub(center, p))) > 8 * length:
            continue  # its center lies more than 8 frame units from the line, beyond any corner
        stretch = clip(p, d, corners)
        if stretch is None:
            continue
        t0, t1 = stretch
        middle = (p[0] + (t0 + t1) / 2 * d[0], p[1] + (t0 + t1) / 2 * d[1])
        crossed = strictly_inside(middle, corners)
        side = 1 if cross(d, sub(center, p)) > 0 else -1
        found.append((x * height + y, crossed, side, t0, t1))
    return found, a > b


def oracle_line(name, a, b, touching):
    traced = oracle(name, a, b)
    if traced is None:
        return None
    cells, backwards = traced
    kept = sorted((t0 + t1, cell) for cell, crossed, _, t0, t1 in cells if crossed or touching)
    order = list(dict.fromkeys(cell for _, cell in kept))
    return order[::-1] if backwards else order


def oracle_visible(name, a, b, walls):
    traced = oracle(name, a, b)
    if traced is None:
        return None
    cells, _ = traced
    blocking = [c for c in cells if walls[c[0]] and c[0] not in (a, b)]
    if any(crossed for _, crossed, _, _, _ in blocking):
        return False
    left = [(t0, t1) for _, _, side, t0, t1 in blocking if side > 0]
    right = [(t0, t1) for _, _, side, t0, t1 in blocking if side < 0]
    return not any(
        max(l0, r0) <= min(l1, r1) for (l0, l1), (r0, r1) in itertools.product(left, right)
    )


@functools.cache
def board(name):
    shape, tiling, periodic = BOARDS[name]
    return LatticeSpace(shape, tiling, periodic)


@pytest.mark.parametrize("name", BOARDS)
@pytest.mark.parametrize("touching", [False, True])
def test_lines_are_the_cells_the_segment_crosses_in_order(name, touching):
    space = board(name)
    for a, b in itertools.product(range(space.n_cells), repeat=2):
        if (a * 7 + b * 3) % 5:
            continue  # a fifth of the pairs, spread over the board
        expected = oracle_line(name, a, b, touching)
        if expected is None:
            continue
        found = line(space, a, b, touching=touching)
        assert found.dtype == np.int64
        assert found.tolist() == expected, (a, b)


@pytest.mark.parametrize("name", BOARDS)
@pytest.mark.parametrize("seed", [0, pytest.param(1, marks=pytest.mark.slow)])
def test_visible_follows_the_rule_for_corners_and_edges(name, seed):
    space = board(name)
    walls = np.random.default_rng(seed).random(space.n_cells) < 0.3
    checked = 0
    for a, b in itertools.combinations(range(space.n_cells), 2):  # sight is symmetric
        expected = oracle_visible(name, a, b, walls)
        if expected is None:
            continue
        assert visible(space, a, b, walls) is expected, (a, b)
        checked += 1
    assert checked > space.n_cells**2 // 4


@pytest.mark.parametrize("name", BOARDS)
def test_sight_is_visible_from_the_viewer_and_symmetric(name):
    space = board(name)
    walls = np.random.default_rng(3).random(space.n_cells) < 0.25
    seen = np.stack([sight(space, cell, walls) for cell in range(space.n_cells)])
    assert seen.dtype == np.bool_
    np.testing.assert_array_equal(seen, seen.T)
    assert seen.diagonal().all()
    for viewer in (0, space.n_cells // 2):
        expected = [visible(space, viewer, cell, walls) for cell in range(space.n_cells)]
        np.testing.assert_array_equal(seen[viewer], expected)


def test_sight_within_a_radius_is_the_visible_cells_in_range():
    space = LatticeSpace((12, 12), tiling="hex")
    walls = np.random.default_rng(4).random(space.n_cells) < 0.2
    viewer = space.ravel([5, 6])
    in_range = np.zeros(space.n_cells, bool)
    in_range[within(space, viewer, 3.5)] = True
    np.testing.assert_array_equal(
        sight(space, viewer, walls, radius=3.5), sight(space, viewer, walls) & in_range
    )


@pytest.mark.parametrize("radius", [None, 2.5])
def test_sight_from_many_viewers_is_each_viewers_sight(any_board, radius):
    space = any_board
    walls = np.random.default_rng(6).random(space.n_cells) < 0.25
    viewers = np.array([space.n_cells - 1, 0, 7, 0])  # unsorted, one twice
    rows = sight(space, viewers, walls, radius, separate=True)
    expected = np.stack([sight(space, viewer, walls, radius) for viewer in viewers])
    assert rows.dtype == np.bool_
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(sight(space, viewers, walls, radius), expected.any(axis=0))


def test_sight_takes_viewers_as_a_mask():
    space = LatticeSpace((9, 7), tiling="hex", periodic=False)
    walls = np.random.default_rng(7).random(space.n_cells) < 0.2
    mask = np.zeros(space.n_cells, bool)
    mask[[40, 3, 17]] = True
    np.testing.assert_array_equal(
        sight(space, mask, walls, 3.0, separate=True),
        sight(space, [3, 17, 40], walls, 3.0, separate=True),
    )


def test_sight_from_no_viewers_sees_nothing():
    space = LatticeSpace((5, 5))
    walls = np.zeros(25, bool)
    assert sight(space, np.zeros(25, bool), walls, 2.0).tolist() == [False] * 25
    assert sight(space, [], walls, separate=True).shape == (0, 25)


@pytest.mark.parametrize(
    "shape",
    [(6, 5), (60, 7)],  # a radius of 4 reaches round both axes, or only the short one
)
def test_sight_on_a_torus_too_short_for_its_radius_looks_to_the_nearest_copy(shape):
    space = LatticeSpace(shape, tiling="hex")
    walls = np.random.default_rng(8).random(space.n_cells) < 0.3
    in_range = np.zeros(space.n_cells, bool)
    for viewer in (0, 13, space.n_cells - 1):
        in_range[:] = False
        in_range[within(space, viewer, 4.0)] = True
        expected = [visible(space, viewer, cell, walls) for cell in range(space.n_cells)]
        np.testing.assert_array_equal(sight(space, viewer, walls, 4.0), expected & in_range)


def test_without_walls_everything_is_seen(any_board):
    space = any_board
    assert sight(space, 0, np.zeros(space.n_cells, bool)).all()


@pytest.fixture(params=list(BOARDS))
def any_board(request):
    return board(request.param)


def test_nothing_sees_between_two_squares_that_meet_at_a_corner():
    space = LatticeSpace((4, 4), periodic=False)
    walls = np.zeros(16, bool)
    walls[space.ravel([[1, 0], [0, 1]])] = True
    assert not visible(space, 0, space.ravel([2, 2]), walls)
    walls[space.ravel([0, 1])] = False  # one square: the line grazes its corner
    assert visible(space, 0, space.ravel([2, 2]), walls)


def test_walls_are_seen_but_hide_what_lies_behind_them():
    space = LatticeSpace((5, 1), periodic=False)
    walls = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(sight(space, 0, walls), [True, True, True, False, False])


def test_a_line_along_an_edge_between_hexes_is_blocked_only_by_both():
    space = LatticeSpace((5, 5), tiling="hex", periodic=False)
    a, b = space.ravel([0, 2]), space.ravel([2, 1])  # along the edge between (1, 1) and (1, 2)
    assert line(space, a, b).tolist() == [a, b]
    assert line(space, a, b, touching=True).tolist() == [
        a,
        space.ravel([1, 1]),
        space.ravel([1, 2]),
        b,
    ]
    walls = np.zeros(25, bool)
    walls[space.ravel([1, 1])] = True
    assert visible(space, a, b, walls)
    walls[space.ravel([1, 2])] = True
    assert not visible(space, a, b, walls)


def test_a_diagonal_line_on_squares_crosses_the_diagonal_cells():
    space = LatticeSpace((5, 5), periodic=False)
    found = line(space, 0, space.ravel([3, 3]))
    np.testing.assert_array_equal(space.unravel(found), [[0, 0], [1, 1], [2, 2], [3, 3]])
    touching = space.unravel(line(space, 0, space.ravel([1, 1]), touching=True))
    np.testing.assert_array_equal(touching, [[0, 0], [0, 1], [1, 0], [1, 1]])


def test_a_line_runs_backwards_when_its_ends_swap():
    space = LatticeSpace((10, 8), tiling="triangle")
    rng = np.random.default_rng(5)
    for a, b in rng.integers(0, space.n_cells, (40, 2)):
        np.testing.assert_array_equal(line(space, a, b), line(space, b, a)[::-1])


def test_a_line_wraps_across_an_edge():
    space = LatticeSpace((8, 3), periodic=(True, False))
    found = line(space, space.ravel([1, 0]), space.ravel([6, 0]))
    np.testing.assert_array_equal(space.unravel(found), [[1, 0], [0, 0], [7, 0], [6, 0]])


def test_a_line_between_two_equally_near_copies_takes_the_lower_turns():
    space = LatticeSpace((4, 4))  # (2, 0) lies 2 away from (0, 0) both ways round
    a, b = 0, space.ravel([2, 0])
    np.testing.assert_array_equal(space.unravel(line(space, a, b)), [[0, 0], [3, 0], [2, 0]])
    np.testing.assert_array_equal(line(space, b, a), line(space, a, b)[::-1])
    hexes = LatticeSpace((4, 4), tiling="hex")  # copies at (2, -2) and (-2, 2) tie
    found = line(hexes, 0, hexes.ravel([2, 2]))
    np.testing.assert_array_equal(hexes.unravel(found), [[0, 0], [3, 1], [2, 2]])


@pytest.mark.parametrize(
    ("shape", "tiling"),
    [
        ((4, 30), "hex"),
        ((30, 4), "hex"),
        ((6, 6), "hex"),
        ((6, 4), "hypercubic"),
        ((8, 4), "triangle"),
    ],
)
def test_lines_run_to_the_nearest_copy_with_the_lowest_turns(shape, tiling):
    space = LatticeSpace(shape, tiling=tiling)
    low, high = np.triu_indices(space.n_cells, 1)
    start, end = _ends(space, low, high)
    np.testing.assert_array_equal(start, space.coords[low])
    turns = (end - space.coords[high]) / np.array(shape)
    assert (turns == np.round(turns)).all()
    # Every copy within 8 turns, by length, then the turns along axis 0, then axis 1.
    grid = np.array(list(itertools.product(range(-8, 9), repeat=2)))
    copies = space.coords[high][:, None, :] + grid * np.array(shape)
    steps = (copies - space.coords[low][:, None, :]) @ space.basis.T
    if tiling == "triangle":
        steps += (
            space._tiling.shift[copies.sum(axis=-1) % 2]
            - space._tiling.shift[space.coords[low].sum(axis=-1) % 2][:, None, :]
        )
    lengths = np.linalg.norm(steps, axis=-1)
    nearest = lengths <= lengths.min(axis=1, keepdims=True) + 1e-9
    rank = np.where(nearest, grid[:, 0] * 100 + grid[:, 1], 10**6)
    np.testing.assert_array_equal(turns, grid[np.argmin(rank, axis=1)])


@pytest.mark.parametrize(
    ("shape", "radius"),
    [((60, 7), 4.0), ((3, 56), 2.5), ((56, 3), 2.5), ((8, 8), 6.0)],
)
def test_sight_agrees_with_visible_where_copies_tie(shape, radius):
    space = LatticeSpace(shape, tiling="hex")  # cells half a period apart have two copies
    walls = np.random.default_rng(9).random(space.n_cells) < 0.3
    viewers = [0, space.n_cells // 3, space.n_cells - 1]
    seen = sight(space, viewers, walls, radius, separate=True)
    for row, viewer in zip(seen, viewers, strict=True):
        for cell in within(space, viewer, radius):
            assert row[cell] == visible(space, viewer, cell, walls), (viewer, cell)


def test_a_line_to_itself_is_its_cell():
    space = LatticeSpace((4, 4))
    assert line(space, 5, 5).tolist() == [5]
    assert visible(space, 5, 5, np.ones(16, bool))


def test_other_spaces_raise():
    blocked = np.zeros(12, bool)
    with pytest.raises(TypeError, match="2-D LatticeSpace"):
        line(icosphere(1), 0, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="2-D LatticeSpace"):
        sight(icosphere(1), 0, blocked)  # ty: ignore[invalid-argument-type]
    cubes = LatticeSpace((3, 3, 3))
    with pytest.raises(ValueError, match="2-D"):
        visible(cubes, 0, 1, np.zeros(27, bool))
    with pytest.raises(ValueError, match="2-D"):
        line(LatticeSpace((5,)), 0, 1)


def test_bad_arguments_raise(not_an_integer, not_a_real):
    space = LatticeSpace((4, 4))
    blocked = np.zeros(16, bool)
    with pytest.raises(TypeError):
        line(space, not_an_integer, 0)
    with pytest.raises(TypeError):
        visible(space, 0, not_an_integer, blocked)
    with pytest.raises(TypeError if np.asarray(not_an_integer).dtype != bool else ValueError):
        sight(space, not_an_integer, blocked)  # a bool is a mask, here of the wrong shape
    if not_a_real is not None:  # None is the default: the whole lattice
        with pytest.raises(TypeError):
            sight(space, 0, blocked, radius=not_a_real)
    with pytest.raises(IndexError):
        line(space, 0, 16)
    with pytest.raises(IndexError):
        sight(space, -1, blocked)
    with pytest.raises(TypeError, match="blocked"):
        visible(space, 0, 1, np.zeros(16))
    with pytest.raises(ValueError, match="blocked"):
        sight(space, 0, np.zeros(15, bool))
    with pytest.raises(ValueError, match="radius"):
        sight(space, 0, blocked, radius=-1.0)


def test_bad_viewers_raise(not_cell_ids):
    space = LatticeSpace((4, 4))
    blocked = np.zeros(16, bool)
    values, error = not_cell_ids
    if values.dtype == np.bool_:
        error = ValueError  # a mask of the wrong shape
    with pytest.raises(error):
        sight(space, values, blocked)
    with pytest.raises(IndexError):
        sight(space, [0, 16], blocked)
    with pytest.raises(ValueError, match="1-D"):
        sight(space, [[0, 1]], blocked)
