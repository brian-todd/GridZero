"""Directions on squares and hexes: `LatticeSpace.directions`, `snap`, `turns`, `in_front`."""

import itertools

import numpy as np
import pytest

from gridzero.core import LatticeSpace, in_front, snap, turns

SPACES = {
    "squares": LatticeSpace((5, 5), periodic=False),
    "hexes": LatticeSpace((5, 5), tiling="hex", periodic=False),
    "wrapped hexes": LatticeSpace((6, 6), tiling="hex", periodic=True),
}


@pytest.fixture(params=list(SPACES.values()), ids=list(SPACES))
def space(request):
    return request.param


def plane(space, v):
    """A lattice offset as a vector of the plane of the centers."""
    return space.basis @ np.asarray(v, dtype=float)


def angle(space, v):
    x, y = plane(space, v)
    return np.degrees(np.arctan2(y, x)) % 360


def vectors(radius=5):
    return [v for v in itertools.product(range(-radius, radius + 1), repeat=2) if v != (0, 0)]


def test_directions_are_the_facet_neighbors_counterclockwise_from_the_first_axis(space):
    directions = space.directions()
    n = len(directions)
    assert n == (6 if space.tiling == "hex" else 4)
    assert directions.dtype == np.int64
    assert not directions.flags.writeable
    assert sorted(map(tuple, directions)) == sorted(map(tuple, space.offsets("facet")))
    angles = [angle(space, d) for d in directions]
    np.testing.assert_allclose(angles, np.arange(n) * 360 / n, atol=1e-9)
    assert space.directions() is directions


def test_the_opposite_of_a_direction_is_half_way_round_as_the_half_turn_maps_it():
    space = LatticeSpace((7, 7), tiling="hex", periodic=False)
    directions, opposite = space.directions(), space.opposite()
    n = len(directions)
    middle = space.ravel([3, 3])
    for d, step in enumerate(directions):
        np.testing.assert_array_equal(directions[(d + n // 2) % n], -step)
        to = space.ravel(space.unravel(middle) + step)
        back = space.unravel(opposite[to]) - space.unravel(opposite[middle])
        np.testing.assert_array_equal(back, directions[(d + n // 2) % n])


@pytest.mark.parametrize(
    ("shape", "tiling"),
    [((4, 4), "triangle"), ((3, 3, 3), "hypercubic"), ((5,), "hypercubic")],
)
def test_only_squares_and_hexes_in_two_dimensions_have_directions(shape, tiling):
    space = LatticeSpace(shape, tiling=tiling, periodic=False)
    with pytest.raises(ValueError, match="squares or hexes"):
        space.directions()
    with pytest.raises(ValueError, match="squares or hexes"):
        snap(space, (1, 0), 0)
    with pytest.raises(ValueError, match="squares or hexes"):
        in_front(space, 0, (1, 0))
    with pytest.raises(ValueError, match="squares or hexes"):
        turns(space, 0, 1)


def test_turns_count_steps_of_direction_the_short_way():
    hexes, squares = SPACES["hexes"], SPACES["squares"]
    assert [turns(hexes, 0, d) for d in range(6)] == [0, 1, 2, 3, 2, 1]
    assert turns(hexes, 5, 1) == 2
    assert [turns(squares, 3, d) for d in range(4)] == [1, 2, 1, 0]


def test_snap_gives_each_direction_for_its_multiples(space):
    n = len(space.directions())
    for previous, (d, step) in itertools.product(range(n), enumerate(space.directions())):
        for k in (1, 2, 7):
            assert snap(space, k * step, previous) == d


def test_snap_picks_the_direction_of_the_smallest_angle_and_ties_toward_the_previous(space):
    n = len(space.directions())
    for previous, v in itertools.product(range(n), vectors()):
        gaps = [abs((angle(space, v) - 360 * d / n + 180) % 360 - 180) for d in range(n)]
        nearest = [d for d in range(n) if gaps[d] - min(gaps) < 1e-9]
        found = snap(space, v, previous)
        assert found in nearest
        if len(nearest) == 2:  # exactly between two: the tie goes toward the previous facing
            assert found == min(nearest, key=lambda d: turns(space, d, previous))


def test_snap_breaks_a_tie_toward_the_previous_facing():
    corner = (1, 1)  # 30 degrees on hexes, between directions 0 and 1; 45 on squares
    assert [snap(SPACES["hexes"], corner, p) for p in range(6)] == [0, 1, 1, 1, 0, 0]
    assert [snap(SPACES["squares"], corner, p) for p in range(4)] == [0, 1, 1, 0]


def test_snap_keeps_the_previous_facing_for_the_zero_vector(space):
    assert snap(space, (0, 0), 3) == 3


def test_snap_is_exact_far_past_where_floats_would_tie():
    hexes = SPACES["hexes"]
    big = 10**8  # (big, big + 1) is just past the corner at 30 degrees, toward direction 1
    assert snap(hexes, (big, big + 1), 0) == 1
    assert snap(hexes, (big + 1, big), 1) == 0


def test_in_front_holds_the_directions_within_a_quarter_turn(space):
    directions = space.directions()
    n = len(directions)
    for facing in range(n):
        front = in_front(space, facing, directions)
        np.testing.assert_array_equal(front, [turns(space, d, facing) <= n // 4 for d in range(n)])


def test_in_front_is_the_half_plane_faced_with_its_edge_in_front(space):
    vs = np.array(vectors())
    for facing, ahead in enumerate(space.directions()):
        cos = np.array([plane(space, ahead) @ plane(space, v) for v in vs])
        found = in_front(space, facing, vs)
        assert found.shape == (len(vs),)
        np.testing.assert_array_equal(found[np.abs(cos) > 1e-9], cos[np.abs(cos) > 1e-9] > 0)
        assert found[np.abs(cos) <= 1e-9].all()  # exactly along the edge
        assert in_front(space, facing, (0, 0))


def test_in_front_turns_with_the_mirror(space):
    vs = np.array(vectors())
    n = len(space.directions())
    for facing in range(n):
        np.testing.assert_array_equal(
            in_front(space, facing, vs), in_front(space, (facing + n // 2) % n, -vs)
        )


def test_directions_must_be_directions_of_the_lattice(space):
    n = len(space.directions())
    for bad in (-1, n):
        with pytest.raises(ValueError, match="direction"):
            snap(space, (1, 0), bad)
        with pytest.raises(ValueError, match="direction"):
            in_front(space, bad, (1, 0))
        with pytest.raises(ValueError, match="direction"):
            turns(space, 0, bad)


def test_directions_must_be_integers(not_an_integer):
    space = SPACES["hexes"]
    with pytest.raises(TypeError):  # a direction is never a float or a bool
        snap(space, (1, 0), not_an_integer)
    with pytest.raises(TypeError):
        in_front(space, not_an_integer, (1, 0))
    with pytest.raises(TypeError):
        turns(space, not_an_integer, 0)


def test_vectors_must_be_integer_offsets_of_two_axes():
    space = SPACES["squares"]
    with pytest.raises(TypeError):  # floats are never truncated
        snap(space, (0.5, 1.0), 0)
    with pytest.raises(TypeError):
        in_front(space, 0, [(True, False)])
    with pytest.raises(ValueError, match="shape"):
        snap(space, (1, 0, 0), 0)
    with pytest.raises(ValueError, match="axis of 2"):
        in_front(space, 0, 3)
