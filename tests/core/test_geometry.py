import numpy as np
import pytest
from scipy.spatial import KDTree

from gridzero.core.geometry import (
    great_circle_angle,
    icosahedron,
    lon_lat,
    sphere_points,
    subdivided_icosahedron,
)


def test_icosahedron_is_regular_and_closed():
    vertices, faces = icosahedron()
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0)
    edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    assert len(edges) == 30
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    np.testing.assert_allclose(lengths, lengths[0])
    np.testing.assert_array_equal(np.bincount(faces.ravel()), np.full(12, 5))


@pytest.mark.parametrize("frequency", [1, 2, 3, 4, 7])
def test_subdivision_has_10f2_plus_2_distinct_unit_points(frequency):
    points = subdivided_icosahedron(frequency)
    assert points.shape == (10 * frequency**2 + 2, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
    nearest_other, _ = KDTree(points).query(points, k=2)
    assert nearest_other[:, 1].min() > 1e-3


def test_subdivision_is_deterministic():
    np.testing.assert_array_equal(subdivided_icosahedron(5), subdivided_icosahedron(5))


@pytest.mark.parametrize("frequency", [0, -2])
def test_subdivision_rejects_a_frequency_below_one(frequency):
    with pytest.raises(ValueError, match="frequency must be positive"):
        subdivided_icosahedron(frequency)


def test_subdivision_rejects_a_frequency_that_is_not_an_integer(not_an_integer):
    with pytest.raises(TypeError, match="frequency must be an integer"):
        subdivided_icosahedron(not_an_integer)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], np.pi / 2),
        ([1, 0, 0], [-1, 0, 0], np.pi),
        ([2, 0, 0], [0, 0, 5], np.pi / 2),
    ],
    ids=["same", "orthogonal", "antipodal", "unnormalized"],
)
def test_great_circle_angle_known_values(a, b, expected):
    assert great_circle_angle(a, b) == pytest.approx(expected)


def test_great_circle_angle_resolves_tiny_angles():
    tiny = 1e-9
    assert great_circle_angle([1, 0, 0], [np.cos(tiny), np.sin(tiny), 0]) == pytest.approx(tiny)


def test_great_circle_angle_broadcasts_over_leading_axes(rng):
    a = rng.normal(size=(4, 1, 3))
    b = rng.normal(size=(1, 5, 3))
    assert great_circle_angle(a, b).shape == (4, 5)


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [(0, 0, [1, 0, 0]), (90, 0, [0, 1, 0]), (180, 0, [-1, 0, 0]), (-90, 0, [0, -1, 0])]
    + [(lon, 90, [0, 0, 1]) for lon in (0, 45)]
    + [(30, -90, [0, 0, -1])],
)
def test_sphere_points_turn_from_x_toward_y_and_rise_toward_z(lon, lat, expected):
    np.testing.assert_allclose(sphere_points(lon, lat), expected, atol=1e-15)


def test_lon_lat_inverts_sphere_points(rng):
    lon, lat = rng.uniform(-180, 180, 100), rng.uniform(-89.9, 89.9, 100)
    points = sphere_points(lon, lat)
    np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 1.0)
    back = lon_lat(3 * points)  # the length of a point does not matter
    np.testing.assert_allclose(back, (lon, lat), atol=1e-12)


def test_sphere_points_broadcast_longitudes_against_latitudes():
    assert sphere_points(np.zeros((4, 1)), np.zeros(5)).shape == (4, 5, 3)
    assert sphere_points(10.0, np.zeros(5)).shape == (5, 3)
