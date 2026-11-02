import numpy as np
import pytest

from gridzero.core.geometry import sphere_points
from gridzero.worlds import noise


def random_points(rng, n):
    return sphere_points(rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n))))


def test_noise_at_a_point_does_not_depend_on_the_other_points(rng):
    points = random_points(rng, 4200)  # more than one block of 4096 points
    field = noise(points, np.random.default_rng(1))
    np.testing.assert_allclose(noise(points[::-1], np.random.default_rng(1))[::-1], field)
    np.testing.assert_allclose(noise(points[4000:], np.random.default_rng(1)), field[4000:])
    np.testing.assert_allclose(
        noise(points.reshape(42, 100, 3), np.random.default_rng(1)), field.reshape(42, 100)
    )


def test_noise_repeats_with_its_seed_and_changes_with_it(rng):
    points = random_points(rng, 100)
    first = noise(points, np.random.default_rng(3))
    np.testing.assert_array_equal(noise(points, np.random.default_rng(3)), first)
    assert not np.allclose(noise(points, np.random.default_rng(4)), first)


def test_noise_has_mean_0_and_variance_1_over_seeds():
    points = sphere_points([0, 50, 120], [0, 30, -60])
    values = np.array([noise(points, np.random.default_rng(seed)) for seed in range(1000)])
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=0.15)
    np.testing.assert_allclose(values.var(axis=0), 1, atol=0.15)


def test_rougher_noise_differs_more_between_neighbors(make_icosphere):
    planet = make_icosphere(16)
    graph = planet.adjacency()

    def step_between_neighbors(roughness):
        field = noise(planet.centers, np.random.default_rng(0), roughness=roughness)
        return np.abs(field[graph.sources] - field[graph.indices]).mean()

    assert step_between_neighbors(0.9) > 2 * step_between_neighbors(0.3)


@pytest.mark.parametrize("periods", [None, [[3.0, 0.0]], [[2.0, 0.0], [1.0, 1.5]]])
def test_noise_has_mean_0_and_variance_1_in_the_plane_too(periods):
    points = np.array([[0.0, 0.0], [1.3, -0.2], [5.0, 7.0]])
    values = np.array(
        [
            noise(points, np.random.default_rng(seed), scales=(1.0, 8.0), periods=periods)
            for seed in range(1000)
        ]
    )
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=0.15)
    np.testing.assert_allclose(values.var(axis=0), 1, atol=0.15)


@pytest.mark.parametrize("periods", [[[3.0, 0.0]], [[2.0, 0.0], [1.0, 1.5]], [[0.0, 0.0, 2.5]]])
def test_noise_repeats_over_each_of_its_periods(rng, periods):
    points = rng.uniform(-5, 5, (200, len(periods[0])))
    field = noise(points, np.random.default_rng(0), periods=periods)
    for period in np.array(periods):
        assert not np.allclose(noise(points + period, np.random.default_rng(0)), field)
        for turns in (1, -2, 7):
            shifted = noise(points + turns * period, np.random.default_rng(0), periods=periods)
            np.testing.assert_allclose(shifted, field, atol=1e-9)


def test_long_periods_barely_change_the_noise(rng):
    points = rng.uniform(-5, 5, (500, 2))
    free = noise(points, np.random.default_rng(0))
    periodic = noise(points, np.random.default_rng(0), periods=[[100.0, 0.0], [0.0, 120.0]])
    assert np.corrcoef(free, periodic)[0, 1] > 0.95
    np.testing.assert_array_equal(
        noise(points, np.random.default_rng(0), periods=np.zeros((0, 2))), free
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rng": 0}, TypeError, "Generator"),
        ({"points": np.zeros((4, 0))}, ValueError, "shape"),
        ({"points": 1.0}, ValueError, "shape"),
        ({"points": np.full((1, 3), np.nan)}, ValueError, "NaN"),
        ({"periods": np.eye(3)[[0, 1, 2, 0]]}, ValueError, "periods must have shape"),
        ({"periods": [[1.0, 0.0]]}, ValueError, "periods must have shape"),
        ({"periods": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]}, ValueError, "linearly independent"),
        ({"periods": [[0.0, 0.0, 0.0]]}, ValueError, "linearly independent"),
        ({"periods": [[np.nan, 1.0, 0.0]]}, ValueError, "periods holds NaN"),
        ({"periods": [["a", "b", "c"]]}, TypeError, "periods"),
        ({"scales": (4.0, 1.0)}, ValueError, "scales"),
        ({"scales": (0.0, 1.0)}, ValueError, "scales"),
        ({"roughness": 0.0}, ValueError, "roughness must be positive"),
        ({"waves": 0}, ValueError, "waves"),
        ({"waves": 2.0}, TypeError, "waves"),
    ],
)
def test_noise_rejects_bad_arguments(kwargs, error, match):
    arguments = {"points": np.eye(3), "rng": np.random.default_rng(0)} | kwargs
    with pytest.raises(error, match=match):
        noise(**arguments)
