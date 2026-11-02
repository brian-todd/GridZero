import dataclasses

import numpy as np
import pytest

from examples.simulations.migration import HERDS, Migration, distance
from gridzero.core import Simulation, icosphere
from model_demo import check_model

MIGRATION = Migration(reach=0.6)  # about two cells at frequency 4
SEASONS = dataclasses.replace(MIGRATION, seasons=True)


def phases(migration):
    """Return a migration's phases by name."""
    return {p.name: p for p in migration.phases()}


@pytest.fixture(scope="module")
def planet():
    return icosphere(4)


@pytest.fixture(scope="module")
def settled(planet):
    """The herds after 1000 steps of still rain, and their distances every 100 steps."""
    start = MIGRATION.start(planet, 0)
    sim = Simulation(start, MIGRATION.phases(), seed=0)
    history = sim.run(1000, observers={"distance": distance}, every=100)
    return start, sim.state, dict(zip(HERDS, history["distance"][-1], strict=True))


def test_the_rain_peaks_at_the_middle_of_the_band():
    z = np.array([-1.0, 0.0, 0.3, 1.0])
    rain = MIGRATION.band(z, 0.3)
    assert rain[2] == pytest.approx(1.1)
    assert rain.argmax() == 2
    assert rain[0] == pytest.approx(MIGRATION.dry, abs=1e-6)


def test_the_distance_is_0_at_the_ideal_free_distribution_and_1_at_its_opposite(planet):
    state = MIGRATION.start(planet, 0)
    rain = state.rain
    ideal = rain * planet.measures
    opposite = np.where(rain == rain.min(), 1.0, 0.0)  # all where the rain is least
    fields = list(HERDS.values())
    ideal_state = state.replace({fields[0]: ideal, fields[1]: opposite})
    d = distance(ideal_state)
    assert d[0] == pytest.approx(0, abs=1e-12)
    assert d[1] == pytest.approx(1 - ideal[rain == rain.min()].sum() / ideal.sum())


def test_every_herd_keeps_its_animals(settled):
    start, end, _ = settled
    for herd in HERDS.values():
        assert end.total(herd) == pytest.approx(start.total(herd), rel=1e-12)


def test_moving_only_where_a_neighbor_offers_more_reaches_the_ideal_free_distribution(settled):
    *_, d = settled
    assert d["Flow down the gradient"] < 0.01
    assert d["Spread, leave if better"] < 0.01
    assert d["same, Ball reach"] < 0.01


def test_always_moving_a_share_settles_short_of_the_ideal(settled):
    *_, d = settled
    assert d["Spread toward food"] > 0.05
    assert d["NeighborAllocation"] > 0.05


def test_the_random_walk_ignores_the_grass(settled):
    start, _, d = settled
    assert d["random walk"] == pytest.approx(distance(start)[-1], abs=0.02)


def test_the_seasons_move_the_rain_band_north_and_south(planet):
    sim = Simulation(SEASONS.start(planet, 0), [phases(SEASONS)["seasons"]], seed=0)
    z = planet.centers[:, 2]
    wettest = []
    for _ in range(4):
        sim.run(MIGRATION.year // 4)
        wettest.append(z[sim.state.rain.argmax()])
    north, equator, south, back = wettest  # each a quarter of a year on
    assert north > 0.3
    assert south < -0.3
    assert abs(equator) < 0.3
    assert abs(back) < 0.3


def test_a_reach_that_finds_no_neighbor_moves_nothing(planet):
    tight = Migration(reach=0.01)
    start = tight.start(planet, 0)
    sim = Simulation(start, [phases(tight)["range_far"]], seed=0)
    sim.run(3)
    np.testing.assert_array_equal(sim.state.ranging, start.ranging)


def test_the_model_keeps_its_promises(planet):
    check_model(SEASONS, planet, steps=50)


def test_the_seasons_come_first_and_only_when_asked():
    first, *rest = phases(SEASONS)
    assert first == "seasons"
    assert "seasons" not in phases(MIGRATION)
    assert rest == list(phases(MIGRATION))


def test_the_seasons_wait_for_their_first_step(planet):
    late = dataclasses.replace(SEASONS, seasons_from=3)
    sim = Simulation(late.start(planet, 0), [phases(late)["seasons"]], seed=0)
    start = sim.state.rain
    sim.run(3)
    np.testing.assert_array_equal(sim.state.rain, start)
    sim.run(1)  # reads step 3
    middle = late.swing * np.sin(2 * np.pi * 3 / late.year)
    np.testing.assert_array_equal(sim.state.rain, late.band(planet.centers[:, 2], middle))
