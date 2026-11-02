import dataclasses

import numpy as np
import pytest

from examples.simulations.biomes import (
    BIOMES,
    Biomes,
    LandState,
    Rates,
    WolvesState,
    by_cover,
)
from gridzero.core import Simulation, icosphere
from model_demo import check_model

BIOMES_ = Biomes()
NO_DEER = dataclasses.replace(BIOMES_, deer=False)
WOLVES = WolvesState.wolves.population


def phases(biomes):
    """Return the phases by name."""
    return {p.name: p for p in biomes.phases()}


def run(biomes, planet, years):
    """Run from the start: the yearly means and the last state."""
    sim = Simulation(biomes.start(planet, 0), biomes.phases(), seed=0)
    history = sim.run(years * biomes.year, observers={"means": biomes.means}, every=biomes.year)
    return history["means"], sim.state


@pytest.fixture(scope="module")
def planet():
    return icosphere(8)


@pytest.fixture(scope="module")
def runs(planet):
    """Thirty years without deer and with: the yearly means and the last states."""
    return run(NO_DEER, planet, 30), run(BIOMES_, planet, 30)


def test_the_model_keeps_its_promises(planet):
    early = dataclasses.replace(BIOMES_, wolves_arrive=1)
    end = check_model(early, planet, steps=early.year + 10)
    assert end.total(WOLVES) > 0


def test_the_wolves_arrive_once_in_their_week_where_each_biome_is_alone(planet):
    early = dataclasses.replace(BIOMES_, wolves_arrive=1)
    sim = Simulation(early.start(planet, 0), [phases(early)["wolves arrive"]], seed=0)
    sim.run(early.year)
    assert (sim.state.wolves.population == 0).all()
    sim.run(1)
    arrived = sim.state.wolves.population > 0
    for biome in BIOMES.values():
        assert (arrived & (sim.state[biome.cover] == 1)).sum() == early.places
    assert arrived.sum() == 2 * early.places
    np.testing.assert_array_equal(sim.state.wolves.population[arrived], early.founders)
    before = sim.state.wolves.population
    sim.run(early.year)
    np.testing.assert_array_equal(sim.state.wolves.population, before)


def test_rstar_in_weekly_steps():
    assert Rates(0.05, 0.2, 0.04, 0.05).rstar == pytest.approx(0.04 / (0.2 * 0.96))


def test_the_biomes_cover_each_cell_together(planet):
    state = BIOMES_.start(planet, 0)
    grass, forest = state[LandState.grassland.cover], state[LandState.forest.cover]
    np.testing.assert_allclose(grass + forest, 1)
    z = planet.centers[:, 2]
    assert (forest[z > BIOMES_.mixed] == 1).all()
    assert (forest[z < -BIOMES_.mixed] == 0).all()


def test_without_deer_the_forest_starts_empty(planet):
    state = NO_DEER.start(planet, 0)
    assert state.total(LandState.forest.grazers.population) == 0
    assert state.total(LandState.grassland.grazers.population) > 0


def test_apart_each_biomes_grazers_hold_its_plants_at_their_rstar(runs):
    (_, _), (history, _) = runs
    arrive = BIOMES_.wolves_arrive
    apart = history[arrive - 5 : arrive].mean(axis=0)
    assert apart[0] == pytest.approx(BIOMES_.grassland.rstar, abs=0.01)
    assert apart[2] == pytest.approx(BIOMES_.forest.rstar, abs=0.01)
    assert (history[:arrive, 4] == 0).all()  # no wolves yet


def test_deer_lower_the_rabbits_most_where_the_forest_covers_most(runs):
    (_, lonely), (_, shared) = runs
    (without, no_deer), (among, deer) = by_cover(lonely), by_cover(shared)
    assert (no_deer == 0).all()
    assert (np.diff(deer) > 0).all()
    assert among[-1] < 0.9 * without[-1]  # apparent competition


def test_wolves_live_on_the_grazers(runs):
    (no_deer, _), (history, _) = runs
    assert history[-1, 4] > 0
    assert no_deer[-1, 4] > 0
    assert history[-1, 4] > no_deer[-1, 4]  # deer feed more wolves


def test_rates_follow_the_order_of_the_biomes():
    assert list(BIOMES_.rates()) == list(BIOMES)
    assert BIOMES_.rates()["forest"] == (LandState.forest, BIOMES_.forest)


def test_wolves_hunt_the_grazers_of_both_biomes(planet):
    state = BIOMES_.start(planet, 0)
    wild = state.replace({WOLVES: np.full(planet.n_cells, 0.1)})
    sim = Simulation(wild, [phases(BIOMES_)["hunt"]], seed=0)
    sim.run(1)
    for biome in BIOMES.values():
        assert sim.state.total(biome.grazers.population) < wild.total(biome.grazers.population)
    caught = sum(
        wild.total(b.grazers.population) - sim.state.total(b.grazers.population)
        for b in BIOMES.values()
    )
    born = sim.state.total(WOLVES) - wild.total(WOLVES)
    assert born == pytest.approx(BIOMES_.efficiency * caught)


def test_by_cover_counts_only_cells_the_biomes_share(planet):
    state = BIOMES_.start(planet, 0)
    forest = state[LandState.forest.cover]
    rabbits = np.where((forest == 0) | (forest == 1), 1.0, 0.0)  # only in unmixed cells
    deer = np.where((forest > 0) & (forest < 1), 1.0, 0.0) * planet.measures
    counted = by_cover(
        state.replace(
            {
                LandState.grassland.grazers.population: rabbits,
                LandState.forest.grazers.population: deer,
            }
        )
    )
    np.testing.assert_array_equal(counted[0], 0)
    np.testing.assert_allclose(counted[1], 1)
