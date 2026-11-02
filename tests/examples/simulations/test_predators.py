import dataclasses

import numpy as np
import pytest

from examples.simulations.predators import (
    BURNING,
    Downwind,
    FoxesState,
    Predators,
    RabbitsState,
    means,
)
from gridzero.core import LatticeSpace, Simulation, icosphere
from model_demo import check_model

PREDATORS = Predators()
CALM = dataclasses.replace(PREDATORS, wildfire=False)
FOXES = FoxesState.foxes.population


def phases(predators):
    """Return the phases by name."""
    return {p.name: p for p in predators.phases()}


@pytest.fixture(scope="module")
def meadow():
    return LatticeSpace((24, 16), tiling="triangle", periodic=True)


@pytest.fixture(scope="module")
def calm(meadow):
    """Forty years without fire: the yearly means, and the last state."""
    sim = Simulation(CALM.start(meadow, 0), CALM.phases(), seed=0)
    history = sim.run(40 * CALM.year, observers={"means": means}, every=CALM.year)
    return history["means"], sim.state


def test_the_model_keeps_its_promises(meadow):
    early = dataclasses.replace(PREDATORS, foxes_arrive=1, lightning=0.05)  # fires, then foxes
    end = check_model(early, meadow, steps=early.year + early.year // 2)  # to a summer's end
    assert end.total(FOXES) > 0
    assert (end.fire > 0).any()


def test_the_foxes_arrive_once_in_their_week_in_a_few_places(meadow):
    early = dataclasses.replace(PREDATORS, foxes_arrive=1)
    sim = Simulation(early.start(meadow, 0), [phases(early)["foxes arrive"]], seed=0)
    sim.run(early.year)
    assert (sim.state.foxes.population == 0).all()
    sim.run(1)
    arrived = sim.state.foxes.population
    assert (arrived > 0).sum() == early.places
    np.testing.assert_array_equal(arrived[arrived > 0], early.founders)
    sim.run(early.year)
    np.testing.assert_array_equal(sim.state.foxes.population, arrived)


def test_the_wind_reaches_only_cells_downwind():
    space = LatticeSpace((8, 8), periodic=True)
    op = Downwind((1.0, 0.0)).compile(space)
    graph = op.graph
    step = space.displacement(space.centers[graph.sources], space.centers[graph.indices])
    assert (step[:, 0] > 0).all()
    assert (np.diff(graph.indptr) == 3).all()  # of 8 corner neighbors, 3 lie ahead


def test_the_wind_needs_a_lattice():
    with pytest.raises(TypeError, match="lattice"):
        Downwind((1.0, 0.0)).compile(icosphere(1))


def test_the_levels_of_the_theory():
    assert PREDATORS.rabbits_hold == pytest.approx(0.2)
    assert PREDATORS.foxes_hold == pytest.approx(0.2)
    assert PREDATORS.greener == pytest.approx(0.5)


def test_rabbits_alone_hold_the_grass_near_their_rstar(calm):
    history, _ = calm
    before = history[PREDATORS.foxes_arrive // 2 : PREDATORS.foxes_arrive].mean(axis=0)
    assert before[0] == pytest.approx(PREDATORS.rabbits_hold, abs=0.02)
    assert before[2] == 0


def test_foxes_hold_the_rabbits_and_the_meadow_grows_greener(calm):
    history, _ = calm
    after = history[-10:].mean(axis=0)
    assert after[1] == pytest.approx(PREDATORS.foxes_hold, abs=0.02)
    assert after[0] == pytest.approx(PREDATORS.greener, abs=0.03)


def test_without_wildfire_nothing_burns(calm):
    history, state = calm
    assert (history[:, 3] == 0).all()
    assert (state.fire == 0).all()


def test_fires_burn_the_grass_and_recover_into_ash(meadow):
    start = PREDATORS.start(meadow, 0)
    fire = np.zeros(meadow.n_cells, np.int8)
    fire[0] = BURNING
    lit = start.replace({RabbitsState.fire: fire})
    by_name = phases(PREDATORS)
    sim = Simulation(lit, [by_name["grow"], by_name["fire"]], seed=0)
    sim.run(1)
    assert sim.state.grass[0] == pytest.approx(0)
    assert sim.state.fire[0] == BURNING + 1
    assert (sim.state.fire[1:] <= BURNING).all()
    sim.run(1)  # grass grows back on ash at a fifth of the rate
    regrown = PREDATORS.regrowth / 5 * PREDATORS.capacity * meadow.measures[0]
    assert sim.state.grass[0] == pytest.approx(regrown)
    sim.run(PREDATORS.ash - 2)
    assert sim.state.fire[0] == PREDATORS.ash + 1  # the last week of ash
    sim.run(1)
    assert sim.state.fire[0] == 0  # recovered


def test_the_start_has_no_foxes_and_a_full_meadow(meadow):
    m = means(PREDATORS.start(meadow, 0))
    assert m[0] == pytest.approx(PREDATORS.capacity)
    assert m[2] == 0
    assert m[3] == 0
