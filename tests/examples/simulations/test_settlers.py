import numpy as np
import pytest

from examples.simulations.settlers import Settlement, SettlersState
from gridzero.core import LatticeSpace, Simulation
from model_demo import check_model

SETTLEMENT = Settlement()
PHASES = {p.name: p for p in SETTLEMENT.phases()}
PEOPLE, STORES = SettlersState.settlers.people, SettlersState.settlers.stores


@pytest.fixture(scope="module")
def world():
    return LatticeSpace((24, 24), tiling="hex", periodic=True)


@pytest.fixture(scope="module")
def start(world):
    return SETTLEMENT.start(world, seed=4)


def test_each_people_starts_in_three_places_on_land_with_a_year_of_food(start):
    people = start.settlers.people
    assert ((people > 0).sum(axis=1) <= 3).all()
    assert (people.sum(axis=1) > 0).all()
    assert start.land[people.sum(axis=0) > 0].all()
    np.testing.assert_array_equal(start.settlers.stores, SETTLEMENT.year * people)


def test_farmers_harvest_more_than_herders_on_the_coasts_but_less_in_the_hills(start):
    land = start.land
    coast = land & (start.height < 0.05)
    hills = land & (start.height > SETTLEMENT.hills)
    assert (start.harvest[0, coast] > start.harvest[1, coast]).all()
    assert (start.harvest[0, hills] < start.harvest[1, hills]).all()


def test_farming_never_yields_less_than_a_tenth_of_its_best(start):
    peaks = start.height >= 0.9 * SETTLEMENT.hills
    assert peaks.any()
    np.testing.assert_allclose(start.harvest[0, peaks], 0.1 * SETTLEMENT.farming)


def test_hunger_runs_from_0_with_a_full_reserve_to_1_with_nothing(start):
    people = np.zeros((2, start.n_cells))
    stores = np.zeros((2, start.n_cells))
    people[:, [0, 1, 2, 4]] = 10.0
    stores[:, 0] = 10.0 * SETTLEMENT.reserve  # full
    stores[:, 1] = 5.0 * SETTLEMENT.reserve  # half
    stores[:, 4] = 20.0 * SETTLEMENT.reserve  # more than full
    hungry = SETTLEMENT.hunger(start.replace({PEOPLE: people, STORES: stores}))
    np.testing.assert_allclose(hungry[:, :5], [[0, 0.5, 1, 0, 0]] * 2)  # cell 3 has no one


def test_raids_conserve_the_stores_and_take_them_from_the_farmers(start):
    people = np.zeros((2, start.n_cells))
    stores = np.zeros((2, start.n_cells))
    people[:, 0] = [50.0, 50.0]
    stores[:, 0] = [100.0, 10.0]
    state = start.replace({PEOPLE: people, STORES: stores})
    sim = Simulation(state, [PHASES["raid"]], seed=0)
    sim.run(1)
    after = sim.state.settlers
    herders = 50.0 / start.space.measures[0]
    looted = SETTLEMENT.loot * SETTLEMENT.raids * herders * 100.0
    np.testing.assert_allclose(after.stores[:, 0], [100.0 - looted, 10.0 + looted])
    assert (after.people[:, 0] < 50.0).all()


def test_raids_take_at_most_all_the_farmers_stores(start):
    people = np.zeros((2, start.n_cells))
    stores = np.zeros((2, start.n_cells))
    people[:, 0] = [1.0, 1e5]  # a horde
    stores[:, 0] = [100.0, 10.0]
    sim = Simulation(start.replace({PEOPLE: people, STORES: stores}), [PHASES["raid"]], seed=0)
    sim.run(1)
    np.testing.assert_allclose(sim.state.settlers.stores[:, 0], [0.0, 110.0])


def test_droughts_strike_once_a_year_in_summer(start):
    sim = Simulation(start, [PHASES["drought"]], seed=0)
    food = [start.food.sum()]
    for _ in range(2 * SETTLEMENT.year):
        sim.run(1)
        food.append(sim.state.food.sum())
    struck = np.flatnonzero(np.diff(food) < 0)
    summer = SETTLEMENT.year // 4
    np.testing.assert_array_equal(struck, [summer, summer + SETTLEMENT.year])


def test_a_people_too_few_in_a_cell_dies_out_there(start):
    people = np.zeros((2, start.n_cells))
    stores = np.ones((2, start.n_cells))
    area = start.space.measures
    # By density, not amount: cell 1 holds less than the minimum, but its area is under 1.
    assert area[1] < 0.95
    people[0, :2] = [0.5 * SETTLEMENT.minimum * area[0], 0.95 * SETTLEMENT.minimum]
    state = start.replace({PEOPLE: people, STORES: stores})
    sim = Simulation(state, [PHASES["die_out"]], seed=0)
    sim.run(1)
    after = sim.state.settlers
    assert after.people[0, 0] == 0
    assert after.stores[0, 0] == 0
    assert after.people[0, 1] == people[0, 1]
    assert after.stores[0, 1] == 1


def test_land_grazed_bare_wears_out_and_recovers_when_left_alone(start):
    people = np.zeros((2, start.n_cells))
    cell = int(np.flatnonzero(start.land)[0])
    people[0, cell] = 10 * SETTLEMENT.settled * start.space.measures[cell]
    food = start.food.copy()
    food[cell] = 0.0
    state = start.replace({PEOPLE: people, SettlersState.food: food})
    sim = Simulation(state, [PHASES["soil"]], seed=0)
    sim.run(1)
    worn = sim.state.fertility[cell]
    assert worn == pytest.approx(1 - SETTLEMENT.wear)
    assert (np.delete(sim.state.fertility, cell) == 1).all()
    left = sim.state.replace({PEOPLE: np.zeros((2, start.n_cells))})
    sim = Simulation(left, [PHASES["soil"]], seed=0)
    sim.run(1)
    assert sim.state.fertility[cell] == pytest.approx(worn + SETTLEMENT.recovery * (1 - worn))


def test_a_run_keeps_every_amount_non_negative_and_the_peoples_alive(start):
    sim = Simulation(start, list(PHASES.values()), seed=0)
    sim.run(5 * SETTLEMENT.year)
    s = sim.state
    assert (s.food >= 0).all()
    assert (s.settlers.people >= 0).all()
    assert (s.settlers.stores >= 0).all()
    assert (s.settlers.people.sum(axis=1) > 0).all()
    assert (s.settlers.people[:, ~s.land] == 0).all()


def test_the_model_keeps_its_promises(world):
    check_model(SETTLEMENT, world, steps=2 * SETTLEMENT.year, seed=7)
