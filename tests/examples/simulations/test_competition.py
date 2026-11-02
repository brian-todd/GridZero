import dataclasses

import numpy as np
import pytest

from examples.simulations.competition import (
    Competition,
    CompetitionState,
)
from gridzero.core import LatticeSpace, Simulation
from model_demo import check_model

COMPETITION = Competition()
SPLIT = dataclasses.replace(COMPETITION, split=True)


@pytest.fixture(scope="module")
def world():
    return LatticeSpace((16, 16), tiling="hex", periodic=True)


def run(competition, space, steps):
    """Run from the start of seed 1, recording each group's total every fifth of the run."""
    sim = Simulation(competition.start(space, 1), competition.phases(), seed=1)
    history = sim.run(
        steps, observers={"people": lambda s: s.total(CompetitionState.people)}, every=steps // 5
    )
    return sim.state, history


def test_the_model_keeps_its_promises(world):
    check_model(SPLIT, world, steps=20)


def test_rstar_is_mortality_over_appetite():
    mortality = np.array([[0.1, 0.2], [0.1, 0.05]])
    np.testing.assert_allclose(COMPETITION.rstar(mortality), [[0.2, 0.4], [0.4, 0.2]])


def test_one_niche_favors_the_lowlanders_everywhere(world):
    height = COMPETITION.start(world, 1).height
    rstar = COMPETITION.rstar(COMPETITION.niches(height))
    assert (rstar[0] < rstar[1]).all()


def test_two_niches_cross_at_the_median_height_of_the_land(world):
    height = SPLIT.start(world, 1).height
    land = height > 0
    rstar = SPLIT.rstar(SPLIT.niches(height))
    lower = np.argmin(rstar, axis=0)[land]
    low_land = height[land] < np.median(height[land])
    np.testing.assert_array_equal(lower, np.where(low_land, 0, 1))


def test_the_start_has_land_by_its_share_food_only_there_and_the_lowlanders_rare(world):
    state = COMPETITION.start(world, 1)
    land = state.land
    assert land.mean() == pytest.approx(COMPETITION.land, abs=0.01)
    assert (state.food[~land] == 0).all()
    assert (state.people[:, ~land] == 0).all()
    np.testing.assert_allclose(state.people[0, land], 0.01 * state.people[1, land])


def test_the_seed_sets_the_terrain(world):
    assert not np.array_equal(
        COMPETITION.start(world, 1).height, COMPETITION.start(world, 2).height
    )


def test_the_group_with_the_lower_rstar_drives_out_the_other(world):
    state, history = run(COMPETITION, world, steps=500)
    assert state.people[1].sum() < 1e-6 * state.people[0].sum()
    assert history["people"][0, 1] > history["people"][-1, 1]


def test_the_winner_holds_the_food_at_its_rstar(world):
    state, _ = run(COMPETITION, world, steps=500)
    meal = COMPETITION.meal(state)[state.land]
    assert meal.mean() == pytest.approx(0.2, abs=0.002)


def test_with_two_niches_each_group_holds_the_land_where_its_rstar_is_lower(world):
    state, _ = run(SPLIT, world, steps=1000)
    land = state.land
    expected = np.argmin(SPLIT.rstar(state.mortality), axis=0)[land]
    holds = np.argmax(state.people, axis=0)[land]
    assert (holds == expected).mean() > 0.9
    assert (state.people[:, land].sum(axis=1) > 0.3 * state.people.sum()).all()
