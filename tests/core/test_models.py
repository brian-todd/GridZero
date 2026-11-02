"""End-to-end models: whole simulations whose outcomes are known, built from every layer."""

import numpy as np
import pytest

from gridzero.core import (
    Adjacency,
    Claim,
    Field,
    Flow,
    NeighborAllocation,
    NeighborChoice,
    Simulation,
    Source,
    Spread,
    State,
    icosphere,
    phase,
)

pytestmark = pytest.mark.usefixtures("each_impl")  # every test on both paths

# Diffusion of heat by finite volumes: the flux between neighbors is proportional to the
# boundary they share and the difference of their densities, over the distance between them.


def finite_volume_rates(space, largest=0.5):
    """Per-edge fractions that diffuse an amount toward uniform density, scaled so no cell
    sends more than ``largest`` of what it holds."""
    graph = space.adjacency()
    edge = graph.edge_data
    unit = edge["boundary"] / (edge["distance"] * space.measures[graph.sources])
    return largest * unit / np.bincount(graph.sources, unit).max()


class HeatState(State):
    heat = Field(np.float64, extensive=True)


def test_heat_diffusing_over_the_planet_is_conserved_and_evens_out_its_density():
    planet = icosphere(2)
    rate = finite_volume_rates(planet)

    @phase(writes=HeatState.heat, hoods={"near": Adjacency()})
    def diffuse(s, ctx):
        return Flow(HeatState.heat, ctx.hoods["near"], rate)

    heat = np.where(np.arange(planet.n_cells) == 0, 1.0, 0.0)
    state = HeatState(planet, {HeatState.heat: heat})
    history = Simulation(state, [diffuse], seed=0).run(
        300,
        observers={
            "total": lambda s: s.total(HeatState.heat),
            "density": lambda s: s.density(HeatState.heat),
        },
    )
    np.testing.assert_allclose(history["total"], 1.0, rtol=1e-12)
    density = history["density"][-1]
    np.testing.assert_allclose(density, 1 / (4 * np.pi), rtol=1e-6)


# Expansion: two kinds of people spread over the land of a planet and grow toward its capacity.
# Explorers leave home ten times as often as settlers.

LEAVE = np.array([[0.02], [0.2]])  # settlers, explorers


class ExpansionState(State):
    pop = Field(np.float64, (2,), extensive=True)
    capacity = Field(np.float64, static=True)
    land = Field(bool, static=True)


def expansion(growth_rate=0.3):
    planet = icosphere(8)
    land = planet.centers[:, 2] > -0.3
    capacity = np.where(land, 10 * planet.measures / planet.measures.mean(), 0.0)
    origin = int(np.argmax(planet.centers[:, 2]))
    pop = np.zeros((2, planet.n_cells))
    pop[:, origin] = 0.5
    state = ExpansionState(
        planet,
        {ExpansionState.pop: pop, ExpansionState.capacity: capacity, ExpansionState.land: land},
    )

    @phase(writes=ExpansionState.pop)
    def grow(s, ctx):
        pop = s[ExpansionState.pop]
        crowding = np.ones(s.n_cells)
        np.divide(
            pop.sum(axis=0),
            s[ExpansionState.capacity],
            out=crowding,
            where=s[ExpansionState.capacity] > 0,
        )
        return Source(ExpansionState.pop, growth_rate * pop * (1 - crowding))

    @phase(writes=ExpansionState.pop, hoods={"near": Adjacency()})
    def disperse(s, ctx):
        room = np.maximum(s[ExpansionState.capacity] - s[ExpansionState.pop].sum(axis=0), 0.0)
        return Spread(
            ExpansionState.pop, ctx.hoods["near"], LEAVE, toward=room, domain=s[ExpansionState.land]
        )

    return Simulation(state, [grow, disperse], seed=0), origin


def test_expansion_never_reaches_the_sea_or_makes_negative_people():
    sim, _ = expansion()
    land = sim.state[ExpansionState.land]
    history = sim.run(
        80,
        observers={
            "lowest": lambda s: s[ExpansionState.pop].min(),
            "at_sea": lambda s: s[ExpansionState.pop][:, ~land].max(),
        },
    )
    assert history["lowest"].min() >= 0
    np.testing.assert_array_equal(history["at_sea"], 0.0)
    assert (
        sim.state[ExpansionState.pop].sum(axis=0) > 1
    ).sum() > 0.9 * land.sum()  # the land fills up


def test_dispersal_alone_conserves_the_people():
    sim, _ = expansion(growth_rate=0.0)
    before = sim.state.total(ExpansionState.pop)
    history = sim.run(50, observers={"total": lambda s: s.total(ExpansionState.pop)})
    np.testing.assert_allclose(history["total"], np.broadcast_to(before, (50, 2)), rtol=1e-12)


def test_explorers_outnumber_settlers_at_the_front_but_not_at_the_origin():
    # Spatial sorting: the fastest dispersers arrive first wherever the population is new.
    sim, origin = expansion()
    sim.run(30)
    planet, pop = sim.space, sim.state[ExpansionState.pop]
    angle = np.arccos(np.clip(planet.centers @ planet.centers[origin], -1, 1))
    occupied = pop.sum(axis=0) > 1e-6
    front = occupied & (angle > np.quantile(angle[occupied], 0.8))
    explorers = pop[1] / pop.sum(axis=0).clip(1e-300)
    assert explorers[origin] < 0.2
    assert explorers[front].min() > 0.99


# Policies: action spaces turn per-cell choices into flows.


class GroupsState(State):
    pop = Field(np.float64, (2,), extensive=True)


def test_a_random_allocation_policy_conserves_and_repeats_with_its_seed():
    planet = icosphere(4)

    @phase(writes=GroupsState.pop, hoods={"near": Adjacency()})
    def wander(s, ctx):
        actions = NeighborAllocation(ctx.hoods["near"], shape=(2,))
        return actions.flow(GroupsState.pop, actions.sample(ctx.rng))

    state = GroupsState(planet, {GroupsState.pop: 1.0})

    def run(seed):
        sim = Simulation(state, [wander], seed=seed)
        return sim.run(20, observers={"pop": lambda s: s[GroupsState.pop]})["pop"]

    first = run(4)
    np.testing.assert_allclose(first.sum(axis=-1), planet.n_cells, rtol=1e-12)
    np.testing.assert_array_equal(run(4), first)
    assert not np.array_equal(run(5), first)


class ForagingState(State):
    pop = Field(np.float64, extensive=True)
    food = Field(np.float64, extensive=True)
    eaten = Field(np.float64, (1,), extensive=True)


@phase(writes=ForagingState.pop, hoods={"near": Adjacency()})
def forage(s, ctx):
    """Half of each cell's people walk to the neighbor with the most food, if it beats home."""
    choice = NeighborChoice(ctx.hoods["near"])
    scores = np.where(choice.valid, s[ForagingState.food][choice.targets], -np.inf)
    return choice.flow(ForagingState.pop, choice.best(scores), rate=0.5)


@phase(writes=(ForagingState.food, ForagingState.eaten))
def eat(s, ctx):
    """Each person asks for 0.1 food. Scarce food is shared in proportion."""
    return Claim(ForagingState.food, 0.1 * s[ForagingState.pop][None], into=ForagingState.eaten)


def foraging_state(planet):
    food = np.random.default_rng(0).uniform(0, 2, planet.n_cells)
    return ForagingState(planet, {ForagingState.pop: 1.0, ForagingState.food: food})


def test_greedy_foragers_gather_on_the_food_peaks():
    planet = icosphere(3)
    state = foraging_state(planet)
    sim = Simulation(state, [forage], seed=0)
    sim.run(20)
    idx = Adjacency().compile(planet).query().idx
    peaks = state[ForagingState.food] > np.max(
        state[ForagingState.food][idx], axis=1, where=idx >= 0, initial=-1
    )
    np.testing.assert_allclose(sim.state.total(ForagingState.pop), planet.n_cells, rtol=1e-12)
    np.testing.assert_allclose(sim.state[ForagingState.pop][peaks].sum(), planet.n_cells, rtol=1e-4)


def test_eating_conserves_food_plus_what_was_eaten():
    planet = icosphere(3)
    state = foraging_state(planet)
    history = Simulation(state, [forage, eat], seed=0).run(
        30,
        observers={
            "stock": lambda s: s.total(ForagingState.food) + s.total(ForagingState.eaten)[0]
        },
    )
    np.testing.assert_allclose(history["stock"], state.total(ForagingState.food), rtol=1e-12)
