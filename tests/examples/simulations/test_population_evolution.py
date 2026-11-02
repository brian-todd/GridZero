import dataclasses

import numpy as np
import pytest
from scipy.stats import spearmanr

from examples.simulations.population_evolution import (
    BINS,
    Evolution,
    EvolutionState,
    clusters,
    species,
    spectrum,
)
from gridzero.core import LatticeSpace, Simulation, icosphere
from gridzero.worlds import generate_planet
from model_demo import check_model

EVOLUTION = Evolution()


@pytest.fixture(scope="module")
def globe():
    return icosphere(4)  # 162 cells


def one_place(model, temperature, steps):
    """Run a model in one land cell of the given temperature, from founders of its best trait."""
    space = LatticeSpace((1,), periodic=False)
    life = np.zeros((BINS, 1))
    life[np.argmin(np.abs(model.traits - temperature)), 0] = 10.0
    state = EvolutionState(
        space,
        {
            EvolutionState.height: [0.5],
            EvolutionState.temperature: [temperature],
            EvolutionState.life: life,
        },
    )
    sim = Simulation(state, model.phases(), seed=0)
    sim.run(steps)
    return sim.state.life[:, 0]


def test_the_model_keeps_its_promises(globe):
    check_model(EVOLUTION, globe, steps=20)


def test_the_model_runs_on_a_planets_map():
    check_model(EVOLUTION, LatticeSpace((16, 8), periodic=(True, False)), steps=5)


def test_births_keep_their_parents_trait_but_for_a_share_into_each_neighbor():
    m = dataclasses.replace(EVOLUTION, mutation=0.1).heredity()
    np.testing.assert_allclose(m.sum(axis=0), 1.0, rtol=0, atol=1e-15)
    assert m[5, 5] == pytest.approx(0.8)
    assert m[4, 5] == m[6, 5] == pytest.approx(0.1)
    assert m[7, 5] == m[3, 5] == 0.0
    assert m[0, 0] == m[-1, -1] == pytest.approx(0.9)  # at the ends they stay


def test_crowding_is_quartic_in_the_distance_between_traits():
    a = EVOLUTION.crowding()
    x = EVOLUTION.traits
    assert np.all(np.diag(a) == 1.0)
    assert a[3, 7] == pytest.approx(np.exp(-(((x[7] - x[3]) / 0.15) ** 4)))
    np.testing.assert_array_equal(a, a.T)


def test_crowding_has_no_subnormal_values():
    a = EVOLUTION.crowding()
    assert not np.any((a > 0) & (a < np.finfo(np.float64).tiny))
    assert np.count_nonzero(a == 0) > 0  # the default's tail reaches them


def test_with_infinite_competition_every_trait_crowds_every_other_alike():
    assert np.all(dataclasses.replace(EVOLUTION, competition=np.inf).crowding() == 1.0)


def test_births_fall_off_with_the_distance_from_the_best_temperature():
    x = EVOLUTION.traits
    g = EVOLUTION.fitness(np.array([x[10], x[10] + 0.3]))
    assert g[10, 0] == 1.0
    assert g[10, 1] == pytest.approx(np.exp(-0.5))
    assert np.argmax(g[:, 0]) == 10


def test_the_start_has_the_founders_in_one_warm_land_cell_in_the_trait_nearest_it(globe):
    state = EVOLUTION.start(globe, 3)
    planet = generate_planet(globe, seed=3, land=EVOLUTION.land)
    np.testing.assert_array_equal(state.height, planet.elevation)
    np.testing.assert_array_equal(state.temperature, planet.temperature)
    bins, cells = np.nonzero(state.life)
    assert bins.size == 1
    (b,), (c,) = bins, cells
    assert state.land[c]
    assert state.temperature[c] >= np.quantile(state.temperature[state.land], 0.9)
    assert b == np.argmin(np.abs(EVOLUTION.traits - state.temperature[c]))
    assert state.life[b, c] == pytest.approx(EVOLUTION.founders * globe.measures[c])


def test_the_founders_land_in_the_warmest_tenth_of_the_land_whatever_the_seed(globe):
    for seed in range(10):
        state = EVOLUTION.start(globe, seed)
        (landing,) = np.flatnonzero(state.life.sum(axis=0))
        warmest = np.quantile(state.temperature[state.land], 0.9)
        assert state.land[landing]
        assert state.temperature[landing] >= warmest


def test_the_seed_sets_the_planet_and_the_landing_cell(globe):
    a, b = EVOLUTION.start(globe, 1), EVOLUTION.start(globe, 2)
    assert not np.array_equal(a.height, b.height)
    landing = [np.flatnonzero(s.life.sum(axis=0)) for s in (a, b, EVOLUTION.start(globe, 1))]
    np.testing.assert_array_equal(landing[0], landing[2])


def test_the_founders_fill_their_landmass_and_go_nowhere_else(globe):
    planet = generate_planet(globe, seed=3, land=EVOLUTION.land)
    start = EVOLUTION.start(globe, 3)
    (landing,) = np.flatnonzero(start.life.sum(axis=0))
    landmass = planet.landmasses == planet.landmasses[landing]
    sim = Simulation(start, EVOLUTION.phases(), seed=3)
    occupied = []
    for _ in range(7):
        sim.run(100)
        lived = sim.state.life.sum(axis=0) > 0
        assert not np.any(lived & ~landmass)
        occupied.append(np.count_nonzero(lived))
    assert occupied == sorted(occupied)
    assert occupied[0] < landmass.sum() == occupied[-1]


def test_a_trait_too_rare_in_a_cell_dies_out_there():
    model = dataclasses.replace(EVOLUTION, minimum=0.01)
    life = one_place(model, 0.5, 2000)
    assert np.all((life == 0) | (life >= 0.01))
    assert np.count_nonzero(life == 0) > 0


# Mutation against selection, in one place: crowding the same for every trait, so it scales
# the whole distribution and leaves its shape to the linear part of a step.
BALANCE = dataclasses.replace(EVOLUTION, competition=np.inf, minimum=0.0)


def settled_shape(model, temperature):
    """Return the dominant eigenvector of a step's linear part, built here from the rules."""
    m = np.zeros((BINS, BINS))
    for j in range(BINS):
        m[j, j] = 1 - 2 * model.mutation
        for i in (j - 1, j + 1):
            m[min(max(i, 0), BINS - 1), j] += model.mutation
    x = np.linspace(model.low, model.high, BINS)
    g = np.exp(-0.5 * ((x - temperature) / model.tolerance) ** 2)
    values, vectors = np.linalg.eig(m @ np.diag(model.births * g) - model.deaths * np.eye(BINS))
    top = np.abs(np.real(vectors[:, np.argmax(np.real(values))]))
    return top / top.sum()


def test_in_one_place_the_traits_settle_in_the_dominant_eigenvectors_shape():
    best = BALANCE.traits[15]  # on a bin: between two, the two settle against each other slowly
    life = one_place(BALANCE, best, 6000)
    np.testing.assert_allclose(life / life.sum(), settled_shape(BALANCE, best), rtol=0, atol=1e-9)


def test_in_one_place_the_traits_settle_around_the_best_one_as_kimura_says():
    model = dataclasses.replace(BALANCE, low=0.3, high=0.7, tolerance=0.1, mutation=0.05)
    life = one_place(model, 0.5, 6000)
    shape = life / life.sum()
    x = model.traits
    mean = shape @ x
    variance = shape @ (x - mean) ** 2
    # D = mutation * h**2 * births and s = births / (2 tolerance**2), so sqrt(D / s) is:
    kimura = (x[1] - x[0]) * model.tolerance * np.sqrt(2 * model.mutation)
    assert mean == pytest.approx(0.5, abs=1e-12)
    assert variance == pytest.approx(kimura, rel=0.02)


def test_clusters_are_runs_of_bins_above_a_share_of_the_fullest():
    counts = np.zeros(BINS)
    counts[[0, 1, 5, 6, 7, 31]] = [100, 50, 2, 100, 1.5, 3]
    counts[3] = 0.5  # below 1% of 100: a gap
    assert clusters(counts) == [(0, 1), (5, 7), (31, 31)]
    assert clusters(counts, cut=0.025) == [(0, 1), (6, 6), (31, 31)]
    assert clusters(counts, cut=0.02) == [(0, 1), (6, 6), (31, 31)]  # 2 of 100: not above
    assert clusters(np.zeros(BINS)) == []
    assert clusters(np.ones(BINS)) == [(0, BINS - 1)]


def test_the_spectrum_sums_each_trait_over_the_cells(globe):
    state = EVOLUTION.start(globe, 3)
    np.testing.assert_array_equal(spectrum(state), state.life.sum(axis=1))


def test_the_mean_trait_is_each_cells_mean_best_temperature_and_nan_where_empty():
    space = LatticeSpace((3,), periodic=False)
    life = np.zeros((BINS, 3))
    life[[2, 6], 0] = [1.0, 3.0]
    life[4, 1] = 2.0
    state = EvolutionState(
        space,
        {
            EvolutionState.height: [0.5, 0.5, 0.5],
            EvolutionState.temperature: [0.1, 0.2, 0.3],
            EvolutionState.life: life,
        },
    )
    x = EVOLUTION.traits
    mean = EVOLUTION.mean_trait(state)
    assert mean[0] == pytest.approx((x[2] + 3 * x[6]) / 4)
    assert mean[1] == pytest.approx(x[4])
    assert np.isnan(mean[2])


def test_the_gradient_is_the_slope_of_the_mean_trait_against_the_temperature():
    space = LatticeSpace((3,), periodic=False)
    life = np.zeros((BINS, 3))
    life[[3, 5, 7], [0, 1, 2]] = 1.0
    x = EVOLUTION.traits
    temperature = [x[3], x[3] + 4 * (x[1] - x[0]), x[3] + 8 * (x[1] - x[0])]  # twice as steep
    state = EvolutionState(
        space,
        {
            EvolutionState.height: [0.5, 0.5, 0.5],
            EvolutionState.temperature: temperature,
            EvolutionState.life: life,
        },
    )
    assert EVOLUTION.gradient(state) == pytest.approx(0.5)


def settled_gradient(dispersal, globe):
    """The gradient after 7,500 steps from seed 3, all traits crowding alike: settled by then."""
    model = dataclasses.replace(EVOLUTION, competition=np.inf, dispersal=dispersal)
    sim = Simulation(model.start(globe, 3), model.phases(), seed=3)
    sim.run(7500)
    return model.gradient(sim.state)


def test_gene_flow_flattens_the_gradient_of_traits(globe):
    """Kirkpatrick and Barton: the more individuals move, the less each cell's traits follow
    its own temperature. Measured: 0.895 at the default dispersal and 0.625 at ten times it."""
    local, mixed = settled_gradient(0.05, globe), settled_gradient(0.5, globe)
    assert 0.8 < local <= 1.0
    assert 0.0 < mixed < local - 0.2


def test_crowding_settles_each_cell_at_a_density_whatever_its_area():
    """With every trait crowding alike, the step's linear part grows the settled shape by its
    dominant eigenvalue, and crowding takes ``births * density / capacity`` off it: they
    balance at a density of ``capacity * eigenvalue / births`` per unit area."""
    globe = icosphere(2)  # 42 cells of areas about 0.3, unequal
    model = dataclasses.replace(BALANCE, dispersal=0.0)
    best = model.traits[15]
    life = np.zeros((BINS, globe.n_cells))
    life[15] = globe.measures
    state = EvolutionState(
        globe,
        {
            EvolutionState.height: np.full(globe.n_cells, 0.5),
            EvolutionState.temperature: np.full(globe.n_cells, best),
            EvolutionState.life: life,
        },
    )
    sim = Simulation(state, model.phases(), seed=0)
    sim.run(6000)
    x = model.traits
    g = np.exp(-0.5 * ((x - best) / model.tolerance) ** 2)
    m = model.heredity()
    growth = np.max(np.real(np.linalg.eigvals(m @ np.diag(model.births * g)))) - model.deaths
    density = sim.state.life.sum(axis=0) / globe.measures
    assert np.ptp(globe.measures) > 0.01
    np.testing.assert_allclose(density, model.capacity * growth / model.births, rtol=1e-9)


def test_a_cells_species_is_the_cluster_holding_most_of_it_and_none_where_empty():
    space = LatticeSpace((4,), periodic=False)
    life = np.zeros((BINS, 4))
    life[[2, 3], 0] = [5.0, 1.0]  # species 0 alone
    life[[3, 10], 1] = [2.0, 3.0]  # species 1 holds more
    life[[2, 10], 2] = [3.0, 3.0]  # a tie goes to the colder
    state = EvolutionState(
        space,
        {
            EvolutionState.height: np.full(4, 0.5),
            EvolutionState.temperature: np.full(4, 0.5),
            EvolutionState.life: life,
        },
    )
    assert clusters(spectrum(state)) == [(2, 3), (10, 10)]
    np.testing.assert_array_equal(species(state), [0, 1, 0, -1])


def settled(model, globe, steps=7500):
    """The state after ``steps`` from seed 3: species have settled by 7,500 on 162 cells."""
    sim = Simulation(model.start(globe, 3), model.phases(), seed=3)
    sim.run(steps)
    return sim.state


@pytest.fixture(scope="module")
def evolved(globe):
    return settled(EVOLUTION, globe)


def centers(model, state):
    """Each species' mean trait."""
    counts, x = spectrum(state), model.traits
    return np.array(
        [counts[a : b + 1] @ x[a : b + 1] / counts[a : b + 1].sum() for a, b in clusters(counts)]
    )


def test_competition_between_nearby_traits_splits_them_into_species(evolved):
    """Measured: 6 species, at -0.11, 0.09, 0.28, 0.48, 0.68 and 0.96, about 0.19 apart."""
    found = centers(EVOLUTION, evolved)
    assert len(found) >= 4
    assert np.all(np.diff(found)[:4] == pytest.approx(0.19, abs=0.03))


def test_each_species_holds_its_own_band_of_climate(evolved):
    """Measured: the cells' main species rank with their temperature at 0.95."""
    main = species(evolved)
    lived = main >= 0
    assert spearmanr(evolved.temperature[lived], main[lived]).statistic > 0.9
    held = np.unique(main[lived])
    medians = [np.median(evolved.temperature[main == i]) for i in held]
    assert len(held) >= 4
    assert np.all(np.diff(medians) > 0)


def test_with_all_traits_crowding_alike_they_stay_one_gradient(globe):
    model = dataclasses.replace(EVOLUTION, competition=np.inf)
    assert len(clusters(spectrum(settled(model, globe)))) == 1


@pytest.mark.slow
def test_species_do_not_come_from_the_ends_of_the_trait_range(globe, evolved):
    """Polechova and Barton: clusters can come from the ends of a range. With the range three
    bins colder or warmer (the bins in the same places), the species from the coldest to the
    middle settle at the same traits."""
    h = EVOLUTION.traits[1] - EVOLUTION.traits[0]
    here = centers(EVOLUTION, evolved)[:5]
    for shift in (-3 * h, 3 * h):
        model = dataclasses.replace(
            EVOLUTION, low=EVOLUTION.low + shift, high=EVOLUTION.high + shift
        )
        np.testing.assert_allclose(centers(model, settled(model, globe))[:5], here, atol=h / 4)


# A warming climate, in one place: theory with continuous traits, crowding the same for every
# trait. Mutation spreads traits as fast as D = mutation * h**2 * births, and selection near
# the best trait is as strong as s = births / (2 tolerance**2).
H = BALANCE.traits[1] - BALANCE.traits[0]
D = BALANCE.mutation * H**2 * BALANCE.births
S = BALANCE.births / (2 * BALANCE.tolerance**2)
CRITICAL = 2 * np.sqrt(D * (BALANCE.births - BALANCE.deaths - np.sqrt(D * S)))


def warmed(model, speed, moving):
    """Settle in one place at the best temperature of bin 5, then warm for ``moving`` steps.

    Returns the lag of the mean trait behind the temperature at each step, and the totals.
    """
    model = dataclasses.replace(model, warming=speed, warming_from=3000)
    space = LatticeSpace((1,), periodic=False)
    life = np.zeros((BINS, 1))
    life[5, 0] = 10.0
    state = EvolutionState(
        space,
        {
            EvolutionState.height: [0.5],
            EvolutionState.temperature: [model.traits[5]],
            EvolutionState.life: life,
        },
    )
    sim = Simulation(state, model.phases(), seed=0)
    history = sim.run(
        3000 + moving,
        observers={
            "lag": lambda s: s.temperature[0] - model.mean_trait(s)[0],
            "total": lambda s: s.life.sum(),
        },
    )
    return history["lag"], history["total"]


def test_warming_raises_the_temperature_from_its_step_on():
    model = dataclasses.replace(EVOLUTION, warming=0.01, warming_from=3)
    state = EVOLUTION.start(icosphere(2), 0)
    sim = Simulation(state, model.phases(), seed=0)
    sim.run(5)
    np.testing.assert_allclose(sim.state.temperature, state.temperature + 0.02, rtol=0, atol=1e-15)


def test_traits_follow_a_slowly_warming_climate_at_the_lag_theory_gives():
    """The lag ``v / (2 sqrt(D s))`` grows with the speed as theory says. Measured at 1/16 and
    1/8 of the critical speed: 0.0267 and 0.0487, against 0.0228 and 0.0457. The bins add
    about 0.004, a twelfth of one, which continuous traits don't have."""
    lags = []
    for speed in (CRITICAL / 16, CRITICAL / 8):
        lag, _ = warmed(BALANCE, speed, moving=4000)
        # The lag ripples by about 2% as the optimum crosses each bin: compare means.
        last, before = lag[-1000:].mean(), lag[-2000:-1000].mean()
        assert last == pytest.approx(before, rel=0.01)  # settled
        lags.append(last)
    per_speed = 1 / (2 * np.sqrt(D * S))
    assert lags[1] == pytest.approx(CRITICAL / 8 * per_speed, rel=0.1)
    assert lags[1] - lags[0] == pytest.approx(CRITICAL / 16 * per_speed, rel=0.1)


def test_a_population_keeps_up_below_the_critical_speed_and_declines_above_it():
    """Theory: growth falls below 0 past ``2 sqrt(D (births - deaths - sqrt(D s)))``."""
    _, slow = warmed(BALANCE, CRITICAL / 2, moving=int(0.9 / (CRITICAL / 2)))
    _, fast = warmed(BALANCE, 2 * CRITICAL, moving=int(0.9 / (2 * CRITICAL)))
    settled = slow[2999]
    assert slow[-1] > 0.5 * settled
    assert fast[-1] < 0.1 * settled


def test_the_threshold_slows_adaptation_to_a_warming_climate():
    """The traits ahead of the population are cut while rare.
    Measured at a quarter of the critical speed: a lag of 0.12 with ``minimum`` 0.01 against
    0.086 without."""
    cut, _ = warmed(dataclasses.replace(BALANCE, minimum=0.01), CRITICAL / 4, moving=3000)
    free, _ = warmed(BALANCE, CRITICAL / 4, moving=3000)
    assert cut[-1000:].mean() > 1.2 * free[-1000:].mean()


def test_warming_is_in_the_schedule_only_when_asked():
    names = [p.name for p in EVOLUTION.phases()]
    warmed = [p.name for p in dataclasses.replace(EVOLUTION, warming=0.001).phases()]
    assert names == ["live", "disperse", "die_out"]
    assert warmed == ["warm", *names]
