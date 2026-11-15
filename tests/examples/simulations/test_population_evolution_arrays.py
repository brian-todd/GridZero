"""The planet's evolution as arrays against the model: the same run, to float32's precision."""

import dataclasses

import jax
import numpy as np
import pytest

from batched_demo import check_array_model
from examples.simulations.population_evolution import Evolution, EvolutionState, spectrum
from examples.simulations.population_evolution import arrays as in_arrays
from examples.simulations.population_evolution.arrays import EvolutionArrays
from gridzero import batched
from gridzero.core import LatticeSpace, Simulation, icosphere


@pytest.fixture(scope="module")
def globe():
    return icosphere(4)  # 162 cells


@pytest.mark.parametrize(
    "model",
    [Evolution(), Evolution(warming=0.002, warming_from=10), Evolution(competition=float("inf"))],
    ids=["default", "warming", "one niche"],
)
def test_the_phases_in_arrays_follow_the_model(globe, model):
    phases = EvolutionArrays(model).phases()
    assert [each.name for each in phases] == [each.name for each in model.phases()]
    check_array_model(model, phases, globe, seed=2, steps=40, rtol=1e-4)


def test_they_run_on_a_planets_map_too():
    model = Evolution()
    space = LatticeSpace((16, 8), periodic=(True, False))
    check_array_model(model, EvolutionArrays(model).phases(), space, seed=1, steps=25, rtol=1e-4)


def test_a_batch_of_planets_is_each_planet_alone(globe):
    model = Evolution()
    sim = batched.Simulation(globe, EvolutionArrays(model).phases())
    starts = [batched.fields(model.start(globe, seed)) for seed in range(3)]
    many = jax.tree.map(lambda *xs: np.stack(xs), *starts)
    key = batched.root(0)
    ends, spectra = jax.jit(jax.vmap(lambda f: sim.run(f, key, 30, in_arrays.spectrum)))(many)
    assert spectra.shape == (3, 30, 32)
    for seed in range(3):
        reference = Simulation(model.start(globe, seed), model.phases(), seed=seed)
        reference.run(30)
        want = spectrum(reference.state)
        np.testing.assert_allclose(spectra[seed, -1], want, rtol=2e-3, atol=2e-3 * want.max())
        life = np.asarray(ends[EvolutionState.life][seed])
        assert life.min() >= 0
        assert not life[:, reference.state.height <= 0].any()  # no one at sea


def test_arrays_of_something_else_raise():
    with pytest.raises(TypeError, match="Evolution"):
        EvolutionArrays(dataclasses.asdict(Evolution()))  # ty: ignore[invalid-argument-type]  # not a model
