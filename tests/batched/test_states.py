"""A state's fields as arrays for JAX, and back."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from demo import DemoState, WorldState
from gridzero import batched
from gridzero.core import Field, LatticeSpace, State, icosphere


class WideState(State):
    """Fields of dtypes JAX narrows."""

    big = Field(np.int64)
    small = Field(np.uint8)
    plain = Field(np.uint64)
    real = Field(np.float64, (2,), extensive=True)
    half = Field(np.float16)
    flag = Field(bool)


def demo(space, rng, step=0):
    return DemoState(
        space,
        {
            DemoState.alive: rng.random(space.n_cells) < 0.5,
            DemoState.pop: rng.random((2, space.n_cells)),
            DemoState.terrain: rng.integers(0, 200, space.n_cells),
        },
        step=step,
    )


@pytest.mark.parametrize("space", [LatticeSpace((4, 3)), icosphere(1)], ids=["squares", "planet"])
def test_fields_hold_a_states_arrays_by_field_in_jaxs_dtypes(space, rng):
    state = demo(space, rng, step=7)
    values = batched.fields(state)
    assert values.kind is DemoState
    assert values.declared == tuple(state)
    assert int(values.step) == 7
    assert values[DemoState.alive].dtype == np.bool_
    assert values[DemoState.pop].dtype == np.float32
    assert values[DemoState.terrain].dtype == np.uint32
    np.testing.assert_array_equal(values[DemoState.alive], state.alive)
    np.testing.assert_array_equal(values[DemoState.terrain], state.terrain)
    np.testing.assert_array_equal(values[DemoState.pop], state.pop.astype(np.float32))


def test_a_state_comes_back_equal_but_for_float32(rng):
    space = LatticeSpace((4, 3))
    state = demo(space, rng, step=3)
    back = batched.state(batched.fields(state), space)
    assert type(back) is DemoState
    assert back.step == 3
    np.testing.assert_array_equal(back.alive, state.alive)
    np.testing.assert_array_equal(back.terrain, state.terrain)
    assert back.terrain.dtype == np.uint8
    assert back.pop.dtype == np.float64
    np.testing.assert_array_equal(back.pop, state.pop.astype(np.float32))


def test_nested_fields_are_fields_too(rng):
    space = LatticeSpace((3, 3))
    people = rng.random((2, 9))
    state = WorldState(space, {WorldState.settlers.people: people})
    values = batched.fields(state)
    np.testing.assert_array_equal(values[WorldState.settlers.people], people.astype(np.float32))
    back = batched.state(values, space)
    np.testing.assert_array_equal(back.settlers.people, people.astype(np.float32))


def test_integers_are_kept_exactly_or_refused():
    space = LatticeSpace((2, 2), periodic=False)
    most = np.iinfo(np.int32)
    ok = WideState(space, {WideState.big: [most.min, -1, 0, most.max], WideState.small: 255})
    values = batched.fields(ok)
    assert values[WideState.big].dtype == np.int32
    assert values[WideState.big].tolist() == [most.min, -1, 0, most.max]
    assert values[WideState.small].dtype == np.uint32
    assert values[WideState.half].dtype == np.float32
    assert batched.state(values, space).big.tolist() == [most.min, -1, 0, most.max]
    for field, value in (
        (WideState.big, most.max + 1),
        (WideState.big, most.min - 1),
        (WideState.plain, 2**32),
    ):
        with pytest.raises(ValueError, match="32 bits"):
            batched.fields(WideState(space, {field: value}))


def test_floats_too_large_for_float32_are_refused_and_infinities_kept():
    space = LatticeSpace((2, 2), periodic=False)
    with pytest.raises(ValueError, match="float32"):
        batched.fields(WideState(space, {WideState.real: 1e39}))
    values = batched.fields(WideState(space, {WideState.real: [np.inf, -np.inf, np.nan, 1e38]}))
    assert np.isinf(values[WideState.real][0, :2]).all()
    assert np.isnan(values[WideState.real][0, 2])


def test_replace_makes_new_fields_in_the_fields_dtypes(rng):
    space = LatticeSpace((4, 3))
    values = batched.fields(demo(space, rng))
    new = values.replace({DemoState.pop: np.ones((2, 12)), DemoState.alive: np.zeros(12, bool)})
    assert new is not values
    assert new[DemoState.pop].dtype == np.float32
    assert float(new[DemoState.pop].sum()) == 24
    assert not new[DemoState.alive].any()
    np.testing.assert_array_equal(new[DemoState.terrain], values[DemoState.terrain])
    assert float(values[DemoState.pop].sum()) != 24
    ones = values.replace({DemoState.alive: np.ones(12, np.int32), DemoState.terrain: np.ones(12)})
    assert ones[DemoState.alive].dtype == np.bool_
    assert ones[DemoState.terrain].dtype == np.uint32
    assert int(new.at(5).step) == 5
    assert int(new.step) == 0


def test_replace_refuses_other_fields_and_other_shapes(rng):
    values = batched.fields(demo(LatticeSpace((4, 3)), rng))
    with pytest.raises(KeyError, match="declares no field"):
        values.replace({WideState.big: np.zeros(12)})
    with pytest.raises(KeyError, match="declares no field"):
        values[WideState.big]
    with pytest.raises(ValueError, match="shape"):
        values.replace({DemoState.pop: np.ones(12)})


def test_fields_stack_into_a_batch_that_vmap_takes_apart(rng):
    space = LatticeSpace((4, 3))
    states = [demo(space, rng, step=i) for i in range(5)]
    many = jax.tree.map(lambda *xs: np.stack(xs), *(batched.fields(s) for s in states))
    assert many[DemoState.pop].shape == (5, 2, 12)

    def total(values):
        return values[DemoState.pop].sum() + values.step

    totals = jax.jit(jax.vmap(total))(many)
    want = [s.pop.astype(np.float32).sum() + s.step for s in states]
    np.testing.assert_allclose(totals, want, rtol=1e-6)
    doubled = jax.vmap(lambda v: v.replace({DemoState.pop: 2 * v[DemoState.pop]}))(many)
    one = jax.tree.map(lambda x: x[3], doubled)
    back = batched.state(one, space)
    np.testing.assert_allclose(back.pop, 2 * states[3].pop, rtol=1e-6)
    assert back.step == 3
    assert isinstance(doubled[DemoState.pop], jnp.ndarray)


def test_fields_of_something_else_raise():
    with pytest.raises(TypeError, match=r"core\.State"):
        batched.fields(np.zeros(3))  # ty: ignore[invalid-argument-type]  # not a state, on purpose
