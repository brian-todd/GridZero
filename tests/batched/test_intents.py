"""Intents as arrays against `core.resolve`: the same fields, to float32's precision."""

import jax
import numpy as np
import pytest

from gridzero import batched, core
from gridzero.core import Adjacency, Ball, Field, LatticeSpace, NeighborOperator, State, icosphere


class LandState(State):
    """Amounts, a density, a count, a flag and a static amount."""

    food = Field(np.float64, extensive=True)
    people = Field(np.float64, (2,), extensive=True)
    packs = Field(np.float64, (3, 2), extensive=True)
    mood = Field(np.float64)
    count = Field(np.int64, extensive=True)
    alive = Field(bool)
    fixed = Field(np.float64, extensive=True, static=True)


def directed(n_cells, size, seed, weighted=False):
    """An operator whose edges go one way, with rows of every length, some empty."""
    rng = np.random.default_rng(seed)
    idx = np.full((n_cells, size), -1)
    for cell in range(n_cells):
        degree = int(rng.integers(0, size + 1))
        idx[cell, :degree] = np.sort(rng.choice(n_cells, size=degree, replace=False))
    weight = rng.random(idx.shape) * (idx >= 0) if weighted else None
    return NeighborOperator.from_table(idx, weight)


SPACES = {
    "hexes": lambda: LatticeSpace((5, 4), tiling="hex"),
    "planet": lambda: icosphere(2),
}
HOODS = {
    "adjacent": lambda space: Adjacency().compile(space),
    "by distance": lambda space: Ball(1.5 if isinstance(space, LatticeSpace) else 0.7).compile(
        space
    ),
    "one way": lambda space: directed(space.n_cells, 4, seed=3),
    "one way, weighted": lambda space: directed(space.n_cells, 4, seed=4, weighted=True),
}


@pytest.fixture(params=list(SPACES))
def space(request):
    return SPACES[request.param]()


@pytest.fixture(params=list(HOODS))
def op(request, space):
    return HOODS[request.param](space)


def land(space, rng) -> LandState:
    n = space.n_cells
    return LandState(
        space,
        {
            LandState.food: rng.random(n) * (rng.random(n) < 0.8),  # some cells hold nothing
            LandState.people: rng.random((2, n)) * 3,
            LandState.packs: rng.random((3, 2, n)),
            LandState.mood: rng.normal(size=n),
            LandState.count: rng.integers(0, 9, n),
            LandState.alive: rng.random(n) < 0.5,
            LandState.fixed: rng.random(n),
        },
    )


def same(found, wanted: State, rtol=2e-5) -> None:
    """Assert that fields in arrays are a state's: exactly, or to float32 for floats."""
    for field in wanted:
        got, want = np.asarray(found[field]), wanted[field]
        assert got.shape == want.shape, field.name
        if want.dtype.kind == "f":
            scale = max(1.0, float(np.abs(want).max()))
            np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * scale, err_msg=field.name)
        else:
            np.testing.assert_array_equal(got, want, err_msg=field.name)


def twin(intent, tables):
    """Return a `core` intent as its twin in arrays, its operator as tables."""
    kind = getattr(batched, type(intent).__name__)
    values = {name: tables if name == "op" else getattr(intent, name) for name in kind._fields}
    return kind(**values)


def check(state, intents, op=None, writes=None):
    """Resolve intents both ways, compiled, and compare every field; return both results."""
    tables = None if op is None else batched.operator(op)
    twins = [twin(intent, tables) for intent in intents]
    wanted = core.resolve(state, intents, writes=writes)
    found = jax.jit(lambda fields: batched.resolve(fields, twins, writes=writes))(
        batched.fields(state)
    )
    same(found, wanted)
    return found, wanted


def test_assign_writes_values_where_asked(space, rng):
    state = land(space, rng)
    n = space.n_cells
    check(state, [core.Assign(LandState.alive, ~state.alive)])
    check(state, [core.Assign(LandState.mood, 0.25)])
    where = rng.random(n) < 0.5
    check(
        state,
        [
            core.Assign(LandState.people, rng.random((2, n)), where=where),
            core.Assign(LandState.count, rng.integers(0, 5, n), where=~where),
        ],
    )


def test_a_source_adds_and_removes_no_more_than_there_is(space, rng):
    state = land(space, rng)
    amount = rng.normal(size=(2, space.n_cells)) * 4  # many removals exceed the amounts
    found, _ = check(state, [core.Source(LandState.people, amount)])
    assert float(found[LandState.people].min()) >= 0
    check(state, [core.Source(LandState.food, 0.5)])
    check(state, [core.Source(LandState.people, np.array([[1.0], [-10.0]]))])


def test_a_spread_moves_amounts_and_conserves_them(space, op, rng):
    state = land(space, rng)
    n = space.n_cells
    for leave, toward, domain in (
        (0.3, None, None),
        (rng.random((2, n)) * 1.4 - 0.2, rng.random(n) - 0.2, None),  # clipped at both ends
        (rng.random(n), rng.random((2, n)), rng.random(n) < 0.6),
        (1.0, None, rng.random(n) < 0.3),
    ):
        found, _ = check(state, [core.Spread(LandState.people, op, leave, toward, domain)], op)
        total = np.asarray(found[LandState.people]).sum(axis=-1)
        np.testing.assert_allclose(total, state.people.sum(axis=-1), rtol=1e-5)
        if domain is not None:  # nothing entered a cell outside the domain
            outside = ~domain
            assert (
                np.asarray(found[LandState.people])[:, outside] <= state.people[:, outside] + 1e-6
            ).all()


def test_a_flow_moves_amounts_edge_by_edge_and_conserves_them(space, op, rng):
    state = land(space, rng)
    n = space.n_cells
    for rate, domain in (
        (0.1, None),
        (rng.random(op.n_edges) - 0.2, None),  # negative rates, and cells that send too much
        (rng.random((2, op.n_edges)) * 0.4, rng.random(n) < 0.5),
        (rng.random((2, op.n_edges)), rng.random(n) < 0.5),
    ):
        found, _ = check(state, [core.Flow(LandState.people, op, rate, domain)], op)
        total = np.asarray(found[LandState.people]).sum(axis=-1)
        np.testing.assert_allclose(total, state.people.sum(axis=-1), rtol=1e-5)
        assert float(found[LandState.people].min()) >= 0


def test_a_claim_rations_a_cell_among_its_claimants(space, rng):
    state = land(space, rng)
    n = space.n_cells
    asked = rng.random((2, n)) * 1.5 - 0.2  # together often more than the food, some negative
    found, _ = check(state, [core.Claim(LandState.food, asked, into=LandState.people)])
    both = np.asarray(found[LandState.food]).sum() + np.asarray(found[LandState.people]).sum()
    assert both == pytest.approx(state.food.sum() + state.people.sum(), rel=1e-5)
    assert float(found[LandState.food].min()) >= 0
    check(
        state,
        [core.Claim(LandState.food, asked, into=LandState.people, efficiency=[[0.5], [2.0]])],
    )
    worth = rng.normal(size=(2, n))  # a negative efficiency counts as 0
    check(state, [core.Claim(LandState.food, asked, into=LandState.people, efficiency=worth)])
    # Claimants of a field with components: three packs claim each group's people.
    check(state, [core.Claim(LandState.people, rng.random((3, 2, n)) * 2, into=LandState.packs)])


def test_takers_of_one_field_share_each_cells_amount(space, rng):
    state = land(space, rng)
    n = space.n_cells
    op = Adjacency().compile(space)
    intents = [
        core.Spread(LandState.people, op, 0.7, domain=rng.random(n) < 0.7),
        core.Source(LandState.people, -state.people * 0.6),  # with the spread: more than all
        core.Flow(LandState.people, op, rng.random(op.n_edges) * 0.3),
        core.Claim(LandState.people, rng.random((3, 2, n)), into=LandState.packs),
        core.Source(LandState.packs, rng.normal(size=(3, 2, n))),
        core.Claim(LandState.food, rng.random((2, n)), into=LandState.people, efficiency=0.5),
        core.Source(LandState.food, 0.05),
        core.Assign(LandState.alive, True),
    ]
    found, _ = check(state, intents, op)
    for field in (LandState.people, LandState.packs, LandState.food):
        assert float(found[field].min()) >= 0


def test_nothing_to_apply_returns_the_fields_themselves(space, rng):
    fields = batched.fields(land(space, rng))
    assert batched.resolve(fields, None) is fields
    assert batched.resolve(fields, []) is fields


def test_one_intent_needs_no_list(space, rng):
    state = land(space, rng)
    found = batched.resolve(batched.fields(state), batched.Source(LandState.food, 1.0))
    same(found, core.resolve(state, core.Source(LandState.food, 1.0)))


def test_resolve_runs_for_a_batch_under_vmap(space, rng):
    op = Adjacency().compile(space)
    tables = batched.operator(op)
    states = [land(space, rng) for _ in range(4)]
    leaves = rng.random((4, space.n_cells))

    def one(fields, leave):
        return batched.resolve(fields, batched.Spread(LandState.people, tables, leave))

    many = jax.tree.map(lambda *xs: np.stack(xs), *(batched.fields(s) for s in states))
    found = jax.jit(jax.vmap(one))(many, leaves)
    for i, state in enumerate(states):
        wanted = core.resolve(state, core.Spread(LandState.people, op, leaves[i]))
        same(jax.tree.map(lambda x, i=i: x[i], found), wanted)


def test_what_a_rule_may_not_write_raises_as_in_core(space, rng):
    fields = batched.fields(land(space, rng))
    op = batched.operator(Adjacency().compile(space))
    with pytest.raises(ValueError, match="static"):
        batched.resolve(fields, batched.Source(LandState.fixed, 1.0))
    with pytest.raises(ValueError, match="not among the fields"):
        batched.resolve(fields, batched.Source(LandState.food, 1.0), writes=[LandState.people])
    twice = [batched.Assign(LandState.mood, 1.0), batched.Assign(LandState.mood, 2.0)]
    with pytest.raises(ValueError, match="assigned twice"):
        batched.resolve(fields, twice)
    mixed = [batched.Assign(LandState.food, 1.0), batched.Source(LandState.food, 2.0)]
    with pytest.raises(ValueError, match="assigned and also changed"):
        batched.resolve(fields, mixed)
    with pytest.raises(ValueError, match="intensive"):
        batched.resolve(fields, batched.Spread(LandState.mood, op, 0.5))
    with pytest.raises(TypeError, match="floating"):
        batched.resolve(fields, batched.Source(LandState.count, 1.0))
    with pytest.raises(ValueError, match="into must be another field"):
        batched.resolve(fields, batched.Claim(LandState.food, 1.0, into=LandState.food))
    with pytest.raises(ValueError, match="into must be another field"):
        batched.resolve(fields, batched.Claim(LandState.packs, 1.0, into=LandState.people))
    with pytest.raises(KeyError, match="declares no field"):
        batched.resolve(fields, batched.Source(Field(np.float64, extensive=True), 1.0))
    with pytest.raises(TypeError, match="must return intents"):
        batched.resolve(fields, np.zeros(3))  # not an intent
    with pytest.raises(TypeError, match="expected an intent"):
        batched.resolve(fields, [core.Source(LandState.food, 1.0)])  # ty: ignore[invalid-argument-type]  # core's, not the arrays'
