import numpy as np
import pytest

from demo import Biome, DemoState, PlanetState, Settlers, WorldState, fields
from gridzero.core import (
    Adjacency,
    Assign,
    CellGraph,
    Claim,
    Explicit,
    Field,
    Flow,
    Hops,
    LatticeSpace,
    NeighborAllocation,
    NeighborOperator,
    Nested,
    NestedState,
    Source,
    Spread,
    State,
    icosphere,
    pull,
    resolve,
)

pytestmark = pytest.mark.usefixtures("each_impl")  # every test on both paths


def test_assign_overwrites_one_field_and_shares_the_others(make_state):
    state = make_state(step=2)
    alive = np.arange(25) % 2 == 0
    new = resolve(state, Assign(DemoState.alive, alive))
    np.testing.assert_array_equal(new[DemoState.alive], alive)
    assert new[DemoState.pop] is state[DemoState.pop]
    assert new.step == 2
    assert not state[DemoState.alive].any()


def test_assigned_values_are_copied_into_a_read_only_array(make_state):
    alive = np.ones(25, bool)
    new = resolve(make_state(), Assign(DemoState.alive, alive))
    alive[0] = False
    assert new[DemoState.alive][0]
    assert not new[DemoState.alive].flags.writeable


def test_assign_broadcasts_values_over_components_and_cells(make_state):
    new = resolve(make_state(), Assign(DemoState.pop, np.array([[1.0], [2.0]])))
    np.testing.assert_array_equal(new[DemoState.pop], np.repeat([[1.0], [2.0]], 25, axis=1))


def test_assign_with_where_writes_only_the_selected_cells(make_state):
    state = make_state(pop=5.0)
    where = np.arange(25) < 3
    new = resolve(state, Assign(DemoState.pop, 0.0, where=where))
    np.testing.assert_array_equal(new[DemoState.pop][:, :3], np.zeros((2, 3)))
    np.testing.assert_array_equal(new[DemoState.pop][:, 3:], np.full((2, 22), 5.0))


def test_assign_with_where_checks_only_the_values_it_writes(make_lattice):
    # -1 does not fit in uint8, but outside ``where`` it is never stored.
    Count = fields(count=Field(np.uint8))
    state = Count(make_lattice())
    where = np.arange(25) < 5
    values = np.where(where, 200, -1)
    new = resolve(state, Assign(Count.count, values, where=where))
    np.testing.assert_array_equal(new[Count.count], np.where(where, 200, 0))
    with pytest.raises(ValueError, match="values from -1 to 200 do not fit in uint8"):
        resolve(state, Assign(Count.count, values))


def test_resolve_accepts_none_one_intent_or_any_iterable(make_state):
    state = make_state()
    assert resolve(state, None) is state
    assert resolve(state, []) is state
    new = resolve(
        state,
        (Assign(field, value) for field, value in [(DemoState.alive, True), (DemoState.pop, 1)]),
    )
    assert new[DemoState.alive].all()
    np.testing.assert_array_equal(new[DemoState.pop], np.ones((2, 25)))


def test_assign_rejects_values_that_would_change_kind(make_state):
    with pytest.raises(TypeError, match="field 'alive'"):
        resolve(make_state(), Assign(DemoState.alive, np.full(25, 0.5)))


def test_assign_rejects_values_of_the_wrong_shape(make_state):
    with pytest.raises(ValueError, match=r"'pop' has shape \(2, 25\); values of shape \(24,\)"):
        resolve(make_state(), Assign(DemoState.pop, np.zeros(24)))


def test_assign_rejects_a_where_that_is_not_boolean_or_does_not_broadcast(make_state):
    with pytest.raises(TypeError, match="where must be boolean"):
        resolve(make_state(), Assign(DemoState.alive, True, where=np.ones(25, int)))
    with pytest.raises(ValueError, match=r"where of shape \(5,\)"):
        resolve(make_state(), Assign(DemoState.alive, True, where=np.ones(5, bool)))


def test_phases_cannot_assign_static_fields(make_state):
    with pytest.raises(ValueError, match="'terrain' is static"):
        resolve(make_state(), Assign(DemoState.terrain, 1))


def test_phases_can_assign_only_the_fields_they_declare(make_state):
    with pytest.raises(ValueError, match=r"'pop' is not among the fields the phase writes: \['al"):
        resolve(make_state(), Assign(DemoState.pop, 1.0), writes={DemoState.alive})


def test_writes_may_give_one_field_or_several(make_state):
    state = make_state()
    assert resolve(state, Assign(DemoState.alive, True), writes=DemoState.alive)[
        DemoState.alive
    ].all()
    assert resolve(state, Assign(DemoState.alive, True), writes=[DemoState.pop, DemoState.alive])[
        DemoState.alive
    ].all()
    with pytest.raises(
        ValueError, match=r"'pop' is not among the fields the phase writes: \['alive'\]"
    ):
        resolve(state, Assign(DemoState.pop, 1.0), writes={DemoState.alive})


def test_fields_are_given_as_fields_not_names(make_state):
    with pytest.raises(TypeError, match="Field attributes of a State class, got 'alive'"):
        resolve(make_state(), Assign("alive", True))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"fields declared on a State class.*got 'alive'"):
        resolve(make_state(), Assign(DemoState.alive, True), writes="alive")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="not declared on a State class"):
        resolve(make_state(), Assign(Field(bool), True))
    other = fields(alive=Field(np.uint8)).alive
    with pytest.raises(KeyError, match="field 'alive' is declared as"):
        resolve(make_state(), Assign(other, 1))


def test_one_phase_cannot_assign_a_field_twice(make_state):
    intents = [
        Assign(DemoState.alive, True),
        Assign(DemoState.alive, False, where=np.ones(25, bool)),
    ]
    with pytest.raises(ValueError, match="'alive' is assigned twice"):
        resolve(make_state(), intents)


def test_assigning_an_unknown_field_raises(make_state):
    with pytest.raises(KeyError, match="no field named 'food'"):
        resolve(make_state(), Assign(AmountsState.food, 1.0))


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        ({DemoState.alive: True}, "must return intents or None, got dict"),
        (np.ones(25, bool), "got ndarray; wrap new values in Assign"),
        (3, "must return intents or None, got int"),
        (["alive"], "expected an intent such as Assign, got str"),
    ],
    ids=["dict", "array", "int", "list-of-names"],
)
def test_resolve_rejects_results_that_are_not_intents(make_state, returned, message):
    with pytest.raises(TypeError, match=message):
        resolve(make_state(), returned)


# Moving amounts: shared fixtures.


class AmountsState(State):
    """Amount fields for the intents that move amounts, plus an intensive and an integer field."""

    pop = Field(np.float64, extensive=True)
    groups = Field(np.float64, (2,), extensive=True)
    food = Field(np.float64, extensive=True)
    fed = Field(np.float64, (2,), extensive=True)
    crowding = Field(np.float64)
    count = Field(np.int64, extensive=True)
    land = Field(bool, static=True)


RING = LatticeSpace((5,))
"""Five cells in a ring. Cell i's neighbors are i - 1 and i + 1, so edges are easy to list."""

RING_OP = Adjacency().compile(RING)
"""Edges in CSR order: 0->1, 0->4, 1->0, 1->2, 2->1, 2->3, 3->2, 3->4, 4->0, 4->3."""


def ring_state(**values):
    """A state of `AmountsState` on the ring, with fields given by attribute name for brevity."""
    values.setdefault("land", np.ones(5, bool))
    return AmountsState(
        RING, {getattr(AmountsState, name): value for name, value in values.items()}
    )


def edge_rates(**rates):
    """Rates on the ring edges, given as ``e_0_4=0.3`` for the edge 0 -> 4. Others are 0."""
    out = np.zeros(RING_OP.n_edges)
    edges = list(zip(RING_OP.graph.sources.tolist(), RING_OP.graph.indices.tolist(), strict=True))
    for name, rate in rates.items():
        _, i, j = name.split("_")
        out[edges.index((int(i), int(j)))] = rate
    return out


def point(value, cell=0, n=5):
    return np.where(np.arange(n) == cell, float(value), 0.0)


def random_amounts(rng, space, shape=(2,)):
    return rng.uniform(0.0, 3.0, (*shape, space.n_cells))


def test_intents_given_a_field_declared_the_same_way_resolve_alike():
    # Fields are values: a field of another class, declared the same way, is the same field.
    Again = fields(groups=Field(np.float64, (2,), extensive=True))
    state = ring_state(groups=point(2.0), food=point(1.0, cell=1) + 0.5)
    by_class = [
        Flow(AmountsState.groups, RING_OP, edge_rates(e_0_1=0.25, e_0_4=0.5)),
        Spread(AmountsState.groups, RING_OP, 0.1, toward=np.arange(5.0)),
    ]
    by_equal = [
        Flow(Again.groups, RING_OP, edge_rates(e_0_1=0.25, e_0_4=0.5)),
        Spread(Again.groups, RING_OP, 0.1, toward=np.arange(5.0)),
    ]
    new, expected = resolve(state, by_equal), resolve(state, by_class)
    for field in AmountsState._schema.values():
        np.testing.assert_array_equal(new[field], expected[field])
    assert not np.array_equal(new[AmountsState.groups], state[AmountsState.groups])


# Flow.


def test_flow_sends_the_given_fraction_of_the_source_along_each_edge():
    new = resolve(
        ring_state(pop=point(10)), Flow(AmountsState.pop, RING_OP, edge_rates(e_0_1=0.1, e_0_4=0.3))
    )
    np.testing.assert_allclose(new[AmountsState.pop], [6.0, 1.0, 0.0, 0.0, 3.0], rtol=1e-15)


def test_flow_rates_are_fractions_of_what_each_source_holds():
    pop = np.array([10.0, 20.0, 0.0, 0.0, 0.0])
    new = resolve(
        ring_state(pop=pop), Flow(AmountsState.pop, RING_OP, edge_rates(e_0_1=0.5, e_1_2=0.5))
    )
    np.testing.assert_allclose(new[AmountsState.pop], [5.0, 15.0, 10.0, 0.0, 0.0], rtol=1e-15)


def test_flow_rates_adding_up_past_one_are_scaled_to_send_everything():
    rates = edge_rates(e_0_1=1.5, e_0_4=0.5)
    new = resolve(ring_state(pop=point(10)), Flow(AmountsState.pop, RING_OP, rates))
    np.testing.assert_allclose(new[AmountsState.pop], [0.0, 7.5, 0.0, 0.0, 2.5], rtol=1e-15)


def test_flow_rates_are_cut_to_one_before_sharing_with_other_takers():
    # The flow asks for 2 x 10 but counts as asking for all 10; the removal asks for 5. The
    # flow then gets 2/3 of the amount, split 3:1, and the removal 1/3.
    intents = [
        Flow(AmountsState.pop, RING_OP, edge_rates(e_0_1=1.5, e_0_4=0.5)),
        Source(AmountsState.pop, point(-5)),
    ]
    new = resolve(ring_state(pop=point(10)), intents)
    np.testing.assert_allclose(new[AmountsState.pop], [0.0, 5.0, 0.0, 0.0, 5 / 3], rtol=1e-15)


def test_flow_rates_are_scaled_before_the_domain_keeps_its_share_home():
    # The rates scale to 0.75 and 0.25 first; the 0.25 meant for cell 4 then stays home.
    domain = np.arange(5) != 4
    rates = edge_rates(e_0_1=1.5, e_0_4=0.5)
    new = resolve(ring_state(pop=point(10)), Flow(AmountsState.pop, RING_OP, rates, domain))
    np.testing.assert_allclose(new[AmountsState.pop], [2.5, 7.5, 0.0, 0.0, 0.0], rtol=1e-15)


def test_flow_ignores_negative_rates():
    rates = edge_rates(e_0_1=-0.5, e_0_4=0.25)
    new = resolve(ring_state(pop=point(8)), Flow(AmountsState.pop, RING_OP, rates))
    np.testing.assert_allclose(new[AmountsState.pop], [6.0, 0.0, 0.0, 0.0, 2.0], rtol=1e-15)


def test_flow_never_enters_cells_outside_its_domain():
    # What was sent toward cell 4 stays home; the flow to cell 1 goes ahead.
    domain = np.arange(5) != 4
    rates = edge_rates(e_0_1=0.25, e_0_4=0.5)
    new = resolve(ring_state(pop=point(8)), Flow(AmountsState.pop, RING_OP, rates, domain))
    np.testing.assert_allclose(new[AmountsState.pop], [6.0, 2.0, 0.0, 0.0, 0.0], rtol=1e-15)


def test_flow_rates_broadcast_over_components_or_differ_per_component():
    groups = np.stack([point(8), point(4)])
    shared = resolve(
        ring_state(groups=groups), Flow(AmountsState.groups, RING_OP, edge_rates(e_0_1=0.5))
    )
    np.testing.assert_allclose(shared[AmountsState.groups][:, 1], [4.0, 2.0], rtol=1e-15)
    rates = np.stack([edge_rates(e_0_1=0.5), edge_rates(e_0_4=0.25)])
    split = resolve(ring_state(groups=groups), Flow(AmountsState.groups, RING_OP, rates))
    np.testing.assert_allclose(
        split[AmountsState.groups][:, [0, 1, 4]], [[4, 4, 0], [3, 0, 1]], rtol=1e-15
    )


@pytest.mark.parametrize("weight", [None, "distance"])
def test_flow_conserves_the_total_and_never_makes_amounts_negative(
    any_space, make_state, rng, weight
):
    op = Adjacency(weight=weight).compile(any_space)
    state = make_state(any_space, pop=random_amounts(rng, any_space))
    rates = rng.uniform(-0.3, 0.7, (2, op.n_edges))  # many cells ask to send more than they hold
    new = resolve(state, Flow(DemoState.pop, op, rates))
    np.testing.assert_allclose(new.total(DemoState.pop), state.total(DemoState.pop), rtol=1e-12)
    assert new[DemoState.pop].min() >= 0


# Spread.


def test_spread_sends_a_fraction_split_equally_over_unweighted_neighbors(make_state):
    state = make_state(pop=point(1.0, cell=12, n=25))
    op = Adjacency().compile(state.space)
    new = resolve(state, Spread(DemoState.pop, op, leave=0.4))
    expected = np.zeros(25)
    expected[12], expected[[7, 11, 13, 17]] = 0.6, 0.1
    np.testing.assert_allclose(new[DemoState.pop], np.broadcast_to(expected, (2, 25)), rtol=1e-15)


def test_spread_splits_in_proportion_to_weight_times_attraction():
    toward = np.array([0.0, 3.0, 0.0, 0.0, 1.0])
    new = resolve(
        ring_state(pop=point(8)), Spread(AmountsState.pop, RING_OP, leave=0.5, toward=toward)
    )
    np.testing.assert_allclose(new[AmountsState.pop], [4.0, 3.0, 0.0, 0.0, 1.0], rtol=1e-15)


def test_spread_keeps_a_uniform_amount_uniform_on_a_regular_grid(make_lattice, make_state):
    space = make_lattice((4, 5), tiling="hex")
    state = make_state(space, pop=1.0)
    new = resolve(state, Spread(DemoState.pop, Adjacency().compile(space), leave=0.3))
    np.testing.assert_allclose(new[DemoState.pop], np.ones((2, 20)), rtol=1e-15)


def test_nothing_leaves_a_cell_with_nowhere_to_go():
    # Neither neighbor of cell 0 attracts anything, and cell 4 is outside the domain.
    toward = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    domain = np.arange(5) != 4
    new = resolve(ring_state(pop=point(8)), Spread(AmountsState.pop, RING_OP, 0.5, toward, domain))
    np.testing.assert_array_equal(new[AmountsState.pop], point(8))


def test_spread_clips_leave_into_zero_and_one():
    pop = np.array([8.0, 0.0, 8.0, 0.0, 0.0])
    leave = np.array([1.5, 0.0, -0.5, 0.0, 0.0])
    new = resolve(ring_state(pop=pop), Spread(AmountsState.pop, RING_OP, leave))
    np.testing.assert_allclose(new[AmountsState.pop], [0.0, 4.0, 8.0, 0.0, 4.0], rtol=1e-15)


@pytest.mark.parametrize("weight", [None, "distance", "boundary"])
def test_spread_equals_the_flow_with_matching_rates(any_space, make_state, rng, weight):
    op = Adjacency(weight=weight).compile(any_space)
    n = any_space.n_cells
    state = make_state(any_space, pop=random_amounts(rng, any_space))
    leave, toward = rng.uniform(-0.2, 1.2, (2, n)), rng.uniform(-0.5, 2.0, (2, n))
    domain = rng.random(n) < 0.7
    spread = resolve(state, Spread(DemoState.pop, op, leave, toward, domain))

    attraction = np.maximum(toward, 0) * domain
    pulled = pull(attraction, op)
    src, dst = op.graph.sources, op.graph.indices
    weights = np.ones(op.n_edges) if weight is None else op.weights
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.clip(leave, 0, 1)[:, src] * weights * attraction[:, dst] / pulled[:, src]
    flow = resolve(state, Flow(DemoState.pop, op, np.nan_to_num(rates), domain))
    np.testing.assert_allclose(spread[DemoState.pop], flow[DemoState.pop], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(spread.total(DemoState.pop), state.total(DemoState.pop), rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-310, 1e-200, 1e200, np.finfo(np.float64).max])
def test_spread_depends_only_on_the_ratios_of_attractions(scale, rng):
    space = icosphere(3)
    op = Adjacency(weight="boundary").compile(space)
    state = AmountsState(
        space, {AmountsState.groups: random_amounts(rng, space), AmountsState.land: True}
    )
    toward = rng.random((2, space.n_cells))
    expected = resolve(state, Spread(AmountsState.groups, op, 0.5, toward))
    new = resolve(state, Spread(AmountsState.groups, op, 0.5, toward * scale))
    rtol = 1e-8 if scale < 1e-300 else 1e-15  # subnormal attractions keep fewer bits
    np.testing.assert_allclose(new[AmountsState.groups], expected[AmountsState.groups], rtol=rtol)


@pytest.mark.parametrize(("faint", "moves"), [(1e-140, True), (1e-160, False)])
def test_attractions_far_below_the_largest_count_as_none(faint, moves):
    # Cell 2's neighbors attract 1e-140 or 1e-160 as much as cell 0. Below 2**-500 (3e-151)
    # of the largest, cell 2 has nowhere to go, and keeps all it holds.
    toward = np.array([1.0, faint, 0.0, faint, 0.0])
    new = resolve(ring_state(pop=point(8, cell=2)), Spread(AmountsState.pop, RING_OP, 0.5, toward))
    expected = [0.0, 2.0, 4.0, 2.0, 0.0] if moves else point(8, cell=2)
    np.testing.assert_array_equal(new[AmountsState.pop], expected)


def test_spread_on_a_directed_neighborhood_sends_along_its_edges(rng):
    # Each cell picks 3 random targets, so the operator differs from its transpose.
    space = icosphere(3)
    n = space.n_cells
    idx = np.argsort(rng.random((n, n)) + np.eye(n), axis=1)[:, :3]
    op = NeighborOperator.from_table(idx, rng.uniform(0.5, 2.0, (n, 3)))
    assert not op.graph.is_symmetric()
    state = AmountsState(
        space, {AmountsState.pop: rng.uniform(0, 2, n), AmountsState.land: np.ones(n, bool)}
    )
    leave, toward = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    spread = resolve(state, Spread(AmountsState.pop, op, leave, toward))
    src, dst = op.graph.sources, op.graph.indices
    rates = leave[src] * op.weights * toward[dst] / pull(toward, op)[src]
    flow = resolve(state, Flow(AmountsState.pop, op, rates))
    np.testing.assert_allclose(spread[AmountsState.pop], flow[AmountsState.pop], rtol=1e-12)
    np.testing.assert_allclose(
        spread.total(AmountsState.pop), state.total(AmountsState.pop), rtol=1e-12
    )


def test_spread_needs_non_negative_weights():
    graph = CellGraph.from_edges([0, 1], [1, 0], 5, edge_data={"w": [1.0, -1.0]})
    op = Explicit(graph, weight="w").compile(RING)
    with pytest.raises(ValueError, match="negative weights"):
        resolve(ring_state(pop=1.0), Spread(AmountsState.pop, op, 0.5))


# Source.


def test_source_adds_and_removes_amounts():
    pop = np.array([4.0, 4.0, 4.0, 0.0, 0.0])
    new = resolve(ring_state(pop=pop), Source(AmountsState.pop, [1.0, -1.0, 0.0, 2.0, 0.0]))
    np.testing.assert_array_equal(new[AmountsState.pop], [5.0, 3.0, 4.0, 2.0, 0.0])


def test_removals_never_take_more_than_a_cell_holds():
    new = resolve(ring_state(pop=point(3)), Source(AmountsState.pop, -5.0))
    np.testing.assert_array_equal(new[AmountsState.pop], np.zeros(5))


def test_amounts_added_in_a_phase_cannot_leave_in_that_phase():
    # Every intent reads the same state, which holds nothing to move.
    new = resolve(
        ring_state(), [Source(AmountsState.pop, point(6)), Spread(AmountsState.pop, RING_OP, 1.0)]
    )
    np.testing.assert_array_equal(new[AmountsState.pop], point(6))


# Several takers of one field.


def test_takers_share_a_scarce_amount_in_proportion_to_what_they_ask():
    # Cell 0 holds 10. The spread asks for 5 and the removal for 8: each gets 10/13 of it.
    intents = [Spread(AmountsState.pop, RING_OP, 0.5), Source(AmountsState.pop, point(-8))]
    new = resolve(ring_state(pop=point(10)), intents)
    np.testing.assert_allclose(new[AmountsState.pop], [0, 25 / 13, 0, 0, 25 / 13], rtol=1e-15)


def test_several_flows_of_one_field_are_scaled_together():
    intents = [
        Flow(AmountsState.pop, RING_OP, edge_rates(e_0_1=0.75)),
        Flow(AmountsState.pop, RING_OP, edge_rates(e_0_4=0.75)),
    ]
    new = resolve(ring_state(pop=point(10)), intents)
    np.testing.assert_allclose(new[AmountsState.pop], [0, 5, 0, 0, 5], rtol=1e-15)


def test_takers_are_rationed_per_component():
    groups = np.stack([point(4), point(4)])
    intents = [
        Spread(AmountsState.groups, RING_OP, 0.5),
        Source(AmountsState.groups, [[-1.0], [-4.0]]),
    ]
    new = resolve(ring_state(groups=groups), intents)
    np.testing.assert_allclose(new[AmountsState.groups][0], [1, 1, 0, 0, 1], rtol=1e-15)
    np.testing.assert_allclose(new[AmountsState.groups][1], [0, 2 / 3, 0, 0, 2 / 3], rtol=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_several_flows_conserve_the_total_stay_non_negative_and_keep_out_of_the_domain(seed):
    rng = np.random.default_rng(seed)
    space = icosphere(3)
    n = space.n_cells
    near = Adjacency(weight="boundary").compile(space)
    far = Hops(2).compile(space)
    land = rng.random(n) < 0.7
    state = AmountsState(
        space, {AmountsState.groups: random_amounts(rng, space), AmountsState.land: land}
    )
    intents = [
        Spread(AmountsState.groups, near, rng.uniform(0, 1.5, (2, n)), rng.random(n), land),
        Flow(AmountsState.groups, far, rng.uniform(-0.1, 0.4, far.n_edges), land),
    ]
    new = resolve(state, intents)
    np.testing.assert_allclose(
        new.total(AmountsState.groups), state.total(AmountsState.groups), rtol=1e-12
    )
    assert new[AmountsState.groups].min() >= 0
    assert np.all(new[AmountsState.groups][:, ~land] <= state[AmountsState.groups][:, ~land])


# Claim.


def test_claims_are_granted_in_full_when_the_supply_suffices():
    claims = np.stack([point(2), point(3)])
    new = resolve(
        ring_state(food=point(10)), Claim(AmountsState.food, claims, into=AmountsState.fed)
    )
    np.testing.assert_array_equal(new[AmountsState.food], point(5))
    np.testing.assert_array_equal(new[AmountsState.fed], claims)


def test_a_scarce_supply_is_split_in_proportion_to_the_claims():
    claims = np.stack([point(6), point(18)])
    new = resolve(
        ring_state(food=point(8)), Claim(AmountsState.food, claims, into=AmountsState.fed)
    )
    np.testing.assert_array_equal(new[AmountsState.food], np.zeros(5))
    np.testing.assert_allclose(new[AmountsState.fed], np.stack([point(2), point(6)]), rtol=1e-15)


def test_claims_add_to_what_the_receiving_field_holds_and_ignore_negative_claims():
    fed = np.stack([point(1), point(1)])
    claims = np.stack([point(2), point(-3)])
    new = resolve(
        ring_state(food=point(10), fed=fed), Claim(AmountsState.food, claims, into=AmountsState.fed)
    )
    np.testing.assert_array_equal(new[AmountsState.food], point(8))
    np.testing.assert_array_equal(new[AmountsState.fed], np.stack([point(3), point(1)]))


def test_claims_conserve_the_sum_of_the_two_fields(rng):
    space = icosphere(3)
    values = {
        AmountsState.food: rng.uniform(0, 2, space.n_cells),
        AmountsState.land: np.ones(space.n_cells, bool),
    }
    state = AmountsState(space, values)
    new = resolve(
        state,
        Claim(AmountsState.food, rng.uniform(0, 2, (2, space.n_cells)), into=AmountsState.fed),
    )
    before = state.total(AmountsState.food) + state.total(AmountsState.fed).sum()
    np.testing.assert_allclose(
        new.total(AmountsState.food) + new.total(AmountsState.fed).sum(), before, rtol=1e-12
    )
    assert new[AmountsState.food].min() >= 0


def test_claims_share_the_supply_with_the_other_takers_of_the_field():
    # 12 food: the removal asks for 8 and the claims for 4 + 4, so each gets 3/4.
    intents = [
        Source(AmountsState.food, point(-8)),
        Claim(AmountsState.food, [[4.0], [4.0]], into=AmountsState.fed),
    ]
    new = resolve(ring_state(food=point(12)), intents)
    np.testing.assert_array_equal(new[AmountsState.food], np.zeros(5))
    np.testing.assert_allclose(new[AmountsState.fed][:, 0], [3.0, 3.0], rtol=1e-15)


def test_an_efficiency_scales_what_reaches_into_but_not_what_leaves():
    # 8 food for claims of 6 and 18: a quarter each is granted, as without an efficiency.
    claims = np.stack([point(6), point(18)])
    claim = Claim(AmountsState.food, claims, into=AmountsState.fed, efficiency=0.5)
    new = resolve(ring_state(food=point(8)), claim)
    np.testing.assert_array_equal(new[AmountsState.food], np.zeros(5))
    np.testing.assert_allclose(new[AmountsState.fed], np.stack([point(1), point(3)]), rtol=1e-15)


def test_each_claimant_may_have_its_own_efficiency_and_negative_ones_count_as_0():
    fed = np.stack([point(1), point(1)])
    claims = np.stack([point(2), point(4)])
    claim = Claim(AmountsState.food, claims, into=AmountsState.fed, efficiency=[[3.0], [-1.0]])
    new = resolve(ring_state(food=point(10), fed=fed), claim)
    np.testing.assert_array_equal(new[AmountsState.food], point(4))
    np.testing.assert_array_equal(new[AmountsState.fed], np.stack([point(7), point(1)]))


def test_an_efficiency_of_1_gives_the_bits_of_no_efficiency(rng):
    state = ring_state(food=rng.uniform(0, 1, 5), fed=random_amounts(rng, RING))
    claims = rng.uniform(0, 1, (2, 5))
    plain = resolve(state, Claim(AmountsState.food, claims, into=AmountsState.fed))
    one = resolve(state, Claim(AmountsState.food, claims, into=AmountsState.fed, efficiency=1))
    assert plain[AmountsState.fed].tobytes() == one[AmountsState.fed].tobytes()


def test_an_efficiency_must_broadcast_to_into():
    with pytest.raises(ValueError, match=r"efficiency has shape \(3, 1\), which does not"):
        resolve(
            ring_state(food=1.0),
            Claim(AmountsState.food, 1.0, into=AmountsState.fed, efficiency=np.ones((3, 1))),
        )


class MealsState(State):
    food = Field(np.float64, (2,), extensive=True)  # grain and fish
    eaten = Field(np.float64, (3, 2), extensive=True)  # by three groups
    summed = Field(np.float64, extensive=True)  # loses the resource axis
    per_group = Field(np.float64, (3,), extensive=True)  # does not end with (2,)


def test_claims_on_a_resource_with_components(make_lattice):
    state = MealsState(make_lattice((3, 3)), {MealsState.food: [[6.0], [1.0]]})
    claims = np.ones((3, 2, 1))
    new = resolve(state, Claim(MealsState.food, claims, into=MealsState.eaten))
    np.testing.assert_allclose(
        new[MealsState.food], np.repeat([[3.0], [0.0]], 9, axis=1), rtol=1e-15
    )
    np.testing.assert_allclose(new[MealsState.eaten][:, :, 0], [[1, 1 / 3]] * 3, rtol=1e-15)


@pytest.mark.parametrize(
    "into", [MealsState.food, MealsState.summed, MealsState.per_group], ids=repr
)
def test_claims_go_into_another_field_shaped_as_claimants_then_the_resource(make_lattice, into):
    state = MealsState(make_lattice((3, 3)), {MealsState.food: 1.0})
    with pytest.raises(
        ValueError, match=r"into must be another field, of shape \(\*claimants, \*\(2,\)\)"
    ):
        resolve(state, Claim(MealsState.food, 1.0, into=into))


def test_a_claim_writes_both_of_its_fields():
    with pytest.raises(ValueError, match="'fed' is not among the fields the phase writes"):
        resolve(
            ring_state(food=point(4)),
            Claim(AmountsState.food, 1.0, into=AmountsState.fed),
            writes={AmountsState.food},
        )


# Errors and edges.


def test_resolved_amounts_are_new_read_only_arrays():
    state = ring_state(pop=point(4), food=point(4))
    intents = [
        Spread(AmountsState.pop, RING_OP, 0.5),
        Claim(AmountsState.food, [[1.0], [1.0]], into=AmountsState.fed),
    ]
    new = resolve(state, intents)
    for field in (AmountsState.pop, AmountsState.food, AmountsState.fed):
        assert not new[field].flags.writeable
        assert not np.shares_memory(new[field], state[field])


def test_an_assign_cannot_be_combined_with_other_intents_on_its_field():
    with pytest.raises(ValueError, match="'pop' is assigned and also changed by a Source"):
        resolve(ring_state(), [Assign(AmountsState.pop, 1.0), Source(AmountsState.pop, 1.0)])
    with pytest.raises(ValueError, match="'fed' is assigned and also changed by a Claim"):
        resolve(
            ring_state(food=1.0),
            [Claim(AmountsState.food, 1.0, into=AmountsState.fed), Assign(AmountsState.fed, 0.0)],
        )


AMOUNT_INTENTS = {
    "flow": lambda field: Flow(field, RING_OP, 0.1),
    "spread": lambda field: Spread(field, RING_OP, 0.1),
    "source": lambda field: Source(field, 1.0),
    "claim-from": lambda field: Claim(field, 1.0, into=AmountsState.fed),
    "claim-into": lambda field: Claim(AmountsState.food, 1.0, into=field),
}


@pytest.mark.parametrize("case", list(AMOUNT_INTENTS))
def test_moving_amounts_needs_an_extensive_field(case):
    with pytest.raises(ValueError, match="field 'crowding' is intensive"):
        resolve(ring_state(food=1.0), AMOUNT_INTENTS[case](AmountsState.crowding))


@pytest.mark.parametrize("case", list(AMOUNT_INTENTS))
def test_moving_amounts_needs_a_floating_field(case):
    # Integer counts would need stochastic rounding to split, which is not built yet.
    with pytest.raises(TypeError, match="field 'count' holds int64"):
        resolve(ring_state(food=1.0), AMOUNT_INTENTS[case](AmountsState.count))


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf], ids=["negative", "nan", "inf"])
def test_moving_amounts_needs_finite_non_negative_amounts(bad):
    with pytest.raises(ValueError, match=r"holds (negative or NaN|infinite) amounts"):
        resolve(ring_state(pop=point(bad, cell=3)), Source(AmountsState.pop, 1.0))


INTENT_VALUES = {
    "flow-rate": lambda value: Flow(AmountsState.pop, RING_OP, value),
    "spread-leave": lambda value: Spread(AmountsState.pop, RING_OP, value),
    "spread-toward": lambda value: Spread(AmountsState.pop, RING_OP, 0.5, toward=value),
    "source-amount": lambda value: Source(AmountsState.pop, value),
    "claim-amount": lambda value: Claim(AmountsState.food, value, into=AmountsState.fed),
    "claim-efficiency": lambda value: Claim(
        AmountsState.food, 1.0, into=AmountsState.fed, efficiency=value
    ),
}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("case", list(INTENT_VALUES))
def test_intents_reject_values_that_are_not_finite(case, bad):
    with pytest.raises(ValueError, match="holds NaN or infinite values"):
        resolve(ring_state(pop=1.0, food=1.0), INTENT_VALUES[case](bad))


@pytest.mark.parametrize("bad", [True, np.array([True] * 5), "0.5"], ids=["bool", "bools", "str"])
@pytest.mark.parametrize("case", list(INTENT_VALUES))
def test_intents_reject_values_that_are_not_numbers(case, bad):
    with pytest.raises(TypeError, match="must hold numbers"):
        resolve(ring_state(pop=1.0, food=1.0), INTENT_VALUES[case](bad))


BROADCAST_INTENTS = {
    "flow-rate": (lambda v: Flow(AmountsState.groups, RING_OP, v), RING_OP.n_edges),
    "spread-leave": (lambda v: Spread(AmountsState.groups, RING_OP, v), 5),
    "spread-toward": (lambda v: Spread(AmountsState.groups, RING_OP, 0.5, toward=v), 5),
    "source-amount": (lambda v: Source(AmountsState.groups, -v), 5),
    "claim-amount": (lambda v: Claim(AmountsState.food, v, into=AmountsState.fed), 5),
    "claim-efficiency": (
        lambda v: Claim(AmountsState.food, 0.3, into=AmountsState.fed, efficiency=v),
        5,
    ),
}
"""Intents with an array of shape (2, last) on the ring, and the length of its last axis."""

BROADCAST_FORMS = {
    "scalar": lambda rng, last: rng.uniform(0.1, 0.9),
    "(1,)": lambda rng, last: rng.uniform(0.1, 0.9, 1),
    "(2, 1)": lambda rng, last: rng.uniform(0.1, 0.9, (2, 1)),
    "(last,)": lambda rng, last: rng.uniform(0.1, 0.9, last),
    "(1, last)": lambda rng, last: rng.uniform(0.1, 0.9, (1, last)),
}


@pytest.mark.parametrize("form", list(BROADCAST_FORMS))
@pytest.mark.parametrize("case", list(BROADCAST_INTENTS))
def test_intent_arrays_in_every_broadcast_form_act_like_the_full_array(case, form, rng):
    make, last = BROADCAST_INTENTS[case]
    values = BROADCAST_FORMS[form](rng, last)
    state = ring_state(groups=random_amounts(rng, RING), food=rng.uniform(0, 1, 5))
    new = resolve(state, make(values))
    full = resolve(state, make(np.broadcast_to(values, (2, last)).copy()))
    for field in (AmountsState.groups, AmountsState.food, AmountsState.fed):
        np.testing.assert_array_equal(new[field], full[field])
    assert not np.array_equal(
        new[AmountsState.groups], state[AmountsState.groups]
    ) or case.startswith("claim")


def test_intent_arrays_must_broadcast_to_the_field():
    with pytest.raises(
        ValueError, match=r"rate has shape \(5,\), which does not broadcast to \(10,\)"
    ):
        resolve(ring_state(pop=1.0), Flow(AmountsState.pop, RING_OP, np.full(5, 0.1)))
    with pytest.raises(ValueError, match=r"amount has shape \(3, 5\), which does not broadcast"):
        resolve(
            ring_state(food=1.0), Claim(AmountsState.food, np.ones((3, 5)), into=AmountsState.fed)
        )


@pytest.mark.parametrize("make", [Flow, Spread])
def test_moving_amounts_needs_an_operator_on_the_same_cells(make, make_lattice):
    with pytest.raises(TypeError, match="op must be a NeighborOperator, got Adjacency"):
        resolve(ring_state(pop=1.0), make(AmountsState.pop, Adjacency(), 0.1))
    op = Adjacency().compile(make_lattice())
    with pytest.raises(ValueError, match="the operator has 25 cells, but the state has 5"):
        resolve(ring_state(pop=1.0), make(AmountsState.pop, op, np.full(op.n_edges, 0.1)))


@pytest.mark.parametrize(
    ("domain", "error"),
    [(np.ones(5), TypeError), (np.ones(4, bool), ValueError), (np.ones((1, 5), bool), ValueError)],
    ids=["floats", "too-short", "two-dimensional"],
)
@pytest.mark.parametrize("make", [Flow, Spread])
def test_a_domain_is_one_flag_per_cell(make, domain, error):
    with pytest.raises(error, match="domain must"):
        resolve(ring_state(pop=1.0), make(AmountsState.pop, RING_OP, 0.1, domain=domain))


def test_amounts_beyond_a_float32_field_raise_instead_of_becoming_infinite():
    big = float(np.finfo(np.float32).max)
    Small = fields(pop=Field(np.float32, extensive=True))
    state = Small(RING, {Small.pop: [big, 0, big, 0, 0]})
    with pytest.raises(ValueError, match="do not fit in float32"):
        resolve(state, Flow(Small.pop, RING_OP, edge_rates(e_0_1=1.0, e_2_1=1.0)))


def test_amounts_beyond_float64_raise_instead_of_becoming_infinite():
    big = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="overflow float64"):
        resolve(ring_state(pop=point(big)), Source(AmountsState.pop, point(big)))
    rates = edge_rates(e_0_1=1.0, e_2_1=1.0)
    with pytest.raises(ValueError, match="overflow float64"):
        resolve(ring_state(pop=[big, 0, big, 0, 0]), Flow(AmountsState.pop, RING_OP, rates))


def test_rates_whose_sum_overflows_raise_instead_of_losing_amounts():
    big = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="rate is too large"):
        resolve(
            ring_state(pop=1.0), Flow(AmountsState.pop, RING_OP, edge_rates(e_0_1=big, e_0_4=big))
        )
    graph = CellGraph.from_edges([0, 0], [1, 4], 5, edge_data={"w": [big, big]})
    op = Explicit(graph, weight="w").compile(RING)
    with pytest.raises(ValueError, match="the operator's weights are too large"):
        resolve(ring_state(pop=1.0), Spread(AmountsState.pop, op, 0.5))


def test_float32_amounts_are_conserved_to_float32_rounding(rng):
    space = icosphere(3)
    Small = fields(pop=Field(np.float32, (2,), extensive=True))
    state = Small(space, {Small.pop: random_amounts(rng, space)})
    new = resolve(state, Spread(Small.pop, Adjacency().compile(space), 0.3, toward=rng.random(92)))
    assert new[Small.pop].dtype == np.float32
    assert new[Small.pop].min() >= 0
    np.testing.assert_allclose(new.total(Small.pop), state.total(Small.pop), rtol=1e-6)


# Nested states: amounts that move together.


def settled(space, rng):
    """A `WorldState` with random amounts in every field, stores at 3 per person."""
    people = rng.random((2, space.n_cells))
    return WorldState(
        space,
        {
            WorldState.settlers.people: people,
            WorldState.settlers.stores: 3 * people,
            WorldState.settlers.mood: rng.random(space.n_cells),
            WorldState.settlers.count: rng.integers(0, 9, space.n_cells),
            WorldState.water: rng.random(space.n_cells),
        },
    )


MOVING = (WorldState.settlers.people, WorldState.settlers.stores)
STAYING = (
    WorldState.settlers.home,
    WorldState.settlers.mood,
    WorldState.settlers.count,
    WorldState.water,
)


def test_a_spread_of_a_nested_state_moves_each_field_as_its_own_spread_would(make_lattice, rng):
    space = make_lattice()
    op = Adjacency().compile(space)
    state = settled(space, rng)
    leave, toward = rng.random((2, space.n_cells)), rng.random(space.n_cells)
    together = resolve(state, Spread(WorldState.settlers, op, leave, toward))
    apart = resolve(state, [Spread(field, op, leave, toward) for field in MOVING])
    for field in MOVING:
        assert together[field].tobytes() == apart[field].tobytes()
    for field in STAYING:
        assert together[field] is state[field]


def test_an_allocation_flow_of_a_nested_state_moves_each_field_as_its_own_would(make_lattice, rng):
    space = make_lattice()
    allocation = NeighborAllocation(Adjacency().compile(space), shape=(2,))
    state = settled(space, rng)
    weights = rng.random(allocation.action_shape)
    together = resolve(state, allocation.flow(WorldState.settlers, weights))
    apart = resolve(state, [allocation.flow(field, weights) for field in MOVING])
    for field in MOVING:
        assert together[field].tobytes() == apart[field].tobytes()
    for field in STAYING:
        assert together[field] is state[field]


def test_a_flow_of_a_nested_state_moves_each_field_as_its_own_flow_would(make_lattice, rng):
    space = make_lattice()
    op = Adjacency().compile(space)
    state = settled(space, rng)
    rate = rng.random((2, op.n_edges)) / 4
    together = resolve(state, Flow(WorldState.settlers, op, rate))
    apart = resolve(state, [Flow(field, op, rate) for field in MOVING])
    for field in MOVING:
        assert together[field].tobytes() == apart[field].tobytes()
    for field in STAYING:
        assert together[field] is state[field]


def test_a_nested_state_carries_its_amounts_in_proportion(make_lattice, rng):
    space = make_lattice()
    state = settled(space, rng)
    moved = resolve(state, Spread(WorldState.settlers, Adjacency().compile(space), 0.4))
    np.testing.assert_allclose(moved.settlers.stores, 3 * moved.settlers.people, rtol=1e-13)
    np.testing.assert_allclose(moved.total(WorldState.settlers.people), state.total(MOVING[0]))


@pytest.mark.parametrize(
    "intent",
    [
        lambda: Source(WorldState.settlers, 1.0),  # ty: ignore[invalid-argument-type] - the point
        lambda: Assign(WorldState.settlers, 1.0),  # ty: ignore[invalid-argument-type] - the point
        lambda: Claim(WorldState.settlers, 1.0, into=WorldState.water),  # ty: ignore[invalid-argument-type] - the point
    ],
    ids=["source", "assign", "claim"],
)
def test_only_flows_and_spreads_take_a_nested_state(make_lattice, rng, intent):
    with pytest.raises(TypeError, match="only Flow and Spread move a nested state's amounts"):
        resolve(settled(make_lattice(), rng), intent())


def test_a_nested_state_with_nothing_to_move_cannot_be_moved(make_lattice):
    marks = type("Marks", (NestedState,), {"flag": Field(bool), "level": Field(np.float64)})
    cls = fields(marks=Nested(marks))
    space = make_lattice()
    with pytest.raises(ValueError, match="no dynamic, extensive, floating field to move"):
        resolve(cls(space), Spread(cls.marks, Adjacency().compile(space), 0.5))


def test_a_nested_state_as_declared_names_no_fields_to_move(make_lattice, rng):
    space = make_lattice()
    with pytest.raises(TypeError, match="nested state as declared"):
        resolve(settled(space, rng), Spread(Settlers, Adjacency().compile(space), 0.5))


def test_writes_of_a_nested_state_stand_for_its_dynamic_fields(make_lattice, rng):
    space = make_lattice()
    state = settled(space, rng)
    spread = Spread(WorldState.settlers, Adjacency().compile(space), 0.5)
    resolve(state, spread, writes=WorldState.settlers)
    with pytest.raises(ValueError, match="'water' is not among the fields the phase writes"):
        resolve(state, Source(WorldState.water, 1.0), writes=WorldState.settlers)


def planet(space, rng):
    """A `PlanetState` with random amounts in every field, rain included."""
    return PlanetState(space, {field: rng.random(space.n_cells) for field in PlanetState(space)})


EUROPE = PlanetState.europe.north
EUROPEAN = (
    EUROPE.rabbits.population,
    EUROPE.rabbits.eaten,
    EUROPE.foxes.population,
    EUROPE.foxes.eaten,
)


def test_a_spread_of_a_nested_state_moves_the_amounts_of_those_nested_in_it(make_lattice, rng):
    space = make_lattice()
    op = Adjacency().compile(space)
    state = planet(space, rng)
    leave, toward = rng.random(space.n_cells), rng.random(space.n_cells)
    together = resolve(state, Spread(PlanetState.europe, op, leave, toward))
    apart = resolve(state, [Spread(field, op, leave, toward) for field in EUROPEAN])
    for field in EUROPEAN:
        assert together[field].tobytes() == apart[field].tobytes()
        assert not np.array_equal(together[field], state[field])
    for field in state:
        if field not in EUROPEAN:
            assert together[field] is state[field]  # rain, grass and the other biomes stay


def test_a_flow_of_a_nested_state_at_depth_moves_only_its_own_amounts(make_lattice, rng):
    space = make_lattice()
    op = Adjacency().compile(space)
    state = planet(space, rng)
    rate = rng.random(op.n_edges) / 4
    moved = resolve(state, Flow(EUROPE.foxes, op, rate))
    for field in state:
        changed = field in (EUROPE.foxes.population, EUROPE.foxes.eaten)
        assert (moved[field] is not state[field]) == changed


def test_a_nested_state_nested_only_in_one_as_declared_names_no_fields_to_move(make_lattice, rng):
    space = make_lattice()
    with pytest.raises(TypeError, match="name it through a state class"):
        resolve(planet(space, rng), Spread(Biome.foxes, Adjacency().compile(space), 0.5))


def test_writes_of_a_nested_state_cover_the_states_nested_in_it(make_lattice, rng):
    space = make_lattice()
    state = planet(space, rng)
    spread = Spread(EUROPE.foxes, Adjacency().compile(space), 0.5)
    resolve(state, spread, writes=PlanetState.europe)
    with pytest.raises(ValueError, match=r"'north\.foxes\.population' is not among the fields"):
        resolve(state, Source(PlanetState.north.foxes.population, 1.0), writes=PlanetState.europe)
