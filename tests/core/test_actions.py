import numpy as np
import pytest

from gridzero.core import (
    ActionSpace,
    Adjacency,
    Box,
    Field,
    LatticeSpace,
    NeighborAllocation,
    NeighborChoice,
    NeighborOperator,
    State,
    icosphere,
    resolve,
)

pytestmark = pytest.mark.usefixtures("each_impl")  # every test on both paths

RING = LatticeSpace((5,))
RING_OP = Adjacency().compile(RING)
"""Five cells in a ring: cell i's slots are stay, then its neighbors in id order."""


class AmountsState(State):
    pop = Field(np.float64, extensive=True)


class GroupsState(State):
    pop = Field(np.float64, (2,), extensive=True)


def ring_state(pop):
    return AmountsState(RING, {AmountsState.pop: pop})


def point(value, cell=0, n=5):
    return np.where(np.arange(n) == cell, float(value), 0.0)


def directed(rng, n=92, k=3):
    """An operator on icosphere(3) where each cell has k random targets, unlike its transpose.

    Also returns each cell's targets in ascending order: its slots 1 to k.
    """
    idx = np.argsort(rng.random((n, n)) + np.eye(n), axis=1)[:, :k]
    op = NeighborOperator.from_table(idx)
    assert not op.graph.is_symmetric()
    return op, np.sort(idx, axis=1)


SPACES = [NeighborAllocation, NeighborChoice]


def test_action_spaces_satisfy_the_protocol():
    for space in (NeighborAllocation(RING_OP), NeighborChoice(RING_OP), Box(5, 0.0, 1.0)):
        assert isinstance(space, ActionSpace)


# Slots.


@pytest.mark.parametrize("make", SPACES)
def test_slots_are_stay_then_the_neighbors_in_id_order(make):
    space = make(RING_OP)
    np.testing.assert_array_equal(
        space.targets, [[0, 1, 4], [1, 0, 2], [2, 1, 3], [3, 2, 4], [4, 0, 3]]
    )
    assert space.valid.all()
    assert space.n_slots == 3


@pytest.mark.parametrize("make", SPACES)
def test_every_cell_gets_the_same_slots_and_smaller_cells_leave_some_invalid(make):
    planet = icosphere(2)
    space = make(Adjacency().compile(planet))
    degree = planet.adjacency().degree
    assert space.n_slots == 7
    np.testing.assert_array_equal(space.valid.sum(axis=1), degree + 1)
    np.testing.assert_array_equal(space.targets[degree == 5, 6], -1)


@pytest.mark.parametrize("make", SPACES)
def test_slots_of_neighbors_outside_the_domain_are_masked(make):
    space = make(RING_OP)
    mask = space.mask(np.arange(5) != 4)
    np.testing.assert_array_equal(mask[[0, 3]], [[True, True, False], [True, True, False]])
    assert mask[[1, 2, 4]].all()  # staying is valid even in a cell outside the domain
    assert space.valid.all()  # the slots that exist, whatever the domain
    assert space.mask() is space.valid


@pytest.mark.parametrize("make", SPACES)
def test_the_mask_follows_the_domain_of_each_call(make, rng):
    op = Adjacency().compile(icosphere(2))
    space = make(op)
    for _ in range(3):
        domain = rng.random(op.n_cells) < 0.5
        inside = np.where(space.targets >= 0, domain[space.targets], False)
        inside[:, 0] = True
        np.testing.assert_array_equal(space.mask(domain), space.valid & inside)


@pytest.mark.parametrize("make", SPACES)
@pytest.mark.parametrize(
    ("domain", "error"),
    [(np.ones(5), TypeError), (np.ones(4, bool), ValueError)],
    ids=["floats", "too-short"],
)
def test_a_domain_is_one_flag_per_cell(make, domain, error):
    with pytest.raises(error, match="domain must"):
        make(RING_OP).mask(domain)


@pytest.mark.parametrize("make", SPACES)
def test_values_are_each_slots_cell_and_fill_elsewhere(make):
    op = Adjacency().compile(icosphere(2))
    space = make(op)
    x = np.arange(op.n_cells, dtype=np.float64) + 0.5
    values = space.values(x, fill=-1.0)
    np.testing.assert_array_equal(values[space.valid], x[space.targets[space.valid]])
    assert (values[~space.valid] == -1.0).all()
    domain = np.arange(op.n_cells) % 3 != 0
    masked = space.values(x, fill=-1.0, domain=domain)
    np.testing.assert_array_equal(masked == -1.0, ~space.mask(domain))


@pytest.mark.parametrize("make", SPACES)
def test_values_carry_leading_axes_and_keep_the_dtype(make):
    space = make(RING_OP)
    x = np.arange(10, dtype=np.int16).reshape(2, 5)
    values = space.values(x)
    assert values.shape == (2, 5, 3)
    assert values.dtype == np.int16
    np.testing.assert_array_equal(values[1, :, 0], x[1])


@pytest.mark.parametrize("make", SPACES)
def test_values_need_a_fill_that_fits(make):
    with pytest.raises(ValueError, match="does not fit"):
        make(RING_OP).values(np.arange(5, dtype=np.uint8), fill=-1)


@pytest.mark.parametrize("make", SPACES)
def test_action_spaces_need_a_compiled_operator(make):
    with pytest.raises(TypeError, match="op must be a NeighborOperator"):
        make(Adjacency())


@pytest.mark.parametrize("make", [*SPACES, lambda op, shape: Box(5, 0.0, 1.0, shape=shape)])
def test_component_shapes_are_integers(make, not_an_integer):
    with pytest.raises(TypeError, match="each shape entry must be an integer"):
        make(RING_OP, shape=(not_an_integer,))


# NeighborAllocation.


def test_allocation_normalizes_weights_over_the_valid_slots():
    space = NeighborAllocation(RING_OP)
    weights = np.array([[1.0, 1.0, 5.0], [2.0, -1.0, 2.0], [0.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0]])
    fractions = space.fractions(weights, domain=np.arange(5) != 4)
    np.testing.assert_allclose(fractions[0], [0.5, 0.5, 0.0], rtol=1e-15)  # slot 2 is cell 4
    np.testing.assert_allclose(fractions[1], [0.5, 0.0, 0.5], rtol=1e-15)  # negative counts as 0
    np.testing.assert_array_equal(fractions[2], [1.0, 0.0, 0.0])  # nothing proposed: stay


def test_allocation_moves_each_fraction_to_its_neighbor():
    space = NeighborAllocation(RING_OP)
    weights = np.zeros((5, 3))
    weights[0] = [2.0, 1.0, 1.0]
    new = resolve(ring_state(point(8)), space.flow(AmountsState.pop, weights))
    np.testing.assert_allclose(new[AmountsState.pop], [4.0, 2.0, 0.0, 0.0, 2.0], rtol=1e-15)


def test_allocation_flow_never_enters_cells_outside_the_domain():
    # Cells 0 and 2 split between staying and their other neighbor. Cell 1 keeps a third.
    space = NeighborAllocation(RING_OP)
    flow = space.flow(AmountsState.pop, np.ones((5, 3)), domain=np.arange(5) != 1)
    new = resolve(ring_state(np.full(5, 3.0)), flow)
    np.testing.assert_allclose(new[AmountsState.pop], [3.5, 1.0, 3.5, 3.5, 3.5], rtol=1e-15)


def test_allocation_flow_rates_are_the_fractions_of_each_edges_slot(any_space, rng):
    op = Adjacency().compile(any_space)
    space = NeighborAllocation(op, shape=(2,))
    weights = rng.uniform(-0.5, 1.0, space.action_shape)
    rates = space.flow(AmountsState.pop, weights).rate
    fractions = space.fractions(weights)
    slot = np.arange(op.n_edges) - op.graph.indptr[op.graph.sources] + 1
    np.testing.assert_array_equal(rates, fractions[:, op.graph.sources, slot])


def test_allocation_flow_follows_the_edges_of_a_directed_neighborhood(rng):
    op, targets = directed(rng)
    space = NeighborAllocation(op)
    np.testing.assert_array_equal(space.targets[:, 1:], targets)
    pop = rng.uniform(0, 2, op.n_cells)
    fractions = space.fractions(rng.random(space.action_shape))
    new = resolve(
        AmountsState(icosphere(3), {AmountsState.pop: pop}), space.flow(AmountsState.pop, fractions)
    )
    received = np.bincount(targets.ravel(), (pop[:, None] * fractions[:, 1:]).ravel(), 92)
    np.testing.assert_allclose(new[AmountsState.pop], pop * fractions[:, 0] + received, rtol=1e-12)


def test_allocation_flows_conserve_the_total(any_space, rng):
    op = Adjacency("vertex").compile(any_space)
    space = NeighborAllocation(op, shape=(2,))
    state = GroupsState(any_space)
    state = state.replace({GroupsState.pop: rng.uniform(0, 2, (2, any_space.n_cells))})
    new = resolve(state, space.flow(GroupsState.pop, space.sample(rng)))
    np.testing.assert_allclose(new.total(GroupsState.pop), state.total(GroupsState.pop), rtol=1e-12)
    assert new[GroupsState.pop].min() >= 0


def test_allocation_samples_are_valid_fractions_and_follow_the_seed():
    space = NeighborAllocation(Adjacency().compile(icosphere(2)), shape=(3,))
    sample = space.sample(np.random.default_rng(1))
    assert sample.shape == space.action_shape == (3, 42, 7)
    assert space.contains(sample)
    np.testing.assert_allclose(sample.sum(axis=-1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(sample[:, ~space.valid], 0.0)
    np.testing.assert_array_equal(space.sample(np.random.default_rng(1)), sample)
    assert not np.array_equal(space.sample(np.random.default_rng(2)), sample)


@pytest.mark.parametrize(
    "action",
    [np.ones((5, 2)), np.ones((2, 5, 3)), -np.ones((5, 3)), np.full((5, 3), np.nan)],
    ids=["too-few-slots", "extra-axis", "negative", "nan"],
)
def test_allocation_contains_only_finite_non_negative_weights_of_its_shape(action):
    assert NeighborAllocation(RING_OP).contains(np.ones((5, 3)))
    assert not NeighborAllocation(RING_OP).contains(action)


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([5e-324, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1e-310, 1e-310], [0.0, 0.5, 0.5]),
        ([1e308, 1e308, 0.0], [0.5, 0.5, 0.0]),
        ([np.finfo(np.float64).max] * 3, [1 / 3] * 3),
    ],
    ids=["smallest", "subnormal", "sum-overflows", "largest"],
)
def test_allocation_fractions_hold_at_the_limits_of_float64(weights, expected):
    fractions = NeighborAllocation(RING_OP).fractions(np.resize(weights, (5, 3)))
    np.testing.assert_allclose(fractions, np.resize(expected, (5, 3)), rtol=1e-15)


@pytest.mark.parametrize("scale", [1e-310, 1e-300, 1e300, np.finfo(np.float64).max])
def test_allocation_fractions_depend_only_on_the_ratios_of_the_weights(scale, rng):
    space = NeighborAllocation(Adjacency().compile(icosphere(2)), shape=(2,))
    weights = rng.random(space.action_shape)
    rtol = 1e-8 if scale < 1e-300 else 1e-15  # subnormal weights keep fewer bits: ~34 at 1e-313
    np.testing.assert_allclose(
        space.fractions(weights * scale), space.fractions(weights), rtol=rtol
    )


def test_allocation_rejects_weights_of_the_wrong_shape_or_not_finite():
    space = NeighborAllocation(RING_OP)
    with pytest.raises(ValueError, match=r"weights must have shape \(\.\.\., 5, 3\)"):
        space.fractions(np.ones((5, 2)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        space.fractions(np.full((5, 3), np.inf))
    with pytest.raises(TypeError, match="must hold numbers"):
        space.fractions(np.ones((5, 3), bool))


# NeighborChoice.


def test_choice_cells_are_the_chosen_neighbors_or_the_cell_itself():
    space = NeighborChoice(RING_OP)
    # Cell 0 picks cell 4, outside the domain, so it stays.
    cells = space.cells([2, 1, 2, 0, 1], domain=np.arange(5) != 4)
    np.testing.assert_array_equal(cells, [0, 0, 3, 3, 0])
    np.testing.assert_array_equal(space.cells([2, 1, 2, 0, 1]), [4, 0, 3, 3, 0])


def test_choice_flow_sends_the_rate_to_the_chosen_neighbor():
    space = NeighborChoice(RING_OP)
    pop = np.array([8.0, 8.0, 8.0, 0.0, 0.0])
    choice = [1, 0, 2, 1, 1]  # 0 -> 1, cell 1 stays, 2 -> 3
    new = resolve(
        ring_state(pop), space.flow(AmountsState.pop, choice, rate=[0.5, 1.0, 0.25, 1.0, 1.0])
    )
    np.testing.assert_allclose(new[AmountsState.pop], [4.0, 12.0, 6.0, 2.0, 0.0], rtol=1e-15)


def test_choice_flow_keeps_choices_outside_the_domain_at_home():
    space = NeighborChoice(RING_OP)
    domain = np.arange(5) != 4
    flow = space.flow(AmountsState.pop, [2, 0, 0, 0, 0], domain=domain)  # cell 0 picks cell 4
    np.testing.assert_array_equal(flow.domain, domain)
    np.testing.assert_array_equal(resolve(ring_state(point(8)), flow)[AmountsState.pop], point(8))


def test_choice_flow_follows_the_edges_of_a_directed_neighborhood(rng):
    op, targets = directed(rng)
    space = NeighborChoice(op)
    pop = rng.uniform(0, 2, op.n_cells)
    choice = space.sample(rng)
    new = resolve(
        AmountsState(icosphere(3), {AmountsState.pop: pop}),
        space.flow(AmountsState.pop, choice, 0.5),
    )
    moves = choice > 0
    chosen = targets[np.arange(op.n_cells), choice - 1][moves]
    expected = pop - 0.5 * pop * moves + np.bincount(chosen, 0.5 * pop[moves], op.n_cells)
    np.testing.assert_allclose(new[AmountsState.pop], expected, rtol=1e-12)


def test_choices_per_component_broadcast_against_the_field(rng):
    planet = icosphere(2)
    op = Adjacency().compile(planet)
    space = NeighborChoice(op, shape=(2,))
    state = GroupsState(planet, {GroupsState.pop: 1.0})
    choice = space.sample(rng)
    new = resolve(state, space.flow(GroupsState.pop, choice, rate=0.5))
    moved = choice > 0
    np.testing.assert_allclose(new.total(GroupsState.pop), state.total(GroupsState.pop), rtol=1e-12)
    for k in range(2):
        received = np.bincount(space.cells(choice)[k][moved[k]], minlength=42)
        np.testing.assert_allclose(
            new[GroupsState.pop][k], 1 - 0.5 * moved[k] + 0.5 * received, rtol=1e-12
        )


def test_actions_of_no_components_give_flows_without_rates(rng):
    allocation = NeighborAllocation(RING_OP, shape=(0,))
    choice = NeighborChoice(RING_OP, shape=(0,))
    assert np.shape(allocation.flow(AmountsState.pop, allocation.sample(rng)).rate) == (
        0,
        RING_OP.n_edges,
    )
    assert allocation.fractions(np.ones((0, 5, 3))).shape == (0, 5, 3)
    assert np.shape(choice.flow(AmountsState.pop, choice.sample(rng)).rate) == (0, RING_OP.n_edges)
    assert choice.best(np.ones((0, 5, 3))).shape == (0, 5)


def test_best_picks_the_highest_scoring_valid_slot_and_stays_on_ties():
    space = NeighborChoice(RING_OP)
    scores = np.array([[0, 1, 9], [3, 3, 1], [0, 0, 0], [0, -1, -2], [-np.inf, -np.inf, 5]])
    np.testing.assert_array_equal(space.best(scores, domain=np.arange(5) != 4), [1, 0, 0, 0, 2])
    np.testing.assert_array_equal(space.best(scores), [2, 0, 0, 0, 2])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_best_compares_integer_scores_exactly(dtype):
    # Cell 4 is outside the domain: it is slot 2 of cells 0 and 3.
    space = NeighborChoice(RING_OP)
    info = np.iinfo(dtype)
    scores = np.array(
        [
            [info.max - 1, info.max, info.max],  # the invalid slot ties the best
            [2**53, 2**53 + 1, 2**53],  # equal as float64
            [info.min] * 3,  # all lowest: stay
            [info.min, info.min, info.min + 1],  # only the invalid slot is higher: stay
            [0, 0, 1],
        ],
        dtype=dtype,
    )
    np.testing.assert_array_equal(space.best(scores, domain=np.arange(5) != 4), [1, 1, 0, 0, 2])


def test_best_rejects_nan_scores():
    with pytest.raises(ValueError, match="NaN"):
        NeighborChoice(RING_OP).best(np.full((5, 3), np.nan))


def test_choice_samples_are_uniform_over_the_valid_slots():
    space = NeighborChoice(Adjacency().compile(icosphere(2)), shape=(3000,))
    sample = space.sample(np.random.default_rng(0))
    assert space.contains(sample)
    pentagon = np.flatnonzero(~space.valid[:, 6])[0]
    counts = np.bincount(sample[:, pentagon], minlength=7)
    assert counts[6] == 0
    np.testing.assert_allclose(counts[:6] / 3000, 1 / 6, atol=0.03)


def test_choice_samples_follow_the_seed():
    space = NeighborChoice(RING_OP, shape=(4,))
    first = space.sample(np.random.default_rng(5))
    np.testing.assert_array_equal(space.sample(np.random.default_rng(5)), first)


@pytest.mark.parametrize(
    "action",
    [np.zeros(4, int), np.zeros(5), np.full(5, 3), np.array([2, 0, 0, 0, 0])],
    ids=["wrong-shape", "floats", "out-of-range", "outside-domain"],
)
def test_choice_contains_only_valid_slots(action):
    space = NeighborChoice(RING_OP)
    domain = np.arange(5) != 4
    assert space.contains(np.zeros(5, int), domain)
    assert not space.contains(action, domain)


def test_choice_rejects_slots_that_are_not_integers(not_cell_ids):
    values, error = not_cell_ids
    with pytest.raises(error):
        NeighborChoice(RING_OP).cells(np.resize(values, 5))


def test_choice_rejects_slots_out_of_range_or_of_the_wrong_shape():
    space = NeighborChoice(RING_OP)
    with pytest.raises(IndexError, match=r"slots must lie in \[0, 3\)"):
        space.cells([0, 0, 3, 0, 0])
    with pytest.raises(IndexError, match=r"slots must lie in \[0, 3\)"):
        space.flow(AmountsState.pop, [0, -1, 0, 0, 0])
    with pytest.raises(ValueError, match=r"choice must have shape \(\.\.\., 5\)"):
        space.cells([0, 0, 0])


# Box.


def test_box_clips_values_into_per_component_bounds():
    box = Box(3, 0.0, [[0.5], [2.0]], shape=(2,))
    clipped = box.clip([[-1.0, 0.25, 9.0], [-1.0, 1.0, 9.0]])
    np.testing.assert_array_equal(clipped, [[0.0, 0.25, 0.5], [0.0, 1.0, 2.0]])
    assert box.contains(clipped)


def test_box_samples_lie_within_per_cell_bounds_and_follow_the_seed():
    low, high = np.arange(4.0), np.arange(4.0) + 0.5
    box = Box(4, low, high)
    sample = box.sample(np.random.default_rng(3))
    assert box.contains(sample)
    assert np.all((sample >= low) & (sample <= high))
    np.testing.assert_array_equal(box.sample(np.random.default_rng(3)), sample)


@pytest.mark.parametrize("bound", [1e308, np.finfo(np.float64).max])
def test_box_samples_between_bounds_as_wide_as_float64_allows(bound):
    box = Box(10_000, -bound, bound)
    sample = box.sample(np.random.default_rng(0))
    assert box.contains(sample)
    assert abs((sample / bound).mean()) < 0.02  # uniform: centered, and spread out
    np.testing.assert_allclose((sample / bound).std(), 1 / np.sqrt(3), rtol=0.02)


@pytest.mark.parametrize("bound", [2.0, np.finfo(np.float64).max, -np.finfo(np.float64).max])
def test_a_box_with_equal_bounds_samples_the_bound(bound):
    np.testing.assert_array_equal(Box(3, bound, bound).sample(np.random.default_rng(0)), bound)


@pytest.mark.parametrize(
    ("low", "high", "match"),
    [
        (1.0, 0.0, "low must not exceed high"),
        (0.0, np.inf, "NaN or infinite"),
        (np.zeros(3), 1.0, r"low has shape \(3,\), which does not broadcast to \(4,\)"),
    ],
    ids=["low-above-high", "infinite", "wrong-shape"],
)
def test_box_rejects_bounds_that_are_not_finite_ordered_and_shaped(low, high, match):
    with pytest.raises(ValueError, match=match):
        Box(4, low, high)


def test_box_needs_an_integer_cell_count(not_an_integer):
    with pytest.raises(TypeError, match="n_cells must be an integer"):
        Box(not_an_integer, 0.0, 1.0)


@pytest.mark.parametrize(
    ("values", "error"),
    [(np.zeros(3), ValueError), (np.full(4, np.nan), ValueError), (np.zeros(4, bool), TypeError)],
    ids=["wrong-shape", "nan", "bools"],
)
def test_box_clip_rejects_values_of_the_wrong_shape_or_kind(values, error):
    with pytest.raises(error):
        Box(4, 0.0, 1.0).clip(values)


@pytest.mark.parametrize(
    "action", [np.zeros(3), np.full(4, 2.0), np.full(4, np.nan)], ids=["shape", "above", "nan"]
)
def test_box_contains_only_values_of_its_shape_within_bounds(action):
    assert Box(4, 0.0, 1.0).contains(np.full(4, 0.5))
    assert not Box(4, 0.0, 1.0).contains(action)
