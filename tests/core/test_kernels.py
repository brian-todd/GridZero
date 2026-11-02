"""The Numba fast paths give the same results as the NumPy/SciPy reference, bit for bit.

Each test runs an operation under both implementations and compares the outcomes: equal
arrays of equal dtype and shape, or the same error. Inputs are chosen to tell implementations
apart: directed operators, weights of both signs, cells without neighbors, every numeric dtype
at its limits, NaN, infinities, subnormals, and component counts that use each block of the
kernels (1, 2, 3, 4, then 4 + 1).
"""

import functools
import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike

from gridzero.core import (
    Adjacency,
    Ball,
    CellGraph,
    Claim,
    Field,
    Flow,
    Hops,
    LatticeSpace,
    NeighborAllocation,
    NeighborChoice,
    NeighborOperator,
    Simulation,
    Source,
    Spread,
    State,
    gather,
    icosphere,
    implementation,
    path,
    phase,
    pull,
    push,
    reach,
    resolve,
    sight,
    steps,
    visible,
)

NUMERIC_DTYPES: list[type[np.generic]] = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
]
NUMERIC_DTYPES += [np.uint32, np.uint64, np.float16, np.float32, np.float64]

COMPONENTS = [(), (1,), (2,), (3,), (4,), (5,), (2, 3), (0,)]
"""Component shapes: one of each block size of the row kernels, a remainder, and none."""


def outcome(impl, fn, *args, **kwargs) -> Any:
    """Return what ``fn`` gives under ``impl``: its result, or the type and text of its error.

    NumPy's floating-point warnings are off: an overflow in a weighted ``"max"`` warns on the
    reference path and not in Numba, but both give inf. Code that sets its own ``errstate``,
    such as the resolver's overflow check, still does.
    """
    with implementation(impl), np.errstate(all="ignore"):
        try:
            return fn(*args, **kwargs)
        except Exception as error:  # compared, not swallowed
            return type(error), str(error)


def assert_same(fn, *args, **kwargs) -> Any:
    """Assert that both implementations give the same outcome, and return it."""
    expected = outcome("reference", fn, *args, **kwargs)
    actual = outcome("numba", fn, *args, **kwargs)
    assert_identical(expected, actual)
    return actual


def assert_identical(expected, actual):
    if isinstance(expected, tuple) and expected and isinstance(expected[0], type):
        assert actual == expected
    elif isinstance(expected, State):
        assert isinstance(actual, State)
        assert list(actual) == list(expected)
        for field in expected:
            assert_identical(expected[field], actual[field])
    else:
        assert isinstance(actual, np.ndarray), actual
        assert isinstance(expected, np.ndarray), expected
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
        np.testing.assert_array_equal(actual, expected)  # NaN matches NaN, and 0.0 -0.0
        if actual.dtype.kind in "fc":
            np.testing.assert_array_equal(float_bits(actual), float_bits(expected))


def float_bits(array):
    """The bits of each float, or of each part of a complex number, with every NaN alike.

    Equal floats can differ in the sign of a zero, which ``assert_array_equal`` ignores. NaN
    payloads are not compared: which NaN a sum keeps depends on the order of its operands.
    """
    size = array.dtype.itemsize // (2 if array.dtype.kind == "c" else 1)
    parts = np.ascontiguousarray(array).view(f"f{size}")
    return np.where(np.isnan(parts), np.nan, parts).view(f"u{size}")


def directed(n_cells, width, rng, weighted=True):
    """A random directed operator: each cell has up to ``width`` random targets, some unused."""
    idx = np.full((n_cells, width), -1)
    for i in range(n_cells):
        k = rng.integers(0, width + 1)
        idx[i, :k] = rng.choice(n_cells, size=k, replace=False)
    weight = rng.normal(size=idx.shape) if weighted else None
    return NeighborOperator.from_table(idx, weight)


@functools.cache  # operators are immutable, so tests can share them
def make_operator(name):
    rng = np.random.default_rng(7)
    planet = icosphere(2)
    if name == "planet":
        return Adjacency().compile(planet)
    if name == "planet-boundary":
        return Adjacency(weight="boundary").compile(planet)
    if name == "planet-hops":
        return Hops(3).compile(planet)
    if name == "directed-signed":
        return directed(30, 6, rng)
    if name == "directed-unweighted":
        return directed(30, 6, rng, weighted=False)
    if name == "lattice-fixed":
        return Adjacency("vertex", weight="distance").compile(LatticeSpace((4, 5), periodic=False))
    if name == "lattice-ball":
        return Ball(2.5).compile(LatticeSpace((7, 7)))
    if name == "no-edges":
        return NeighborOperator(CellGraph(np.zeros(6, dtype=np.int64), []))
    if name == "no-cells":
        return NeighborOperator(CellGraph([0], []))
    raise KeyError(name)


OPERATORS = [
    "planet",
    "planet-boundary",
    "planet-hops",
    "directed-signed",
    "directed-unweighted",
    "lattice-fixed",
    "lattice-ball",
    "no-edges",
    "no-cells",
]


def values(dtype: DTypeLike, shape, rng):
    """Values of ``dtype``: random, plus the dtype's limits, and for floats NaN, +-inf, -0.0
    and the smallest subnormal, each in about three random places.

    Random places matter: a NaN only in cell 0 is always a row's first neighbor, where even a
    max that drops NaN keeps it.
    """
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return rng.random(shape) < 0.5
    if dtype.kind == "f":
        x = rng.normal(scale=100.0, size=shape).astype(dtype)
        info = np.finfo(dtype)
        specials = [np.nan, np.inf, -np.inf, -0.0, info.smallest_subnormal, info.max, info.min]
    else:
        info = np.iinfo(dtype)
        x = rng.integers(max(info.min, -1000), min(info.max, 1000), size=shape, dtype=dtype)
        specials = [info.min, info.max, 0]
    flat = x.reshape(-1)
    spots = rng.choice(flat.size, size=min(flat.size, 3 * len(specials)), replace=False)
    flat[spots] = np.resize(np.array(specials, dtype=dtype), spots.size)
    return x


def small(dtype, shape, rng):
    """Values of ``dtype`` from -2 to 2 (0 to 2 if unsigned), whose sums never overflow."""
    low = 0 if np.dtype(dtype).kind in "bu" else -2
    return rng.integers(low, 3, size=shape).astype(dtype)


@pytest.fixture(params=OPERATORS)
def op(request):
    return make_operator(request.param)


# ---------------------------------------------------------------------------------------------
# Operators.


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
def test_pull_matches_the_reference_for_every_dtype_and_component_count(op, dtype, reduce):
    rng = np.random.default_rng(1)
    for shape in COMPONENTS:
        x = values(dtype, (*shape, op.n_cells), rng)
        assert_same(pull, x, op, reduce)
        # Small values: no overflow error, so the arithmetic itself is compared.
        assert_same(pull, small(dtype, x.shape, rng), op, reduce)


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
def test_push_matches_the_reference_for_every_dtype(op, dtype):
    rng = np.random.default_rng(2)
    for shape in COMPONENTS:
        x = values(dtype, (*shape, op.n_cells), rng)
        assert_same(push, x, op)
        assert_same(push, small(dtype, x.shape, rng), op)


@pytest.mark.parametrize(
    ("reduce", "fill"),
    [("mean", np.nan), ("mean", -2.5), ("max", np.nan), ("max", 7), ("min", -np.inf)],
)
def test_pull_fills_cells_without_neighbors_as_the_reference_does(op, reduce, fill):
    rng = np.random.default_rng(3)
    for dtype in (np.float32, np.float64, np.int16):
        x = values(dtype, (2, op.n_cells), rng)
        assert_same(pull, x, op, reduce, fill=fill)


def test_pull_matches_the_reference_on_views_that_are_not_contiguous():
    op = make_operator("planet-boundary")
    rng = np.random.default_rng(4)
    table = rng.normal(size=(op.n_cells, 6))
    for x in (table.T, table.T[::2], np.broadcast_to(table[:, 0], (5, op.n_cells))):
        assert not x.flags.c_contiguous
        for reduce in ("sum", "mean", "max", "min"):
            assert_same(pull, x, op, reduce)
        assert_same(push, x, op)


DTYPES_TO_GATHER = [*NUMERIC_DTYPES, np.complex64, np.complex128, "datetime64[s]", "U3"]


@pytest.mark.parametrize("dtype", DTYPES_TO_GATHER, ids=lambda d: np.dtype(d).name)
def test_gather_matches_the_reference_for_every_dtype(op, dtype):
    rng = np.random.default_rng(5)
    dtype = np.dtype(dtype)
    raw = rng.integers(0, 100, size=(2, 3, op.n_cells))
    x = raw.astype(dtype) if dtype.kind != "U" else raw.astype(str).astype(dtype)
    fill = {"M": np.datetime64(5, "s"), "U": "-"}.get(dtype.kind, 1)
    some = np.arange(op.n_cells)[::2][::-1]
    for cells in (None, some):
        assert_same(gather, x, op, cells, fill=fill)
        assert_same(gather, x[0, 0], op, cells, fill=fill)
    if dtype.kind == "f":
        assert_same(gather, x, op, fill=np.nan)


# ---------------------------------------------------------------------------------------------
# Intents.

PLANET = icosphere(2)


class StocksState(State):
    pop = Field(np.float64, (3,), extensive=True)
    grid = Field(np.float64, (2, 3), extensive=True)
    heat = Field(np.float32, extensive=True)
    food = Field(np.float64, extensive=True)
    fed = Field(np.float64, (3,), extensive=True)


class LivingState(State):
    pop = StocksState.pop
    food = StocksState.food
    eaten = Field(np.float64, (3,), extensive=True)


def planet_state(rng):
    n = PLANET.n_cells
    amounts = {
        StocksState.pop: rng.exponential(size=(3, n)),
        StocksState.grid: rng.exponential(size=(2, 3, n)),
        StocksState.heat: rng.exponential(size=n),
        StocksState.food: rng.exponential(size=n),
    }
    amounts[StocksState.pop][:, :5] = 0.0  # empty cells
    amounts[StocksState.pop][0, 5] = 5e-324  # a subnormal amount
    return StocksState(PLANET, amounts)


@pytest.fixture
def planet_op():
    return Adjacency(weight="boundary").compile(PLANET)


def flow_rates(op, rng):
    """Rates in every documented form, with negative rates and cells whose rates add past 1."""
    edges = op.n_edges
    per_edge = rng.normal(scale=0.4, size=(3, edges))
    per_edge[:, ::7] = -0.0
    return {
        "scalar": 0.1,
        "past-one": 0.4,
        "shared": per_edge[0],
        "per-component": per_edge,
        "column": np.array([[0.05], [0.3], [-0.1]]),
        "broadcast-one": per_edge[:1],
    }


@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize(
    "field", [StocksState.pop, StocksState.grid, StocksState.heat], ids=lambda f: f.name
)
def test_flows_resolve_as_the_reference_does(planet_op, field, domain, rng):
    state = planet_state(rng)
    inside = rng.random(PLANET.n_cells) < 0.6 if domain else None
    for name, rate in flow_rates(planet_op, rng).items():
        if field is StocksState.heat and np.ndim(rate) == 2:
            continue  # one component: no per-component rates
        flow = Flow(field, planet_op, rate, inside)
        # Alone, and with a removal, so that rationing decides how much each taker gets.
        for intents in (flow, [flow, Source(field, -0.7 * state[field])]):
            try:
                assert_same(resolve, state, intents)
            except AssertionError as error:
                error.add_note(f"rates: {name}")
                raise


def test_flows_along_directed_operators_resolve_as_the_reference_does(rng):
    space = LatticeSpace((6, 5))
    op = directed(space.n_cells, 5, rng, weighted=False)
    state = StocksState(space, {StocksState.pop: rng.exponential(size=(3, space.n_cells))})
    rate = rng.normal(scale=0.5, size=(3, op.n_edges))
    for inside in (None, rng.random(space.n_cells) < 0.5):
        assert_same(resolve, state, Flow(StocksState.pop, op, rate, inside))
        assert_same(
            resolve, state, Spread(StocksState.pop, op, 0.3, state[StocksState.pop][0], inside)
        )


def test_flow_errors_match_the_reference(planet_op, rng):
    state = planet_state(rng)
    huge = np.full(planet_op.n_edges, np.finfo(np.float64).max)
    assert isinstance(assert_same(resolve, state, Flow(StocksState.pop, planet_op, huge))[0], type)
    nan = np.full(planet_op.n_edges, np.nan)
    assert isinstance(assert_same(resolve, state, Flow(StocksState.pop, planet_op, nan))[0], type)


@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize("toward", ["none", "shared", "per-component", "faint"])
def test_spreads_resolve_as_the_reference_does(planet_op, toward, domain, rng):
    state = planet_state(rng)
    n = PLANET.n_cells
    attraction = {
        "none": None,
        "shared": rng.exponential(size=n),
        "per-component": rng.normal(size=(3, n)),
        "faint": np.where(rng.random(n) < 0.5, 1e-200, 1.0),
    }[toward]
    inside = rng.random(n) < 0.6 if domain else None
    for leave in (0.2, rng.random((3, n)), 1.5):
        spread = Spread(StocksState.pop, planet_op, leave, attraction, inside)
        assert_same(resolve, state, spread)
        assert_same(
            resolve, state, [spread, Source(StocksState.pop, -0.9 * state[StocksState.pop])]
        )


def test_sources_and_claims_ration_as_the_reference_does(rng):
    state = planet_state(rng)
    n = PLANET.n_cells
    intents = [
        Source(StocksState.food, rng.normal(size=n)),
        Claim(StocksState.food, rng.exponential(scale=0.6, size=(3, n)), into=StocksState.fed),
        Source(StocksState.pop, -rng.exponential(size=(3, n))),
        Source(StocksState.pop, -0.5),
        Source(StocksState.heat, rng.normal(size=n)),
    ]
    assert_same(resolve, state, intents)
    for intent in intents:
        assert_same(resolve, state, intent)


# ---------------------------------------------------------------------------------------------
# Action spaces.


def slot_weights(space, rng, shape=(3,)):
    weights = rng.normal(size=(*shape, *space.action_shape[-2:]))
    weights[..., :4, :] = 0.0  # cells that propose nothing
    weights[..., 4, :] = np.finfo(np.float64).max  # a row whose total overflows
    weights[..., 5, :] = 5e-324  # subnormal weights
    weights[..., 6, 1:] = np.finfo(np.float64).max / 3  # overflow only once added up
    weights[..., 7, :] = -0.0  # negative zeros count as 0, as every negative weight does
    weights[..., 8, ::2] = -0.0
    return weights


@pytest.mark.parametrize("domain", [False, True])
def test_allocations_match_the_reference(planet_op, domain, rng):
    inside = rng.random(PLANET.n_cells) < 0.6 if domain else None
    for shape in ((), (3,), (2, 3), (0,)):
        space = NeighborAllocation(planet_op, shape=shape)
        weights = slot_weights(space, rng, shape)
        assert_same(lambda w, s=space: s.fractions(w, inside), weights)
        assert_same(lambda w, s=space: s.flow(StocksState.pop, w, inside).rate, weights)
        assert_same(lambda s=space: s.sample(np.random.default_rng(3), inside))
    space = NeighborAllocation(planet_op, shape=(3,))
    state, weights = planet_state(rng), slot_weights(space, rng)
    assert_same(lambda: resolve(state, space.flow(StocksState.pop, weights, inside)))


@pytest.mark.parametrize("make", [NeighborAllocation, NeighborChoice])
def test_masks_match_the_reference(planet_op, make, rng):
    space = make(planet_op)
    for share in (0.0, 0.5, 1.0):
        inside = rng.random(PLANET.n_cells) < share
        x = rng.normal(size=(2, PLANET.n_cells))
        assert_same(lambda d=inside: space.mask(d))
        assert_same(lambda d=inside, x=x: space.values(x, -1.0, d))


def scores(dtype: DTypeLike, shape, rng):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        x = rng.normal(size=shape)
        x[..., 0, :] = -np.inf  # every slot at the lowest score: stay wins the tie
        x[..., 1, 2:] = x[..., 1, 1:2]  # ties between neighbors: the lowest slot wins
        x[..., 2, :] = np.where(np.arange(shape[-1]) % 2, -0.0, 0.0)  # 0.0 ties -0.0
        return x.astype(dtype)
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    x[..., 0, :] = info.min
    x[..., 1, :] = info.max
    x[..., 2, :] = rng.integers(0, 2, size=shape[-1])  # many ties
    if dtype == np.int64:
        x[..., 3, :] = 2**53 + np.arange(shape[-1])  # beyond what float64 tells apart
    return x


@pytest.mark.parametrize("domain", [False, True])
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int8, np.uint64], ids=lambda d: np.dtype(d).name
)
def test_choices_match_the_reference(planet_op, dtype, domain, rng):
    inside = rng.random(PLANET.n_cells) < 0.6 if domain else None
    space = NeighborChoice(planet_op, shape=(3,))
    picks = assert_same(
        lambda x: space.best(x, inside), scores(dtype, (3, *space.valid.shape), rng)
    )
    for rate in (0.5, rng.normal(size=(3, PLANET.n_cells)), rng.random(PLANET.n_cells)):
        assert_same(lambda r=rate: space.flow(StocksState.pop, picks, r, inside).rate)
    every_slot = rng.integers(0, space.n_slots, size=(2, 3, PLANET.n_cells))  # invalid ones too
    assert_same(lambda: space.flow(StocksState.grid, every_slot, domain=inside).rate)
    assert_same(lambda: space.flow(StocksState.grid, every_slot[:0], domain=inside).rate)
    assert_same(lambda: space.sample(np.random.default_rng(3), inside))


# ---------------------------------------------------------------------------------------------
# Whole models.


def test_a_model_runs_identically_on_both_paths():
    planet = icosphere(4)
    land = np.random.default_rng(0).random(planet.n_cells) < 0.7

    @phase(writes=(StocksState.pop, StocksState.food, LivingState.eaten))
    def live(s, ctx):
        pop = s[StocksState.pop]
        return [
            Source(StocksState.food, 0.1 * planet.measures),
            Claim(StocksState.food, 0.3 * pop, into=LivingState.eaten),
            Source(StocksState.pop, 0.05 * pop - 0.02 * pop**2 / planet.measures),
        ]

    @phase(writes=StocksState.pop, hoods={"near": Adjacency(weight="boundary")})
    def spread(s, ctx):
        return Spread(StocksState.pop, ctx.hoods["near"], 0.1, s[StocksState.food] + 1e-3, land)

    @phase(writes=StocksState.pop, hoods={"near": Adjacency()})
    def forage(s, ctx):
        choice = NeighborChoice(ctx.hoods["near"], shape=(3,))
        smell = choice.values(s[StocksState.food], domain=land)
        best = choice.best(np.broadcast_to(smell, (3, *smell.shape)), land)
        return choice.flow(StocksState.pop, best, 0.3, land)

    @phase(writes=StocksState.pop, hoods={"near": Hops(2)})
    def wander(s, ctx):
        allocation = NeighborAllocation(ctx.hoods["near"], shape=(3,))
        return allocation.flow(StocksState.pop, allocation.sample(ctx.rng))

    @phase(writes=LivingState.eaten, hoods={"near": Adjacency()}, every=3)
    def share(s, ctx):
        return Flow(LivingState.eaten, ctx.hoods["near"], 0.05)

    pop = np.where(land, 1.0, 0.0) * planet.measures
    state = LivingState(planet, {LivingState.pop: np.stack([pop, 0.5 * pop, 0.1 * pop])})

    def run(impl):
        with implementation(impl):
            sim = Simulation(state, [live, spread, forage, wander, share], seed=11)
            sim.run(30)
            return sim.state

    assert_identical(run("reference"), run("numba"))


PATH_SPACES = {
    "hex": lambda: LatticeSpace((40, 30), tiling="hex", periodic=(True, False)),
    "squares": lambda: LatticeSpace((30, 30), periodic=False),
    "triangles": lambda: LatticeSpace((30, 20), tiling="triangle"),
    "icosphere": lambda: icosphere(8),
}

PATH_COSTS = {
    "unit": lambda n, rng: None,
    "uniform": lambda n, rng: rng.uniform(0.1, 3.0, n),
    "integers": lambda n, rng: rng.integers(1, 4, n).astype(float),  # many ties
    "walls": lambda n, rng: np.where(rng.random(n) < 0.3, np.inf, rng.uniform(1, 2, n)),
    "extremes": lambda n, rng: rng.choice([1e-300, 5e-324, 1.0, 1e300, 1e308], n),
}

PATH_WALLS = {  # ``blocked``: unit costs search breadth first around them, others by Dijkstra
    "open": lambda n, rng: None,
    "no walls": lambda n, rng: np.zeros(n, dtype=bool),
    "walls": lambda n, rng: rng.random(n) < 0.3,
}


def same_bits(fn, *args, **kwargs):
    """Assert that both paths give the same outcome, zeros' and infinities' bits included."""
    actual = assert_same(fn, *args, **kwargs)
    expected = outcome("reference", fn, *args, **kwargs)
    if isinstance(actual, np.ndarray) and actual.dtype.kind == "f":
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    return actual


@pytest.mark.parametrize("space", PATH_SPACES)
@pytest.mark.parametrize("costs", PATH_COSTS)
@pytest.mark.parametrize("walls", PATH_WALLS)
@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_steps_match_the_reference_bit_for_bit(space, costs, walls, kind, rng):
    space = PATH_SPACES[space]()
    cost = PATH_COSTS[costs](space.n_cells, rng)
    blocked = PATH_WALLS[walls](space.n_cells, rng)
    sources = rng.choice(space.n_cells, 3, replace=False)  # some on walls: they start there
    hood = {"kind": kind, "cost": cost, "blocked": blocked}
    for limit in (math.inf, 0.0, 1.0, 2.5, 7.0):
        same_bits(steps, space, sources[0], limit=limit, **hood)
    same_bits(steps, space, sources, **hood)
    same_bits(steps, space, sources, separate=True, limit=4.0, **hood)
    same_bits(reach, space, sources, 3.0, **hood)


@pytest.mark.parametrize("space", PATH_SPACES)
@pytest.mark.parametrize("costs", PATH_COSTS)
@pytest.mark.parametrize("walls", PATH_WALLS)
@pytest.mark.parametrize("kind", ["facet", "vertex"])
def test_paths_match_the_reference(space, costs, walls, kind, rng):
    space = PATH_SPACES[space]()
    cost = PATH_COSTS[costs](space.n_cells, rng)
    blocked = PATH_WALLS[walls](space.n_cells, rng)
    for start, goal in rng.integers(0, space.n_cells, (10, 2)):  # on walls too
        same_bits(path, space, start, goal, kind=kind, cost=cost, blocked=blocked)


def test_path_errors_match_the_reference():
    space = LatticeSpace((3,), periodic=False)
    error = same_bits(path, space, 0, 2, cost=[1.0, 1e20, 1e-10])
    assert error[0] is ValueError


SIGHT_BOARDS = {
    "squares": lambda: LatticeSpace((40, 30), periodic=False),
    "squares-wrapped": lambda: LatticeSpace((40, 30)),
    "hex": lambda: LatticeSpace((30, 40), tiling="hex", periodic=(True, False)),
    "hex-wrapped": lambda: LatticeSpace((30, 40), tiling="hex"),
    "triangles": lambda: LatticeSpace((40, 30), tiling="triangle", periodic=False),
    "triangles-wrapped": lambda: LatticeSpace((40, 30), tiling="triangle"),
}


@pytest.mark.parametrize("board", SIGHT_BOARDS)
@pytest.mark.parametrize("density", [0.0, 0.15, 0.4])
def test_sight_matches_the_reference(board, density, rng):
    space = SIGHT_BOARDS[board]()
    walls = rng.random(space.n_cells) < density
    for viewer in rng.integers(0, space.n_cells, 3):
        same_bits(sight, space, viewer, walls, radius=6.5)
    viewers = rng.integers(0, space.n_cells, 8)
    for radius in (0.0, 1.0, 2.5, 6.5, 14.5):  # 14.5 reaches round the shorter wrapped axes
        same_bits(sight, space, viewers, walls, radius=radius, separate=True)
    for a, b in rng.integers(0, space.n_cells, (40, 2)):  # long lines, across wrapped edges
        same_bits(lambda *args: np.array(visible(*args)), space, a, b, walls)
