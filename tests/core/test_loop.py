import dataclasses
import itertools
import os
import pickle
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from demo import DemoState, EcosystemState, PlanetState, Species, WorldState, fields
from gridzero.core import (
    Adjacency,
    Assign,
    Context,
    Field,
    LatticeSpace,
    NeighborOperator,
    Phase,
    Simulation,
    Source,
    Space,
    State,
    StateNeighborhood,
    icosphere,
    phase,
    pull,
    resolve,
)


def noop(state, ctx):
    return None


class CountersState(State):
    x = Field(np.int64)
    y = Field(np.int64)


class LifeState(State):
    alive = Field(bool)


class MarksState(State):
    marked = Field(bool)
    x = CountersState.x


def increment(field):
    """A phase that adds 1 to an integer field."""
    rule = lambda s, ctx: Assign(field, s[field] + 1)  # noqa: E731 - a one-line rule
    return Phase(rule, writes=field, name=f"increment_{field.name}")


@pytest.fixture
def counters(make_lattice):
    """A 3x3 lattice with two integer fields, ``x`` and ``y``, both zero."""
    return CountersState(make_lattice((3, 3)))


# Phase.


def test_phase_name_defaults_to_the_function_name():
    assert Phase(noop).name == "noop"


def test_phase_name_of_a_callable_object_defaults_to_its_class_name():
    class Grow:
        def __call__(self, state, ctx):
            return None

    assert Phase(Grow()).name == "Grow"
    assert Phase(Grow(), name="grow_fast").name == "grow_fast"


def test_phase_writes_one_field_or_several():
    assert Phase(noop, writes=CountersState.x).writes == (CountersState.x,)
    assert Phase(noop, writes=[CountersState.x, CountersState.y]).writes == (
        CountersState.x,
        CountersState.y,
    )
    assert phase(writes=(CountersState.x, CountersState.y))(noop).writes == (
        CountersState.x,
        CountersState.y,
    )


def test_phase_writes_are_fields_declared_on_a_state_class():
    for writes in ("x", ["x"], [CountersState.x, "y"], [Field(np.int64)], [3]):
        with pytest.raises(TypeError, match="writes must be fields declared on a State class"):
            Phase(noop, writes=writes)  # ty: ignore[invalid-argument-type]


def test_phase_decorator_makes_a_phase_that_calls_its_rule(counters):
    @phase(writes=CountersState.x, every=2, offset=1)
    def bump(state, ctx):
        return Assign(CountersState.x, state[CountersState.x] + 1)

    assert isinstance(bump, Phase)
    assert (bump.name, bump.writes, bump.every, bump.offset) == ("bump", (CountersState.x,), 2, 1)
    np.testing.assert_array_equal(
        resolve(counters, bump(counters, Context()))[CountersState.x], np.ones(9)
    )


@pytest.mark.parametrize(("every", "offset"), [(0, 0), (-1, 0), (2, 2), (2, -1)])
def test_phase_rejects_a_schedule_that_is_not_every_nth_step(every, offset):
    with pytest.raises(ValueError, match="every must be positive"):
        Phase(noop, every=every, offset=offset)


def test_phase_rejects_a_rule_that_is_not_callable():
    with pytest.raises(TypeError, match="callable"):
        Phase("noop")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("which", ["every", "offset"])
def test_phase_schedule_must_be_integers(which, not_an_integer):
    with pytest.raises(TypeError, match=f"{which} must be an integer"):
        Phase(noop, **{which: not_an_integer})


@pytest.mark.parametrize(
    ("name", "error"), [(5, TypeError), (b"grow", TypeError), ("", ValueError)]
)
def test_phase_name_must_be_a_non_empty_string(name, error):
    # The name keys the phase's random stream, so a bad one must fail here, not mid-run.
    with pytest.raises(error, match="phase name"):
        Phase(noop, name=name)


def test_phase_keeps_a_read_only_copy_of_its_neighborhoods():
    hoods = {"near": Adjacency()}
    rule = Phase(noop, hoods=hoods)
    hoods["far"] = Adjacency("vertex")
    assert list(rule.hoods) == ["near"]
    with pytest.raises(TypeError):
        rule.hoods["far"] = Adjacency("vertex")  # ty: ignore[invalid-assignment]


def test_the_simulation_runs_a_phase_by_calling_it(counters):
    # So a Phase subclass behaves the same when called directly and when run.
    class Doubled(Phase[CountersState]):
        def __call__(self, state, ctx):
            intent = self.fn(state, ctx)
            assert isinstance(intent, Assign)
            return Assign(CountersState.x, 2 * np.asarray(intent.values))

    rule = Doubled(
        lambda s, ctx: Assign(CountersState.x, s[CountersState.x] + 1),
        writes=CountersState.x,
        name="doubled",
    )
    assert Simulation(counters, [rule], seed=0).step()[CountersState.x][0] == 2


# Context.


def test_context_hoods_are_read_only(make_lattice):
    ctx = Context({"near": Adjacency().compile(make_lattice())})
    with pytest.raises(TypeError):
        ctx.hoods["far"] = ctx.hoods["near"]  # ty: ignore[invalid-assignment]


def test_context_streams_are_selected_by_seed_and_spawn_key():
    def draws(seed, spawn_key):
        return Context(seed=seed, spawn_key=spawn_key).rng.random(4)

    np.testing.assert_array_equal(draws(1, (2, 3)), draws(1, (2, 3)))
    assert not np.array_equal(draws(1, (2, 3)), draws(1, (2, 4)))
    assert not np.array_equal(draws(1, (2, 3)), draws(2, (2, 3)))


def test_context_rng_is_the_generator_of_its_seed_sequence():
    ctx = Context(seed=5, spawn_key=(1, 2))
    assert ctx.rng is ctx.rng
    assert ctx.seed.spawn_key == (1, 2)
    expected = np.random.default_rng(np.random.SeedSequence(5, spawn_key=(1, 2))).random(3)
    np.testing.assert_array_equal(ctx.rng.random(3), expected)


def test_context_drew_says_whether_its_stream_was_touched():
    untouched = Context(seed=5, spawn_key=(1, 2))
    assert not untouched.drew
    assert not untouched.drew  # asking doesn't touch it
    by_rng, by_seed = Context(seed=5), Context(seed=5)
    by_rng.rng.random()
    by_seed.seed.spawn(1)
    assert by_rng.drew
    assert by_seed.drew


# Building a simulation.


def test_simulation_rejects_a_state_that_is_not_a_state(counters):
    with pytest.raises(TypeError, match="State"):
        Simulation(dict(counters), [])  # ty: ignore[invalid-argument-type]


def test_simulation_rejects_rules_that_are_not_phases(counters):
    with pytest.raises(TypeError, match="wrap rules"):
        Simulation(counters, [noop])  # ty: ignore[invalid-argument-type]


def test_simulation_rejects_two_phases_with_one_name(counters):
    with pytest.raises(ValueError, match="two phases are named 'noop'"):
        Simulation(counters, [Phase(noop), Phase(noop)])


def test_simulation_rejects_a_phase_that_writes_a_static_field(make_state):
    with pytest.raises(ValueError, match="phase 'noop' writes static field 'terrain'"):
        Simulation(make_state(), [Phase(noop, writes=DemoState.terrain)])


def test_simulation_rejects_a_phase_that_writes_an_unknown_field(counters):
    food = fields(food=Field(np.float64)).food
    with pytest.raises(KeyError, match="phase 'noop' writes 'food': no field named 'food'"):
        Simulation(counters, [Phase(noop, writes=food)])


def test_simulation_rejects_a_neighborhood_without_compile(counters):
    with pytest.raises(TypeError, match="neighborhood 'near' has no compile method"):
        Simulation(counters, [Phase(noop, hoods={"near": "vertex"})])  # ty: ignore[invalid-argument-type]


@dataclass(frozen=True)
class Compiles:
    """A neighborhood that compiles to whatever it is given."""

    result: object

    def compile(self, space: Space) -> object:
        return self.result


def test_simulation_rejects_a_neighborhood_that_compiles_to_something_else(counters):
    with pytest.raises(TypeError, match="compiled to str, not a NeighborOperator"):
        Simulation(counters, [Phase(noop, hoods={"near": Compiles("operator")})])  # ty: ignore[invalid-argument-type]


def test_simulation_rejects_an_operator_on_a_different_number_of_cells(counters):
    small = NeighborOperator.from_table(np.zeros((4, 1), int))
    with pytest.raises(ValueError, match="operator on 4 cells, but the space has 9"):
        Simulation(counters, [Phase(noop, hoods={"near": Compiles(small)})])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        (-1, ValueError),
        (True, TypeError),
        (np.True_, TypeError),
        (1.5, TypeError),
        ("7", TypeError),
    ],
)
def test_simulation_rejects_a_seed_that_is_not_a_non_negative_integer(counters, seed, error):
    with pytest.raises(error, match=r"seed|integer"):
        Simulation(counters, [], seed=seed)


# Steps.


def test_each_phase_sees_what_the_earlier_phases_of_the_step_wrote(counters):
    copy_x = Phase(
        lambda s, ctx: Assign(CountersState.y, s[CountersState.x]),
        writes=CountersState.y,
        name="copy_x",
    )
    sim = Simulation(counters, [increment(CountersState.x), copy_x], seed=0)
    sim.step()
    assert (sim.state[CountersState.x][0], sim.state[CountersState.y][0]) == (1, 1)


def test_the_order_of_phases_is_part_of_the_model(counters):
    copy_x = Phase(
        lambda s, ctx: Assign(CountersState.y, s[CountersState.x]),
        writes=CountersState.y,
        name="copy_x",
    )
    sim = Simulation(counters, [copy_x, increment(CountersState.x)], seed=0)
    sim.step()
    assert (sim.state[CountersState.x][0], sim.state[CountersState.y][0]) == (1, 0)


def test_within_a_phase_every_read_sees_the_state_before_the_phase(counters):
    def swap(s, ctx):
        return [
            Assign(CountersState.x, s[CountersState.y]),
            Assign(CountersState.y, s[CountersState.x]),
        ]

    sim = Simulation(
        counters.replace({CountersState.x: 1, CountersState.y: 2}),
        [Phase(swap, writes=(CountersState.x, CountersState.y))],
        seed=0,
    )
    sim.step()
    assert (sim.state[CountersState.x][0], sim.state[CountersState.y][0]) == (2, 1)


def test_a_step_returns_the_new_state_and_counts_the_step(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    new = sim.step()
    assert new is sim.state
    assert new.step == 1
    assert counters.step == 0
    assert counters[CountersState.x][0] == 0


def test_a_failing_phase_leaves_the_simulation_at_the_start_of_the_step(counters):
    def fail(s, ctx):
        raise RuntimeError("boom")

    sim = Simulation(counters, [increment(CountersState.x), Phase(fail)], seed=0)
    with pytest.raises(RuntimeError, match="boom") as info:
        sim.step()
    assert sim.state is counters
    assert info.value.__notes__ == ["in phase 'fail' at step 0"]


def test_an_invalid_intent_is_reported_with_its_phase_and_step(counters):
    sneaky = Phase(lambda s, ctx: Assign(CountersState.y, 1), writes=CountersState.x, name="sneaky")
    sim = Simulation(counters.replace({}), [sneaky], seed=0)
    with pytest.raises(ValueError, match="'y' is not among the fields the phase writes") as info:
        sim.step()
    assert info.value.__notes__ == ["in phase 'sneaky' at step 0"]


def test_phases_run_on_the_steps_their_schedule_selects(counters):
    steps = []

    def record(s, ctx):
        steps.append(s.step)

    Simulation(counters, [Phase(record, every=3, offset=1)], seed=0).run(8)
    assert steps == [1, 4, 7]


def test_states_left_behind_never_change(make_lattice):
    # A blinker: horizontal at even steps, vertical at odd steps.
    space = make_lattice()
    alive = np.zeros(25, bool)
    alive[[11, 12, 13]] = True
    sim = Simulation(life_state(space, alive), [life_like(CONWAY, Adjacency("vertex"))], seed=0)
    states = [sim.state] + [sim.step() for _ in range(4)]
    for state in states:
        cells = [11, 12, 13] if state.step % 2 == 0 else [7, 12, 17]
        assert np.flatnonzero(state[LifeState.alive]).tolist() == cells


# Running and observing.


def test_run_records_observers_after_every_selected_step(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    observers = {"x0": lambda s: s[CountersState.x][0], "x": lambda s: s[CountersState.x]}
    history = sim.run(6, observers=observers, every=2)
    np.testing.assert_array_equal(history["step"], [2, 4, 6])
    np.testing.assert_array_equal(history["x0"], [2, 4, 6])
    np.testing.assert_array_equal(history["x"], np.repeat([[2], [4], [6]], 9, axis=1))
    assert sim.state.step == 6


def test_run_continues_from_the_current_step(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    sim.run(3)
    history = sim.run(4, observers={"x0": lambda s: s[CountersState.x][0]}, every=2)
    np.testing.assert_array_equal(history["step"], [4, 6])
    np.testing.assert_array_equal(history["x0"], [4, 6])


def test_run_without_records_returns_empty_histories(counters):
    history = Simulation(counters, [], seed=0).run(3, observers={"x0": lambda s: 0}, every=5)
    assert history["step"].shape == (0,)
    assert history["x0"].shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n_steps": -1}, "n_steps"), ({"every": 0}, "every"), ({"observers": {"step": len}}, "step")],
)
def test_run_rejects_bad_arguments(counters, kwargs, match):
    with pytest.raises(ValueError, match=match):
        Simulation(counters, [], seed=0).run(**kwargs)


@pytest.mark.parametrize("which", ["n_steps", "every"])
def test_run_needs_integer_counts(counters, which, not_an_integer):
    with pytest.raises(TypeError, match=f"{which} must be an integer"):
        Simulation(counters, [], seed=0).run(**{which: not_an_integer})


def test_an_observer_that_changes_shape_fails_at_the_step_where_it_does(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    with pytest.raises(ValueError, match=r"'grows' returned shape \(2,\) after step 2, but shape"):
        sim.run(5, observers={"grows": lambda s: np.zeros(s.step)})
    assert sim.state.step == 2  # the steps already taken stay taken


# Errors name where they happened.


class Broken:
    """A neighborhood whose compile fails."""

    depends_on = (CountersState.x,)

    def compile(self, space, state=None):
        raise RuntimeError("cannot compile")


def _fail(s, ctx):
    raise RuntimeError("rule failed")


def _assign_half(s, ctx):
    return Assign(CountersState.x, 0.5)  # x is an integer field


ERROR_CASES = {
    "compiled-type": (
        lambda st: Simulation(st, [Phase(noop, hoods={"near": Compiles(None)})]),  # ty: ignore[invalid-argument-type]
        TypeError,
        ["phase 'noop'", "neighborhood 'near'"],
    ),
    "compile": (
        lambda st: Simulation(st, [Phase(noop, name="p", hoods={"near": Adjacency(5)})]),
        ValueError,
        ["neighborhood 'near' of phase 'p'"],
    ),
    "state-compile": (
        lambda st: Simulation(st, [Phase(noop, name="p", hoods={"near": Broken()})]).step(),
        RuntimeError,
        ["neighborhood 'near'", "in phase 'p' at step 0"],
    ),
    "rule": (
        lambda st: Simulation(st, [Phase(_fail, name="p")]).step(),
        RuntimeError,
        ["in phase 'p' at step 0"],
    ),
    "resolver": (
        lambda st: Simulation(st, [Phase(_assign_half, CountersState.x, name="p")]).step(),
        TypeError,
        ["in phase 'p' at step 0"],
    ),
    "observer": (
        lambda st: Simulation(st, []).run(2, observers={"o": lambda s: 1 / 0}),
        ZeroDivisionError,
        ["in observer 'o' after step 1"],
    ),
}
"""Ways that building or running a simulation can fail: the call, its error, and the context
the error must name."""


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_errors_name_the_phase_neighborhood_or_observer_and_the_step(counters, case):
    fail, error, context = ERROR_CASES[case]
    with pytest.raises(error) as caught:
        fail(counters)
    text = "\n".join([str(caught.value), *getattr(caught.value, "__notes__", [])])
    for expected in context:
        assert expected in text


# Snapshots, restoring and forking.


def test_assigning_a_saved_state_restores_it(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    saved = sim.state
    sim.run(3)
    sim.state = saved
    assert sim.state is saved
    assert sim.step()[CountersState.x][0] == 1


def test_a_restored_state_must_fit_the_simulation(counters, make_lattice):
    sim = Simulation(counters, [], seed=0)
    with pytest.raises(ValueError, match="different space"):
        sim.state = CountersState(make_lattice((3, 3)))
    with pytest.raises(ValueError, match="the state is a Some, not a CountersState"):
        sim.state = fields(x=Field(np.int64))(counters.space)
    with pytest.raises(TypeError, match="State"):
        sim.state = dict(counters)


def test_a_changed_fork_leaves_the_original_alone(counters):
    sim = Simulation(counters, [increment(CountersState.x)], seed=0)
    fork = sim.fork()
    fork.state = fork.state.replace({CountersState.x: 10})
    fork.run(2)
    assert (sim.state.step, sim.state[CountersState.x][0]) == (0, 0)
    assert (fork.state.step, fork.state[CountersState.x][0]) == (2, 12)


# Neighborhoods in the loop.


def test_equal_neighborhoods_compile_once_and_are_shared_between_phases(make_lattice):
    space = make_lattice()
    seen = []

    def look(s, ctx):
        seen.append(ctx.hoods["near"])

    phases = [Phase(look, hoods={"near": Adjacency("vertex")}, name=name) for name in "ab"]
    Simulation(State(space), phases, seed=0).run(2)
    assert len(seen) == 4
    assert all(op is seen[0] for op in seen)
    assert seen[0].graph is space.adjacency("vertex")


class Ring:
    """An unhashable custom neighborhood: each cell's neighbor is the next cell id."""

    def __eq__(self, other):
        return isinstance(other, Ring)

    def compile(self, space):
        return NeighborOperator.from_table((np.arange(space.n_cells)[:, None] + 1) % space.n_cells)


def test_custom_neighborhoods_need_not_be_hashable(counters):
    def shift(s, ctx):
        return Assign(CountersState.y, pull(s[CountersState.x], ctx.hoods["next"]))

    state = counters.replace({CountersState.x: np.arange(9)})
    sim = Simulation(state, [Phase(shift, writes=CountersState.y, hoods={"next": Ring()})], seed=0)
    np.testing.assert_array_equal(sim.step()[CountersState.y], (np.arange(9) + 1) % 9)


@dataclass(frozen=True, eq=False)
class MarkedCells:
    """A state neighborhood: the neighbors of every cell are the cells marked in a field.

    ``compiled`` records the marked cells each time it is compiled.
    """

    field: Field[np.bool_]
    compiled: list[list[int]] = dataclasses.field(default_factory=list)

    @property
    def depends_on(self) -> tuple[Field[np.bool_], ...]:
        return (self.field,)

    def compile(self, space: Space, state: State) -> NeighborOperator:
        marked = np.flatnonzero(state[self.field])
        self.compiled.append(marked.tolist())
        return NeighborOperator.from_table(np.broadcast_to(marked, (space.n_cells, marked.size)))


def test_state_neighborhoods_satisfy_their_protocol():
    assert isinstance(MarkedCells(MarksState.marked), StateNeighborhood)
    assert not isinstance(Adjacency(), StateNeighborhood)


def test_a_state_neighborhood_is_recompiled_only_when_its_fields_change(make_lattice):
    hood = MarkedCells(MarksState.marked)
    state = MarksState(make_lattice((3, 3)), {MarksState.marked: np.arange(9) == 0})
    seen = []

    def use(s, ctx):
        seen.append(int(ctx.hoods["marked"].degree[0]))
        return Assign(MarksState.x, s[MarksState.x] + 1)

    def mark_next(s, ctx):
        return Assign(MarksState.marked, s[MarksState.marked] | np.roll(s[MarksState.marked], 1))

    phases = [
        Phase(use, writes=MarksState.x, hoods={"marked": hood}),
        Phase(mark_next, writes=MarksState.marked, every=2),
    ]
    Simulation(state, phases, seed=0).run(6)
    # mark_next runs after use at steps 0, 2 and 4, so use sees the new marks one step later.
    assert seen == [1, 2, 2, 3, 3, 4]
    assert hood.compiled == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]


def test_a_state_neighborhood_must_depend_on_fields_of_the_state(counters):
    with pytest.raises(KeyError, match="no field named 'marked'") as caught:
        Simulation(counters, [Phase(noop, hoods={"near": MarkedCells(MarksState.marked)})])
    assert "neighborhood 'near' depends on" in "\n".join(caught.value.__notes__)
    with pytest.raises(TypeError, match="Field attributes of a State class, got 'x'"):
        Simulation(counters, [Phase(noop, hoods={"near": MarkedCells("x")})])  # ty: ignore[invalid-argument-type]


def test_rules_may_read_fields_as_attributes(counters):
    def by_field(s, ctx):
        return [
            Assign(CountersState.x, s[CountersState.x] + s[CountersState.y]),
            Assign(CountersState.y, s[CountersState.y] + 1),
        ]

    def by_attribute(s: CountersState, ctx: Context) -> list[Assign]:
        return [Assign(CountersState.x, s.x + s.y), Assign(CountersState.y, s.y + 1)]

    writes = (CountersState.x, CountersState.y)
    sims = [
        Simulation(counters, [Phase(rule, writes=writes)], seed=0)
        for rule in (by_field, by_attribute)
    ]
    for sim in sims:
        sim.run(3)
    for counter in writes:
        np.testing.assert_array_equal(sims[0].state[counter], sims[1].state[counter])
    np.testing.assert_array_equal(sims[1].state.x, np.full(9, 3))


class TalliedState(CountersState):
    tally = Field(np.int64)


def test_a_phase_for_a_base_class_runs_on_states_of_a_subclass(make_lattice):
    @phase(writes=CountersState.x)
    def count(s: CountersState, ctx: Context) -> Assign:
        return Assign(CountersState.x, s.x + 1)

    @phase(writes=TalliedState.tally)
    def tally(s: TalliedState, ctx: Context) -> Assign:
        return Assign(TalliedState.tally, s.tally + s.x)

    sim = Simulation(TalliedState(make_lattice((3, 3))), [count, tally], seed=0)
    sim.run(3)
    assert type(sim.state) is TalliedState  # every state the simulation makes keeps the class
    np.testing.assert_array_equal(sim.state.x, np.full(9, 3))
    np.testing.assert_array_equal(sim.state.tally, np.full(9, 1 + 2 + 3))


def test_a_state_is_made_on_a_space_not_on_another_state(counters):
    # As a view was made before states were typed by their class: now a mistake, not a copy.
    with pytest.raises(TypeError, match="CountersState is made on a space, got CountersState"):
        CountersState(counters)
    with pytest.raises(TypeError, match="is made on a space, got int"):
        CountersState(9)  # ty: ignore[invalid-argument-type]


# Randomness.


def noise(rate):
    """A phase that flips each cell's ``alive`` flag with probability ``rate``."""

    def flip(s, ctx):
        return Assign(LifeState.alive, s[LifeState.alive] ^ (ctx.rng.random(s.n_cells) < rate))

    return Phase(flip, writes=LifeState.alive, name="noise")


def noisy_hex_life(seed, extra=()):
    space = LatticeSpace((8, 8), tiling="hex")
    alive = np.random.default_rng(0).random(space.n_cells) < 0.3
    phases = [life_like(HEX_RULE, Adjacency()), noise(0.05), *extra]
    return Simulation(life_state(space, alive), phases, seed=seed)


def alive_history(sim, n_steps):
    return sim.run(n_steps, observers={"alive": lambda s: s[LifeState.alive]})["alive"]


def test_equal_seeds_give_identical_runs():
    np.testing.assert_array_equal(
        alive_history(noisy_hex_life(7), 30), alive_history(noisy_hex_life(7), 30)
    )


def test_different_seeds_give_different_runs():
    assert not np.array_equal(
        alive_history(noisy_hex_life(7), 30), alive_history(noisy_hex_life(8), 30)
    )


def test_an_omitted_seed_is_drawn_and_kept_so_the_run_can_be_repeated():
    first = noisy_hex_life(None)
    history = alive_history(first, 10)
    np.testing.assert_array_equal(alive_history(noisy_hex_life(first.seed), 10), history)
    assert noisy_hex_life(None).seed != first.seed


def draw_phase(name, draws):
    def draw(s, ctx):
        draws.append((name, s.step, ctx.rng.random(4)))

    return Phase(draw, name=name)


def test_streams_differ_between_steps_and_between_phases(counters):
    draws = []
    Simulation(counters, [draw_phase("a", draws), draw_phase("b", draws)], seed=0).run(2)
    assert len({values.tobytes() for _, _, values in draws}) == 4


def test_adding_a_phase_leaves_the_streams_of_the_others_unchanged(counters):
    alone, together = [], []
    Simulation(counters, [draw_phase("a", alone)], seed=0).run(3)
    phases = [draw_phase("new", []), draw_phase("a", together), increment(CountersState.x)]
    Simulation(counters, phases, seed=0).run(3)
    for (_, _, expected), (_, _, actual) in zip(alone, together, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_restoring_a_saved_state_replays_the_same_steps():
    sim = noisy_hex_life(3)
    sim.run(5)
    saved = sim.state
    first = alive_history(sim, 10)
    sim.state = saved
    np.testing.assert_array_equal(alive_history(sim, 10), first)


def test_a_run_resumed_from_a_pickled_state_continues_exactly():
    # A state from disk lives on a rebuilt space, so it resumes in a new simulation of the
    # same phases and seed. Its streams depend only on the seed, the step and the phase.
    straight = noisy_hex_life(3)
    straight.run(5)
    expected = alive_history(straight, 10)
    first = noisy_hex_life(3)
    first.run(5)
    loaded = pickle.loads(pickle.dumps(first.state))
    with pytest.raises(ValueError, match="different space"):
        first.state = loaded
    resumed = Simulation(loaded, first.phases, seed=first.seed)
    np.testing.assert_array_equal(alive_history(resumed, 10), expected)


def test_a_fork_repeats_what_the_original_does():
    sim = noisy_hex_life(3)
    sim.run(5)
    fork = sim.fork()
    np.testing.assert_array_equal(alive_history(fork, 10), alive_history(sim, 10))


FINGERPRINT = textwrap.dedent(
    """
    import hashlib

    import numpy as np

    from gridzero.core import (
        Adjacency, Assign, Field, LatticeSpace, Phase, Simulation, State, pull,
    )

    class LifeState(State):
        alive = Field(bool)

    def life(s, ctx):
        n = pull(s[LifeState.alive], ctx.hoods["near"])
        return Assign(LifeState.alive, (n == 2) | (s[LifeState.alive] & (n == 3)))

    def noise(s, ctx):
        return Assign(LifeState.alive, s[LifeState.alive] ^ (ctx.rng.random(s.n_cells) < 0.1))

    state = LifeState(LatticeSpace((8, 8), tiling="hex"))
    phases = [Phase(life, LifeState.alive, {"near": Adjacency()}), Phase(noise, LifeState.alive)]
    observers = {"alive": lambda s: s[LifeState.alive]}
    history = Simulation(state, phases, seed=11).run(20, observers=observers)
    print(hashlib.sha256(history["alive"].tobytes()).hexdigest())
    """
)


@pytest.mark.slow
def test_equal_seeds_give_identical_runs_in_separate_processes():
    # String hashing differs between processes, so phase streams must not depend on it.
    def fingerprint(hash_seed):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        command = [sys.executable, "-c", FINGERPRINT]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert fingerprint("1") == fingerprint("2")


# Life on every kind of space: the milestone 3 acceptance tests.

CONWAY = ((3,), (2, 3))
"""Conway's rule on squares with 8 neighbors: born with 3, survives with 2 or 3."""

HEX_RULE = ((2,), (3, 4))
"""A hexagonal Life rule, B2/S34: born with 2 neighbors, survives with 3 or 4."""

SQUARE_OFFSETS = [(dr, dc) for dr, dc in itertools.product((-1, 0, 1), repeat=2) if dr or dc]
HEX_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def life_state(space, alive):
    return LifeState(space, {LifeState.alive: alive})


def life_like(rule, hood):
    """A Life-like phase: ``rule`` is (birth counts, survival counts)."""
    born, survive = rule

    @phase(writes=LifeState.alive, hoods={"near": hood})
    def life(state, ctx):
        n = pull(state[LifeState.alive], ctx.hoods["near"])
        alive = state[LifeState.alive]
        return Assign(LifeState.alive, np.where(alive, np.isin(n, survive), np.isin(n, born)))

    return life


def run_life(space, rule, hood, alive, n_steps):
    sim = Simulation(life_state(space, alive), [life_like(rule, hood)], seed=0)
    return alive_history(sim, n_steps)


def shifted_life(grid, offsets, rule, n_steps):
    """Reference Life on a periodic grid: neighbor counts from shifted copies of the grid."""
    born, survive = rule
    history = []
    for _ in range(n_steps):
        n = sum(np.roll(grid, (-dr, -dc), axis=(0, 1)).astype(int) for dr, dc in offsets)
        grid = np.where(grid, np.isin(n, survive), np.isin(n, born))
        history.append(grid)
    return np.array(history)


@pytest.mark.parametrize(
    ("tiling", "offsets", "rule", "kind"),
    [("hypercubic", SQUARE_OFFSETS, CONWAY, "vertex"), ("hex", HEX_OFFSETS, HEX_RULE, "facet")],
    ids=["square", "hex"],
)
def test_life_on_a_periodic_lattice_matches_shifted_copies(rng, tiling, offsets, rule, kind):
    space = LatticeSpace((12, 10), tiling=tiling)
    alive = rng.random(space.n_cells) < 0.35
    history = run_life(space, rule, Adjacency(kind), alive, 25)
    expected = shifted_life(space.grid_view(alive), offsets, rule, 25)
    np.testing.assert_array_equal(history.reshape(expected.shape), expected)
    assert len({frame.tobytes() for frame in history}) > 10


def test_life_on_the_icosphere_matches_the_delaunay_neighbors(rng):
    space = icosphere(4)
    adjacency = np.zeros((space.n_cells, space.n_cells), int)
    for a, b in itertools.permutations(ConvexHull(space.centers).simplices.T, 2):
        adjacency[a, b] = 1
    alive = rng.random(space.n_cells) < 0.3
    history = run_life(space, HEX_RULE, Adjacency(), alive, 25)
    born, survive = HEX_RULE
    expected = []
    for _ in range(25):
        n = adjacency @ alive
        alive = np.where(alive, np.isin(n, survive), np.isin(n, born))
        expected.append(alive)
    np.testing.assert_array_equal(history, expected)
    assert len({frame.tobytes() for frame in history}) > 10


def test_a_glider_moves_one_cell_diagonally_every_four_steps():
    space = LatticeSpace((8, 8))
    grid = np.zeros((8, 8), bool)
    grid[[0, 1, 2, 2, 2], [1, 2, 0, 1, 2]] = True
    history = run_life(space, CONWAY, Adjacency("vertex"), grid.ravel(), 32)
    for k in range(1, 9):
        expected = np.roll(grid, (k, k), axis=(0, 1))
        np.testing.assert_array_equal(space.grid_view(history[4 * k - 1]), expected)


# Nested states.


def test_writes_of_a_nested_state_are_its_dynamic_fields():
    rule = Phase(lambda s, ctx: None, writes=WorldState.settlers)
    settlers = WorldState.settlers
    assert rule.writes == (settlers.people, settlers.stores, settlers.mood, settlers.count)


def test_writes_of_a_nested_state_include_those_nested_in_it():
    rule = Phase(lambda s, ctx: None, writes=PlanetState.europe)
    assert [field.name for field in rule.writes] == [
        "europe.north.rain",
        "europe.north.rabbits.population",
        "europe.north.rabbits.eaten",
        "europe.north.foxes.population",
        "europe.north.foxes.eaten",
    ]


def dying(name: str, species: type[Species], rate: float) -> Phase[State]:
    """Return a phase in which a share ``rate`` of a species dies: one rule for any species."""

    @phase(writes=species, name=f"die {name}")
    def die(s: State, ctx: Context) -> Source:
        return Source(species.population, -rate * s[species.population])

    return die


def test_a_phase_factory_makes_a_phase_for_each_nested_state(make_lattice):
    values = {EcosystemState.foxes.population: 1.0, EcosystemState.rabbits.population: 1.0}
    start = EcosystemState(make_lattice(), values)
    phases = [
        dying("foxes", EcosystemState.foxes, 0.5),
        dying("rabbits", EcosystemState.rabbits, 0.25),
    ]
    sim = Simulation(start, phases, seed=0)
    sim.step()
    assert (sim.state.foxes.population == 0.5).all()
    assert (sim.state.rabbits.population == 0.75).all()
