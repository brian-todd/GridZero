"""The loop in arrays against `core.Simulation`: the same runs, to float32's precision."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_demo import check_array_model
from gridzero import batched, core
from gridzero.core import Adjacency, Field, LatticeSpace, State, icosphere


class LifeState(State):
    alive = Field(bool)


@dataclass(frozen=True)
class Life:
    """The README's first simulation: hexagonal Life, and its phases once more as arrays."""

    density: float = 0.3

    def start(self, space, seed) -> LifeState:
        rng = np.random.default_rng(seed)
        return LifeState(space, {LifeState.alive: rng.random(space.n_cells) < self.density})

    def phases(self):
        @core.phase(writes=LifeState.alive, hoods={"near": Adjacency()})
        def life(s: LifeState, ctx: core.Context) -> core.Assign:
            n = core.pull(s.alive, ctx.hoods["near"])
            return core.Assign(
                LifeState.alive, (~s.alive & (n == 2)) | (s.alive & ((n == 3) | (n == 4)))
            )

        return [life]

    def array_phases(self):
        @batched.phase(writes=LifeState.alive, hoods={"near": Adjacency()})
        def life(s, ctx):
            alive = s[LifeState.alive]
            n = batched.pull(alive, ctx.hoods["near"])
            return batched.Assign(
                LifeState.alive, (~alive & (n == 2)) | (alive & ((n == 3) | (n == 4)))
            )

        return [life]


class HerdState(State):
    grass = Field(np.float64, extensive=True)
    herd = Field(np.float64, (2,), extensive=True)
    season = Field(np.int64)


@dataclass(frozen=True)
class Herds:
    """Grass grows every third step, two herds eat it and roam, and a season counts years."""

    growth: float = 0.3
    year: int = 4

    def start(self, space, seed) -> HerdState:
        rng = np.random.default_rng(seed)
        n = space.n_cells
        values = {HerdState.grass: rng.random(n), HerdState.herd: rng.random((2, n))}
        return HerdState(space, values)

    def phases(self):
        @core.phase(writes=HerdState.grass, every=3, offset=1)
        def grow(s: HerdState, ctx: core.Context) -> core.Source:
            return core.Source(HerdState.grass, self.growth * s.space.measures)

        @core.phase(writes=(HerdState.grass, HerdState.herd))
        def graze(s: HerdState, ctx: core.Context) -> list[core.Claim | core.Source]:
            asked = 0.4 * s.density(HerdState.grass) * s.herd
            return [
                core.Claim(HerdState.grass, asked, into=HerdState.herd, efficiency=0.5),
                core.Source(HerdState.herd, -0.1 * s.herd),
            ]

        @core.phase(writes=HerdState.herd, hoods={"near": Adjacency()})
        def roam(s: HerdState, ctx: core.Context) -> core.Spread:
            return core.Spread(HerdState.herd, ctx.hoods["near"], 0.2, toward=s.grass)

        @core.phase(writes=HerdState.season, every=self.year, offset=self.year - 1)
        def turn(s: HerdState, ctx: core.Context) -> core.Assign:
            return core.Assign(HerdState.season, s.season + 1)

        @core.phase()
        def idle(s: HerdState, ctx: core.Context) -> None:
            return None

        return [grow, graze, roam, turn, idle]

    def array_phases(self):
        @batched.phase(writes=HerdState.grass, every=3, offset=1)
        def grow(s, ctx):
            return batched.Source(HerdState.grass, self.growth * ctx.measures)

        @batched.phase(writes=(HerdState.grass, HerdState.herd))
        def graze(s, ctx):
            asked = 0.4 * (s[HerdState.grass] / ctx.measures) * s[HerdState.herd]
            return [
                batched.Claim(HerdState.grass, asked, into=HerdState.herd, efficiency=0.5),
                batched.Source(HerdState.herd, -0.1 * s[HerdState.herd]),
            ]

        @batched.phase(writes=HerdState.herd, hoods={"near": Adjacency()})
        def roam(s, ctx):
            return batched.Spread(HerdState.herd, ctx.hoods["near"], 0.2, toward=s[HerdState.grass])

        @batched.phase(writes=HerdState.season, every=self.year, offset=self.year - 1)
        def turn(s, ctx):
            return batched.Assign(HerdState.season, s[HerdState.season] + 1)

        @batched.phase()
        def idle(s, ctx):
            return None

        return [grow, graze, roam, turn, idle]


SPACES = {
    "hex torus": lambda: LatticeSpace((12, 10), tiling="hex", periodic=True),
    "planet": lambda: icosphere(2),
}


@pytest.fixture(params=list(SPACES))
def space(request):
    return SPACES[request.param]()


def test_life_in_arrays_is_life_bit_for_bit(space):
    life = Life()
    check_array_model(life, life.array_phases(), space, seed=3, steps=30, rtol=0.0)


def test_phases_run_in_order_on_their_steps_as_in_core(space):
    herds = Herds()
    check_array_model(herds, herds.array_phases(), space, seed=1, steps=14, rtol=1e-4)


def test_a_run_is_its_steps_and_records_what_it_is_asked(space):
    herds = Herds()
    sim = batched.Simulation(space, herds.array_phases())
    start = batched.fields(herds.start(space, 2))
    key = batched.root(5)
    stepped = start
    totals = []
    for _ in range(7):
        stepped = sim.step(stepped, key)
        totals.append(float(stepped[HerdState.herd].sum()))
    end, history = jax.jit(lambda f: sim.run(f, key, 7, lambda f: f[HerdState.herd].sum()))(start)
    assert int(end.step) == 7
    assert history.shape == (7,)
    np.testing.assert_allclose(history, totals, rtol=1e-5)
    np.testing.assert_allclose(end[HerdState.herd], stepped[HerdState.herd], rtol=1e-5)
    same, nothing = sim.run(start, key, 0)
    assert nothing is None
    assert int(same.step) == 0
    np.testing.assert_array_equal(same[HerdState.grass], start[HerdState.grass])


def test_a_batch_of_runs_is_each_run_alone(space):
    herds = Herds()
    sim = batched.Simulation(space, herds.array_phases())
    starts = [batched.fields(herds.start(space, seed)) for seed in range(4)]
    many = jax.tree.map(lambda *xs: np.stack(xs), *starts)
    key = batched.root(0)
    ends, _ = jax.jit(jax.vmap(lambda f: sim.run(f, key, 6)))(many)
    for i, start in enumerate(starts):
        alone, _ = sim.run(start, key, 6)
        np.testing.assert_allclose(ends[HerdState.herd][i], alone[HerdState.herd], rtol=1e-5)
        np.testing.assert_array_equal(ends[HerdState.season][i], alone[HerdState.season])


class NoiseState(State):
    drawn = Field(np.int64, (2,))


def test_each_phase_draws_from_its_own_stream_of_the_key_at_each_step():
    space = LatticeSpace((4, 4))

    def roller(name, row):
        @batched.phase(writes=NoiseState.drawn, name=name)
        def roll(s, ctx):
            drawn = s[NoiseState.drawn].at[row].set(batched.d(ctx.key, 1000, (16,)))
            return batched.Assign(NoiseState.drawn, drawn)

        return roll

    sim = batched.Simulation(space, [roller("first", 0), roller("second", 1)])
    start = batched.fields(NoiseState(space))
    one = sim.step(start, batched.root(1))
    again = sim.step(start, batched.root(1))
    other = sim.step(start, batched.root(2))
    two = sim.step(one, batched.root(1))
    drawn = np.asarray(one[NoiseState.drawn])
    np.testing.assert_array_equal(drawn, again[NoiseState.drawn])
    assert (drawn[0] != drawn[1]).any()  # two phases, two streams
    assert (drawn != np.asarray(other[NoiseState.drawn])).any()  # another key
    assert (drawn != np.asarray(two[NoiseState.drawn])).any()  # another step
    assert drawn.min() >= 1
    assert drawn.max() <= 1000
    # A phase keeps its stream when others join the schedule.
    alone = batched.Simulation(space, [roller("second", 1)]).step(start, batched.root(1))
    np.testing.assert_array_equal(np.asarray(alone[NoiseState.drawn])[1], drawn[1])


def test_neighborhoods_are_compiled_once_per_description():
    space = LatticeSpace((4, 4))
    seen = []

    @batched.phase(hoods={"a": Adjacency(), "b": Adjacency(), "c": Adjacency("vertex")})
    def look(s, ctx):
        seen.append(ctx.hoods)
        return None

    batched.Simulation(space, [look]).step(batched.fields(NoiseState(space)), batched.root(0))
    assert seen[0]["a"] is seen[0]["b"]
    assert seen[0]["c"] is not seen[0]["a"]
    assert seen[0]["a"].idx.shape == (16, 4)


def test_a_phase_declares_what_core_phases_declare(not_an_integer):
    def rule(s, ctx):
        return None

    made = batched.phase(writes=LifeState.alive, every=4, offset=3, name="named")(rule)
    assert (made.name, made.every, made.offset) == ("named", 4, 3)
    assert made.writes == (LifeState.alive,)
    assert batched.phase()(rule).name == "rule"
    with pytest.raises(TypeError, match="every"):
        batched.phase(every=not_an_integer)
    with pytest.raises(TypeError, match="offset"):
        batched.phase(offset=not_an_integer)
    for every, offset in ((0, 0), (3, 3), (3, -1)):
        with pytest.raises(ValueError, match="every must be positive"):
            batched.phase(every=every, offset=offset)
    with pytest.raises(TypeError, match="writes fields"):
        batched.phase(writes="alive")  # ty: ignore[invalid-argument-type]  # a name, on purpose
    with pytest.raises(ValueError, match="must not be empty"):
        batched.phase(name="")(rule)


class AroundTheLiving:
    """A neighborhood that depends on the state (`core.StateNeighborhood`)."""

    depends_on = ("alive",)

    def compile(self, state):
        return Adjacency().compile(state.space)


def test_a_simulation_refuses_what_it_cannot_run():
    space = LatticeSpace((4, 4))

    def rule(s, ctx):
        return None

    with pytest.raises(TypeError, match="Space"):
        batched.Simulation("board", [])  # ty: ignore[invalid-argument-type]  # not a space
    with pytest.raises(TypeError, match=r"batched\.phase"):
        batched.Simulation(space, [rule])  # ty: ignore[invalid-argument-type]  # not a phase
    with pytest.raises(ValueError, match="distinct names"):
        batched.Simulation(space, [batched.phase()(rule), batched.phase()(rule)])
    with pytest.raises(TypeError, match="compile on a space alone"):
        batched.Simulation(space, [batched.phase(hoods={"near": AroundTheLiving()})(rule)])  # ty: ignore[invalid-argument-type]  # it depends on the state
    with pytest.raises(TypeError, match="compile on a space alone"):
        batched.Simulation(space, [batched.phase(hoods={"near": "adjacent"})(rule)])  # ty: ignore[invalid-argument-type]  # a name
    sim = batched.Simulation(space, [batched.phase()(rule)])
    with pytest.raises(ValueError, match="steps"):
        sim.run(batched.fields(LifeState(space)), batched.root(0), -1)


def test_a_rule_that_writes_what_it_did_not_declare_raises():
    space = LatticeSpace((4, 4))

    @batched.phase()
    def sneaky(s, ctx):
        return batched.Assign(LifeState.alive, True)

    with pytest.raises(ValueError, match="not among the fields"):
        batched.Simulation(space, [sneaky]).step(batched.fields(LifeState(space)), batched.root(0))


def test_steps_count_in_the_fields():
    space = LatticeSpace((4, 4))
    life = Life()
    sim = batched.Simulation(space, life.array_phases())
    fields = batched.fields(life.start(space, 0)).at(41)
    after = sim.step(fields, batched.root(0))
    assert int(after.step) == 42
    assert after.step.dtype == jnp.int32
