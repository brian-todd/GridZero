"""Time the core operations and whole models on the reference path and the fast paths.

Run from the repository root, pinned to one performance core for steady numbers:

    taskset -c 4 uv run python -m benchmarks.bench            # everything, about 2 minutes
    taskset -c 4 uv run python -m benchmarks.bench models     # only the whole models
    taskset -c 4 uv run python -m benchmarks.bench queries    # paths and sight

Each figure is the best of several repeats after a warm-up, in microseconds per call or per
step. The script prints Markdown tables, headed by the machine and library versions, and ends
with the steps-per-second target of the 3-phase model.
"""

import argparse
import platform
import sys
import timeit
from collections.abc import Callable, Sequence
from importlib.metadata import version

import numpy as np

from gridzero.core import (
    Adjacency,
    Assign,
    Claim,
    Field,
    Flow,
    Hops,
    LatticeSpace,
    NeighborAllocation,
    NeighborChoice,
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

TARGET_STEPS_PER_SECOND = 1_000
"""The milestone target: the 3-phase population model at f=32 (10,242 cells), on one core."""

IMPLS = ("reference", "numba")


def best_time(fn: Callable[[], object], budget: float = 0.25, repeat: int = 7) -> float:
    """Return the best time of one call of ``fn``, in microseconds."""
    fn()  # warm up: compile kernels, fill caches
    once = timeit.timeit(fn, number=1)
    number = max(1, int(budget / repeat / max(once, 1e-7)))
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number * 1e6


def table(title: str, rows: Sequence[tuple[str, Callable[[], object]]]) -> None:
    print(f"\n### {title}\n")
    print("| Case | Reference (us) | Numba (us) | Speed-up |")
    print("|---|---|---|---|")
    for label, fn in rows:
        times = []
        for impl in IMPLS:
            with implementation(impl):
                times.append(best_time(fn))
        print(f"| {label} | {times[0]:,.1f} | {times[1]:,.1f} | {times[0] / times[1]:.2f}x |")
        sys.stdout.flush()


def operations(frequency: int = 32) -> None:
    planet = icosphere(frequency)
    rng = np.random.default_rng(0)
    near = Adjacency().compile(planet)
    weighted = Adjacency(weight="boundary").compile(planet)
    wide = Hops(2).compile(planet)
    one, three = rng.random(planet.n_cells), rng.random((3, planet.n_cells))
    rows = [
        ("pull sum, 1 component", lambda: pull(one, near)),
        ("pull sum, 3 components", lambda: pull(three, near)),
        ("pull sum, 3 components, weighted", lambda: pull(three, weighted)),
        ("pull sum, 3 components, `Hops(2)`", lambda: pull(three, wide)),
        ("pull mean, 3 components, weighted", lambda: pull(three, weighted, "mean")),
        ("pull max, 3 components", lambda: pull(three, near, "max")),
        ("push, 3 components", lambda: push(three, near)),
        ("gather, 3 components", lambda: gather(three, near)),
    ]
    table(f"Operators at f={frequency} ({planet.n_cells:,} cells, {near.n_edges:,} edges)", rows)


class StocksState(State):
    pop = Field(np.float64, (3,), extensive=True)
    food = Field(np.float64, extensive=True)
    fed = Field(np.float64, (3,), extensive=True)


class GroupsState(State):
    pop = Field(np.float64, (3,), extensive=True)
    capacity = Field(np.float64, static=True)


class ForagersState(State):
    pop = Field(np.float64, (3,), extensive=True)
    food = Field(np.float64, extensive=True)
    eaten = Field(np.float64, (3,), extensive=True)


class LifeState(State):
    alive = Field(bool)


def intents(frequency: int = 32) -> None:
    planet = icosphere(frequency)
    n = planet.n_cells
    rng = np.random.default_rng(0)
    near = Adjacency().compile(planet)
    state = StocksState(
        planet, {StocksState.pop: rng.random((3, n)), StocksState.food: rng.random(n)}
    )
    leave, toward, land = 0.05 * rng.random((3, n)), rng.random(n), rng.random(n) < 0.7
    shared, per_component = 0.02 * rng.random(near.n_edges), 0.02 * rng.random((3, near.n_edges))
    rows = [
        (
            "`Spread`, uniform attraction",
            lambda: resolve(state, Spread(StocksState.pop, near, leave)),
        ),
        (
            "`Spread`, attraction and domain",
            lambda: resolve(state, Spread(StocksState.pop, near, leave, toward, land)),
        ),
        (
            "`Flow`, rates shared by 3 components",
            lambda: resolve(state, Flow(StocksState.pop, near, shared)),
        ),
        (
            "`Flow`, rates per component",
            lambda: resolve(state, Flow(StocksState.pop, near, per_component)),
        ),
        (
            "`Flow` with a domain, and a removal (rationing)",
            lambda: resolve(
                state,
                [
                    Flow(StocksState.pop, near, per_component, land),
                    Source(StocksState.pop, -0.99 * state[StocksState.pop]),
                ],
            ),
        ),
        (
            "`Source`",
            lambda: resolve(state, Source(StocksState.pop, 0.01 * state[StocksState.pop])),
        ),
        (
            "`Claim` by 3 groups",
            lambda: resolve(
                state, Claim(StocksState.food, 0.1 * state[StocksState.pop], StocksState.fed)
            ),
        ),
    ]
    table(f"Intents at f={frequency}, 3 components", rows)


def actions(frequency: int = 32) -> None:
    planet = icosphere(frequency)
    rng = np.random.default_rng(0)
    near = Adjacency().compile(planet)
    allocation = NeighborAllocation(near, shape=(3,))
    choice = NeighborChoice(near, shape=(3,))
    weights = rng.exponential(size=allocation.action_shape)
    scores = rng.random((3, planet.n_cells, choice.n_slots))
    slots = choice.sample(rng)
    rows = [
        ("`NeighborAllocation.fractions`", lambda: allocation.fractions(weights)),
        ("`NeighborAllocation.flow`", lambda: allocation.flow(StocksState.pop, weights)),
        ("`NeighborAllocation.sample` (NumPy draws)", lambda: allocation.sample(rng)),
        ("`NeighborChoice.best`", lambda: choice.best(scores)),
        ("`NeighborChoice.flow`", lambda: choice.flow(StocksState.pop, slots, 0.5)),
        ("`NeighborChoice.sample` (NumPy draws)", lambda: choice.sample(rng)),
    ]
    table(f"Action spaces at f={frequency}, 3 components", rows)


def queries(side: int = 200, board: int = 64) -> None:
    rng = np.random.default_rng(0)
    hexes = LatticeSpace((side, side), tiling="hex")
    cost = rng.uniform(1.0, 3.0, hexes.n_cells)
    far = int(np.argmax(steps(hexes, 0)))
    rows = [
        ("steps, unit costs", lambda: steps(hexes, 0)),
        ("steps, costs", lambda: steps(hexes, 0, cost=cost)),
        ("reach, budget 5", lambda: reach(hexes, 0, 5)),
        ("reach, budget 5, costs", lambda: reach(hexes, 0, 5.0, cost=cost)),
        ("path to the farthest cell, costs", lambda: path(hexes, 0, far, cost=cost)),
        ("path to a cell 7 steps away, costs", lambda: path(hexes, 0, 7, cost=cost)),
    ]
    table(f"Paths on {side}x{side} hexes ({hexes.n_cells:,} cells)", rows)
    rows = []
    for tiling in ("hypercubic", "hex", "triangle"):
        space = LatticeSpace((board, board), tiling=tiling, periodic=False)
        walls = rng.random(space.n_cells) < 0.15
        viewer = space.n_cells // 2 + board // 2
        viewers = rng.choice(space.n_cells, 12, replace=False)
        rows += [
            (f"sight within 10, {tiling}", lambda s=space, w=walls, v=viewer: sight(s, v, w, 10.0)),
            (
                f"sight within 10 from 12 viewers, {tiling}",
                lambda s=space, w=walls, v=viewers: sight(s, v, w, 10.0, separate=True),
            ),
            (
                f"visible across the board, {tiling}",
                lambda s=space, w=walls: visible(s, 0, s.n_cells - 1, w),
            ),
        ]
    table(f"Sight on {board}x{board} boards, 15% walls", rows)


def three_phase_model(frequency: int) -> Simulation[GroupsState]:
    """Logistic growth, spread toward free room, and random shocks."""
    planet = icosphere(frequency)
    values = {GroupsState.pop: 0.1 * planet.measures, GroupsState.capacity: planet.measures}
    state = GroupsState(planet, values)

    @phase(writes=GroupsState.pop)
    def growth(s, ctx):
        pop = s[GroupsState.pop]
        return Source(GroupsState.pop, 0.1 * pop * (1 - pop.sum(axis=0) / s[GroupsState.capacity]))

    @phase(writes=GroupsState.pop, hoods={"near": Adjacency()})
    def spread(s, ctx):
        room = s[GroupsState.capacity] - s[GroupsState.pop].sum(axis=0)
        return Spread(GroupsState.pop, ctx.hoods["near"], 0.05, toward=room)

    @phase(writes=GroupsState.pop, every=10)
    def shocks(s, ctx):
        hit = ctx.rng.random(s.n_cells) < 0.01
        return Source(GroupsState.pop, -0.5 * s[GroupsState.pop] * hit)

    return Simulation(state, [growth, spread, shocks], seed=42)


def forager_model(frequency: int) -> Simulation[ForagersState]:
    """Greedy foragers choosing the neighbor with the most food, eating it, and wandering."""
    planet = icosphere(frequency)
    rng = np.random.default_rng(1)
    state = ForagersState(
        planet,
        {ForagersState.pop: rng.random((3, planet.n_cells)), ForagersState.food: planet.measures},
    )

    @phase(writes=(ForagersState.food, ForagersState.eaten))
    def eat(s, ctx):
        return [
            Source(ForagersState.food, 0.05 * planet.measures),
            Claim(ForagersState.food, 0.2 * s[ForagersState.pop], ForagersState.eaten),
        ]

    @phase(writes=ForagersState.pop, hoods={"near": Adjacency()})
    def forage(s, ctx):
        choice = NeighborChoice(ctx.hoods["near"], shape=(3,))
        smell = np.take(s[ForagersState.food], choice.targets)
        return choice.flow(
            ForagersState.pop, choice.best(np.broadcast_to(smell, (3, *smell.shape))), 0.5
        )

    @phase(writes=ForagersState.pop, hoods={"near": Adjacency()})
    def wander(s, ctx):
        allocation = NeighborAllocation(ctx.hoods["near"], shape=(3,))
        return allocation.flow(ForagersState.pop, allocation.sample(ctx.rng))

    return Simulation(state, [eat, forage, wander], seed=7)


def life_model(side: int) -> Simulation[LifeState]:
    """Conway's Life (B3/S23) on a periodic square lattice: an `Assign` of a neighbor count."""
    space = LatticeSpace((side, side))
    rng = np.random.default_rng(2)
    state = LifeState(space, {LifeState.alive: rng.random(space.n_cells) < 0.3})

    @phase(writes=LifeState.alive, hoods={"moore": Adjacency("vertex")})
    def life(s, ctx):
        alive = s[LifeState.alive]
        count = pull(alive, ctx.hoods["moore"])
        return Assign(LifeState.alive, (count == 3) | (alive & (count == 2)))

    return Simulation(state, [life], seed=0)


def models() -> float:
    print("\n### Whole models, per step\n")
    print("| Model | Reference (us) | Numba (us) | Speed-up | Numba steps/s |")
    print("|---|---|---|---|---|")
    cases = [
        ("3-phase population, f=16", lambda: three_phase_model(16)),
        ("3-phase population, f=32", lambda: three_phase_model(32)),
        ("3-phase population, f=64", lambda: three_phase_model(64)),
        ("foragers (choice, claim, allocation), f=32", lambda: forager_model(32)),
        ("Life, 256x256 lattice", lambda: life_model(256)),
        ("Life, 1024x1024 lattice", lambda: life_model(1024)),
    ]
    target = 0.0
    for label, build in cases:
        times = []
        for impl in IMPLS:
            with implementation(impl):
                sim = build()
                sim.run(20)
                times.append(best_time(sim.step, budget=0.5))
        rate = 1e6 / times[1]
        if label == "3-phase population, f=32":
            target = rate
        print(
            f"| {label} | {times[0]:,.0f} | {times[1]:,.0f} | {times[0] / times[1]:.2f}x "
            f"| {rate:,.0f} |"
        )
        sys.stdout.flush()
    return target


def header() -> None:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as info:
            cpu = next(line.split(":", 1)[1].strip() for line in info if "model name" in line)
    except OSError, StopIteration:
        pass
    libraries = ", ".join(f"{name} {version(name)}" for name in ("numpy", "scipy", "numba"))
    print(f"**Machine:** {cpu}. Python {platform.python_version()}, {libraries}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "parts",
        nargs="*",
        choices=["operations", "intents", "actions", "queries", "models"],
        help="what to time (default: everything)",
    )
    parts = parser.parse_args().parts or ["operations", "intents", "actions", "queries", "models"]
    header()
    for part in ("operations", "intents", "actions", "queries"):
        if part in parts:
            globals()[part]()
    if "models" in parts:
        rate = models()
        verdict = "meets" if rate >= TARGET_STEPS_PER_SECOND else "MISSES"
        print(
            f"\nThe 3-phase model at f=32 runs {rate:,.0f} steps/s: it {verdict} the target of "
            f"{TARGET_STEPS_PER_SECOND:,} steps/s."
        )


if __name__ == "__main__":
    main()
