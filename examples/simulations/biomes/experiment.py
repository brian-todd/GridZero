"""Apparent competition between two biomes: `examples.simulations.biomes`, run and checked.

Run with ``uv run python run.py biomes [years]``, by default 60 years each run,
with deer and without. It prints the means of each run against the theory.
"""

import dataclasses
import itertools
import time

import numpy as np
from numpy.typing import NDArray

from examples.simulations.biomes import COVER, Biomes, WolvesState, by_cover
from gridzero.core import MeshSpace, Simulation, icosphere

SEED = 0  # the start draws nothing: the seed sets where the wolves arrive


def run(biomes: Biomes, planet: MeshSpace, years: int) -> tuple[NDArray[np.float64], WolvesState]:
    """Run from the start of `SEED`.

    Parameters
    ----------
    biomes : Biomes
        The parameters.
    planet : MeshSpace
        The planet.
    years : int
        How long to run.

    Returns
    -------
    history : ndarray of float64, shape (years, 5)
        `Biomes.means` at the end of each year.
    state : WolvesState
        The last state.
    """
    sim = Simulation(biomes.start(planet, SEED), biomes.phases(), seed=SEED)
    history = sim.run(years * biomes.year, observers={"means": biomes.means}, every=biomes.year)
    return history["means"], sim.state


def main(years: str = "60") -> None:
    """Run the biomes with deer and without, and print the means against the theory.

    Parameters
    ----------
    years : str, default "60"
        How many years each run lasts.
    """
    n_years = int(years)
    biomes = Biomes()
    arrive = biomes.wolves_arrive
    planet = icosphere(16)
    no_deer, lonely = run(dataclasses.replace(biomes, deer=False), planet, n_years)
    clock = time.perf_counter()
    history, shared = run(biomes, planet, n_years)
    seconds = time.perf_counter() - clock
    steps = n_years * biomes.year
    print(f"{shared.n_cells} cells; the biomes alone for {arrive} years, then wolves")
    print(f"{steps} steps with deer in {seconds:.1f} s: {steps / seconds:.0f} steps/s")

    # 1. Apart: each biome's grazers hold its plants at their R*, at meal time.
    apart = history[arrive - 5 : arrive].mean(axis=0)
    print(f"\napart, years {arrive - 4} to {arrive}: plants per unit area where a biome is alone")
    for i, (name, (_, rates)) in enumerate(biomes.rates().items()):
        print(f"  {name:9s}: {apart[2 * i]:.3f}, against the grazers' R* {rates.rstar:.3f}")

    # 2. and 3. Wolves, with rabbits only, then with rabbits and deer.
    print(
        f"\ntheory, in continuous time: wolves hold the grazers at {biomes.held:.3f} on one biome,"
    )
    for name, (_, rates) in biomes.rates().items():
        print(f"  and the {name}'s plants then grow back to {biomes.greener(rates):.3f}")
    for label, h in (("wolves, no deer", no_deer), ("wolves and deer", history)):
        m = h[-10:].mean(axis=0)
        print(
            f"  {label}, the last 10 years: grass {m[0]:.3f}, rabbits {m[1]:.3f}; "
            f"forest plants {m[2]:.3f}, deer {m[3]:.3f}; wolves {m[4]:.4f}"
        )

    print("\napparent competition: grazers per unit area where the biomes mix, by forest cover")
    (without, _), (among, deer) = by_cover(lonely), by_cover(shared)
    bands = " ".join(f"{a:4.2f}-{b:4.2f}" for a, b in itertools.pairwise(COVER))
    print(f"  forest cover       {bands}")
    print("  rabbits, no deer   " + " ".join(f"{x:9.3f}" for x in without))
    print("  rabbits, with deer " + " ".join(f"{x:9.3f}" for x in among))
    print("  deer               " + " ".join(f"{x:9.3f}" for x in deer))
    print("  change in rabbits  " + " ".join(f"{x:9.0%}" for x in among / without - 1))
