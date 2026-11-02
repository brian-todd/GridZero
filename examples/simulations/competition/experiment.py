"""Two groups competing for one food supply: `examples.simulations.competition`, run and checked.

Run with ``uv run python run.py competition [prefix]``. It prints the checks against the R*
rule and saves a map of the two niches to ``{prefix}_map.png`` (by default
``competition_map.png`` in the working directory).
"""

import dataclasses
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples.simulations.competition import (
    GROUPS,
    Competition,
    CompetitionState,
)
from gridzero.core import LatticeSpace, Simulation
from gridzero.render import flat_map, mix
from gridzero.worlds import terrain_colors

SEED = 1  # a terrain with land enough for both niches


def run(
    competition: Competition, space: LatticeSpace, steps: int
) -> tuple[CompetitionState, dict[str, NDArray[Any]]]:
    """Run from the start of `SEED`.

    Parameters
    ----------
    competition : Competition
        The parameters.
    space : LatticeSpace
        The lattice.
    steps : int
        How long to run.

    Returns
    -------
    state : CompetitionState
        The last state.
    history : dict of str to ndarray
        The step (int64, shape (5,)) and each group's total people (float64, shape (5, 2)),
        every fifth of the run.
    """
    sim = Simulation(competition.start(space, SEED), competition.phases(), seed=SEED)
    totals = {"people": lambda s: s.total(CompetitionState.people)}
    history = sim.run(steps, observers=totals, every=steps // 5)
    return sim.state, history


def main(prefix: str = "competition") -> None:
    """Run one niche and two, print the checks and save the map.

    Parameters
    ----------
    prefix : str, default "competition"
        Where to save the map of the two niches: ``{prefix}_map.png``.
    """
    competition = Competition()
    space = LatticeSpace((64, 64), tiling="hex", periodic=True)

    # One niche: the group with the lower R* takes over from 1% of the other's numbers.
    state, history = run(competition, space, steps=1000)
    land = state.land
    print(f"{space.n_cells} hexes, {land.mean():.0%} land")
    rstar = competition.rstar(state.mortality)[:, land].mean(axis=1)
    print(f"\none niche: R* {rstar[0]:.2f} for {GROUPS[0]}, {rstar[1]:.2f} for {GROUPS[1]}")
    for step, people in zip(history["step"], history["people"], strict=True):
        print(f"  step {step:4d}: {people[0]:7.1f} {GROUPS[0]}, {people[1]:7.1f} {GROUPS[1]}")
    meal = competition.meal(state)[land]
    print(
        f"  food density when people eat: {meal.mean():.4f} on average "
        f"(R* of the winner {rstar.min():.2f}), from {meal.min():.4f} to {meal.max():.4f}"
    )
    share = state.people[1].sum() / state.people.sum()
    print(f"  {GROUPS[1]} are {share:.2%} of the people: {'excluded' if share < 0.01 else 'NOT'}")

    # Two niches: each group should hold the half of the land where its R* is lower.
    split = dataclasses.replace(competition, split=True)
    state, history = run(split, space, steps=2000)
    expected = np.argmin(competition.rstar(state.mortality), axis=0)[land]
    holds = np.argmax(state.people, axis=0)[land]
    print("\ntwo niches:")
    for step, people in zip(history["step"], history["people"], strict=True):
        print(f"  step {step:4d}: {people[0]:7.1f} {GROUPS[0]}, {people[1]:7.1f} {GROUPS[1]}")
    print(f"  the group with the lower R* holds {(holds == expected).mean():.1%} of land cells")
    mixed = state.people[:, land].min(axis=0) / state.people[:, land].sum(axis=0)
    print(f"  {(mixed > 0.1).mean():.1%} of land cells have more than 10% of the other group")

    # The picture: land colored by the share of each group, over the sea.
    colors = terrain_colors(state.height)
    colors[land] = mix(state.people[:, land], ["tab:orange", "tab:purple"])
    plt.imsave(f"{prefix}_map.png", flat_map(space, 1024).image(colors))
    print(f"\nsaved {prefix}_map.png")
