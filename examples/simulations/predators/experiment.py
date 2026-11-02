"""A trophic cascade on a board of triangles: `examples.simulations.predators`, run and checked.

Run with ``uv run python run.py predators [years] [prefix]``, by default
100 years and ``predators`` (giving ``predators_map.png`` and ``predators_history.png`` in the
working directory).
It runs the meadow with fires and without, and prints the means against the theory.
"""

import dataclasses
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples.simulations.predators import (
    BURNING,
    FoxesState,
    Predators,
    means,
)
from gridzero.core import LatticeSpace, Simulation
from gridzero.render import colorize, flat_map, over

SEED = 0  # sets the rabbits' numbers, where the foxes arrive and the fires

SPECIES = (  # name, field, line color, colormap
    ("grass", FoxesState.grass, "tab:green", "YlGn"),
    ("rabbits", FoxesState.rabbits.population, "tab:purple", "Purples"),
    ("foxes", FoxesState.foxes.population, "tab:orange", "Oranges"),
)


def cascade(predators: Predators, name: str, history: NDArray[np.float64]) -> None:
    """Print the means before the foxes and over the last 10 years, against the theory.

    Parameters
    ----------
    predators : Predators
        The parameters of the run.
    name : str
        The run, as printed.
    history : ndarray of float64, shape (years, 4)
        `means` at the end of each year.
    """
    arrive = predators.foxes_arrive
    before = history[arrive // 2 : arrive].mean(axis=0)
    after = history[-10:].mean(axis=0)
    print(
        f"  {name}: grass {before[0]:.3f} with rabbits alone, then rabbits {after[1]:.3f} "
        f"and grass {after[0]:.3f} with foxes: {after[0] / before[0]:.1f}x greener"
    )


def pictures(
    predators: Predators,
    state: FoxesState,
    history: NDArray[np.float64],
    calm: NDArray[np.float64],
    prefix: str,
) -> None:
    """Save the board at the end, and the history of both runs.

    Parameters
    ----------
    predators : Predators
        The parameters of the runs.
    state : FoxesState
        The last state of the run with fire.
    history, calm : ndarray of float64, shape (years, 4)
        `means` at the end of each year, with fire and without.
    prefix : str
        Where to save: ``{prefix}_map.png`` and ``{prefix}_history.png``.
    """
    space = state.space
    raster = flat_map(space, 768)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
    for ax, (name, field, _, cmap) in zip(axes, SPECIES, strict=True):
        density = state.density(field)
        colors = colorize(density, cmap, vmin=0)
        if name == "grass":  # fires in red, ash in gray
            burning = np.zeros((space.n_cells, 4))
            burning[state.fire > BURNING] = [0.4, 0.4, 0.4, 0.6]
            burning[state.fire == BURNING] = [0.9, 0.1, 0.0, 1.0]
            colors = over(burning, colors)
        ax.imshow(raster.image(colors))
        ax.set_title(f"{name}, year {state.step // predators.year}")
        ax.axis("off")
    fig.savefig(f"{prefix}_map.png", dpi=100)
    plt.close(fig)

    arrive = predators.foxes_arrive
    years = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8), layout="constrained")
    levels = (None, predators.foxes_hold, None)
    for i, ((name, _, color, _), level) in enumerate(zip(SPECIES, levels, strict=True)):
        axes[i].plot(years, calm[:, i], color="0.7", linewidth=2, label="without fire")
        axes[i].plot(years, history[:, i], color=color, linewidth=2, label="with fire")
        if level is not None:
            axes[i].axhline(level, color="0.4", linewidth=1)
            axes[i].annotate(
                "held by foxes",
                (years[0], level),
                xytext=(4, -12),
                textcoords="offset points",
                ha="left",
                color="0.3",
            )
        axes[i].set_ylabel(f"{name} per unit area")
        axes[i].legend(frameon=False, loc="upper right")
    axes[3].plot(years, 100 * history[:, 3], color="tab:red", linewidth=2)
    axes[3].set_ylabel("% of cells burning")
    axes[3].set_xlabel("year")
    for ax in axes:
        ax.axvline(arrive, color="0.4", linewidth=1)
        ax.grid(color="0.9", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axes[0].annotate(
        "foxes arrive", (arrive, 0), xytext=(4, 4), textcoords="offset points", color="0.3"
    )
    fig.savefig(f"{prefix}_history.png", dpi=150)
    plt.close(fig)


def run(
    predators: Predators, space: LatticeSpace, years: int
) -> tuple[NDArray[np.float64], FoxesState]:
    """Run from the start of `SEED`.

    Parameters
    ----------
    predators : Predators
        The parameters.
    space : LatticeSpace
        The board.
    years : int
        How long to run.

    Returns
    -------
    history : ndarray of float64, shape (years, 4)
        `means` at the end of each year.
    state : FoxesState
        The last state.
    """
    sim = Simulation(predators.start(space, SEED), predators.phases(), seed=SEED)
    history = sim.run(years * predators.year, observers={"means": means}, every=predators.year)
    return history["means"], sim.state


def main(years: str = "100", prefix: str = "predators") -> None:
    """Run the meadow with fires and without, print the means and save the pictures.

    Parameters
    ----------
    years : str, default "100"
        How many years each run lasts.
    prefix : str, default "predators"
        Where to save the pictures: ``{prefix}_map.png`` and ``{prefix}_history.png``.
    """
    n_years = int(years)
    predators = Predators()
    space = LatticeSpace((192, 128), tiling="triangle", periodic=True)
    clock = time.perf_counter()
    history, state = run(predators, space, n_years)
    seconds = time.perf_counter() - clock
    steps = n_years * predators.year
    print(
        f"{state.n_cells} triangles; rabbits alone for {predators.foxes_arrive} years, then foxes"
    )
    print(f"{steps} steps in {seconds:.1f} s: {steps / seconds:.0f} steps/s")
    for year in range(5, n_years + 1, 5):
        grass, rabbits, foxes, burning = history[year - 1]
        print(
            f"year {year:3d}: grass {grass:.3f}, rabbits {rabbits:.3f}, foxes {foxes:.3f} "
            f"per unit area; {burning:.2%} burning"
        )

    # The cascade, against the levels each consumer holds its food at. Fires burn the grass
    # the rabbits leave, so the theory is checked on a calm meadow, run without them.
    calm, _ = run(dataclasses.replace(predators, wildfire=False), space, n_years)
    print(
        f"\ntheory: rabbits hold grass at {predators.rabbits_hold:.3f}; foxes hold rabbits at "
        f"{predators.foxes_hold:.3f}, and grass then grows to {predators.greener:.3f}"
    )
    cascade(predators, "without fire", calm)
    cascade(predators, "with fire", history)

    pictures(predators, state, history, calm, prefix)
    print(f"saved {prefix}_map.png and {prefix}_history.png")
