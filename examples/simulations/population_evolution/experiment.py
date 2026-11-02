"""Species evolving on a planet: `examples.simulations.population_evolution`, run and drawn.

Run with ``uv run python run.py population_evolution [steps] [prefix] [warming]``, by default
6000 steps, ``population_evolution`` (giving ``population_evolution_map.png`` and
``population_evolution_history.png`` in the working directory) and no warming. A founder
lands on a planet of 2,562 cells. Every 1000 steps it prints the share of its continent
occupied, the species (clusters of traits) and how steeply the mean trait follows the
temperature. With ``warming``, the
temperature rises by that much a step from halfway through the run: 1e-4 is about a tenth of
the speed at which one place's population can't keep up. The cold species die out first, as
the coldest climates disappear. Traits can't follow past the last bin (1.25): over 6000
steps, 1e-4 warms the hottest land to about 1.6, and the warmest traits pile up there.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples.simulations.population_evolution import (
    Evolution,
    EvolutionState,
    clusters,
    species,
    spectrum,
)
from gridzero.core import MeshSpace, Simulation, icosphere, regions
from gridzero.core.geometry import lon_lat
from gridzero.render import colorize, over, world_map
from gridzero.worlds import terrain_colors

SEED = 1  # the founders land on a continent with room for several species
RECORD = 50  # steps between records of the spectrum


def report(evolution: Evolution, state: EvolutionState, continent: NDArray[np.bool_]) -> str:
    """Return a line on a state: its continent occupied, its species and its gradient.

    Parameters
    ----------
    evolution : Evolution
        The parameters of the run.
    state : EvolutionState
        A state.
    continent : ndarray of bool, shape (n_cells,)
        The landmass the founders landed on.

    Returns
    -------
    str
        The line to print.
    """
    x = evolution.traits
    lived = state.life.sum(axis=0) > 0
    found = clusters(spectrum(state))
    traits = ", ".join(f"{x[a]:.2f}-{x[b]:.2f}" for a, b in found)
    return (
        f"step {state.step:6d}: {lived[continent].mean():4.0%} of the continent, "
        f"{len(found)} species ({traits}), gradient {evolution.gradient(state):.3f}"
    )


def pictures(
    evolution: Evolution,
    planet: MeshSpace,
    state: EvolutionState,
    history: NDArray[np.float64],
    landing: int,
    prefix: str,
) -> None:
    """Draw the planet, its traits and species, and the traits over time.

    Parameters
    ----------
    evolution : Evolution
        The parameters of the run.
    planet : MeshSpace
        The cells.
    state : EvolutionState
        The last state.
    history : ndarray of float64, shape (records, BINS)
        `spectrum` every `RECORD` steps.
    landing : int
        The founders' cell: the maps are centered on its longitude.
    prefix : str
        Where to save: ``{prefix}_map.png`` and ``{prefix}_history.png``.
    """
    terrain = terrain_colors(state.height)
    raster = world_map(planet, 1024, center=float(lon_lat(planet.centers[landing])[0]))
    temperature = np.where(state.land, state.temperature, np.nan)
    hottest = max(1.0, float(np.nanmax(temperature)))  # past 1 after warming
    labels = species(state)
    main = np.where(labels >= 0, labels, np.nan)
    panels = (
        ("the planet", terrain),
        (
            "temperature on land",
            over(colorize(temperature, "coolwarm", vmin=0, vmax=hottest), terrain),
        ),
        (
            "mean trait: the best temperature",
            over(colorize(evolution.mean_trait(state), "coolwarm", vmin=0, vmax=hottest), terrain),
        ),
        (
            "species: the cluster of traits holding most of a cell",
            over(colorize(main, "tab20"), terrain),
        ),
    )
    fig, axes = plt.subplots(4, 1, figsize=(8, 16), layout="constrained")
    for ax, (title, colors) in zip(axes, panels, strict=True):
        ax.imshow(raster.image(colors))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(f"{prefix}_map.png", dpi=100)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    shares = history / history.sum(axis=1, keepdims=True)
    x = evolution.traits
    step = (x[1] - x[0]) / 2
    image = left.imshow(
        np.log10(np.maximum(shares.T, 1e-6)),
        origin="lower",
        aspect="auto",
        extent=(0, len(history) * RECORD, x[0] - step, x[-1] + step),
        cmap="magma",
    )
    fig.colorbar(image, ax=left, label="log10 share of the individuals")
    left.set(xlabel="step", ylabel="trait: best temperature", title="evolution of the traits")
    mean = evolution.mean_trait(state)
    lived = ~np.isnan(mean)
    right.scatter(state.temperature[lived], mean[lived], s=6, alpha=0.5)
    right.plot([0, 1], [0, 1], color="gray", linestyle="--", label="traits matching the climate")
    right.set(xlabel="temperature", ylabel="mean trait", title="gradient of traits")
    right.legend()
    fig.savefig(f"{prefix}_history.png", dpi=150)
    plt.close(fig)


def main(steps: str = "6000", prefix: str = "population_evolution", warming: str = "0") -> None:
    """Run the planet's evolution, print it every 1000 steps, and draw it.

    Parameters
    ----------
    steps : str, default "6000"
        How many steps to run.
    prefix : str, default "population_evolution"
        Where to save the pictures (see `pictures`).
    warming : str, default "0"
        Rise in temperature per step, from halfway through the run.
    """
    n_steps = int(steps)
    evolution = Evolution(warming=float(warming), warming_from=n_steps // 2)
    planet = icosphere(16)
    start = evolution.start(planet, SEED)
    landmasses = regions(planet, start.land)
    (landing,) = np.flatnonzero(start.life.sum(axis=0))
    continent = landmasses == landmasses[landing]
    print(f"{planet.n_cells} cells; the founders land on a continent of {continent.sum()}")
    if evolution.warming:
        print(f"warming {evolution.warming:g} a step from step {evolution.warming_from}")
    sim = Simulation(start, evolution.phases(), seed=SEED)
    records = []
    clock = time.perf_counter()
    while sim.state.step < n_steps:
        stretch = min(1000, n_steps - sim.state.step)
        history = sim.run(stretch, observers={"spectrum": spectrum}, every=RECORD)
        records.append(history["spectrum"])
        print(report(evolution, sim.state, continent))
    seconds = time.perf_counter() - clock
    print(f"{n_steps} steps in {seconds:.1f} s: {n_steps / seconds:.0f} steps/s")
    pictures(evolution, planet, sim.state, np.concatenate(records), int(landing), prefix)
    print(f"saved {prefix}_map.png and {prefix}_history.png")
