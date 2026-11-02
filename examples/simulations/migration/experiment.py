"""Herds following the rains: `examples.simulations.migration`, run and checked.

Still rain for 10,000 steps, then five years of seasons, in one run on an icosphere of 2,562
cells. Run with ``uv run python run.py migration [prefix]``, in about 15 s. It prints each
herd's distance from the ideal free distribution, and saves globes of the rain and the herds
in the seasons to ``{prefix}_globes.png`` (by default
``migration_globes.png`` in the working directory).
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples.simulations.migration import HERDS, Migration, distance
from gridzero.core import Simulation, icosphere
from gridzero.render import colorize, globe

SEED = 0  # the start is the same for every seed: herds spread evenly, rain at the equator
STILL = 10_000  # steps of still rain before the seasons


def report(title: str, rows: list[tuple[str, NDArray[np.float64]]]) -> None:
    """Print each herd's distance at the given records.

    Parameters
    ----------
    title : str
        What the records are of.
    rows : list of (str, ndarray of float64, shape (n_herds,))
        Each record's label and every herd's `distance`, as columns.
    """
    print(f"\n{title}: share of each herd away from the ideal free distribution")
    print(f"  {'':24s}" + "".join(f"{label:>12s}" for label, _ in rows))
    for i, name in enumerate(HERDS):
        print(f"  {name:24s}" + "".join(f"{value[i]:12.4f}" for _, value in rows))


def main(prefix: str = "migration") -> None:
    """Run still rain, then seasons, print the distances and save the globes.

    Parameters
    ----------
    prefix : str, default "migration"
        Where to save the globes of the rain and the herds in the seasons:
        ``{prefix}_globes.png``.
    """
    migration = Migration(seasons=True, seasons_from=STILL)
    year = migration.year
    planet = icosphere(16)
    print(
        f"{planet.n_cells} cells, areas from {planet.measures.min():.4f} to "
        f"{planet.measures.max():.4f}"
    )
    start = migration.start(planet, SEED)
    sim = Simulation(start, migration.phases(), seed=SEED)

    # 1. Still rain, at the equator.
    history = sim.run(STILL, observers={"distance": distance}, every=100)
    d = history["distance"]
    report(
        "still rain",
        [
            ("start", distance(start)),
            ("step 100", d[0]),
            ("step 1000", d[9]),
            ("step 10000", d[-1]),
        ],
    )

    # 2. Seasons: the band swings with the year, from the herds of the still rain.
    history = sim.run(5 * year, observers={"distance": distance})
    d = history["distance"]
    last = d[-year:]
    report(
        "seasons",
        [
            ("year 1", d[:year].mean(axis=0)),
            ("year 5", last.mean(axis=0)),
            ("worst, yr 5", last.max(axis=0)),
        ],
    )

    # Every herd keeps its animals over all 15,000 steps, to rounding.
    drift = max(abs(sim.state.total(herd) / start.total(herd) - 1) for herd in HERDS.values())
    print(f"\nlargest change in a herd's total over {sim.state.step} steps: {drift:.1e}")

    # The picture: rain and three herds, at the height of the northern summer.
    sim.run((year // 4 - sim.state.step) % year)
    state = sim.state
    panels = [("rain", state.rain)] + [
        (name, state.density(HERDS[name]))
        for name in ("Spread toward food", "Flow down the gradient", "NeighborChoice, best")
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4), layout="constrained")
    for ax, (name, values) in zip(axes, panels, strict=True):
        ax.imshow(globe(planet, view=(0.0, 20.0), size=400).image(colorize(values, "YlGn", vmin=0)))
        ax.set_title(name)
        ax.axis("off")
    fig.savefig(f"{prefix}_globes.png", dpi=100)
    print(f"\nsaved {prefix}_globes.png")
