"""Farmers and herders settling a world: `examples.simulations.settlers`, run and timed.

Run with ``uv run python run.py settlers [years] [prefix]``, by default
1000 years and ``settlers`` (giving ``settlers_map.png`` and ``settlers_history.png`` in the
working directory). It
prints the totals 20 times along the run and how fast each stretch ran, and saves a map of the
end and a chart of the history.
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from examples.simulations.settlers import (
    PEOPLES,
    Settlement,
    SettlersState,
)
from gridzero.core import LatticeSpace, Simulation
from gridzero.render import flat_map, mix
from gridzero.worlds import terrain_colors

COLORS = ("tab:orange", "tab:purple")  # farmers, herders: the same on the map and the chart

SEED = 4  # sets the terrain, the homes and the droughts


def chart(history: dict[str, NDArray[Any]], year: int, path: str) -> None:
    """Save the history: each people's numbers, and the fertility of the land, by year.

    Parameters
    ----------
    history : dict of str to ndarray
        What `main` records at the end of each year: at least ``step`` (int64, shape
        (years,)), ``people`` (float64, shape (years, 2)) and ``fertility`` (float64, shape
        (years,)).
    year : int
        Steps in a year.
    path : str
        Where to save the chart.
    """
    years = history["step"] / year
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5), height_ratios=(3, 1), layout="constrained"
    )
    for group, (name, color) in enumerate(zip(PEOPLES, COLORS, strict=True)):
        people = history["people"][:, group]
        top.plot(years, people, color=color, linewidth=2, label=name)
        top.annotate(
            f"{name} {people[-1]:,.0f}",
            (years[-1], people[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color="0.25",
        )
    bottom.plot(years, history["fertility"], color="0.35", linewidth=2)
    top.set_ylabel("people")
    top.yaxis.set_major_formatter(lambda y, _: f"{y / 1000:.0f}K" if y else "0")
    top.legend(frameon=False, loc="lower right")
    bottom.set_ylabel("fertility of land")
    bottom.set_xlabel("year")
    for ax in (top, bottom):
        ax.grid(color="0.9", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0)
    top.set_ylim(bottom=0)
    top.set_xlim(right=years[-1] * 1.15)  # room for the labels at the ends of the lines
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(years: str = "1000", prefix: str = "settlers") -> None:
    """Run the settlement, print the totals and save the map and the chart.

    Parameters
    ----------
    years : str, default "1000"
        How many years to run.
    prefix : str, default "settlers"
        Where to save the pictures: ``{prefix}_map.png`` and ``{prefix}_history.png``.
    """
    n_years = int(years)
    settlement = Settlement()
    year, settled = settlement.year, settlement.settled
    space = LatticeSpace((128, 128), tiling="hex", periodic=True)
    sim = Simulation(settlement.start(space, SEED), settlement.phases(), seed=SEED)
    print(
        f"{space.n_cells} hexes, {sim.state.land.mean():.0%} land, "
        f"{len(settlement.phases())} phases"
    )

    def smallest(s: SettlersState) -> float:
        """Return the smallest amount of people above 0 in any cell, to watch for subnormals."""
        alive = s.settlers.people[s.settlers.people > 0]
        return float(alive.min()) if alive.size else np.nan

    observers = {
        "people": lambda s: s.total(SettlersState.settlers.people),
        "stores": lambda s: s.total(SettlersState.settlers.stores),
        "fertility": lambda s: s.fertility[s.land].mean(),
        "settled": lambda s: (s.density(SettlersState.settlers.people) > settled).sum(axis=1),
        "smallest": smallest,
        "clock": lambda s: time.perf_counter(),  # to see whether the steps slow down
    }
    start = time.perf_counter()
    history = sim.run(n_years * year, observers=observers, every=year)
    seconds = time.perf_counter() - start
    every = max(n_years // 20, 1)
    clock = np.concatenate([[start], history["clock"]])
    for i in range(every - 1, n_years, every):
        people, cells = history["people"][i], history["settled"][i]
        speed = every * year / (clock[i + 1] - clock[i + 1 - every])
        print(
            f"year {i + 1:5d}: {people[0]:9.1f} farmers in {cells[0]:5d} cells, "
            f"{people[1]:9.1f} herders in {cells[1]:5d}, "
            f"fertility {history['fertility'][i]:.3f}, smallest {history['smallest'][i]:.1e}, "
            f"{speed:4.0f} steps/s"
        )
    print(f"{n_years * year} steps in {seconds:.1f} s: {n_years * year / seconds:.0f} steps/s")

    state = sim.state
    colors = terrain_colors(state.height)
    peopled = state.land & (state.density(SettlersState.settlers.people).sum(axis=0) > settled)
    colors[peopled] = mix(state.settlers.people[:, peopled], COLORS)
    plt.imsave(f"{prefix}_map.png", flat_map(space, 1024).image(colors))
    chart(history, year, f"{prefix}_history.png")
    print(f"saved {prefix}_map.png and {prefix}_history.png")
