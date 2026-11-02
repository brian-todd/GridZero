"""Show training runs generation by generation: any metric of any game's runs, kept current.

Reads each run's ``{prefix}_generations.jsonl`` (`examples.games.curves`), as every training
experiment writes it, and opens a window with one metric at a time for every run, chosen from
the list on the left or with the up and down keys. The runs are read again every 5 s, so a
run still training adds each generation as it ends. With a prefix, every metric goes to
``{prefix}_curves.png`` instead, each on a chart of its own.

Run with ``uv run python run.py games.curves [prefix] runs...``: a window by default (an empty
prefix, ``""``), then the runs, each a file of generations or the prefix it was run with,
such as ``runs/muzero/colonies_scratch_all``.
"""

import matplotlib.pyplot as plt

from examples.games.browse import figure
from examples.games.curves import Curves, grid, log, metrics, name, read


def main(prefix: str = "", *runs: str) -> None:
    """Show the runs' metrics in a window, or save them all.

    Parameters
    ----------
    prefix : str, default ""
        Where the charts go, ``{prefix}_curves.png``, or empty for a window.
    *runs : str
        The runs, at least one.
    """
    if not runs:
        raise ValueError("name at least one run")
    if prefix:
        read_runs = [(name(path), *read(path)) for path in map(log, runs)]
        rows = -(-len(metrics(read_runs)) // 5)
        shown = figure(prefix, (12.5, 2 * rows + 0.6))
        grid(shown, read_runs)
        shown.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2 * rows + 0.6)))
        shown.savefig(f"{prefix}_curves.png", dpi=100)
        print(f"{len(metrics(read_runs))} metrics of {len(runs)} runs: {prefix}_curves.png.")
        return
    curves = Curves(figure(prefix, (12, 7)), runs)
    print(f"{len(curves.metrics)} metrics of {len(runs)} runs; up and down choose one.")
    plt.show()
