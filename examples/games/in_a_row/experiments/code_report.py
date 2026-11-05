"""Write a report on a code player: where it and its teacher disagree, and who is right.

Loads weights saved by ``zero_scratch`` (32 channels, 4 blocks), the teacher, and what
``code_fit`` wrote under a prefix: ``{prefix}_lessons.npz`` and ``{prefix}_round<k>.json``. On
those lessons it measures how each fit does on each round's positions, whether a search steers
into its own errors, the labels against a deeper search and exact values, what candidate
features would add, and the widest disagreements. Writes ``{prefix}_report.html``, one file that
needs nothing else to open.

Run with ``uv run python run.py in_a_row.code_report weights.pt [shown] [prefix]``: by default
4 cards per range of plies in each gallery, prefix ``code``.
"""

import time
from pathlib import Path

from examples.games import CORES
from examples.games.in_a_row import FittedEvaluator, InARow, Lessons
from examples.games.in_a_row.report import analyze, page
from gridzero.agents.net import NetEvaluator

from .zero_check import saved_net

SEED = 6_000_000  # the report's samples, the games it keeps out, and its searches' ties


def main(weights: str, shown: str = "4", prefix: str = "code") -> None:
    """Measure a code player against its teacher, and write the report.

    Parameters
    ----------
    weights : str
        The teacher's saved weights.
    shown : str, default "4"
        Cards per range of plies in each gallery.
    prefix : str, default "code"
        What ``code_fit`` wrote under, and what the report is written under.
    """
    game = InARow()
    net = saved_net(game, weights)
    folder = Path(prefix).parent
    name = Path(prefix).name
    paths = sorted(
        folder.glob(f"{name}_round*.json"),
        key=lambda path: int(path.stem.removeprefix(f"{name}_round")),
    )
    fits = [FittedEvaluator.load(path) for path in paths]
    lessons = Lessons.load(f"{prefix}_lessons.npz")
    start = time.perf_counter()
    report = analyze(
        game, NetEvaluator(net), fits, lessons, shown=int(shown), seed=SEED, workers=CORES
    )
    written = f"{prefix}_report.html"
    Path(written).write_text(page(game, report))
    print(f"Wrote {written} in {time.perf_counter() - start:.0f} s.")
