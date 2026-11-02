"""Training runs generation by generation, from the figures each run writes as it goes.

Every training experiment (``zero_scratch``, ``muzero_scratch``) appends a line
of JSON a generation to ``{prefix}_generations.jsonl``. `read` turns such a file into one
series per figure, whatever the game: numbers by their key, keys inside objects joined with
dots (``scores.muzero.raider``), a list of numbers as one metric of as many values (the losses
of each unrolled step, wins, losses and draws), and other lists by their positions
(``opening[1]``). `Curves` shows one metric at a time for any number of runs, chosen from a
list beside the chart, and reads the files again as they grow; `grid` draws every metric at
once.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import RadioButtons
from numpy.typing import NDArray

RUNS = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c2417a", "#7a7a7a", "#c99a1a", "#2fa3b5")
"""Each run's color, in order: runs past the eighth take them again from the first."""
STYLES = ("-", "--", ":", "-.")
"""Each value's line style, in order, for a metric of several values a generation."""

type Run = dict[str, NDArray[np.float64]]
"""A run's metrics by name: each of float64 of shape (generations, values), NaN where a
generation didn't record it."""


def log(path: str | Path) -> Path:
    """Return a run's file: the path itself, or ``{path}_generations.jsonl``.

    Parameters
    ----------
    path : str or Path
        A file of generations, or the prefix a run was given.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If neither exists.
    """
    path = Path(path)
    for candidate in (path, Path(f"{path}_generations.jsonl")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no run at {path} or {path}_generations.jsonl")


def new_log(prefix: str) -> Path:
    """Return the file of generations of a new run, ``{prefix}_generations.jsonl``.

    Training experiments append each generation to it, so a run given the prefix of an
    earlier one would carry on its file, and replace its weights, without a word.

    Parameters
    ----------
    prefix : str
        The run's prefix.

    Returns
    -------
    Path
        The file, not yet made.

    Raises
    ------
    FileExistsError
        If a run already used the prefix.
    """
    path = Path(f"{prefix}_generations.jsonl")
    if path.exists():
        raise FileExistsError(f"{path} exists: give the new run a prefix of its own")
    return path


def name(path: Path) -> str:
    """Return a run's name: its file's without ``_generations``, or its folder's."""
    stem = path.stem.removesuffix("_generations")
    return path.parent.name if stem == "generations" else stem


def read(path: str | Path) -> tuple[NDArray[np.float64], Run]:
    """Read a run's generations so far.

    Parameters
    ----------
    path : str or Path
        A file of generations, or the prefix a run was given (`log`).

    Returns
    -------
    generations : ndarray of float64, shape (generations,)
        Each line's ``generation``, or its line number from 1 where it has none.
    metrics : Run
        Each metric, in the order first seen. A line not yet ended (a run writing it) is left
        out.

    Raises
    ------
    FileNotFoundError
        If there is no run there.
    """
    lines = log(path).read_text().split("\n")[:-1]  # the last is empty, or still being written
    entries = [_flat(json.loads(line)) for line in lines if line.strip()]
    generations = np.array(
        [e.pop("generation", [k + 1.0])[0] for k, e in enumerate(entries)], np.float64
    )
    widths: dict[str, int] = {}
    for entry in entries:
        for key, values in entry.items():
            widths[key] = max(widths.get(key, 0), len(values))
    metrics = {key: np.full((len(entries), width), np.nan) for key, width in widths.items()}
    for g, entry in enumerate(entries):
        for key, values in entry.items():
            metrics[key][g, : len(values)] = values
    return generations, metrics


def _flat(value: object, key: str = "") -> dict[str, list[float]]:
    """Return the numbers in a value of JSON by their keys, a list of numbers as one."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return {}
    if isinstance(value, int | float):
        return {key: [float(value)]}
    if isinstance(value, dict):
        out: dict[str, list[float]] = {}
        for k, v in value.items():
            out |= _flat(v, f"{key}.{k}" if key else str(k))
        return out
    if isinstance(value, list):
        if value and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
            return {key: [float(v) for v in value]}
        out = {}
        for i, v in enumerate(value):
            out |= _flat(v, f"{key}[{i}]")
        return out
    return {}


def plot(ax: Axes, runs: Sequence[tuple[str, NDArray[np.float64], Run]], metric: str) -> None:
    """Draw one metric of every run that has it: a color per run, a line style per value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Where to draw; cleared first.
    runs : sequence of (str, ndarray, Run)
        Each run's name, generations and metrics, as `read` gives them.
    metric : str
        The metric's name.
    """
    ax.clear()
    for k, (run, generations, metrics) in enumerate(runs):
        if metric not in metrics:
            continue
        values = metrics[metric]
        for i in range(values.shape[1]):
            ax.plot(
                generations,
                values[:, i],
                color=RUNS[k % len(RUNS)],
                linestyle=STYLES[i % len(STYLES)],
                linewidth=1.5,
                label=run if i == 0 else None,
            )
    ax.set_title(metric, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def metrics(runs: Sequence[tuple[str, NDArray[np.float64], Run]]) -> list[str]:
    """Return every metric any run has, in the order first seen."""
    return list(dict.fromkeys(key for _, _, m in runs for key in m))


class Curves:
    """One metric at a time for any number of runs, chosen beside the chart, kept current.

    The list on the left chooses the metric, as do the up and down keys. Every ``interval``
    milliseconds the runs are read again, and the chart redrawn when one has grown, so a run
    still training shows each generation as it ends. A metric of several values a generation
    draws each in a line style of its own (`STYLES`: the first solid). In a window, keep it
    referenced while it is open: matplotlib holds its widgets only weakly.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to draw on, from ``pyplot.figure()`` to show it in a window.
    paths : sequence of str or Path
        The runs: files of generations, or the prefixes they were given (`log`).
    interval : int, default 5000
        Milliseconds between reading the runs again.

    Attributes
    ----------
    runs : list of (str, ndarray, Run)
        Each run's name, generations and metrics, as last read.
    metrics : list of str
        Every metric, as first read: the list's choices.
    buttons : matplotlib.widgets.RadioButtons
        The list, whose selected label is the metric shown.
    timer : matplotlib.backend_bases.TimerBase
        The timer that reads the runs again.

    Raises
    ------
    ValueError
        If no path is given.
    FileNotFoundError
        If a path holds no run.
    """

    def __init__(
        self, figure: Figure, paths: Sequence[str | Path], *, interval: int = 5000
    ) -> None:
        if not paths:
            raise ValueError("need at least one run")
        self.figure = figure
        self._paths = [log(p) for p in paths]
        self.runs = [(name(p), *read(p)) for p in self._paths]
        self.metrics = metrics(self.runs)
        self.ax = figure.add_axes((0.33, 0.1, 0.62, 0.82))
        self.buttons = RadioButtons(
            figure.add_axes((0.01, 0.02, 0.26, 0.96), frameon=False),
            self.metrics,
            label_props={"fontsize": [7]},
            radio_props={"s": [20]},
        )
        self.buttons.on_clicked(lambda _: self.draw())
        figure.canvas.mpl_connect("key_press_event", self._key)
        self.timer = figure.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.draw()

    @property
    def metric(self) -> str:
        """The metric shown."""
        return self.buttons.value_selected

    def draw(self) -> None:
        """Draw the metric shown, with a legend of the runs."""
        plot(self.ax, self.runs, self.metric)
        self.ax.set_xlabel("generation")
        self.ax.legend(loc="best", frameon=False, fontsize=8)
        self.figure.canvas.draw_idle()

    def refresh(self) -> bool:
        """Read the runs again, and redraw if one has grown.

        Returns
        -------
        bool
            Whether one had grown.
        """
        runs = [(name(p), *read(p)) for p in self._paths]
        grown = [len(g) for _, g, _ in runs] != [len(g) for _, g, _ in self.runs]
        self.runs = runs
        if grown:
            self.draw()
        return grown

    def _tick(self) -> None:
        """Refresh, returning None: a timer drops a callback that returns False."""
        self.refresh()

    def _key(self, event: Event) -> None:
        """Choose the next or the previous metric."""
        if not isinstance(event, KeyEvent) or event.key not in ("up", "down"):
            return
        step = 1 if event.key == "down" else -1
        self.buttons.set_active((self.metrics.index(self.metric) + step) % len(self.metrics))


def grid(figure: Figure, runs: Sequence[tuple[str, NDArray[np.float64], Run]]) -> None:
    """Draw every metric of the runs, each on axes of its own, with one legend.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to draw on: give it about 2.5 inches a column and 2 a row, 5 columns.
    runs : sequence of (str, ndarray, Run)
        Each run's name, generations and metrics, as `read` gives them.
    """
    names = metrics(runs)
    rows = -(-len(names) // 5)
    for k, metric in enumerate(names):
        ax = figure.add_subplot(rows, 5, k + 1)
        plot(ax, runs, metric)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], color=RUNS[k % len(RUNS)]) for k in range(len(runs))]
    labels = [run for run, _, _ in runs]
    figure.legend(handles, labels, loc="upper center", ncols=min(len(runs), 4), frameon=False)
