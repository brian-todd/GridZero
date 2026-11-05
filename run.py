"""Run an experiment on one of the examples: the one way to run them.

    uv run python run.py                      # list every experiment
    uv run python run.py predators 100        # run one, with its arguments
    uv run python run.py tactics.zero_check --help

Every experiment is a module beside its example (a simulation's ``experiment.py``, a game's
``experiments/``) with a docstring and ``main(*args: str)``, and is named in `EXPERIMENTS`
below. Run from the repository's root, so that ``examples`` imports.

The experiments that play many games run a process per core, so each process gets one thread
for Torch on the CPU (``OMP_NUM_THREADS=1``, unless set already): with a thread per core in
each, the smallest ``tactics.zero_check`` ran for over 3 minutes instead of 21 s.
"""

import importlib
import inspect
import os
import sys
from collections.abc import Callable, Sequence

EXPERIMENTS = {
    "competition": "examples.simulations.competition.experiment",
    "migration": "examples.simulations.migration.experiment",
    "predators": "examples.simulations.predators.experiment",
    "biomes": "examples.simulations.biomes.experiment",
    "settlers": "examples.simulations.settlers.experiment",
    "population_evolution": "examples.simulations.population_evolution.experiment",
    "tactics.tournament": "examples.games.tactics.experiments.tournament",
    "tactics.search": "examples.games.tactics.experiments.search",
    "tactics.zero_check": "examples.games.tactics.experiments.zero_check",
    "tactics.zero_scratch": "examples.games.tactics.experiments.zero_scratch",
    "tactics.watch": "examples.games.tactics.experiments.watch",
    "in_a_row.zero_scratch": "examples.games.in_a_row.experiments.zero_scratch",
    "in_a_row.zero_check": "examples.games.in_a_row.experiments.zero_check",
    "in_a_row.code_ladder": "examples.games.in_a_row.experiments.code_ladder",
    "in_a_row.code_fit": "examples.games.in_a_row.experiments.code_fit",
    "in_a_row.code_report": "examples.games.in_a_row.experiments.code_report",
    "in_a_row.play": "examples.games.in_a_row.experiments.play",
    "voronoi.tournament": "examples.games.voronoi.experiments.tournament",
    "voronoi.watch": "examples.games.voronoi.experiments.watch",
    "colonies.tournament": "examples.games.colonies.experiments.tournament",
    "colonies.muzero_check": "examples.games.colonies.experiments.muzero_check",
    "colonies.muzero_scratch": "examples.games.colonies.experiments.muzero_scratch",
    "colonies.muzero_ladder": "examples.games.colonies.experiments.muzero_ladder",
    "colonies.watch": "examples.games.colonies.experiments.watch",
    "games.curves": "examples.games.experiments.curves",
}
"""Every experiment, by the name it runs under: its module."""


def experiment(name: str) -> Callable[..., None]:
    """Import an experiment and return its ``main``.

    Parameters
    ----------
    name : str
        A name in `EXPERIMENTS`.

    Returns
    -------
    callable
        Its ``main``, which takes its arguments as strings.
    """
    return importlib.import_module(EXPERIMENTS[name]).main


def usage(name: str | None = None) -> str:
    """Describe how to run an experiment, or list them all.

    Parameters
    ----------
    name : str or None, optional
        An experiment: its arguments and whole docstring. None (the default) lists them all,
        each with the first line of its docstring (which imports every one of them).

    Returns
    -------
    str
        The text to print.
    """
    if name is None:
        rows = [f"  {n:<22} {_doc(n).split(chr(10), 1)[0]}" for n in EXPERIMENTS]
        return "\n".join(["uv run python run.py <experiment> [args]", "", *rows])
    args = " ".join(
        f"<{p.name}>" if p.default is p.empty and p.kind is not p.VAR_POSITIONAL else f"[{p.name}]"
        for p in inspect.signature(experiment(name)).parameters.values()
    )
    return f"uv run python run.py {name} {args}\n\n{_doc(name)}"


def main(argv: Sequence[str]) -> int:
    """Run an experiment from the command line.

    Parameters
    ----------
    argv : sequence of str
        The experiment's name, then its arguments. Nothing, ``-h`` or ``--help`` lists the
        experiments; ``-h`` or ``--help`` after a name describes that one.

    Returns
    -------
    int
        The exit status: 0 when the experiment ran or help was asked for, 2 when the
        arguments didn't fit.
    """
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    name, *args = argv
    if name not in EXPERIMENTS:
        print(f"no experiment {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if args[:1] in (["-h"], ["--help"]):
        print(usage(name))
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # before the experiment imports Torch
    run = experiment(name)
    try:
        inspect.signature(run).bind(*args)
    except TypeError as error:
        print(f"{error}\n\n{usage(name)}", file=sys.stderr)
        return 2
    run(*args)
    return 0


def _doc(name: str) -> str:
    """Return an experiment's module docstring."""
    return inspect.getdoc(importlib.import_module(EXPERIMENTS[name])) or ""


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
