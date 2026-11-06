"""Games built on the `game` layer: `tactics`, `in_a_row`, `voronoi` and `colonies`.

`tactics` is a two-player skirmish on hexes; `in_a_row` a family of games of lines, such as
Connect Four and tic-tac-toe; `voronoi` stones placed on a planet by any number of players; and
`colonies` orders on a planet under fog of war, each round resolved by a `core.Simulation`.

Each game is a package listed in its `__init__.py`: `game.py` (the rules, with its state,
position and events, or those in `state.py`, as in `in_a_row`, and `events.py` too, as in
Tactics), `players.py` (scripted players), `match.py` (``EVAL_SEED``, the starts every player
is scored on, and what sets of games show), `picture.py`, `pictures.py` or `text.py` (how a
position is drawn), its experiments in `experiments/`, and a `README.md`. Each game but
`in_a_row` has a `watch` experiment, built on `browse.py`. `experiments/` here holds the
experiments over any game's output, run as ``games.<name>``: `games.curves` shows training runs
(`curves.py`). `fitting.py` holds the arithmetic code players share,
and `regression.py` how their constants are fit.
"""

import os

CORES = os.process_cpu_count() or 1
"""The cores this process may use: the experiments play their series on all of them."""
