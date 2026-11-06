"""Set the code player against random and greedy, on the small board or the full one.

The code player is `agents.AlphaZeroPlayer` on `code_player.HAND_PICKED`, without noise: boards
differ from start to start, so games do too. Prints what a decision costs over one game of the
player against itself, then at each budget of simulations its record against each rival on the
games players are scored on (`EVAL_SEED`), on every core.

Run with ``uv run python run.py tactics.code_ladder [starts] [size] [simulations...]``: by default
48 starts on 15x15 (`small_rules`) at 32 and 64 simulations. Any other size is the full rules.
"""

import dataclasses
import time

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    HAND_PICKED,
    GreedyPlayer,
    RandomPlayer,
    Tactics,
    TacticsPosition,
    small_rules,
)
from gridzero.agents import AlphaZeroPlayer
from gridzero.game import Player, ladder

SEED = 1  # the draws of the timed game


def main(starts: str = "48", size: str = "15", *simulations: str) -> None:
    """Print what a decision costs and the player's record at each budget.

    Parameters
    ----------
    starts : str, default "48"
        Starts against each rival, each played once per rotation of the seats.
    size : str, default "15"
        The side of the board: 15 for `small_rules`, else the full rules on that size.
    *simulations : str
        The budgets, in simulations a decision; by default 32 and 64.
    """
    side = int(size)
    game = small_rules() if side == 15 else dataclasses.replace(Tactics(), shape=(side, side))
    rivals: dict[str, Player[Tactics, TacticsPosition]] = {
        "random": RandomPlayer(),
        "greedy": GreedyPlayer(),
    }
    for budget in [int(s) for s in simulations or ("32", "64")]:
        player = AlphaZeroPlayer(HAND_PICKED, simulations=budget)
        print(f"{budget} simulations: {_cost(game, player) * 1e3:.1f} ms a decision.", flush=True)
        start = time.perf_counter()
        scores = ladder(game, player, rivals, int(starts), EVAL_SEED, workers=CORES)
        text = ", ".join(f"{rival} {s.won}-{s.lost}" for rival, s in scores.items())
        print(f"Against {text} (wins-losses), {time.perf_counter() - start:.0f} s.", flush=True)


def _cost(game: Tactics, player: AlphaZeroPlayer[Tactics, TacticsPosition]) -> float:
    """Return the median seconds a decision takes over one game of the player against itself."""
    rng = np.random.default_rng(SEED)
    position = game.start(EVAL_SEED)
    times = []
    while not position.over:
        legal = game.legal(position)
        start = time.perf_counter()
        action = player(game, game.player(position), position, legal, rng)
        times.append(time.perf_counter() - start)
        position = game.step(position, action)
    return float(np.median(times))
