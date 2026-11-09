"""Set the code player against rivals, on the small board or the full one.

The code player is `agents.AlphaZeroPlayer` on `code_player.HAND_PICKED` or on constants saved by
``tactics.code_fit``, without noise: boards differ from start to start, so games do too. Prints
what a decision costs, on average over one game of each player against itself, then at each budget
of simulations its record against each rival, on every core. A budget of ``equal`` buys it the time
the teacher, the search player the code player is fit to, takes a decision. Replica 0 is the games
players are scored on (`EVAL_SEED`); others are starts of their own, for a score not chosen on them.

Run with ``uv run python run.py tactics.code_ladder [weights] [starts] [size] [budgets] [replica]
[rivals...]``: by default `HAND_PICKED`, 48 starts on 15x15 (`small_rules`) at budgets ``32,64``,
replica 0, against ``random`` and ``greedy``. Any other size is the full rules. ``teacher`` is the
third rival, and takes about a minute a game on 15x15.
"""

import dataclasses
import time

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    HAND_PICKED,
    SEARCH_C_SCALE,
    CodeEvaluator,
    GreedyPlayer,
    RandomPlayer,
    SearchPlayer,
    Tactics,
    TacticsPosition,
    small_rules,
)
from gridzero.agents import AlphaZeroPlayer
from gridzero.game import Player, ladder

SEED = 1  # the draws of the timed games
SEEDS_PER_REPLICA = 100_000  # keeps replicas' starts apart
TEACHER_SIMULATIONS = 50  # as in `code_games`, which it was fit to
TIMED_AT = 32  # simulations a decision when timing a simulation of the code player


def main(
    weights: str = "hand",
    starts: str = "48",
    size: str = "15",
    budgets: str = "32,64",
    replica: str = "0",
    *rivals: str,
) -> None:
    """Print what a decision costs and the player's record at each budget.

    Parameters
    ----------
    weights : str, default "hand"
        Constants saved by ``tactics.code_fit``, or ``hand`` for `HAND_PICKED`.
    starts : str, default "48"
        Starts against each rival, each played once per rotation of the seats.
    size : str, default "15"
        The side of the board: 15 for `small_rules`, else the full rules on that size.
    budgets : str, default "32,64"
        The budgets, in simulations a decision, or ``equal``, separated by commas.
    replica : str, default "0"
        Which set of starts: 0 is the games players are scored on.
    *rivals : str
        Names of the rivals: ``random``, ``greedy``, ``teacher``; by default the first two.

    Raises
    ------
    KeyError
        If a rival has no such name.
    """
    side = int(size)
    game = small_rules() if side == 15 else dataclasses.replace(Tactics(), shape=(side, side))
    evaluator = HAND_PICKED if weights == "hand" else CodeEvaluator.load(weights)
    teacher = SearchPlayer(simulations=TEACHER_SIMULATIONS)
    known: dict[str, Player[Tactics, TacticsPosition]] = {
        "random": RandomPlayer(),
        "greedy": GreedyPlayer(),
        "teacher": teacher,
    }
    chosen = {name: known[name] for name in rivals or ("random", "greedy")}
    seed = first_start(int(replica))
    a_simulation = (
        _cost(game, AlphaZeroPlayer(evaluator, simulations=TIMED_AT, c_scale=SEARCH_C_SCALE))
        / TIMED_AT
    )
    print(f"Code player: {a_simulation * 1e6:.0f} us a simulation.", flush=True)
    for budget in budgets.split(","):
        if budget == "equal":
            a_decision = _cost(game, teacher)
            print(f"Teacher: {a_decision * 1e3:.1f} ms a decision.", flush=True)
            simulations = max(1, round(a_decision / a_simulation))
        else:
            simulations = int(budget)
        player = AlphaZeroPlayer(evaluator, simulations=simulations, c_scale=SEARCH_C_SCALE)
        print(f"{simulations} simulations: {_cost(game, player) * 1e3:.1f} ms.", flush=True)
        start = time.perf_counter()
        scores = ladder(game, player, chosen, int(starts), seed, workers=CORES)
        text = ", ".join(f"{rival} {s.won}-{s.lost}" for rival, s in scores.items())
        print(f"Against {text} (wins-losses), {time.perf_counter() - start:.0f} s.", flush=True)


def first_start(replica: int) -> int:
    """Return the seed of a replica's first start.

    Parameters
    ----------
    replica : int
        Which set of starts: 0 is the games players are scored on.

    Returns
    -------
    int
        The seed.
    """
    return EVAL_SEED + replica * SEEDS_PER_REPLICA


def _cost(game: Tactics, player: Player[Tactics, TacticsPosition]) -> float:
    """Return the mean seconds a decision takes over one game of the player against itself.

    The mean, since the teacher thinks at a decision in three and follows its plan at the others.
    """
    rng = np.random.default_rng(SEED)
    position = game.start(EVAL_SEED)
    decisions = 0
    spent = 0.0
    while not position.over:
        legal = game.legal(position)
        start = time.perf_counter()
        action = player(game, game.player(position), position, legal, rng)
        spent += time.perf_counter() - start
        decisions += 1
        position = game.step(position, action)
    return spent / decisions
