"""Set the code player against rivals, on the small board or the full one.

The code player is `agents.AlphaZeroPlayer` on `code_player.HAND_PICKED` or on constants saved by
``tactics.code_fit``, without noise: boards differ from start to start, so games do too. It deploys
as `GreedyPlayer` does (`Fielded`), as it does in ``tactics.fog`` and ``tactics.rules``: deploying
by its own search it won as often against greedy and about 11 games in 100 fewer against the search
player. Prints what a decision and a unit's whole turn cost in battle, on average over one game of
each player against itself, then at each budget of simulations its record against each rival, on
every core. A budget of ``equal`` buys it the time the teacher, the search player the code player
is fit to, takes a decision. Replica 0 is the games players are scored on (`EVAL_SEED`); others
are starts of their own, for a score not chosen on them.

Run with ``uv run python run.py tactics.code_ladder [weights] [starts] [size] [budgets] [replica]
[rivals...]``: by default `HAND_PICKED`, 48 starts on 15x15 (`small_rules`) at budgets ``32,64``,
replica 0, against ``random`` and ``greedy``. Any other size is the full rules. ``teacher`` is the
third rival, and takes about a minute a game on 15x15.
"""

import dataclasses
import time
from typing import NamedTuple

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    HAND_PICKED,
    SEARCH_C_SCALE,
    CodeEvaluator,
    Fielded,
    GreedyPlayer,
    RandomPlayer,
    SearchPlayer,
    Tactics,
    TacticsPosition,
    Verb,
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
    """Print what a decision and a turn cost and the player's record at each budget.

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
    a_simulation = _cost(game, _player(evaluator, TIMED_AT)).decision / TIMED_AT
    print(f"Code player: {a_simulation * 1e6:.0f} us a simulation.", flush=True)
    for budget in budgets.split(","):
        if budget == "equal":
            a_decision = _cost(game, teacher).decision
            print(f"Teacher: {a_decision * 1e3:.1f} ms a decision.", flush=True)
            simulations = max(1, round(a_decision / a_simulation))
        else:
            simulations = int(budget)
        player = _player(evaluator, simulations)
        cost = _cost(game, player)
        print(
            f"{simulations} simulations: {cost.decision * 1e3:.1f} ms a decision, "
            f"{cost.turn * 1e3:.0f} ms a unit's turn.",
            flush=True,
        )
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


def _player(evaluator: CodeEvaluator, simulations: int) -> Player[Tactics, TacticsPosition]:
    """Return the code player at a budget: the search in battle, greedy's deployment."""
    return Fielded(AlphaZeroPlayer(evaluator, simulations=simulations, c_scale=SEARCH_C_SCALE))


class _Cost(NamedTuple):
    """What a player's thinking costs in battle, in seconds, over one game against itself."""

    decision: float
    """A decision, on average."""
    turn: float
    """A unit's whole turn, on average: its decisions up to and with ending it."""


def _cost(game: Tactics, player: Player[Tactics, TacticsPosition]) -> _Cost:
    """Return the mean seconds a decision and a unit's turn take in battle, over one game.

    The player plays itself. Means, since the teacher thinks at a decision in three and follows
    its plan at the others; and of the battle, since every player here deploys as greedy does.
    """
    rng = np.random.default_rng(SEED)
    position = game.start(EVAL_SEED)
    decisions = turns = 0
    spent = 0.0
    while not position.over:
        legal = game.legal(position)
        start = time.perf_counter()
        action = player(game, game.player(position), position, legal, rng)
        if not position.deploying:
            spent += time.perf_counter() - start
            decisions += 1
            turns += action.verb == Verb.END
        position = game.step(position, action)
    return _Cost(spent / max(decisions, 1), spent / max(turns, 1))
