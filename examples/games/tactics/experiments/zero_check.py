"""Play the AlphaZero player on Tactics' trained network against its rivals (T8, step 2).

The AlphaZero player (`zero.AlphaZeroPlayer` with `net.NetEvaluator`: Gumbel tree search over
single decisions, the network's prior and value, no noise) plays 2 * ``starts`` games on the
boards players are scored on (`EVAL_SEED`), with the sides swapped and dice of their own, on
every core, against each rival:
``net`` (the network alone, its best action), ``greedy`` or ``search`` (the search player with
its defaults). This is AlphaZero's go/no-go check on Tactics: if the search
can't beat the network it searches with, self-play has nothing to learn from. It prints each
record and how the games went: rounds, the side edge, and hit points removed by each kind of
unit per unit deployed.

Run with ``uv run python run.py tactics.zero_check weights.pt [starts]
[simulations] [rivals ...]``: by default 24 starts, 32 simulations a decision, and the rival
``net``. A decision took 166 ms at 32 simulations on
one core of an i9-12900K, most of it the network on the CPU.
"""

import time

import torch

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    SearchPlayer,
    Tactics,
    balance,
    summarize,
)
from gridzero.agents.net import NetEvaluator, NetPlayer, PolicyNet
from gridzero.agents.zero import AlphaZeroPlayer
from gridzero.game import series


def main(weights: str, starts: str = "24", simulations: str = "32", *rivals: str) -> None:
    """Play the AlphaZero player against its rivals, and print each record.

    Parameters
    ----------
    weights : str
        The saved weights of the network.
    starts : str, default "24"
        Pairs of games against each rival, with the sides swapped.
    simulations : str, default "32"
        The player's simulations a decision.
    *rivals : str
        Rivals by name: ``net`` (the network alone), ``greedy`` or ``search``; ``net`` if none.
    """
    game = Tactics()
    net = PolicyNet(game)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    zero = AlphaZeroPlayer(NetEvaluator(net), simulations=int(simulations))
    others = {"net": NetPlayer(net), "greedy": GreedyPlayer(), "search": SearchPlayer()}
    chosen = list(rivals) or ["net"]
    print(
        f"AlphaZero player on {weights}, {simulations} simulations a decision, {starts} starts "
        f"against each of {', '.join(chosen)}, {CORES} processes.\n",
        flush=True,
    )
    for name in chosen:
        start = time.perf_counter()
        records = series(
            game,
            [("zero", zero), (name, others[name])],
            int(starts),
            EVAL_SEED,
            same_dice=False,
            workers=CORES,
        )
        summary = summarize(game, records)
        counts = summary.tally
        scores = balance(game, summary)
        fair = scores.fairness
        impact = " / ".join(f"{x:.2f}" for x in scores.impact)
        won, lost, drawn = counts.score("zero")
        print(
            f"Against {name}: {won} wins, {lost} losses, "
            f"{drawn} draws in {(time.perf_counter() - start) / 60:.1f} minutes; "
            f"{summary.rounds:.1f} rounds; side edge {fair.seat_edges[0]:+.3f}; impact {impact} "
            f"(footsoldier, archer, knight, captain).",
            flush=True,
        )
