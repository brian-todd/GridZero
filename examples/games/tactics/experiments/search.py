"""The search player against the greedy player, at a few budgets of simulations.

Each budget plays a series on the boards players are scored on (`EVAL_SEED`): every board
twice, the sides swapped, with the same dice, the games in parallel on every core. It prints
who won, from which side, how long games ran, and what each kind of unit did, and checks what a
search that sees ahead should show:

- **search beats greedy:** it wins more than half of the decided games, by more than two
  standard deviations of a fair coin, at the largest budget.

Run with ``uv run python run.py tactics.search [starts] [simulations ...]``: by
default 24 starts at 100 simulations a turn, with the default rollouts of 16 unit
turns. A turn takes about 2.2 s on one performance core of an i9-12900K, so a game about 7
minutes, and 48 games about 22 minutes on its 24 threads.
"""

import time

import numpy as np

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    SearchPlayer,
    Tactics,
    summarize,
    summary_text,
)
from gridzero.game import series


def main(starts: str = "24", *budgets: str) -> None:
    """Play search against greedy at each budget, and check that search wins at the largest.

    Parameters
    ----------
    starts : str, default "24"
        Pairs of games a budget, with the sides swapped.
    *budgets : str
        Simulations a turn, one series each: 100 if none.
    """
    simulations_tried = [int(b) for b in budgets] or [100]
    game = Tactics()
    verdict = "NO"
    for simulations in simulations_tried:
        player = SearchPlayer(simulations=simulations)
        start = time.perf_counter()
        records = series(
            game,
            [("search", player), ("greedy", GreedyPlayer())],
            int(starts),
            EVAL_SEED,
            workers=CORES,
        )
        elapsed = time.perf_counter() - start
        summary = summarize(game, records)
        title = f"Search ({simulations} simulations a turn) against greedy"
        print(f"\n### {title}\n\n{summary_text(game, summary)}")
        print(f"\n{len(records)} games in {elapsed / 60:.1f} minutes on {CORES} processes.")
        decided = summary.tally.wins["search"] + summary.tally.wins["greedy"]
        share = summary.tally.wins["search"] / max(decided, 1)
        spread = 2 * np.sqrt(0.25 / max(decided, 1))
        print(
            f"Search won {share:.0%} of {decided} decided games; a fair coin lands within "
            f"{spread:.0%} of 50% 95% of the time."
        )
        verdict = "yes" if decided and share > 0.5 + spread else "NO"
    print(f"\nCheck: search beats greedy at {simulations_tried[-1]} simulations: {verdict}.")
