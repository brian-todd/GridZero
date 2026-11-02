"""A tournament of Tactics players: greedy against random, and greedy against itself.

Each pairing plays a series on the boards players are scored on (`EVAL_SEED`): every board
twice, the sides swapped, on every core. Greedy against random rolls the same dice in both
games of a board; greedy against itself rolls dice of their own, since with the same dice its
two games would be one game played twice. It prints who won, from which side, how long games
ran, and what each kind of unit did, and checks two things a fair game and a sensible player
should show:

- **greedy beats random** in nearly every game,
- **neither side has an edge:** between two greedy players, team 0's edge over half the
  decided games stays within a fair game's noise (`gridzero.game.fairness`).

It saves a picture of the end of the first greedy game as ``{prefix}_end.png``, and prints
its last events.

Run with ``uv run python run.py tactics.tournament [starts] [prefix]``: by default 10 starts
a pairing, and the picture in ``tactics_end.png`` in the working directory.
"""

import time

from matplotlib.figure import Figure

from examples.games import CORES
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    RandomPlayer,
    Tactics,
    event_text,
    summarize,
    summary_text,
)
from examples.games.tactics.pictures import draw
from gridzero.game import fairness, replay, series


def main(starts: str = "10", prefix: str = "tactics") -> None:
    """Play both pairings, check them, and save the end of the first greedy game.

    Parameters
    ----------
    starts : str, default "10"
        Starts a pairing, each played twice with the sides swapped.
    prefix : str, default "tactics"
        Where the picture goes: ``{prefix}_end.png``.
    """
    game = Tactics()
    greedy, rand = ("greedy", GreedyPlayer()), ("random", RandomPlayer())

    start = time.perf_counter()
    against_random = series(game, [greedy, rand], int(starts), EVAL_SEED, workers=CORES)
    summary = summarize(game, against_random)
    print(f"\n### Greedy against random\n\n{summary_text(game, summary)}")
    share = summary.tally.wins["greedy"] / summary.tally.games
    print(f"\nCheck: greedy won {share:.0%} of its games against random (expected: nearly all).")

    mirror = series(
        game,
        [greedy, ("greedy too", GreedyPlayer())],
        int(starts),
        EVAL_SEED,
        same_dice=False,
        workers=CORES,
    )
    summary = summarize(game, mirror)
    print(f"\n### Greedy against greedy\n\n{summary_text(game, summary)}")
    fair = fairness(summary.tally)
    verdict = "even" if fair.excess_edge() == 0 else "UNEVEN"
    print(
        f"\nCheck: team 0's edge over half of {fair.decided} decided games is "
        f"{fair.seat_edges[0]:+.2f}, {fair.excess_edge():.2f} beyond a fair game's noise: "
        f"{verdict}."
    )
    games = len(against_random) + len(mirror)
    print(f"\n{games} games in {time.perf_counter() - start:.0f} s on {CORES} processes.")

    record = mirror[0]
    final = record.final
    figure = Figure(figsize=(12, 7))
    ax = draw(game, final, figure.add_subplot())
    ax.set_title(f"{record.names[0]} (red) against {record.names[1]} (blue): round {final.round}")
    picture = f"{prefix}_end.png"
    figure.savefig(picture, dpi=120, bbox_inches="tight")
    print(f"\nThe end of the first greedy game ({picture}):\n")
    events = [e for position in replay(game, record) for e in position.events]  # typed, unlike
    for event in events[-6:]:  # record.events
        print(" ", event_text(game, final.state, event))
