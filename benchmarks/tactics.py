"""Time the game of Tactics: making boards, masks of legal actions, steps, players and games.

Run from the repository root, pinned to one performance core for steady numbers:

    taskset -c 4 uv run python -m benchmarks.tactics

Two kinds of game on the default rules (25x25 boards since 2026-11-05, 24x24 from 2026-10-25,
32x32 before): random
players (who end a unit's turn one time in ten when they could act), and two greedy players.
Figures are means over every decision of a few games of each, in microseconds, after a warm-up
game that compiles the kernels.
"""

import time

import numpy as np

from benchmarks.bench import header
from examples.games.tactics import GreedyPlayer, RandomPlayer, Tactics, TacticsPosition
from gridzero.game import Player


def play(
    game: Tactics, player: Player[Tactics, TacticsPosition], seed: int
) -> dict[str, list[float]]:
    """Play one game with ``player`` on both sides, timing each call, in seconds."""
    times: dict[str, list[float]] = {"start": [], "legal": [], "player": [], "step": []}
    start = time.perf_counter()
    position = game.start(seed)
    times["start"].append(time.perf_counter() - start)
    while not position.over:
        rng = np.random.default_rng(position.ply)
        start = time.perf_counter()
        legal = game.legal(position)
        times["legal"].append(time.perf_counter() - start)
        start = time.perf_counter()
        action = player(game, game.player(position), position, legal, rng)
        times["player"].append(time.perf_counter() - start)
        start = time.perf_counter()
        position = game.step(position, action)
        times["step"].append(time.perf_counter() - start)
    return times


def main() -> None:
    header()
    game = Tactics()
    for name, player in (("random", RandomPlayer()), ("greedy", GreedyPlayer())):
        play(game, player, 0)  # warm-up: compiles the kernels
        games = [play(game, player, seed) for seed in range(1, 6)]
        print(f"\n**{name.capitalize()} players**\n")
        print("| Call | Mean (us) | Calls per game |")
        print("|---|---|---|")
        for kind in ("start", "legal", "player", "step"):
            values = [t for g in games for t in g[kind]]
            print(f"| `{kind}` | {1e6 * np.mean(values):.0f} | {len(values) / len(games):.0f} |")
        total = np.mean([sum(sum(v) for v in g.values()) for g in games])
        print(f"\nA game takes {total:.2f} s.")


if __name__ == "__main__":
    main()
