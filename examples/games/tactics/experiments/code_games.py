"""Play the search player against itself, to teach a code player: the games, kept to fit to.

On `small_rules`, one game per start, since a player against itself would play the same game twice
with the seats swapped. Saves ``{prefix}_lessons.npz`` as it goes (`lessons.teach`), so a run cut
short keeps what it finished, then prints how the games went.

Run with ``uv run python run.py tactics.code_games [games] [prefix] [work_plans] [replica]
[minutes]``: by default 600 games, prefix ``code``, 4 plans to work, replica 0 and 60 minutes.
Replicas play starts of their own. A game takes about 1 minute of one process at 50 simulations.
"""

import time
from pathlib import Path

import numpy as np

from examples.games import CORES
from examples.games.tactics import SearchPlayer, Verb, small_rules, teach

SEED = 6_000_000  # the lessons' starts, clear of EVAL_SEED
SEEDS_PER_REPLICA = 100_000  # keeps replicas' starts apart
SIMULATIONS = 50  # a turn: the search player at this budget beat greedy 34-14 here


def main(
    games: str = "600",
    prefix: str = "code",
    work_plans: str = "4",
    replica: str = "0",
    minutes: str = "60",
) -> None:
    """Play the games, save them and print how they went.

    Parameters
    ----------
    games : str, default "600"
        Games to play.
    prefix : str, default "code"
        Every file written starts with it.
    work_plans : str, default "4"
        The search player's plans to work; 0 leaves its engineer to the greedy player's work.
    replica : str, default "0"
        Which set of starts.
    minutes : str, default "60"
        No game starts after this long; those under way finish.

    Raises
    ------
    FileExistsError
        If a run already used the prefix.
    """
    path = Path(f"{prefix}_lessons.npz")
    if path.exists():
        raise FileExistsError(f"{prefix} has been used: give the new run a prefix of its own")
    game = small_rules()
    teacher = SearchPlayer(simulations=SIMULATIONS, work_plans=int(work_plans))
    first = SEED + int(replica) * SEEDS_PER_REPLICA
    seeds = range(first, first + int(games))
    start = time.perf_counter()
    lessons = teach(game, teacher, seeds, path, seconds=float(minutes) * 60, workers=CORES)
    elapsed = time.perf_counter() - start
    work = (lessons.verb == Verb.BUILD) | (lessons.verb == Verb.BREAK)
    print(f"{len(lessons)} of {games} games in {elapsed / 60:.1f} minutes on {CORES} processes.")
    print(
        f"{lessons.verb.size} decisions, {lessons.searched.size} searches, "
        f"team 0 won {np.mean(lessons.winner == 0):.0%}, "
        f"{np.mean(lessons.rounds):.1f} rounds a game, {work.sum() / len(lessons):.1f} tiles "
        f"worked on a game."
    )
    print(f"Wrote {path}.")
