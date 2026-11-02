"""Rate saved MuZero players of Colonies against each other and the scripted players.

Every pair among the uniform player, the settler, the raider and a MuZero player per weights
spec (`examples.games.colonies.learned`: ``path@root`` for a network trained before
2026-10-30, ``path@64`` for 64 simulations; its search, no noise)
plays a series on the games players are scored on (`EVAL_SEED`), on every core. It prints the
table of wins, and a rating per player on the Elo scale, the uniform player at 0, fitted to every
pair's results (Bradley-Terry, with one drawn game added to each pair so that a perfect record
has a finite rating). Unlike a score against the scripted players alone, the ratings keep
telling generations apart once all of them beat the scripted players.

Run with ``uv run python run.py colonies.muzero_ladder [starts] [simulations] weights...``: by
default 10 starts a pair and 32 simulations (unless a spec sets them), then any number of
weights specs, named by their file names and tags.
"""

import itertools
import time

import numpy as np
from numpy.typing import NDArray

from examples.games import CORES
from examples.games.colonies import EVAL_SEED, Colonies, ColoniesView, Raider, Settler
from examples.games.colonies.learned import player
from gridzero.agents import UniformPlayer
from gridzero.game import Player, series


def main(starts: str = "10", simulations: str = "32", *weights: str) -> None:
    """Play every pair of players, and print their wins and ratings.

    Parameters
    ----------
    starts : str, default "10"
        Starts each pair plays, each once per rotation of the seats.
    simulations : str, default "32"
        The MuZero players' simulations a decision.
    *weights : str
        Specs of saved weights, one MuZero player each.

    Raises
    ------
    ValueError
        If two players have one name: two specs of one file and tags, or of files named alike.
    """
    game = Colonies()
    players: dict[str, Player[Colonies, ColoniesView]] = {
        "uniform": UniformPlayer(),
        "settler": Settler(),
        "raider": Raider(),
    }
    for spec in weights:
        name, searcher = player(spec, int(simulations))
        if name in players:
            raise ValueError(f"two players named {name!r}: give each spec a name of its own")
        players[name] = searcher
    names = list(players)
    wins = np.zeros((len(names), len(names)))
    begin = time.perf_counter()
    for i, j in itertools.combinations(range(len(names)), 2):
        pair = [(names[i], players[names[i]]), (names[j], players[names[j]])]
        for record in series(game, pair, int(starts), EVAL_SEED, workers=CORES):
            if record.winner is None:
                wins[i, j] += 0.5
                wins[j, i] += 0.5
            else:
                won = names.index(record.winner)
                wins[won, j if won == i else i] += 1
    width = max(len(n) for n in names)
    print(f"Wins of each row against each column ({time.perf_counter() - begin:.0f} s):")
    print(" " * width + "".join(f"{k:>6}" for k in range(len(names))))
    for k, name in enumerate(names):
        cells = "".join("     -" if k == m else f"{wins[k, m]:6.1f}" for m in range(len(names)))
        print(f"{name:>{width}}{cells}  ({k})")
    ratings = elo(wins)
    print("Ratings (Elo scale, uniform 0):")
    for k in np.argsort(-ratings):
        print(f"  {names[k]:>{width}} {ratings[k]:7.0f}")


def elo(wins: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Bradley-Terry ratings on the Elo scale, the first player at 0.

    Parameters
    ----------
    wins : ndarray of float64, shape (n, n)
        ``wins[i, j]`` is how many games ``i`` won against ``j``, draws counted half to each.

    Returns
    -------
    ndarray of float64, shape (n,)
        Ratings such that ``i`` beats ``j`` with probability ``1 / (1 + 10**((r_j - r_i) /
        400))``, fitted with one drawn game added to every pair.
    """
    n = wins.shape[0]
    w = wins + 0.5 * (1 - np.eye(n))
    games = w + w.T
    strength = np.ones(n)
    for _ in range(10_000):  # the minorize-maximize iteration of Hunter (2004)
        denominator = (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = w.sum(axis=1) / denominator
        updated /= updated[0]
        if np.allclose(updated, strength, rtol=1e-10, atol=0):
            break
        strength = updated
    return 400 * np.log10(strength)
