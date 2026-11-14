"""Measure a saved Connect Four network: alone, and as an AlphaZero agent with its search.

Loads weights saved by ``zero_scratch`` (32 channels, 4 blocks), then plays 2 * ``starts``
games of the network alone against the uniform player and `ThreatPlayer`, and as many of the
agent (`AlphaZeroPlayer` on the network, without noise) against `ThreatPlayer`, on the games
players are scored on (`EVAL_SEED`), on every core, and prints its opening. The network runs on
the CPU, in each worker.

Run with ``uv run python run.py in_a_row.zero_check weights.pt [starts] [simulations]``:
by default 50 starts and 96 simulations, as in training.
"""

import time

import torch

from examples.games import CORES
from examples.games.in_a_row import (
    EVAL_SEED,
    InARow,
    ThreatPlayer,
    opening,
)
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.agents.net import NetEvaluator, NetPlayer, PolicyNet
from gridzero.game import ladder

CHANNELS, BLOCKS = 32, 4  # as zero_scratch saved them: the weights don't record it


def saved_net(game: InARow, weights: str) -> PolicyNet:
    """Return a network saved by ``zero_scratch``, on the CPU, ready to play.

    Parameters
    ----------
    game : InARow
        The rules it plays.
    weights : str
        The saved weights.

    Returns
    -------
    PolicyNet
        The network, in evaluation mode.
    """
    net = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    return net


def main(weights: str, starts: str = "50", simulations: str = "96") -> None:
    """Print the network's opening, and play it alone and as the agent against its rivals.

    Parameters
    ----------
    weights : str
        The saved weights.
    starts : str, default "50"
        Pairs of games against each rival, with the sides swapped.
    simulations : str, default "96"
        The agent's simulations a decision.
    """
    game = InARow()
    net = saved_net(game, weights)
    first = opening(game, NetEvaluator(net))
    print(
        f"{weights}: opens in column {first.column}, prior "
        f"{[round(p, 2) for p in first.prior]}, value for player 0 {first.value:.2f}.",
        flush=True,
    )
    agent = AlphaZeroPlayer(NetEvaluator(net), simulations=int(simulations), considered=7)
    for label, player, rivals in (
        ("network alone", NetPlayer(net), {"uniform": UniformPlayer(), "threat": ThreatPlayer()}),
        (f"agent, {simulations} simulations", agent, {"threat": ThreatPlayer()}),
    ):
        start = time.perf_counter()
        scores = ladder(game, player, rivals, int(starts), EVAL_SEED, workers=CORES)
        text = ", ".join(f"{name} {s.won}-{s.lost}-{s.drawn}" for name, s in scores.items())
        print(
            f"{label} against {text} (wins-losses-draws), {time.perf_counter() - start:.0f} s.",
            flush=True,
        )
