"""Time self-play of the AlphaZero player on Tactics, on the GPU (T8, step 3).

Plays ``actors * games`` self-play games of the AlphaZero player on a trained network
(`zero_train.generate`: actor processes playing their games at once, judged in batches by an
inference server on the GPU), with Gumbel noise, and prints what they cost: games an hour,
decisions and leaves a second, the actors' time in the evaluator, the server's batches, busy
time and peak memory, and the whole GPU's memory and utilization, polled from ``nvidia-smi``
every second.

Run from the repository root, unpinned (the actors use every core):

    uv run python -m benchmarks.zero_selfplay weights.pt [games] [actors] [simulations]

by default 8 games for each of 12 actors, 64 simulations a decision.
"""

import subprocess
import sys
import threading
import time
from types import TracebackType
from typing import Self

import numpy as np
import torch

from examples.games.tactics import Tactics
from gridzero.agents.net import NetEvaluator, PolicyNet
from gridzero.agents.zero import AlphaZeroPlayer
from gridzero.agents.zero_train import generate

SEED = 7_000_000  # the self-play boards, from SEED onward


class GpuPoll:
    """Polls the GPU's memory in use and utilization once a second, in a thread, while open."""

    def __init__(self) -> None:
        self.memory: list[float] = []
        """MiB in use, a poll each."""
        self.busy: list[float] = []
        """Percent utilization, a poll each."""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        """Poll ``nvidia-smi`` each second until stopped."""
        while not self._stop.wait(1.0):
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            used, busy = (float(x) for x in out.strip().split(","))
            self.memory.append(used)
            self.busy.append(busy)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()


def main(
    weights: str,
    games: str = "8",
    actors: str = "12",
    simulations: str = "64",
) -> None:
    """Play the self-play games on the GPU and print what they cost.

    Parameters
    ----------
    weights : str
        The saved weights of the network.
    games : str, default "8"
        Games an actor.
    actors : str, default "12"
        Actor processes.
    simulations : str, default "64"
        Simulations a decision.

    Raises
    ------
    SystemExit
        If no GPU is available.
    """
    game = Tactics()
    net = PolicyNet(game)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    if not torch.cuda.is_available():
        raise SystemExit("this measures self-play on the GPU, and no GPU is available")
    net.cuda()  # the server runs the network where its weights are
    player = AlphaZeroPlayer(NetEvaluator(net), simulations=int(simulations), noise=True)
    n_actors, n_games = int(actors), int(actors) * int(games)
    seeds = range(SEED, SEED + n_games)
    print(
        f"{n_games} self-play games on {weights}: {n_actors} actors of {games} games, "
        f"{simulations} simulations a decision.",
        flush=True,
    )
    idle = GpuPoll()
    with idle:
        time.sleep(2)
    base = float(np.median(idle.memory)) if idle.memory else 0.0
    start = time.perf_counter()
    with GpuPoll() as gpu:
        experience, played, costs = generate(game, player, seeds, actors=n_actors)
    wall = time.perf_counter() - start
    searched = sum(p.searched for p in played)
    stats = costs.actors
    print(f"Wall time {wall / 60:.1f} minutes: {n_games / wall * 3600:.0f} games an hour.")
    print(
        f"{searched / wall:.0f} decisions searched a second ({searched / n_games:.0f} a game), "
        f"{costs.positions / wall:.0f} leaves judged a second."
    )
    share = [s.evaluator_seconds / s.seconds for s in stats]
    print(
        f"Actors: {np.mean([s.seconds for s in stats]) / 60:.1f} minutes each (longest "
        f"{max(s.seconds for s in stats) / 60:.1f}); evaluator {np.mean(share):.0%} of their "
        f"time, {np.mean([s.evaluator_seconds / s.calls for s in stats]) * 1e3:.1f} ms a call."
    )
    print(
        f"Server: {costs.batches} batches of {costs.positions / costs.batches:.1f} positions, "
        f"busy {costs.server_seconds / wall:.0%} of the time, "
        f"{costs.server_seconds / costs.positions * 1e6:.0f} us a position; "
        f"peak allocated {costs.peak_mib:.0f} MiB."
    )
    if gpu.memory:
        print(
            f"GPU: {max(gpu.memory):.0f} MiB at most in use ({base:.0f} before); utilization "
            f"mean {np.mean(gpu.busy):.0f}%, median {np.median(gpu.busy):.0f}%."
        )
    winners = [p.winner for p in played]
    print(
        f"Games: {winners.count(0)}-{winners.count(1)} by team, {winners.count(None)} draws, "
        f"{np.mean([p.final.round for p in played]):.1f} rounds; {len(experience)} decisions kept, "
        f"{experience.obs.nbytes / 2**20:.0f} MiB of observations."
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
