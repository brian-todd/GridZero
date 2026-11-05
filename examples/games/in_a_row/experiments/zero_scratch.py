"""Train an AlphaZero agent on Connect Four's rules from scratch, in a time box.

A new, small network (32 channels, 4 blocks) trains by `agents.zero_train.train` on `InARow()`
(7 columns of 6, four in a row, gravity), with nothing learned before. Before training and after
each generation, the network alone (its best action, on the CPU) plays 2 * ``starts`` games
against the uniform player and as many against `ThreatPlayer`, the agent (its search with the
network, 96 simulations, no noise) plays as many against `ThreatPlayer`, and the script prints
its first move on the empty board and its value there for player 0: theory says the first
player wins, only by opening in the center column (3). No generation starts that would end
past the time box; run it under ``timeout`` as a hard stop beside it. Every series is on the
games players are scored on (`EVAL_SEED`).

96 simulations a decision end sequential halving evenly for any number of legal actions from 2
to 7, which mattered before the search chose among the actions a round cut short still
considered (fixed 2026-10-29): agents A and B were trained that way.

Run with ``uv run python run.py in_a_row.zero_scratch minutes [prefix] [starts] [seed]``:
by default the files in the working directory as ``zero_scratch_*``, 50 starts and seed 0 (a
replica: agents A and B were seeds 0 and 1). Weights go to ``{prefix}_gen_NNN.pt``, and each
generation's figures to ``{prefix}_generations.jsonl``.
"""

import copy
import dataclasses
import json
import time

import torch

from examples.games import CORES
from examples.games.curves import new_log
from examples.games.in_a_row import EVAL_SEED, InARow, Opening, ThreatPlayer, opening
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.agents.net import NetEvaluator, NetPlayer, PolicyNet
from gridzero.agents.zero_train import Zero, train
from gridzero.game import Score, ladder

SEED = 30_000_000  # the self-play games: replica r plays from SEED + 1_000_000 * r
CHANNELS, BLOCKS = 32, 4
SIMULATIONS = 96  # ends sequential halving evenly for 2 to 7 legal actions (see above)
RIVALS = {"uniform": UniformPlayer(), "threat": ThreatPlayer()}
SETTINGS = Zero(
    games=256,
    simulations=SIMULATIONS,
    considered=7,
    keep=1.0,
    buffer=100_000,
    steps=200,
    batch_size=256,
    lr=1e-3,
    value_weight=1.0,
)


def records(game: InARow, net: PolicyNet, starts: int) -> dict[str, dict[str, Score]]:
    """Play the network alone against `RIVALS`, and the agent (its search) against ThreatPlayer.

    Parameters
    ----------
    game : InARow
        The rules.
    net : PolicyNet
        The network, on any device: a copy plays on the CPU.
    starts : int
        Pairs of games against each rival, with the sides swapped.

    Returns
    -------
    dict of str to dict of str to Score
        The scores of ``"alone"`` and ``"agent"``, by rival.
    """
    cpu = copy.deepcopy(net).cpu()
    alone = ladder(game, NetPlayer(cpu), RIVALS, starts, EVAL_SEED, workers=CORES)
    agent = AlphaZeroPlayer(NetEvaluator(cpu), simulations=SIMULATIONS, considered=7)
    searched = ladder(game, agent, {"threat": ThreatPlayer()}, starts, EVAL_SEED, workers=CORES)
    return {"alone": alone, "agent": searched}


def show(label: str, result: dict[str, dict[str, Score]], first: Opening) -> None:
    """Print a generation's scores and its opening.

    Parameters
    ----------
    label : str
        The generation, as printed.
    result : dict of str to dict of str to Score
        What `records` returned.
    first : Opening
        The network's first move on the empty board.
    """
    text = "; ".join(
        f"{who} against " + ", ".join(f"{name} {s.won}-{s.lost}-{s.drawn}" for name, s in r.items())
        for who, r in result.items()
    )
    prior = [round(p, 2) for p in first.prior]
    print(
        f"{label}: {text} (wins-losses-draws); opens in column {first.column}, prior {prior}, "
        f"value for player 0 {first.value:.2f}.",
        flush=True,
    )


def main(minutes: str, prefix: str = "zero_scratch", starts: str = "50", seed: str = "0") -> None:
    """Train from scratch for at most ``minutes``, measuring each generation.

    Parameters
    ----------
    minutes : str
        The time box: no generation starts that would end past it.
    prefix : str, default "zero_scratch"
        Where the weights and the figures go: ``{prefix}_gen_NNN.pt`` and
        ``{prefix}_generations.jsonl``.
    starts : str, default "50"
        Pairs of games against each rival after each generation.
    seed : str, default "0"
        The replica: sets the network's first weights and the self-play games.

    Raises
    ------
    SystemExit
        If no GPU is available.
    FileExistsError
        If a run already used the prefix.
    """
    if not torch.cuda.is_available():
        raise SystemExit("this trains on the GPU, and no GPU is available")
    generations = new_log(prefix)
    game = InARow()
    torch.manual_seed(int(seed))
    net = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS).cuda()
    print(
        f"AlphaZero from scratch on {game} for at most {minutes} minutes, seed {seed}: {SETTINGS}."
    )
    show(
        "Generation 0",
        records(game, net, int(starts)),
        opening(game, NetEvaluator(copy.deepcopy(net).cpu())),
    )
    start = time.perf_counter()
    first_game = SEED + 1_000_000 * int(seed)
    box = float(minutes) * 60
    for g in train(net, game, SETTINGS, generations=1000, seed=first_game, seconds=box):
        torch.save(
            {k: v.cpu() for k, v in net.state_dict().items()}, f"{prefix}_gen_{g.generation:03d}.pt"
        )
        result, first = (
            records(game, net, int(starts)),
            opening(game, NetEvaluator(copy.deepcopy(net).cpu())),
        )
        record = dataclasses.asdict(g) | {"records": result, "opening": first._asdict()}
        record.pop("played")  # positions, not JSON
        record["costs"] = {
            "seconds": g.costs.seconds,
            "batches": g.costs.batches,
            "positions": g.costs.positions,
            "peak_mib": g.costs.peak_mib,
        }
        with generations.open("a") as f:
            f.write(json.dumps(record) + "\n")
        wins = sum(p.winner == 0 for p in g.played)
        print(
            f"Generation {g.generation} at {(time.perf_counter() - start) / 60:.1f} min: "
            f"{g.games} games (player 0 won {wins}, {g.draws} draws, {g.plies:.1f} plies), "
            f"buffer {g.buffer}; before training policy {g.fresh_policy_loss:.3f} (target "
            f"entropy {g.target_entropy:.3f}) value {g.fresh_value_loss:.3f}; play "
            f"{g.play_seconds:.0f} s ({g.costs.positions / g.play_seconds:.0f} leaves/s), "
            f"train {g.train_seconds:.0f} s, GPU peak {g.costs.peak_mib:.0f} MiB.",
            flush=True,
        )
        show(f"Generation {g.generation}", result, first)
