"""Check that AlphaZero's loop learns: from scratch on a small board, in a time box (T8, step 4).

A new, small network (32 channels, 4 blocks) trains by `zero_train.train` on a 9x9 board with
five units a side and a 12-round cap, with nothing learned before. Before training and after
each generation, the network alone (its best action, on the CPU) plays 2 * ``starts`` games
against the random player and as many against greedy, on the boards players are scored on
(`EVAL_SEED`). If the loop learns, its records climb: past random first, then toward greedy. No
generation starts that would end past the time box; run it under ``timeout`` as a hard stop
beside it. Each generation's weights go to ``{prefix}_gen_NNN.pt``, and its figures to
``{prefix}_generations.jsonl``.

Run with ``uv run python run.py tactics.zero_scratch minutes [prefix] [starts] [games]
[actors] [steps] [value_weight]``: by default the files in the working directory as
``zero_scratch_*``, 24 starts, 128 games a generation, every core but two playing them, 150
steps and a value weight of 0.25.
"""

import dataclasses
import json
import time

import torch

from examples.games import CORES
from examples.games.curves import new_log
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    RandomPlayer,
    Tactics,
    TacticsPosition,
)
from gridzero.agents.net import NetPlayer, PolicyNet
from gridzero.agents.zero_train import ACTORS, Generation, Zero, train
from gridzero.game import Score, ladder

SEED = 20_000_000  # the self-play boards, from SEED onward, and the network's first weights
CHANNELS, BLOCKS = 32, 4


def rounds(generation: Generation[TacticsPosition]) -> float:
    """Return the mean rounds of a generation's self-play games, from their ends.

    Parameters
    ----------
    generation : Generation of TacticsPosition
        A generation of training.

    Returns
    -------
    float
        The mean rounds: Tactics' own count.
    """
    return sum(p.final.round for p in generation.played) / len(generation.played)


def records(game: Tactics, net: PolicyNet, starts: int) -> dict[str, Score]:
    """Play the network alone against random and greedy.

    Parameters
    ----------
    game : Tactics
        The rules.
    net : PolicyNet
        The network, on any device: a copy plays on the CPU.
    starts : int
        Pairs of games against each rival, with the sides swapped.

    Returns
    -------
    dict of str to Score
        Its wins, losses and draws, by rival.
    """
    cpu = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS)
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    rivals = {"random": RandomPlayer(), "greedy": GreedyPlayer()}
    return ladder(game, NetPlayer(cpu), rivals, starts, EVAL_SEED, same_dice=False, workers=CORES)


def show(label: str, result: dict[str, Score]) -> None:
    """Print a generation's scores.

    Parameters
    ----------
    label : str
        The generation, as printed.
    result : dict of str to Score
        What `records` returned.
    """
    text = ", ".join(f"{name} {won}-{lost}-{drawn}" for name, (won, lost, drawn) in result.items())
    print(f"{label}: network alone against {text} (wins-losses-draws).", flush=True)


def main(
    minutes: str,
    prefix: str = "zero_scratch",
    starts: str = "24",
    games: str = "128",
    actors: str = str(ACTORS),
    steps: str = "150",
    value_weight: str = "0.25",
) -> None:
    """Train from scratch for at most ``minutes``, measuring each generation.

    Five units a side on a 9x9 board, with a 12-round cap: short games, so generations are
    quick.

    Parameters
    ----------
    minutes : str
        The time box: no generation starts that would end past it.
    prefix : str, default "zero_scratch"
        Where the weights and the figures go: ``{prefix}_gen_NNN.pt`` and
        ``{prefix}_generations.jsonl``.
    starts : str, default "24"
        Pairs of games against each rival after each generation.
    games : str, default "128"
        Self-play games a generation.
    actors : str, default every core but two
        Processes playing them (`zero_train.ACTORS`).
    steps : str, default "150"
        Training steps a generation.
    value_weight : str, default "0.25"
        The value loss's weight.

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
    game = Tactics(shape=(9, 9), counts=(2, 1, 1, 1), home_depth=3, max_rounds=12)
    torch.manual_seed(SEED)
    net = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS).cuda()
    settings = Zero(
        games=int(games), actors=int(actors), steps=int(steps), value_weight=float(value_weight)
    )
    print(f"AlphaZero from scratch on {game.shape} for at most {minutes} minutes: {settings}.")
    show("Generation 0", records(game, net, int(starts)))
    start = time.perf_counter()
    box = float(minutes) * 60
    for g in train(net, game, settings, generations=1000, seed=SEED, seconds=box):
        torch.save(
            {k: v.cpu() for k, v in net.state_dict().items()}, f"{prefix}_gen_{g.generation:03d}.pt"
        )
        result = records(game, net, int(starts))
        record = dataclasses.asdict(g) | {"records": result}
        record.pop("played")  # positions, not JSON
        record["costs"] = {"seconds": g.costs.seconds, "batches": g.costs.batches}
        with generations.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"Generation {g.generation} at {(time.perf_counter() - start) / 60:.1f} min: "
            f"{g.games} games ({g.draws} draws, {rounds(g):.1f} rounds), buffer {g.buffer}; "
            f"before training policy {g.fresh_policy_loss:.3f} (target entropy "
            f"{g.target_entropy:.3f}) value {g.fresh_value_loss:.3f}; play "
            f"{g.play_seconds / 60:.1f} min, train {g.train_seconds:.0f} s.",
            flush=True,
        )
        show(f"Generation {g.generation}", result)
