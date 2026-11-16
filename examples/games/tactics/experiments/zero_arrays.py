"""Train AlphaZero on Tactics with self-play wholly on the GPU, from scratch or continuing a run.

The search in arrays plays 1,024 games at once on the rules as arrays (`TacticsArrays`), under
fog, each decision searched from its mover's belief with the rules that hide nothing; the network
trains on every decision with a choice, toward the search's improved policy and a value target
half the game's outcome, half the search's own value of the position
(`agents.zero_arrays_train`). On outcomes alone the value never learned to steer the search, and
the network stalled at greedy's level; with the blend it passed the MCTS teacher (9x9,
2026-11-16). Training only: the weights are scored elsewhere.

Each generation (ten trainings, about a minute on 9x9) saves the weights as
``{prefix}_gen_NNN.pt`` and ``{prefix}_last.pt``, AdamW's state as ``{prefix}_optimizer.pt``,
and its figures as a line of ``{prefix}_generations.jsonl``. A run continued from another's
prefix starts from its last weights, its optimizer, its generation count and the first seed it
left unplayed.

Run with ``uv run python run.py tactics.zero_arrays minutes [prefix] [board] [root_value]
[continues]``: by default ``zero_arrays_*`` in the working directory, the 15x15 board
(``small``, `small_rules`; ``nine`` for `nine_rules`), half the value target from the search,
and a new network. Under ``timeout`` as a hard stop beside the time box.
"""

import dataclasses
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # Torch shares the GPU

import torch  # after the setting above, which JAX reads when it starts

from examples.games.curves import new_log
from examples.games.tactics import nine_rules, small_rules
from examples.games.tactics.arrays import TacticsArrays
from gridzero.agents.net import PolicyNet
from gridzero.agents.zero_arrays_train import ArrayZero, train

SEED = 40_000_000  # the self-play boards, from SEED onward, and the network's first weights
CHANNELS, BLOCKS = 64, 6
BOARDS = {"small": small_rules, "nine": nine_rules}


def main(
    minutes: str,
    prefix: str = "zero_arrays",
    board: str = "small",
    root_value: str = "0.5",
    continues: str | None = None,
) -> None:
    """Train for at most ``minutes``, saving every generation.

    Parameters
    ----------
    minutes : str
        The time box: no generation starts that would end past it.
    prefix : str, default "zero_arrays"
        Where the weights and the figures go (see the module).
    board : str, default "small"
        The rules: ``small`` (`small_rules`, 15x15) or ``nine`` (`nine_rules`, 9x9).
    root_value : str, default "0.5"
        The share of the value target taken from the search's value (`ArrayZero.root_value`).
    continues : str, optional
        The prefix of a run to continue, on the same board.

    Raises
    ------
    SystemExit
        If no GPU is available, or the board is not one of `BOARDS`.
    FileExistsError
        If a run already used the prefix.
    """
    if not torch.cuda.is_available():
        raise SystemExit("this trains on the GPU, and no GPU is available")
    if board not in BOARDS:
        raise SystemExit(f"board must be one of {', '.join(BOARDS)}, got {board!r}")
    log = new_log(prefix)
    game = BOARDS[board]()
    rules, searched = TacticsArrays(game), TacticsArrays(game, hidden=False)
    torch.manual_seed(SEED)
    net = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS).cuda()
    settings = ArrayZero(root_value=float(root_value))
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    seed, done = SEED, 0
    if continues is not None:
        net.load_state_dict(torch.load(f"{continues}_last.pt", map_location="cuda"))
        optimizer.load_state_dict(torch.load(f"{continues}_optimizer.pt", map_location="cuda"))
        with open(f"{continues}_generations.jsonl") as f:
            last = json.loads(f.readlines()[-1])
        seed, done = last["next_seed"], last["generation"]
    print(
        f"AlphaZero in arrays on {game.shape} under fog for at most {minutes} minutes"
        f"{f', continuing {continues} after generation {done}' if continues else ''}: "
        f"{settings}.",
        flush=True,
    )
    start = time.perf_counter()
    generations = train(
        net,
        rules,
        lambda s: rules.to_arrays(game.start(s)),
        settings,
        searched=searched,
        generations=100_000,
        seed=seed,
        seconds=float(minutes) * 60,
        optimizer=optimizer,
    )
    for g in generations:
        number = done + g.generation
        weights = {k: v.cpu() for k, v in net.state_dict().items()}
        torch.save(weights, f"{prefix}_gen_{number:03d}.pt")
        torch.save(weights, f"{prefix}_last.pt")
        torch.save(optimizer.state_dict(), f"{prefix}_optimizer.pt")
        record = dataclasses.asdict(g) | {"generation": number}
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        won = g.wins[0] / g.games if g.games else float("nan")
        print(
            f"Generation {number} at {(time.perf_counter() - start) / 60:.1f} min: "
            f"{g.decisions / g.seconds:.0f} decisions a second, {g.games} games "
            f"({g.plies:.0f} plies, {g.draws} draws, the first player won {won:.2f}); "
            f"buffer {g.buffer} ({g.ready} ended); policy {g.policy_loss:.3f} (target "
            f"entropy {g.target_entropy:.3f}), value {g.value_loss:.3f}.",
            flush=True,
        )
