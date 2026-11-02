"""Train a MuZero player on Colonies from scratch, in a time box.

A new network (`WorldNet`, 64 channels, 4 blocks) learns by `agents.muzero_train.train`: games
of a MuZero player against itself under fog, each player planning on the network's latent states
from its own view, then learning from every decision: the search's improved policy, the outcome,
and the whole position it couldn't see (the belief), unrolled 3 rounds through the dynamics.
Before training and after each generation, the network alone (its policy's best order) and the
MuZero player (its search, 32 simulations, no noise) each play the uniform player, the settler
and the raider in every other seat, on the games players are scored on (`EVAL_SEED`), on the CPU
of every core. Measuring counts against the time box: no generation starts that the last one's
duration says would end past it. Run it under ``timeout`` too.

Run with ``uv run python run.py colonies.muzero_scratch minutes [prefix] [starts] [seed]
[belief] [consistency] [reanalyze] [codes]``: by default the files in the working directory as
``muzero_scratch_*``, 10 starts against each rival, seed 0 (a replica), the belief's loss
weighing 1 against the policy's (0 trains no belief: does learning what is hidden help play?),
no consistency loss (EfficientZero's weighs 2: do unrolled states held to represented ones plan
better?), no reanalysis (the stalest policy targets searched again each generation: do
fresher targets learn faster?), and no chance codes (Stochastic MuZero's afterstates and codes:
do dynamics that can tell outcomes apart, rather than average them, plan better?). Weights go to
``{prefix}_gen_NNN.pt``, and each generation's figures to ``{prefix}_generations.jsonl``.
"""

import copy
import dataclasses
import json
import time

import torch

from examples.games import CORES
from examples.games.colonies import EVAL_SEED, Colonies, Raider, Settler
from examples.games.colonies.learned import network
from examples.games.curves import new_log
from gridzero.agents import UniformPlayer
from gridzero.agents.muzero import MuZeroPlayer
from gridzero.agents.muzero_net import WorldNet, world_model
from gridzero.agents.muzero_train import Fit, MuZero, train
from gridzero.game import Score, ladder

SEED = 7_000_000  # the self-play games: replica r plays from SEED + 1_000_000 * r
SIMULATIONS = 32
RIVALS = {"uniform": UniformPlayer(), "settler": Settler(), "raider": Raider()}
SETTINGS = MuZero(
    games=128,
    simulations=SIMULATIONS,
    considered=16,
    buffer=100_000,
    steps=300,
    fit=Fit(unroll=3, batch_size=256, lr=1e-3),
)


def records(game: Colonies, net: WorldNet, starts: int) -> dict[str, dict[str, Score]]:
    """Play the network alone and the MuZero player against `RIVALS`.

    Parameters
    ----------
    game : Colonies
        The rules.
    net : WorldNet
        The network, on any device: a copy plays on the CPU.
    starts : int
        Starts against each rival, each once per rotation of the seats.

    Returns
    -------
    dict of str to dict of str to Score
        The scores of ``"alone"`` and ``"muzero"``, by rival.
    """
    cpu = copy.deepcopy(net).cpu().eval()
    players = {
        "alone": MuZeroPlayer(world_model(cpu), simulations=0),
        "muzero": MuZeroPlayer(world_model(cpu), simulations=SIMULATIONS),
    }
    return {
        who: ladder(game, player, RIVALS, starts, EVAL_SEED, workers=CORES)
        for who, player in players.items()
    }


def show(label: str, result: dict[str, dict[str, Score]]) -> str:
    """Print a generation's scores, and return them as printed.

    Parameters
    ----------
    label : str
        The generation, as printed.
    result : dict of str to dict of str to Score
        What `records` returned.

    Returns
    -------
    str
        The line printed.
    """
    text = "; ".join(
        f"{who} against " + ", ".join(f"{name} {s.won}-{s.lost}-{s.drawn}" for name, s in r.items())
        for who, r in result.items()
    )
    line = f"{label}: {text} (wins-losses-draws)."
    print(line, flush=True)
    return line


def main(
    minutes: str,
    prefix: str = "muzero_scratch",
    starts: str = "10",
    seed: str = "0",
    belief: str = "1",
    consistency: str = "0",
    reanalyze: str = "0",
    codes: str = "0",
) -> None:
    """Train from scratch for at most ``minutes``, measuring each generation.

    Parameters
    ----------
    minutes : str
        The time box: no generation starts that would end past it.
    prefix : str, default "muzero_scratch"
        Where the weights and the figures go: ``{prefix}_gen_NNN.pt`` and
        ``{prefix}_generations.jsonl``.
    starts : str, default "10"
        Starts against each rival after each generation.
    seed : str, default "0"
        The replica: sets the network's first weights and the self-play games.
    belief : str, default "1"
        The weight of the belief's loss against the policy's, at least 0.
    consistency : str, default "0"
        The weight of the consistency loss against the policy's, at least 0.
    reanalyze : str, default "0"
        Decisions whose policy targets are searched again each generation, at least 0.
    codes : str, default "0"
        The network's chance codes, at least 0: 0 for MuZero's deterministic dynamics.

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
    game = Colonies()
    first_game = SEED + 1_000_000 * int(seed)
    torch.manual_seed(first_game)
    net = network(codes=int(codes)).cuda()
    weights = {"belief_weight": float(belief), "consistency_weight": float(consistency)}
    settings = dataclasses.replace(
        SETTINGS, fit=dataclasses.replace(SETTINGS.fit, **weights), reanalyze=int(reanalyze)
    )
    print(
        f"MuZero from scratch on {game} for at most {minutes} minutes, seed {seed}, "
        f"{net.codes} codes: {settings}."
    )
    show("Generation 0", records(game, net, int(starts)))
    start = time.perf_counter()
    box = float(minutes) * 60
    for g in train(net, game, settings, seconds=box, seed=first_game):
        torch.save(net.state_dict(), f"{prefix}_gen_{g.generation:03d}.pt")
        wins = [sum(p.winner == s for p in g.played) for s in range(game.players)]
        draws = sum(p.winner is None for p in g.played)
        print(
            f"Generation {g.generation}: {len(g.played)} games "
            f"(wins by seat {wins}, {draws} draws), {g.decisions} decisions, "
            f"buffer {g.buffer}; losses policy "
            f"{g.losses.policy.round(3).tolist()}, value {g.losses.value.round(3).tolist()}, "
            f"belief {g.losses.belief.round(3).tolist()}, consistency "
            f"{g.losses.consistency.round(3).tolist()}, chance "
            f"{g.losses.chance.round(3).tolist()}, afterstate "
            f"{g.losses.afterstate.round(3).tolist()}, commitment "
            f"{g.losses.commitment.round(3).tolist()}, diversity "
            f"{g.losses.diversity.round(3).tolist()}; play {g.play_seconds:.0f} s, "
            f"reanalyzed {g.reanalyzed} in {g.reanalyze_seconds:.0f} s, "
            f"train {g.train_seconds:.0f} s, {time.perf_counter() - start:.0f} s in all.",
            flush=True,
        )
        result = records(game, net, int(starts))
        show(f"Generation {g.generation}", result)
        with generations.open("a") as out:
            entry = {
                "generation": g.generation,
                "seat_wins": wins,
                "draws": draws,
                "policy_loss": g.losses.policy.tolist(),
                "value_loss": g.losses.value.tolist(),
                "belief_loss": g.losses.belief.tolist(),
                "consistency_loss": g.losses.consistency.tolist(),
                "chance_loss": g.losses.chance.tolist(),
                "afterstate_loss": g.losses.afterstate.tolist(),
                "commitment_loss": g.losses.commitment.tolist(),
                "diversity_loss": g.losses.diversity.tolist(),
                "play_seconds": g.play_seconds,
                "train_seconds": g.train_seconds,
                "reanalyzed": g.reanalyzed,
                "reanalyze_seconds": g.reanalyze_seconds,
                "scores": {
                    who: {name: list(s) for name, s in r.items()} for who, r in result.items()
                },
            }
            out.write(json.dumps(entry) + "\n")
