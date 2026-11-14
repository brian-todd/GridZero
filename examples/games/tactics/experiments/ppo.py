"""Warm a network up by PPO self-play on Tactics, in a time box.

A new network (`PolicyNet`'s defaults) first imitates ``taught`` games of greedy against itself
(`agents.imitate`), then trains by `ppo.self_play` on the rules of `small_rules`: each iteration
plays ``games`` games with actions drawn from its policy, a quarter of them against greedy and
the rest against itself, and learns from their results. PPO plays many more games an hour than
AlphaZero's self-play, since no decision is searched, and a network's value needs thousands of
games before it stops memorizing them; what it gives is a network for AlphaZero to start from.

Before training and every `EVERY` iterations, the network alone (its best action, on the CPU,
deploying as greedy does: `Fielded`) plays 2 * ``starts`` games against greedy on the boards
players are scored on (`EVAL_SEED`), and its weights go to ``{prefix}_iter_NNN.pt``; each
iteration's figures go to ``{prefix}_generations.jsonl``, which ``games.curves`` shows. No
iteration starts that would end past the time box (the first always runs); run it under
``timeout`` as a hard stop beside it.

Run with ``uv run python run.py tactics.ppo minutes [prefix] [starts] [games] [taught]
[entropy]``: by default the files in the working directory as ``ppo_*``, 24 starts, 96 games an
iteration, 200 games of greedy to imitate and an entropy bonus of 0.001.
"""

import dataclasses
import json
import time

import torch

from examples.games import CORES
from examples.games.curves import new_log
from examples.games.tactics import EVAL_SEED, Fielded, GreedyPlayer, Tactics, small_rules
from gridzero.agents.imitate import collect, imitate
from gridzero.agents.net import NetPlayer, PolicyNet
from gridzero.agents.ppo import PPO, Iteration, self_play
from gridzero.game import Score, ladder

SEED = 40_000_000  # the self-play boards, from SEED onward, and the network's first weights
TAUGHT_SEED = 30_000_000  # the boards of the games it imitates, if any
EVERY = 10
"""Iterations between two scorings of the network against greedy."""


def record(game: Tactics, net: PolicyNet, starts: int) -> Score:
    """Play the network alone against greedy, greedy deploying for both.

    Parameters
    ----------
    game : Tactics
        The rules.
    net : PolicyNet
        The network, on any device: a copy plays on the CPU.
    starts : int
        Pairs of games, with the sides swapped.

    Returns
    -------
    Score
        Its wins, losses and draws.
    """
    cpu = PolicyNet(game, net.channels, net.blocks)
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    player = Fielded(NetPlayer(cpu))
    rivals = {"greedy": GreedyPlayer()}
    return ladder(game, player, rivals, starts, EVAL_SEED, same_dice=False, workers=CORES)["greedy"]


def rounds(iteration: Iteration) -> float:
    """Return the mean rounds of an iteration's games, from their ends.

    Parameters
    ----------
    iteration : Iteration
        An iteration of training.

    Returns
    -------
    float
        The mean rounds: Tactics' own count.
    """
    return sum(p.final.round for p in iteration.played) / len(iteration.played)


def main(
    minutes: str,
    prefix: str = "ppo",
    starts: str = "24",
    games: str = "96",
    taught: str = "200",
    entropy: str = "0.001",
) -> None:
    """Train by PPO self-play for at most ``minutes``, scoring the network as it goes.

    Parameters
    ----------
    minutes : str
        The time box: no iteration starts that would end past it.
    prefix : str, default "ppo"
        Where the weights and the figures go: ``{prefix}_iter_NNN.pt`` and
        ``{prefix}_generations.jsonl``.
    starts : str, default "24"
        Pairs of games against greedy at each scoring.
    games : str, default "96"
        Games an iteration.
    taught : str, default "200"
        Games of greedy to imitate before training, if any.
    entropy : str, default "0.001"
        The weight of the policy's entropy, a bonus.

    Raises
    ------
    SystemExit
        If no GPU is available.
    FileExistsError
        If a run already used the prefix.
    """
    if not torch.cuda.is_available():
        raise SystemExit("this trains on the GPU, and no GPU is available")
    figures = new_log(prefix)
    game = small_rules()
    torch.manual_seed(SEED)
    net = PolicyNet(game).cuda()
    settings = PPO(games=int(games), opponent=GreedyPlayer(), entropy_weight=float(entropy))
    print(f"PPO on {game.shape} for at most {minutes} minutes, {CORES} processes: {settings}.")
    start = time.perf_counter()
    box = float(minutes) * 60
    if int(taught) > 0:
        lesson = collect(game, GreedyPlayer(), int(taught), TAUGHT_SEED, workers=CORES)
        print(f"Imitating {len(lesson)} decisions of {taught} games of greedy.", flush=True)
        for epoch in imitate(net, lesson, seed=SEED):
            print(f"  {epoch}", flush=True)
        del lesson

    def score(done: int) -> Score:
        """Score the network after ``done`` iterations, and save its weights."""
        weights = {k: v.cpu() for k, v in net.state_dict().items()}
        torch.save(weights, f"{prefix}_iter_{done:03d}.pt")
        won, lost, drawn = result = record(game, net, int(starts))
        print(
            f"Iteration {done}: network alone against greedy {won}-{lost}-{drawn} "
            f"(wins-losses-draws).",
            flush=True,
        )
        return result

    score(0)
    scored, done = 0, 0
    left = max(box - (time.perf_counter() - start), 0.0)  # at 0, it trains one iteration
    for it in self_play(
        net, game, settings, iterations=1_000_000, seed=SEED, workers=CORES, seconds=left
    ):
        done = it.iteration
        figure = dataclasses.asdict(it) | {"rounds": rounds(it)}
        figure.pop("played")  # positions, not JSON
        print(
            f"Iteration {done} at {(time.perf_counter() - start) / 60:.1f} min: "
            f"{it.decisions} decisions; against greedy {it.opponent_wins}-{it.opponent_losses} "
            f"of {it.opponent_games}; {figure['rounds']:.1f} rounds, {it.draws} draws; policy "
            f"{it.policy_loss:+.4f}, value {it.value_loss:.3f}, entropy {it.entropy:.3f}, KL "
            f"{it.kl:.4f}, clipped {it.clipped:.1%}, {it.passes} passes; play "
            f"{it.play_seconds:.0f} s, learn {it.learn_seconds:.0f} s.",
            flush=True,
        )
        if done % EVERY == 0:
            figure["record"] = score(done)
            scored = done
        with figures.open("a") as f:
            f.write(json.dumps(figure) + "\n")
    if scored != done:
        score(done)
    print(
        f"{done} iterations in {(time.perf_counter() - start) / 60:.1f} minutes. Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB."
    )
