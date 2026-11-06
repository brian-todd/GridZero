"""Check that AlphaZero's loop learns: from scratch on a small board, in a time box (T8, step 4).

A new, small network (32 channels, 4 blocks) trains by `zero_train.train` on the rules of
`small_rules`, with nothing learned before. Before training and after each generation, the network
alone (its best action, on the CPU) plays 2 * ``starts`` games against the random player and as
many against greedy, on the boards players are scored on (`EVAL_SEED`), and as many against itself,
to show what it fields and does. If the loop learns, its records climb: past random first, then
toward greedy. Each generation also reports how its self-play games ended: by what reason, how
many on a flip in the last round (where turn order may decide the game), and from which seat. No
generation starts that would end past the time box; run it under ``timeout`` as a hard stop
beside it. Each generation's weights go to ``{prefix}_gen_NNN.pt``, and its figures to
``{prefix}_generations.jsonl``.

Run with ``uv run python run.py tactics.zero_scratch minutes [prefix] [starts] [games]
[actors] [steps] [value_weight] [lr] [taught]``: by default the files in the working directory as
``zero_scratch_*``, 24 starts, 128 games a generation, every core but two playing them, 1500
steps, a value weight of 0.25 and a learning rate of 1e-3. A network that starts from nothing
needs far more than `Zero`'s defaults for warm starts: at 150 steps and 2e-4 (2026-11-05), it
trained for 0.6 s a generation against 210 s of play, and hardly moved. With ``taught`` above 0
it first imitates that many games of greedy against itself (`agents.imitate`).
"""

import dataclasses
import json
import time
from collections import Counter
from collections.abc import Sequence
from typing import TypedDict

import torch

from examples.games import CORES
from examples.games.curves import new_log
from examples.games.tactics import (
    EVAL_SEED,
    Captured,
    GameOver,
    GreedyPlayer,
    RandomPlayer,
    Tactics,
    TacticsPosition,
    summarize,
)
from examples.games.tactics.events import Ending
from gridzero.agents.imitate import collect, imitate
from gridzero.agents.net import NetPlayer, PolicyNet
from gridzero.agents.zero_train import ACTORS, Generation, Zero, train
from gridzero.game import Score, ladder, series

SEED = 20_000_000  # the self-play boards, from SEED onward, and the network's first weights
TAUGHT_SEED = 30_000_000  # the boards of the games it imitates, if any
CHANNELS, BLOCKS = 32, 4


class Habits(TypedDict):
    """What a network fields and does against itself (`habits`)."""

    picks: dict[str, float]
    built: float
    broke: float
    shots: float
    rounds: float
    endings: dict[Ending, int]


class Endings(TypedDict):
    """How self-play games ended (`endings`)."""

    reasons: dict[Ending, int]
    last_round_flips: int
    team_0: int
    games: int


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
    rivals = {"random": RandomPlayer(), "greedy": GreedyPlayer()}
    player = NetPlayer(on_cpu(game, net))
    return ladder(game, player, rivals, starts, EVAL_SEED, same_dice=False, workers=CORES)


def on_cpu(game: Tactics, net: PolicyNet) -> PolicyNet:
    """Return a copy of a network on the CPU, to play on.

    Parameters
    ----------
    game : Tactics
        The rules.
    net : PolicyNet
        The network, on any device, of this experiment's size.

    Returns
    -------
    PolicyNet
        The copy.
    """
    cpu = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS)
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    return cpu


def habits(game: Tactics, net: PolicyNet, starts: int) -> Habits:
    """Play the network alone against itself, and return what it fields and does.

    Parameters
    ----------
    game : Tactics
        The rules.
    net : PolicyNet
        The network, on any device.
    starts : int
        Pairs of games, with dice of their own.

    Returns
    -------
    Habits
        ``picks``: units fielded a side, by kind's name; ``built``, ``broke`` and ``shots``: tiles
        and deferred attacks a game; ``rounds``: its mean; and ``endings``: games by reason.
    """
    player = NetPlayer(on_cpu(game, net))
    records_ = series(
        game, [("a", player), ("b", player)], starts, EVAL_SEED, same_dice=False, workers=CORES
    )
    summary = summarize(game, records_)
    games = summary.tally.games
    return {
        "picks": {
            stats.name: float(n / (2 * games))
            for stats, n in zip(game.units, summary.placed, strict=True)
        },
        "built": summary.built / games,
        "broke": summary.broke / games,
        "shots": summary.shots / games,
        "rounds": summary.rounds,
        "endings": dict(summary.endings),
    }


def endings(finals: Sequence[TacticsPosition]) -> Endings:
    """Return how self-play games ended, from their last positions.

    Parameters
    ----------
    finals : sequence of TacticsPosition
        The last position of each game.

    Returns
    -------
    Endings
        ``reasons``: games by `events.Ending`; ``last_round_flips``: games won at the cap by a flip
        in that last round; and ``team_0``: team 0's wins, of ``games``.
    """
    reasons: Counter[Ending] = Counter()
    last_round_flips = 0
    for final in finals:
        over = final.events[-1]
        if isinstance(over, GameOver):
            reasons[over.reason] += 1
            flipped = any(isinstance(e, Captured) and e.flip for e in final.events)
            last_round_flips += over.reason == "cap" and flipped
    return {
        "reasons": dict(reasons),
        "last_round_flips": last_round_flips,
        "team_0": sum(final.winner == 0 for final in finals),
        "games": len(finals),
    }


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


def small_rules() -> Tactics:
    """Return the rules this trains on: the default team and caps, on a 15x15 board.

    Chosen with greedy players (2026-11-05) to end as the full board does, 38% of games by holding
    and 62% by killing, in 5.2 rounds and about 176 decisions. On 9x9, 97% to 100% end by killing,
    in 3 rounds, and the objective hardly matters.

    Returns
    -------
    Tactics
        The rules: 4 rounds to hold, a deadline at round 8 and a cap at round 16.
    """
    return Tactics(shape=(15, 15), home_depth=4, hold_rounds=4, capture_by=8, max_rounds=16)


def main(
    minutes: str,
    prefix: str = "zero_scratch",
    starts: str = "24",
    games: str = "128",
    actors: str = str(ACTORS),
    steps: str = "1500",
    value_weight: str = "0.25",
    lr: str = "1e-3",
    taught: str = "0",
) -> None:
    """Train from scratch for at most ``minutes``, measuring each generation.

    On `small_rules`, whose games are short, so that generations are quick.

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
    steps : str, default "1500"
        Training steps a generation.
    value_weight : str, default "0.25"
        The value loss's weight.
    lr : str, default "1e-3"
        The learning rate.
    taught : str, default "0"
        Games of greedy to imitate before training, if any.

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
    game = small_rules()
    torch.manual_seed(SEED)
    net = PolicyNet(game, channels=CHANNELS, blocks=BLOCKS).cuda()
    settings = Zero(
        games=int(games),
        actors=int(actors),
        steps=int(steps),
        value_weight=float(value_weight),
        lr=float(lr),
    )
    print(f"AlphaZero from scratch on {game.shape} for at most {minutes} minutes: {settings}.")
    start = time.perf_counter()
    box = float(minutes) * 60
    if int(taught) > 0:
        lesson = collect(game, GreedyPlayer(), int(taught), TAUGHT_SEED, workers=CORES)
        print(f"Imitating {len(lesson)} decisions of {taught} games of greedy.", flush=True)
        for epoch in imitate(net, lesson, seed=SEED):
            print(f"  {epoch}", flush=True)
        del lesson
    show("Generation 0", records(game, net, int(starts)))
    left = max(box - (time.perf_counter() - start), 0.0)  # at 0, it trains one generation
    for g in train(net, game, settings, generations=1000, seed=SEED, seconds=left):
        torch.save(
            {k: v.cpu() for k, v in net.state_dict().items()}, f"{prefix}_gen_{g.generation:03d}.pt"
        )
        result = records(game, net, int(starts))
        ended = endings([p.final for p in g.played])
        does = habits(game, net, int(starts))
        record = dataclasses.asdict(g) | {"records": result, "endings": ended, "habits": does}
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
        print(f"  self-play ended: {ended}; network against itself: {does}", flush=True)
