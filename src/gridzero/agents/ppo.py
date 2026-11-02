"""Training a `PolicyNet` by self-play with proximal policy optimization (PPO), in any game.

Each iteration plays games with the current network, drawing its actions from its policy, in
processes on the CPU, then improves the network on what they showed, on its own device. Games
are the network against itself, both sides learning; with an opponent (`PPO.opponent`), a share
are against it, only the network's side learning, which keeps it honest and measures it as it
trains. Written for Tactics (T7, stage 2), against its greedy player.

A decision's reward is the game's result for its player, +1, -1 or 0, at the player's last
decision with a choice, and nothing before. Advantages are generalized advantage estimates
(GAE) along each player's own decisions, with the network's values; the moves of the other side
between two of them are the environment's. The network's value then keeps its meaning: the
expected result of the position for the player, from -1 to 1.

This module imports Torch, so the package does not import it.
"""

import multiprocessing
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from gridzero.agents.frame import Learnable
from gridzero.agents.net import (
    Decision,
    NetPlayer,
    PolicyNet,
    channels_last,
    outputs,
)
from gridzero.agents.zero import Experience, Played
from gridzero.core import State
from gridzero.core._arrays import check_int, check_real
from gridzero.game import Action, Player, Position, play


@dataclass(frozen=True)
class PPO:
    """The settings of self-play training.

    Parameters
    ----------
    games : int, default 96
        Games per iteration, at least 1.
    opponent : Player, optional
        A player the network also plays against, such as a game's scripted player. None for
        self-play only.
    against_opponent : float, default 0.25
        The share of an iteration's games against the opponent, in [0, 1], the network as
        player 0 and player 1 in turn; unused without one. The rest are the network against
        itself.
    gamma : float, default 1.0
        Discount per decision of a player, in [0, 1].
    lam : float, default 0.95
        GAE's lambda, in [0, 1]: 0 bootstraps from the next value, 1 waits for the result.
    clip : float, default 0.2
        How far the ratio of new to old probabilities may move before its gain is cut, above 0.
    epochs : int, default 4
        Passes over an iteration's decisions, at least 1.
    batch_size : int, default 512
        Decisions per step, at least 1.
    lr : float, default 3e-4
        Adam's learning rate, above 0.
    value_weight : float, default 0.5
        Weight of the value's squared error, at least 0.
    entropy_weight : float, default 0.01
        Weight of the policy's entropy over legal actions, a bonus, at least 0.
    max_grad_norm : float, default 0.5
        The largest norm of a step's gradient, above 0.
    target_kl : float or None, default 0.03
        Stop an iteration's passes early once a pass's mean approximate KL divergence from the
        old policy exceeds it; None never stops early.

    Raises
    ------
    TypeError, ValueError
        If a setting is not a number in its range.
    """

    games: int = 96
    opponent: Player[Any, Any] | None = None  # Any: an opponent of any game, seeing positions
    against_opponent: float = 0.25
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch_size: int = 512
    lr: float = 3e-4
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    def __post_init__(self) -> None:
        check_int(self.games, "games", minimum=1)
        check_int(self.epochs, "epochs", minimum=1)
        check_int(self.batch_size, "batch_size", minimum=1)
        if self.opponent is not None and not callable(self.opponent):
            raise TypeError(f"opponent must be a player, got {type(self.opponent).__name__}")
        for name in ("against_opponent", "gamma", "lam"):
            check_real(getattr(self, name), name, 0.0, 1.0)
        for name in ("value_weight", "entropy_weight"):
            check_real(getattr(self, name), name, 0.0)
        for name in ("clip", "lr", "max_grad_norm"):
            if check_real(getattr(self, name), name, 0.0) == 0:
                raise ValueError(f"{name} must be above 0")
        if self.target_kl is not None and check_real(self.target_kl, "target_kl", 0.0) == 0:
            raise ValueError("target_kl must be above 0")


@dataclass(frozen=True)
class Iteration:
    """What one iteration of self-play did.

    Parameters
    ----------
    iteration : int
        Iterations done, from 1.
    decisions : int
        Decisions learned from.
    opponent_games, opponent_wins, opponent_losses : int
        Games against the opponent this iteration, and the network's wins and losses in them.
    draws : int
        Games drawn.
    played : tuple of Played
        Each game as `zero.Played` (its seed, winner, plies, and final position, for what is
        the game's own, such as Tactics' rounds), ``searched`` and ``kept`` both counting the
        decisions with a choice kept to learn from.
    policy_loss, value_loss, entropy, kl, clipped : float
        Means over the last pass: PPO's clipped surrogate loss, the value's squared error, the
        policy's entropy over legal actions, the approximate KL divergence from the old policy,
        and the share of decisions whose ratio was clipped.
    passes : int
        Passes run, fewer than `PPO.epochs` if `PPO.target_kl` stopped them.
    play_seconds, learn_seconds : float
        Wall time playing and learning.
    """

    iteration: int
    decisions: int
    opponent_games: int
    opponent_wins: int
    opponent_losses: int
    draws: int
    played: tuple[Played[Any], ...]  # Any: positions of any game
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    clipped: float
    passes: int
    play_seconds: float
    learn_seconds: float


def self_play(
    net: PolicyNet,
    game: Learnable[Any],
    settings: PPO | None = None,
    *,
    iterations: int = 10,
    seed: int = 0,
    workers: int = 1,
) -> Iterator[Iteration]:
    """Train a network by self-play, one iteration at a time.

    Iteration ``i`` (from 0) plays games with seeds ``seed + i * games`` onward, so every
    iteration plays on new boards. Games are played by a copy of the network on the CPU, in
    ``workers`` processes that last for the whole run (one Torch thread each); the network
    learns on its own device, where on a GPU it runs in bfloat16.

    Parameters
    ----------
    net : PolicyNet
        The network, trained in place. Starting from one taught by `imitate.imitate` saves
        learning the rules' basics by trial.
    game : Learnable
        The rules the network was made for, of two players.
    settings : PPO, optional
        The settings; `PPO()` by default.
    iterations : int, default 10
        Iterations to run, at least 1.
    seed : int, default 0
        Seed of the first game, at least 0. It also seeds the order of decisions in learning.
    workers : int, default 1
        Processes playing games, at least 1. With 1, games are played here.

    Yields
    ------
    Iteration
        What each iteration did. Training runs as the iterations are taken, and the processes
        end when the generator does.

    Raises
    ------
    TypeError, ValueError
        If ``iterations``, ``seed`` or ``workers`` is not an integer in range, or the game
        hasn't two players.
    """
    if game.n_players != 2:
        raise ValueError(f"PPO self-play plays games of two players, not {game.n_players}")
    settings = PPO() if settings is None else settings
    iterations = check_int(iterations, "iterations", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    workers = check_int(workers, "workers", minimum=1)
    return _self_play(net, game, settings, iterations, seed, workers)


def _self_play(
    net: PolicyNet, game: Learnable[Any], settings: PPO, iterations: int, seed: int, workers: int
) -> Iterator[Iteration]:
    """Run the iterations of `self_play` once its arguments are checked."""
    device = channels_last(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr)
    rng = np.random.default_rng(seed)
    pool = None
    if workers > 1:
        context = multiprocessing.get_context("spawn")  # as in series
        pool = ProcessPoolExecutor(workers, mp_context=context)
    try:
        for i in range(iterations):
            start = time.perf_counter()
            weights = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            jobs = _jobs(game, net, weights, settings, seed + i * settings.games, workers)
            if pool is None:
                parts = [job() for job in jobs]
            else:
                parts = list(pool.map(_Play.__call__, jobs))
            played = time.perf_counter() - start
            batch = _Batch.join([p.batch for p in parts])
            results = [r for p in parts for r in p.results]
            start = time.perf_counter()
            stats, passes = _learn(net, optimizer, batch, settings, device, rng)
            against = [r for r in results if r.learner is not None]
            yield Iteration(
                iteration=i + 1,
                decisions=len(batch.experience),
                opponent_games=len(against),
                opponent_wins=sum(r.played.winner == r.learner for r in against),
                opponent_losses=sum(
                    r.played.winner is not None and r.played.winner != r.learner for r in against
                ),
                draws=sum(r.played.winner is None for r in results),
                played=tuple(r.played for r in results),
                policy_loss=stats[0],
                value_loss=stats[1],
                entropy=stats[2],
                kl=stats[3],
                clipped=stats[4],
                passes=passes,
                play_seconds=played,
                learn_seconds=time.perf_counter() - start,
            )
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)


def _jobs(
    game: Learnable[Any],
    net: PolicyNet,
    weights: dict[str, Tensor],
    settings: PPO,
    first: int,
    workers: int,
) -> list[_Play]:
    """Split an iteration's games into jobs, a few per process so that slow games even out."""
    n_against = 0
    if settings.opponent is not None:
        n_against = round(settings.against_opponent * settings.games)
    games = [
        (first + g, (g % 2) if g < n_against else None, settings.gamma, settings.lam)
        for g in range(settings.games)
    ]
    n_jobs = min(len(games), 1 if workers == 1 else 4 * workers)
    chunks = np.array_split(np.arange(len(games)), n_jobs)
    spec = (net.channels, net.blocks)
    return [
        _Play(game, spec, weights, settings.opponent, tuple(games[k] for k in chunk))
        for chunk in chunks
    ]


@dataclass(frozen=True)
class _Result:
    """How one game went, and the network's player against the opponent (None in self-play)."""

    played: Played[Any]  # Any: positions of any game
    learner: int | None


@dataclass(frozen=True, eq=False)
class _Batch:
    """Decisions to learn from, with what PPO needs beside the experience."""

    experience: Experience
    """Each decision, its policy all on the action drawn."""
    action: NDArray[np.int64]
    """The action drawn, in the player's frame, ``verb * n_cells + cell``, shape (n,)."""
    logp: NDArray[np.float32]
    """The log-probability of each action when it was drawn, shape (n,)."""
    value: NDArray[np.float32]
    """The value then, shape (n,)."""
    advantage: NDArray[np.float32]
    """GAE's advantage, shape (n,)."""
    target: NDArray[np.float32]
    """The value's target, the lambda-return ``advantage + value``, shape (n,)."""

    def take(self, index: NDArray[np.integer]) -> _Batch:
        """Return the decisions at ``index``."""
        return _Batch(
            self.experience.take(index),
            self.action[index],
            self.logp[index],
            self.value[index],
            self.advantage[index],
            self.target[index],
        )

    @staticmethod
    def join(parts: Sequence[_Batch]) -> _Batch:
        """Return decisions one after another; at least one part."""
        return _Batch(
            Experience.join([p.experience for p in parts]),
            np.concatenate([p.action for p in parts]),
            np.concatenate([p.logp for p in parts]),
            np.concatenate([p.value for p in parts]),
            np.concatenate([p.advantage for p in parts]),
            np.concatenate([p.target for p in parts]),
        )


@dataclass(frozen=True)
class _Played:
    """What a job of games gave: decisions to learn from and how the games ended."""

    batch: _Batch
    results: tuple[_Result, ...]


_NETS: dict[tuple[object, tuple[int, int]], PolicyNet] = {}
"""Networks built in this process, kept between jobs so that each job only loads weights."""


@dataclass(frozen=True)
class _Play:
    """Games played by the network, here or in another process.

    ``games`` holds, per game, its seed, the network's player against the opponent (None for
    the network against itself), gamma and lambda.
    """

    game: Learnable[Any]
    spec: tuple[int, int]
    weights: dict[str, Tensor]
    opponent: Player[Any, Any] | None
    games: tuple[tuple[int, int | None, float, float], ...]

    def __call__(self) -> _Played:
        """Play the games and return their decisions and results."""
        threads = torch.get_num_threads()
        # One process per core, and the same arithmetic here as in a worker. Restored after,
        # since with one worker this is the caller's process.
        torch.set_num_threads(1)
        try:
            key = (self.game, self.spec)
            if key not in _NETS:
                _NETS[key] = PolicyNet(self.game, *self.spec)
            net = _NETS[key]
            net.load_state_dict(self.weights)
            parts, results = [], []
            for seed, learner, gamma, lam in self.games:
                batch, result = _play(self.game, net, self.opponent, seed, learner, gamma, lam)
                parts.append(batch)
                results.append(result)
        finally:
            torch.set_num_threads(threads)
        return _Played(_Batch.join(parts), tuple(results))


def _play(
    game: Learnable[Any],
    net: PolicyNet,
    opponent: Player[Any, Any] | None,
    seed: int,
    learner: int | None,
    gamma: float,
    lam: float,
) -> tuple[_Batch, _Result]:
    """Play one game; return the learning side's decisions, with advantages, and the result."""
    actor = _Actor(NetPlayer(net, temperature=1.0))
    players: tuple[Player[Any, Any], Player[Any, Any]] = (actor, actor)
    if learner is not None and opponent is not None:
        players = (actor, opponent) if learner == 0 else (opponent, actor)
    record = play(game, players, game.start(seed))
    decisions = actor.decisions
    n = len(decisions)
    player = np.array([d.player for d in decisions], dtype=np.int64)
    value = np.array([d.value for d in decisions], dtype=np.float32)
    advantage = np.zeros(n, dtype=np.float32)
    winner = record.final.winner
    for side in (0, 1):
        steps = np.flatnonzero(player == side)
        if steps.size == 0:
            continue
        result = 0.0 if winner is None else (1.0 if winner == side else -1.0)
        advantage[steps] = gae(value[steps], result, gamma, lam)
    outcome = np.zeros(n, dtype=np.float32)
    if winner is not None:
        outcome = np.where(player == winner, 1.0, -1.0).astype(np.float32)
    n_planes, n_cells = len(game.planes), game.space.n_cells
    action = np.array([d.index for d in decisions], dtype=np.int64)
    lengths = np.array([np.count_nonzero(d.legal) for d in decisions], dtype=np.int64)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=start[1:])
    policy = np.zeros(start[-1], dtype=np.float32)
    at = [np.count_nonzero(d.legal[: d.index]) for d in decisions]  # among the legal ones
    policy[start[:-1] + np.array(at, dtype=np.int64)] = 1.0
    experience = Experience(
        obs=np.array([np.rint(d.obs * 255) for d in decisions], dtype=np.uint8).reshape(
            n, n_planes, n_cells
        ),
        legal=np.array([np.packbits(d.legal) for d in decisions], dtype=np.uint8).reshape(n, -1),
        policy=policy,
        start=start,
        outcome=outcome,
        game=np.full(n, seed, dtype=np.int64),
        ply=np.array(actor.plies, dtype=np.int64),
    )
    logp = np.array([d.logp for d in decisions], dtype=np.float32)
    batch = _Batch(experience, action, logp, value, advantage, advantage + value)
    final = record.final
    played = Played(seed, winner, final.ply, n, n, final)
    return batch, _Result(played, learner)


def gae(
    values: NDArray[np.float32], result: float, gamma: float, lam: float
) -> NDArray[np.float32]:
    """Return generalized advantage estimates along one player's decisions in a game.

    Parameters
    ----------
    values : ndarray of float32, shape (n,)
        The network's value at each of the player's decisions, in order, at least one.
    result : float
        The game's result for the player, the only reward, after the last decision.
    gamma, lam : float
        The discount per decision and GAE's lambda, each in [0, 1].

    Returns
    -------
    ndarray of float32, shape (n,)
        ``A[t] = delta[t] + gamma * lam * A[t + 1]``, where ``delta[t] = gamma * values[t + 1] -
        values[t]``, and at the last decision ``delta = result - values[-1]``.
    """
    n = values.size
    advantage = np.zeros(n, dtype=np.float64)
    following = 0.0
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            delta = result - float(values[t])
        else:
            delta = gamma * float(values[t + 1]) - float(values[t])
        following = delta + gamma * lam * following
        advantage[t] = following
    return advantage.astype(np.float32)


@dataclass(eq=False)
class _Actor:
    """A player that plays as its `NetPlayer` and keeps each decision with a choice."""

    player: NetPlayer
    decisions: list[Decision] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    def __call__(
        self,
        game: Learnable[Any],
        player: int,
        position: Position[State, object],
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return the network's action, keeping the decision if there was a choice."""
        decision = self.player.decide(game, position, legal, rng)
        if np.count_nonzero(legal) > 1:
            self.decisions.append(decision)
            self.plies.append(position.ply)
        return decision.action


def _learn(
    net: PolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: _Batch,
    settings: PPO,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    """Run PPO's passes over a batch; return the last pass's mean statistics and the passes."""
    n = len(batch.experience)
    stats = [np.nan] * 5
    passes = 0
    if n == 0:
        return stats, passes
    for _ in range(settings.epochs):
        net.train()
        order = rng.permutation(n)
        sums = np.zeros(5)
        for start in range(0, n, settings.batch_size):
            index = np.sort(order[start : start + settings.batch_size])
            part = batch.take(index)
            e = part.experience
            scores, value, legal = outputs(net, e.obs, e.legal, device)
            logp_all = torch.log_softmax(scores, dim=1).masked_fill(~legal, 0.0)
            action = torch.from_numpy(part.action).to(device)
            logp = logp_all.gather(1, action[:, None])[:, 0]
            old = torch.from_numpy(part.logp).to(device)
            advantage = torch.from_numpy(part.advantage).to(device)
            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            log_ratio = logp - old
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - settings.clip, 1 + settings.clip)
            policy = -torch.minimum(ratio * advantage, clipped * advantage)
            target = torch.from_numpy(part.target).to(device)
            value_loss = (value - target) ** 2
            entropy = -(logp_all.exp() * logp_all).sum(dim=1)
            loss = (
                policy.mean()
                + settings.value_weight * value_loss.mean()
                - settings.entropy_weight * entropy.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), settings.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = (ratio - 1) - log_ratio
                cut = (ratio - 1).abs() > settings.clip
                sums += [
                    policy.sum().item(),
                    value_loss.sum().item(),
                    entropy.sum().item(),
                    kl.sum().item(),
                    cut.float().sum().item(),
                ]
        passes += 1
        stats = [float(s) for s in sums / n]
        if settings.target_kl is not None and stats[3] > settings.target_kl:
            break
    return stats, passes
