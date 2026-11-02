"""Self-play and training for the AlphaZero player on a GPU, for any learnable game.

`train` runs AlphaZero's loop on a `PolicyNet` for a game that is `agents.frame.Learnable`, one
generation at a time: the AlphaZero player
on the network plays games against itself (`generate`), with Gumbel noise; a share of its
searched decisions goes into a replay buffer of the latest; and the network trains on batches
drawn from the buffer, toward the search's improved policy and the games' outcomes.

`generate` splits games between actor processes, which play them on the CPU with
`agents.zero.self_play`, and judges their leaves in an inference server, a thread of this process
that holds the network on its device (a GPU, in bfloat16). Each actor plays its games at once:
it writes the observations of one leaf per game into memory it shares with the server, as
bytes, asks through its pipe, and waits; the server takes every request waiting, runs them
through the network as one batch, writes the scores and values back, and answers each actor.
Batches therefore span all actors, and one CUDA context serves them all. On a GPU the server
keeps two batches in flight, gathering one while the GPU runs the other.

It was written for Tactics, where these were measured.
Actors that each held a copy of the network on the GPU (2026-10-26) ran 545 games an hour
with 16 actors: their 16 CUDA contexts took turns on the GPU, each with small batches. A
server that ran one batch at a time, on float32 observations, ran 683 with 22 actors; it was
busy 97% of the time, with the GPU idle more than half of it.

This module imports Torch, so the package does not import it.
"""

import ctypes
import multiprocessing
import multiprocessing.context
import multiprocessing.queues
import os
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection, wait
from typing import Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from gridzero.agents.frame import Learnable
from gridzero.agents.net import (
    NetEvaluator,
    PolicyNet,
    as_input,
    channels_last,
    judgements,
    observations,
    outputs,
)
from gridzero.agents.zero import AlphaZeroPlayer, Experience, Played, self_play
from gridzero.core import State
from gridzero.core._arrays import check_int, check_real
from gridzero.game import Position


class ActorStats(NamedTuple):
    """What one actor's games cost."""

    games: int
    seconds: float
    """Wall time of the actor's games."""
    evaluator_seconds: float
    """Time in the evaluator: observations, waiting for the server, and its answers back."""
    calls: int
    """Requests to the server."""
    positions: int
    """Positions judged."""


class Costs(NamedTuple):
    """What a call of `generate` cost."""

    seconds: float
    """Wall time, from starting the actors to their last result."""
    actors: list[ActorStats]
    """Each actor's costs, in order of the actors."""
    batches: int
    """Batches the server ran."""
    positions: int
    """Positions the server judged."""
    server_seconds: float
    """Time the server spent other than waiting for requests: gathering, launching, waiting
    for and answering batches."""
    peak_mib: float
    """The most memory Torch allocated on the device, in MiB (0 on the CPU)."""


def generate[G: Learnable[Any], P: Position[State, object]](
    game: G,
    player: AlphaZeroPlayer[G, P],
    seeds: Sequence[int],
    *,
    actors: int = 1,
    keep: float = 0.25,
) -> tuple[Experience, list[Played[P]], Costs]:
    """Play self-play games of ``player`` in actor processes, judged by an inference server.

    ``G`` is bounded by ``Learnable[Any]``: any learnable game, whatever its position type.

    Parameters
    ----------
    game : Learnable
        The rules, sent to every actor (games are values that pickle).
    player : AlphaZeroPlayer
        The player on both sides. Its evaluator must be a `agents.net.NetEvaluator`, whose network
        the server runs where its parameters are (in bfloat16 on a GPU).
    seeds : sequence of int
        One game per seed, split between the actors in order, in near-equal runs.
    actors : int, default 1
        Actor processes, at least 1.
    keep : float, default 0.25
        The share of searched decisions kept (see `agents.zero.self_play`).

    Returns
    -------
    experience : Experience
        The decisions kept, in order of the seeds.
    played : list of Played
        How each game went, in order of the seeds.
    costs : Costs
        What the actors and the server spent.

    Raises
    ------
    TypeError, ValueError
        If ``actors`` is not an integer of at least 1, the evaluator is not a
        `agents.net.NetEvaluator`, or `agents.zero.self_play` rejects an argument.
    RuntimeError
        If an actor or the server fails; an actor's traceback is in the message.
    """
    actors = check_int(actors, "actors", minimum=1)
    if not isinstance(player.evaluator, NetEvaluator):
        raise TypeError(
            f"the player's evaluator must be a NetEvaluator, got {type(player.evaluator).__name__}"
        )
    split = np.array_split(np.asarray(list(seeds), dtype=np.int64), actors)
    runs = [r.tolist() for r in split if r.size]
    if not runs:
        empty, played = self_play(game, player, [], keep=keep)
        return empty, played, Costs(0.0, [], 0, 0, 0.0, 0.0)
    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")  # as in series
    server = _Server(game, player.evaluator.net, [len(r) for r in runs], context)
    sent: AlphaZeroPlayer[G, P] = replace(player, evaluator=_unset)  # the network stays here
    results: _Results = context.Queue()
    processes = [
        context.Process(
            target=_actor,
            args=(i, game, sent, run, keep, server.slots[i], results),
            daemon=True,
        )
        for i, run in enumerate(runs)
    ]
    for p in processes:
        p.start()
    server.start()
    parts: dict[int, _Part[P]] = {}
    try:
        while len(parts) < len(runs):
            try:
                i, part, error = results.get(timeout=1.0)
            except queue.Empty:
                server.check()
                continue
            if part is None:
                raise RuntimeError(f"actor {i} failed:\n{error}")
            parts[i] = part
    finally:
        try:
            server.stop()  # raises what stopped the server, chained to any error raised above
        finally:
            for p in processes:  # stopped even then
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
    ordered = [parts[i] for i in range(len(runs))]
    costs = Costs(
        time.perf_counter() - start,
        [p[2] for p in ordered],
        server.batches,
        server.positions,
        server.seconds,
        server.peak_mib,
    )
    return Experience.join([p[0] for p in ordered]), [g for p in ordered for g in p[1]], costs


ACTORS = max(1, (os.process_cpu_count() or 1) - 2)
"""Actor processes by default: every core this process may use but two, which the inference
server and the learner keep busy (22 of 24 on the machine the defaults were tuned on)."""


@dataclass(frozen=True)
class Zero:
    """The settings of AlphaZero training.

    Parameters
    ----------
    games : int, default 256
        Self-play games per generation, at least 1.
    actors : int, default ACTORS
        Actor processes playing them (`generate`), at least 1.
    simulations, considered : int, default 64 and 16
        The AlphaZero player's simulations a decision and actions considered at the root
        (`zero.AlphaZeroPlayer`), which plays with Gumbel noise.
    keep : float, default 0.25
        The share of searched decisions kept for training, in [0, 1].
    buffer : int, default 200_000
        Decisions the replay buffer holds, at least 1: the latest.
    steps : int, default 500
        Training steps per generation, at least 0.
    batch_size : int, default 256
        Decisions per step, drawn at random from the buffer, at least 1.
    lr, weight_decay : float, default 2e-4 and 1e-4
        AdamW's learning rate (above 0) and weight decay (at least 0). The rate is low since
        training starts from a trained network.
    value_weight : float, default 1.0
        Weight of the value's squared error beside the policy's cross-entropy, at least 0.

    Raises
    ------
    TypeError, ValueError
        If a setting is not a number in its range.
    """

    games: int = 256
    actors: int = ACTORS
    simulations: int = 64
    considered: int = 16
    keep: float = 0.25
    buffer: int = 200_000
    steps: int = 500
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    value_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("games", "actors", "simulations", "considered", "buffer", "batch_size"):
            check_int(getattr(self, name), name, minimum=1)
        check_int(self.steps, "steps", minimum=0)
        check_real(self.keep, "keep", 0.0, 1.0)
        if check_real(self.lr, "lr", 0.0) == 0:
            raise ValueError("lr must be above 0")
        check_real(self.weight_decay, "weight_decay", 0.0)
        check_real(self.value_weight, "value_weight", 0.0)


@dataclass(frozen=True, eq=False)
class Generation[P]:
    """What one generation of AlphaZero training did.

    Parameters
    ----------
    generation : int
        Generations done, from 1.
    games, draws : int
        Self-play games played, and drawn.
    plies : float
        Mean plies a game.
    searched, kept : int
        Decisions searched in the games, and kept for training.
    buffer : int
        Decisions in the replay buffer after this generation's were added.
    fresh_policy_loss, fresh_value_loss : float
        The losses on the decisions kept, before training on them: how well the network
        already predicted the search's policy and the outcome. NaN if none were kept.
    policy_loss, value_loss : float
        Mean losses over the generation's training steps (NaN without steps): the
        cross-entropy of the improved policy and the squared error of the value.
    target_entropy : float
        The mean entropy of the improved policies kept (the floor of the policy's
        cross-entropy), NaN if none were kept.
    play_seconds, train_seconds : float
        Wall time of self-play and of training.
    costs : Costs
        What self-play cost (`generate`).
    played : tuple of Played
        How each game went, with its last position, for what is the game's own (such as the
        rounds of Tactics).
    """

    generation: int
    games: int
    draws: int
    plies: float
    searched: int
    kept: int
    buffer: int
    fresh_policy_loss: float
    fresh_value_loss: float
    policy_loss: float
    value_loss: float
    target_entropy: float
    play_seconds: float
    train_seconds: float
    costs: Costs
    played: tuple[Played[P], ...]


def train[P: Position[State, object]](
    net: PolicyNet,
    game: Learnable[P],
    settings: Zero | None = None,
    *,
    generations: int = 10,
    seed: int = 0,
    seconds: float | None = None,
) -> Iterator[Generation[P]]:
    """Train a network by AlphaZero's loop, one generation at a time.

    Generation ``g`` (from 0) plays self-play games with seeds ``seed + g * games`` onward,
    with the network as it stands, in `generate`'s actors and inference server; adds the
    decisions kept to the replay buffer, dropping the oldest past `Zero.buffer`; and trains
    the network on `Zero.steps` batches drawn at random from the buffer, with AdamW kept
    across generations. The loss is the cross-entropy of the search's improved policy over
    legal actions plus `Zero.value_weight` times the squared error of the value against the
    outcome. On a GPU, self-play and training run in bfloat16.

    Parameters
    ----------
    net : PolicyNet
        The network, trained in place, on the device its parameters are on.
    game : Learnable
        The rules the network was made for (`PolicyNet(game)`).
    settings : Zero, optional
        The settings; `Zero()` by default.
    generations : int, default 10
        The most generations to run, at least 1.
    seed : int, default 0
        Seed of the first game, at least 0. It also seeds the batches drawn.
    seconds : float, optional
        A time box: no generation starts that would end after ``seconds`` from the start, as
        estimated from the last generation's length. The first always runs. None for no box.
        The box is wall time from the first generation's start, so it counts what the caller
        does between generations too (such as evaluating the network), and so does each
        generation's length: a caller's work shortens the run by its own time. To compare runs,
        a fixed number of ``generations`` is fairer than a box.

    Yields
    ------
    Generation
        What each generation did. Training runs as the generations are taken.

    Raises
    ------
    TypeError, ValueError
        If ``generations``, ``seed`` or ``seconds`` is not a number in range.
    """
    settings = Zero() if settings is None else settings
    generations = check_int(generations, "generations", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    if seconds is not None:
        check_real(seconds, "seconds", 0.0)
    return _train(net, game, settings, generations, seed, seconds)


def _train[P: Position[State, object]](
    net: PolicyNet,
    game: Learnable[P],
    settings: Zero,
    generations: int,
    seed: int,
    seconds: float | None,
) -> Iterator[Generation[P]]:
    """Run the generations of `train` once its arguments are checked."""
    device = channels_last(net)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    player: AlphaZeroPlayer[Learnable[P], P] = AlphaZeroPlayer(
        NetEvaluator(net),
        simulations=settings.simulations,
        considered=settings.considered,
        noise=True,
    )
    rng = np.random.default_rng(seed)
    buffer: Experience | None = None
    begin = time.perf_counter()
    last = 0.0
    for g in range(generations):
        if g > 0 and not _room(time.perf_counter() - begin, last, seconds):
            return
        start = time.perf_counter()
        first = seed + g * settings.games
        net.eval()
        fresh, played, costs = generate(
            game,
            player,
            range(first, first + settings.games),
            actors=settings.actors,
            keep=settings.keep,
        )
        playing = time.perf_counter() - start
        buffer = _add(buffer, fresh, settings.buffer)
        fresh_losses = _mean_losses(net, fresh, settings.batch_size, device)
        start = time.perf_counter()
        trained = _learn(net, optimizer, buffer, settings, device, rng)
        training = time.perf_counter() - start
        yield Generation(
            generation=g + 1,
            games=len(played),
            draws=sum(p.winner is None for p in played),
            plies=float(np.mean([p.plies for p in played])),
            searched=sum(p.searched for p in played),
            kept=len(fresh),
            buffer=len(buffer),
            fresh_policy_loss=fresh_losses[0],
            fresh_value_loss=fresh_losses[1],
            policy_loss=trained[0],
            value_loss=trained[1],
            target_entropy=_entropy(fresh),
            play_seconds=playing,
            train_seconds=training,
            costs=costs,
            played=tuple(played),
        )
        last = time.perf_counter() - start + playing


def _room(elapsed: float, last: float, seconds: float | None) -> bool:
    """Return whether a generation as long as the last fits in the time box after ``elapsed``."""
    return seconds is None or elapsed + last <= seconds


def _add(buffer: Experience | None, fresh: Experience, size: int) -> Experience:
    """Return the buffer with fresh decisions after it, keeping the latest ``size``."""
    joined = fresh if buffer is None else Experience.join([buffer, fresh])
    if len(joined) > size:
        joined = joined.take(np.arange(len(joined) - size, len(joined)))
    return joined


def _learn(
    net: PolicyNet,
    optimizer: torch.optim.Optimizer,
    buffer: Experience,
    settings: Zero,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Train on `Zero.steps` random batches of the buffer; return the mean losses, or NaN."""
    if settings.steps == 0 or len(buffer) == 0:
        return np.nan, np.nan
    net.train()
    sums = np.zeros(2)
    for _ in range(settings.steps):
        index = np.sort(rng.integers(len(buffer), size=min(settings.batch_size, len(buffer))))
        policy, value = losses(net, buffer.take(index), device)
        loss = policy.mean() + settings.value_weight * value.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sums += [policy.mean().item(), value.mean().item()]
    net.eval()
    policy_loss, value_loss = sums / settings.steps
    return float(policy_loss), float(value_loss)


def _mean_losses(
    net: PolicyNet, experience: Experience, batch_size: int, device: torch.device
) -> tuple[float, float]:
    """Return the mean losses of the network on decisions, without training; NaN for none."""
    if len(experience) == 0:
        return np.nan, np.nan
    net.eval()
    sums = np.zeros(2)
    with torch.no_grad():
        for start in range(0, len(experience), batch_size):
            index = np.arange(start, min(start + batch_size, len(experience)))
            policy, value = losses(net, experience.take(index), device)
            sums += [policy.sum().item(), value.sum().item()]
    policy_loss, value_loss = sums / len(experience)
    return float(policy_loss), float(value_loss)


def losses(net: PolicyNet, batch: Experience, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return the AlphaZero losses of each decision of a batch.

    Parameters
    ----------
    net : PolicyNet
        The network, on ``device``; on a GPU it runs in bfloat16.
    batch : Experience
        Decisions, from `zero.self_play` with the network's rules.
    device : torch.device
        Where the network is.

    Returns
    -------
    policy : Tensor of float32, shape (n,)
        The cross-entropy of each decision's improved policy against the network's policy
        over its legal actions: ``-sum(target * log_softmax(scores over legal actions))``.
    value : Tensor of float32, shape (n,)
        The squared error of the network's value against the outcome.
    """
    return _losses_of(*outputs(net, batch.obs, batch.legal, device), batch)


def _losses_of(
    scores: Tensor, value: Tensor, legal: Tensor, batch: Experience
) -> tuple[Tensor, Tensor]:
    """Return `losses` from the network's `outputs` on ``batch``, for callers that need both."""
    log_policy = torch.where(legal, torch.log_softmax(scores, dim=1), 0.0)
    target = torch.zeros(legal.shape, device=scores.device)
    target[legal] = torch.from_numpy(batch.policy).to(scores.device)  # row by row, as in the CSR
    outcome = torch.from_numpy(batch.outcome).to(scores.device)
    return -(target * log_policy).sum(dim=1), (value - outcome) ** 2


def _entropy(experience: Experience) -> float:
    """Return the mean entropy of the improved policies of decisions, NaN for none."""
    if len(experience) == 0:
        return np.nan
    p = experience.policy.astype(np.float64)
    terms = -p * np.log(p, where=p > 0, out=np.zeros_like(p))
    return float(np.add.reduceat(terms, experience.start[:-1]).mean())


type _Part[P] = tuple[Experience, list[Played[P]], ActorStats]
"""What an actor sends back: its experience, its games and its costs."""

# Any: the queue carries any game's results, whose position type the server doesn't know.
type _Results = multiprocessing.queues.Queue[tuple[int, _Part[Any] | None, str | None]]
"""What actors send back: the actor, its part or None, and None or a traceback."""


@dataclass(frozen=True)
class _Slot:
    """An actor's memory shared with the server, and its line to the server.

    ``obs`` holds up to ``size`` observations (uint8, shape (size, n_planes, n_cells), as
    `agents.net.observations` makes them), ``scores`` the network's scores for them (float32, shape
    (size, n_verbs, n_cells)) and ``value`` its values (float32, shape (size,)), as flat
    shared arrays.
    """

    size: int
    obs: ctypes.Array[ctypes.c_ubyte]
    scores: ctypes.Array[ctypes.c_float]
    value: ctypes.Array[ctypes.c_float]
    line: Connection
    """The actor's end of its pipe: it sends how many observations it wrote, as 4 bytes
    (little-endian), and the server answers with no bytes once their judgements are in."""


def _views(
    game: Learnable[Any], slot: _Slot
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
    """Return a slot's shared arrays as NumPy arrays: observations, scores and values."""
    n_planes, n_cells = len(game.planes), game.space.n_cells
    obs = np.frombuffer(slot.obs, dtype=np.uint8)
    scores = np.frombuffer(slot.scores, dtype=np.float32)
    return (
        obs.reshape(slot.size, n_planes, n_cells),
        scores.reshape(slot.size, game.n_verbs, n_cells),
        np.frombuffer(slot.value, dtype=np.float32),
    )


_GRAPHED = 512
"""The largest batch replayed from a CUDA graph; larger ones run as they come."""


def _sizes(most: int) -> list[int]:
    """Return the batch sizes to capture as graphs, up to ``most``.

    They are multiples of 8, each about 1.5 times the last, so padding a batch to the next
    costs at most half of it.
    """
    sizes = [8]
    while sizes[-1] < most:
        sizes.append(-(-sizes[-1] * 3 // 16) * 8)
    return sizes


class _Forward:
    """The network's forward pass, from observations as bytes on the host.

    On a GPU each batch is padded to the next of a few sizes and replayed from a CUDA graph
    captured for that size: replaying costs about 2 us of the CPU, where launching the
    network's kernels one by one cost 580 us (an RTX 3080, a 6-block network), whatever the
    batch. On the CPU the network runs as it is.
    """

    def __init__(
        self, net: PolicyNet, device: torch.device, shape: tuple[int, int], most: int
    ) -> None:
        self.net = net
        self.device = device
        self.shape = shape
        self.graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
        self.graphs = {}
        if device.type == "cuda":
            pool = torch.cuda.MemPool()  # shared: one graph replays at a time
            for size in reversed(_sizes(min(most, _GRAPHED))):
                self.graphs[size] = self._capture(size, pool)

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scores and values of ``obs`` (uint8, on the host), float32 on the device.

        On a GPU the results may be the graph's own outputs, which its next replay
        overwrites: copy them first, on the same stream.
        """
        n = len(obs)
        size = min((s for s in self.graphs if s >= n), default=None)
        if size is None:
            gpu = self.device.type == "cuda"
            with torch.no_grad(), torch.autocast(self.device.type, torch.bfloat16, enabled=gpu):
                scores, value = self.net(as_input(obs.to(self.device, non_blocking=True)))
            return scores.float(), value.float()
        graph, static, scores, value = self.graphs[size]
        static[:n].copy_(obs, non_blocking=True)  # rows past n keep old observations
        graph.replay()
        return scores[:n], value[:n]

    def _capture(
        self, size: int, pool: torch.cuda.MemPool
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture the forward pass of ``size`` observations: graph, input and outputs."""
        static = torch.zeros((size, *self.shape), dtype=torch.uint8, device=self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(3):  # warm up, as torch.cuda.graph asks
                self.net(as_input(static))
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool), torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            scores, value = self.net(as_input(static))
            scores, value = scores.float(), value.float()
        return graph, static, scores, value


@dataclass(frozen=True)
class _Batch:
    """A batch in flight: who asked for how many, and where the answers will be."""

    waiting: list[tuple[int, int]]
    """Each request: the actor and how many observations it wrote."""
    scores: torch.Tensor
    """The scores, on the host once ``done`` is reached (float32, (n, n_verbs, n_cells))."""
    value: torch.Tensor
    """The values, likewise (float32, shape (n,))."""
    done: torch.cuda.Event | None
    """Reached when the outputs are on the host; None where they already are (on the CPU)."""


class _Server:
    """The inference server: a thread that judges the actors' requests in batches."""

    def __init__(
        self,
        game: Learnable[Any],
        net: PolicyNet,
        sizes: list[int],
        context: multiprocessing.context.SpawnContext,
    ) -> None:
        n_planes, n_cells = len(game.planes), game.space.n_cells
        self.net = net
        self.shape = (n_planes, n_cells)
        self.slots: list[_Slot] = []
        self._lines: list[Connection] = []
        for size in sizes:
            mine, theirs = context.Pipe()
            obs = context.RawArray("B", size * n_planes * n_cells)
            scores = context.RawArray("f", size * game.n_verbs * n_cells)
            self.slots.append(_Slot(size, obs, scores, context.RawArray("f", size), theirs))
            self._lines.append(mine)
        self._actor = {id(line): i for i, line in enumerate(self._lines)}
        self.views = [_views(game, slot) for slot in self.slots]
        # Room for every actor's whole slot, twice over: a batch gathers into one half while
        # the other half's batch is in flight.
        self.most = sum(sizes)
        shapes = [(self.most, *self.shape), (self.most, game.n_verbs, n_cells), (self.most,)]
        dtypes = [torch.uint8, torch.float32, torch.float32]
        pin = next(net.parameters()).device.type == "cuda"
        self._host = [
            [
                torch.empty(shape, dtype=dtype, pin_memory=pin)
                for shape, dtype in zip(shapes, dtypes, strict=True)
            ]
            for _ in range(2)
        ]
        self.batches = self.positions = 0
        self.seconds = self.peak_mib = 0.0
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        """Start serving."""
        self._thread.start()

    def check(self) -> None:
        """Raise what stopped the server early, if anything did."""
        if self._error is not None:
            raise RuntimeError("the inference server failed") from self._error

    def stop(self) -> None:
        """Stop serving, and raise what stopped the server early, if anything did."""
        self._done.set()
        self._thread.join()
        self.check()

    def _serve(self) -> None:
        """Gather the requests waiting into batches and answer them, until stopped.

        Up to two batches are in flight. With none, the requests waiting go at once; with
        one, the next goes once it is at least as large, so that batches don't shrink to
        what arrives while the GPU is busy. A batch is answered when it is done, or when no
        request arrived. On the CPU a batch is done when it goes.
        """
        try:
            device = channels_last(self.net)
            self.net.eval()
            gpu = device.type == "cuda"
            if gpu:
                torch.cuda.reset_peak_memory_stats(device)
            forward = _Forward(self.net, device, self.shape, self.most)
            flight: deque[_Batch] = deque()
            pending: list[tuple[int, int]] = []
            count = 0
            while not self._done.is_set():
                busy = {i for batch in flight for i, _ in batch.waiting}
                busy.update(i for i, _ in pending)
                idle = [line for i, line in enumerate(self._lines) if i not in busy]
                asking = wait(idle, timeout=1e-4 if flight or pending else 0.05)
                start = time.perf_counter()
                pending.extend(self._request(self._actor[id(line)]) for line in asking)
                n = sum(k for _, k in pending)
                if pending and (not flight or (len(flight) == 1 and n >= len(flight[0].value))):
                    flight.append(self._launch(pending, self._host[count % 2], forward))
                    pending, count = [], count + 1
                elif flight and (not asking or _done(flight[0])):
                    self._answer(flight.popleft())
                self.seconds += time.perf_counter() - start
            if gpu:
                self.peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
        except BaseException as error:  # raised by check, in the caller's thread
            self._error = error

    def _request(self, i: int) -> tuple[int, int]:
        """Read actor ``i``'s request: the actor and how many observations it wrote."""
        return i, int.from_bytes(self._lines[i].recv_bytes(), "little")

    def _launch(
        self, waiting: list[tuple[int, int]], host: list[torch.Tensor], forward: _Forward
    ) -> _Batch:
        """Gather the waiting requests into ``host`` and start them through the network."""
        obs_host, scores_host, value_host = host
        n = sum(k for _, k in waiting)
        np.concatenate([self.views[i][0][:k] for i, k in waiting], out=obs_host[:n].numpy())
        scores, value = forward(obs_host[:n])
        scores_host[:n].copy_(scores, non_blocking=True)
        value_host[:n].copy_(value, non_blocking=True)
        done = None
        if forward.device.type == "cuda":
            done = torch.cuda.Event()
            done.record()
        self.batches += 1
        self.positions += n
        return _Batch(waiting, scores_host[:n], value_host[:n], done)

    def _answer(self, batch: _Batch) -> None:
        """Wait for a batch, write its judgements to the actors' slots, and answer them."""
        if batch.done is not None:
            batch.done.synchronize()
        scores, value = batch.scores.numpy(), batch.value.numpy()
        at = 0
        for i, k in batch.waiting:
            self.views[i][1][:k] = scores[at : at + k]
            self.views[i][2][:k] = value[at : at + k]
            at += k
            self._lines[i].send_bytes(b"")


def _done(batch: _Batch) -> bool:
    """Return whether a batch's outputs are on the host."""
    return batch.done is None or batch.done.query()


def _unset(
    game: object, positions: Sequence[object], legal: Sequence[NDArray[np.bool_]], /
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Stand in for the evaluator of a player sent to an actor, which gives it its own."""
    raise RuntimeError("an actor's player has no evaluator until the actor gives it one")


@dataclass(eq=False)
class _Remote:
    """An actor's evaluator: observations out to the server, its judgements back."""

    game: Learnable[Any]
    slot: _Slot
    seconds: float = 0.0
    calls: int = 0
    positions: int = 0

    def __post_init__(self) -> None:
        self.obs, self.scores, self.value = _views(self.game, self.slot)

    def __call__[P: Position[State, object]](
        self, game: Learnable[P], positions: Sequence[P], legal: Sequence[NDArray[np.bool_]], /
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the network's judgements of ``positions``, as `agents.net.NetEvaluator` does."""
        start = time.perf_counter()
        n = len(positions)
        players, obs = observations(game, positions)
        self.obs[:n] = obs
        self.slot.line.send_bytes(n.to_bytes(4, "little"))
        self.slot.line.recv_bytes()
        scores = self.scores[:n].astype(np.float64)
        result = judgements(game, scores, self.value[:n].astype(np.float64), legal, players)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.positions += n
        return result


def _actor[G: Learnable[Any], P: Position[State, object]](
    index: int,
    game: G,
    player: AlphaZeroPlayer[G, P],
    seeds: list[int],
    keep: float,
    slot: _Slot,
    results: _Results,
) -> None:
    """Play an actor's games with the server as evaluator, and put the result on ``results``."""
    try:
        torch.set_num_threads(1)  # the actors share the machine's cores
        remote = _Remote(game, slot)
        start = time.perf_counter()
        experience, played = self_play(game, replace(player, evaluator=remote), seeds, keep=keep)
        seconds = time.perf_counter() - start
        stats = ActorStats(len(seeds), seconds, remote.seconds, remote.calls, remote.positions)
        results.put((index, (experience, played, stats), None))
    except BaseException:  # sent to the caller, which raises it
        results.put((index, None, traceback.format_exc()))
