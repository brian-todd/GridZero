"""Training a MuZero player's network (`agents.muzero_net.WorldNet`) on trajectories.

`fit` takes steps of gradient descent on decisions drawn from `agents.muzero.Trajectories`.
From each decision drawn, the network represents its observation, then unrolls ``unroll``
steps of its dynamics through the actions the player went on to take; at each step it predicts
that decision's policy, the game's outcome for the player, and the whole position as the
player would see it (`losses`). Past the end of a trajectory only the value learns, toward
the outcome, which no later step changes (MuZero's absorbing end), so that a search that looks
past the last decision finds the result there rather than noise. As in
MuZero, each unrolled step's losses weigh ``1 / unroll`` of the first's and the gradient
through each step of the dynamics is halved, so that a longer unroll doesn't swamp the
representation.

`train` closes MuZero's loop: generations of self-play (`agents.muzero.self_play`, the
network on its device judging every game's leaves at once), each added to a buffer of the
latest trajectories, then `fit` on the buffer, within a time box.

Three additions were tried on Colonies in 2026-10 and removed on 2026-11-09, none having raised
its ceiling: EfficientZero's consistency loss, MuZero Reanalyze, and Stochastic MuZero's chance
codes (worse).

This module imports Torch, so the package does not import it: import it as
``gridzero.agents.muzero_train``.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from gridzero.agents.frame import Revealable
from gridzero.agents.muzero import MuZeroPlayer, Trajectories, self_play
from gridzero.agents.muzero_net import NetModel, WorldNet
from gridzero.agents.net import unpack
from gridzero.agents.zero import Played
from gridzero.core import State, check_int, check_real
from gridzero.game import Position


@dataclass(frozen=True)
class Fit:
    """How `fit` trains.

    Parameters
    ----------
    unroll : int, default 0
        Steps of the dynamics unrolled from each decision drawn, at least 0. With 0, only the
        representation and the predictions train.
    batch_size : int, default 256
        Decisions drawn per step, at least 1.
    lr : float, default 1e-3
        AdamW's learning rate, positive.
    weight_decay : float, default 1e-4
        AdamW's weight decay, at least 0.
    value_weight, belief_weight : float, default 1.0
        Weights of the value's and the belief's losses against the policy's, at least 0.
    """

    unroll: int = 0
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    belief_weight: float = 1.0

    def __post_init__(self) -> None:
        check_int(self.unroll, "unroll", minimum=0)
        check_int(self.batch_size, "batch_size", minimum=1)
        if not check_real(self.lr, "lr") > 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        check_real(self.weight_decay, "weight_decay", 0.0)
        check_real(self.value_weight, "value_weight", 0.0)
        check_real(self.belief_weight, "belief_weight", 0.0)


class Losses(NamedTuple):
    """Mean losses over the decisions and unrolled steps that count, per step of the unroll."""

    policy: NDArray[np.float64]
    """The policy's cross-entropy against its target, shape (unroll + 1,)."""
    value: NDArray[np.float64]
    """The value's squared error against the outcome, shape (unroll + 1,)."""
    belief: NDArray[np.float64]
    """The belief's binary cross-entropy against the revealed planes, shape (unroll + 1,)."""


class Unrolled(NamedTuple):
    """Each decision's losses at each step unrolled from it, as `losses` gives them.

    Each is float32 of shape (unroll + 1, n), but ``live``, bool. The policy's and the belief's
    are 0 past a trajectory's end; the value's counts there, against the outcome.
    """

    policy: Tensor
    """The cross-entropy of the target against the policy over every action."""
    value: Tensor
    """The value's squared error against the outcome."""
    belief: Tensor
    """The belief's binary cross-entropy, the mean over its planes and cells."""
    live: Tensor
    """Whether each step lies within its trajectory."""


def losses(
    net: WorldNet, data: Trajectories, rows: NDArray[np.int64], unroll: int, device: torch.device
) -> Unrolled:
    """Return the losses of decisions and the steps unrolled from them.

    Parameters
    ----------
    net : WorldNet
        The network, on ``device``.
    data : Trajectories
        Decisions of its game.
    rows : ndarray of int64, shape (n,)
        The decisions to start from.
    unroll : int
        Steps of the dynamics after each, at least 0.
    device : torch.device
        Where the network is.

    Returns
    -------
    Unrolled
        Each decision's losses at each step. The policy's is over every action, legal or not,
        so that it learns to put nothing on illegal ones, which a search below the root relies
        on.
    """
    remaining = data.remaining()[rows]
    n_actions = net.n_verbs * net.n_cells
    obs = torch.from_numpy(data.obs[rows]).to(device).float() / 255
    latent = net.represent(obs)
    out: list[tuple[Tensor, Tensor, Tensor]] = []
    live = []
    for k in range(unroll + 1):
        at = rows + np.minimum(k, remaining)  # past the end: the last decision, masked
        if k > 0:  # the action taken before step k, or past the end the last one, repeated
            before = rows + np.minimum(k - 1, remaining)
            action = torch.from_numpy(data.action[before]).to(device)
            latent = _halved(net.dynamics(latent, action))
        pred = net.predict(latent)
        legal = unpack(torch.from_numpy(data.legal[at]).to(device), n_actions)
        log_policy = torch.log_softmax(pred.logits.flatten(1), dim=1)  # over every action
        target = torch.zeros(legal.shape, device=device)
        target[legal] = torch.from_numpy(_policies(data, at)).to(device)  # row by row
        outcome = torch.from_numpy(data.outcome[at]).to(device)
        revealed = torch.from_numpy(data.revealed[at]).to(device).float() / 255
        belief = F.binary_cross_entropy_with_logits(pred.belief, revealed, reduction="none")
        out.append(
            (
                -(target * log_policy).sum(dim=1),
                (pred.value - outcome) ** 2,
                belief.mean(dim=(1, 2)),
            )
        )
        live.append(torch.from_numpy(remaining >= k).to(device))
    mask = torch.stack(live)
    policy, value, belief = (torch.stack(x) for x in zip(*out, strict=True))
    return Unrolled(policy * mask, value, belief * mask, mask)


def weighted(unrolled: Unrolled, settings: Fit) -> Tensor:
    """Return the loss `fit` descends: each step's losses weighted, summed, over the batch.

    Parameters
    ----------
    unrolled : Unrolled
        What `losses` returns.
    settings : Fit
        The weights.

    Returns
    -------
    Tensor of float32, shape ()
        The sum over steps and decisions of ``policy + value_weight * value + belief_weight *
        belief``, the unrolled steps weighing ``1 / unroll`` of the first, divided by ``n``.
    """
    steps, n = unrolled.policy.shape
    weight = torch.ones(steps, device=unrolled.policy.device)
    weight[1:] = 1.0 / max(steps - 1, 1)
    per_step = (
        unrolled.policy
        + settings.value_weight * unrolled.value
        + settings.belief_weight * unrolled.belief
    ).sum(1)
    return per_step @ weight / n


def fit(
    net: WorldNet,
    data: Trajectories,
    settings: Fit | None = None,
    *,
    seconds: float,
    steps: int | None = None,
    seed: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
) -> Losses:
    """Train a network on trajectories, for a time box or a number of steps.

    Parameters
    ----------
    net : WorldNet
        The network, trained in place where its parameters are.
    data : Trajectories
        Decisions of its game.
    settings : Fit, optional
        How to train; `Fit` by default.
    seconds : float
        The time box: no step starts after it, positive.
    steps : int, optional
        At most this many steps, at least 1.
    seed : int, default 0
        Which decisions are drawn, non-negative.
    optimizer : torch.optim.Optimizer, optional
        The optimizer of the network's parameters, carried from an earlier call so that it
        keeps its statistics (as `train` does between generations). By default a new AdamW
        with ``settings.lr`` and ``settings.weight_decay``.

    Returns
    -------
    Losses
        The mean losses over the last 50 steps, or as many as were taken.

    Raises
    ------
    TypeError, ValueError
        If an argument is not a number in range.
    """
    settings = Fit() if settings is None else settings
    if not check_real(seconds, "seconds") > 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    limit = None if steps is None else check_int(steps, "steps", minimum=1)
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    device = next(net.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
        )
    net.train()
    recent: list[NDArray[np.float64]] = []
    end = time.perf_counter() + seconds
    taken = 0
    while time.perf_counter() < end and (limit is None or taken < limit):
        rows = rng.integers(len(data), size=settings.batch_size)
        unrolled = losses(net, data, rows, settings.unroll, device)
        total = weighted(unrolled, settings)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        each = unrolled[:-1]  # every loss, in `Losses`' order
        counts = torch.stack([unrolled.live.sum(dim=1).clamp(min=1)] * len(each))
        counts[1] = settings.batch_size  # the value counts at every step
        recent.append(
            (torch.stack([x.sum(dim=1) for x in each]) / counts).detach().cpu().double().numpy()
        )
        recent = recent[-50:]
        taken += 1
    net.eval()
    if not recent:
        return Losses(*[np.full(settings.unroll + 1, np.nan)] * len(Losses._fields))
    return Losses(*np.mean(recent, axis=0))


@dataclass(frozen=True)
class MuZero:
    """How `train` closes the loop.

    Parameters
    ----------
    games : int, default 128
        Self-play games a generation, played at once, at least 1.
    simulations : int, default 32
        Simulations a decision in self-play, at least 1.
    considered : int, default 16
        Actions sequential halving considers at the root, at least 1.
    buffer : int, default 100_000
        Decisions kept to learn from, at least 1: the latest whole trajectories that fit.
    steps : int, default 300
        Steps of `fit` a generation, at least 1.
    fit : Fit, default Fit(unroll=3)
        How each generation trains.
    """

    games: int = 128
    simulations: int = 32
    considered: int = 16
    buffer: int = 100_000
    steps: int = 300
    fit: Fit = field(default_factory=lambda: Fit(unroll=3))

    def __post_init__(self) -> None:
        for name in ("games", "simulations", "considered", "buffer", "steps"):
            check_int(getattr(self, name), name, minimum=1)
        if not isinstance(self.fit, Fit):
            raise TypeError(f"fit must be a Fit, got {type(self.fit).__name__}")


class Generation[P](NamedTuple):
    """What one generation of `train` did."""

    generation: int
    """Generations done, from 1, as `zero_train.Generation` counts them."""
    played: tuple[Played[P], ...]
    """How each self-play game went."""
    decisions: int
    """Decisions the games added."""
    buffer: int
    """Decisions in the buffer it trained on."""
    losses: Losses
    """`fit`'s mean losses over its last steps."""
    play_seconds: float
    train_seconds: float


def train[P: Position[State, object]](
    net: WorldNet,
    game: Revealable[P, Any],
    settings: MuZero | None = None,
    *,
    seconds: float,
    generations: int | None = None,
    seed: int = 0,
) -> Iterator[Generation[P]]:
    """Train a network by MuZero's loop: self-play, then learning from the latest games.

    Each generation plays ``settings.games`` games of `MuZeroPlayer` (with Gumbel noise) on the
    network against itself, from seeds ``seed + generation * games`` on, adds their
    trajectories to the buffer, drops the oldest whole trajectories past ``settings.buffer``
    decisions, and takes ``settings.steps`` steps of `fit` on it, with one optimizer for the
    whole run. No generation starts that the last one's
    duration, the caller's work between them included, says would end past the time box; the
    first always starts.

    Parameters
    ----------
    net : WorldNet
        The network, trained in place where its parameters are (a GPU, for speed).
    game : Revealable
        The rules.
    settings : MuZero, optional
        How to train; `MuZero` by default.
    seconds : float
        The time box, positive.
    generations : int, optional
        At most this many generations, at least 1.
    seed : int, default 0
        The first self-play game's seed, and the root of `fit`'s draws, non-negative.

    Yields
    ------
    Generation
        After each generation, with the network trained so far.

    Raises
    ------
    TypeError, ValueError
        If an argument is not a number in range.
    """
    settings = MuZero() if settings is None else settings
    if not check_real(seconds, "seconds") > 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    limit = None if generations is None else check_int(generations, "generations", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    end = time.perf_counter() + seconds
    buffer: Trajectories | None = None
    g = 0
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=settings.fit.lr, weight_decay=settings.fit.weight_decay
    )
    begin = time.perf_counter()
    while limit is None or g < limit:
        now = time.perf_counter()
        last = now - begin if g > 0 else 0.0  # until the next starts: the caller's work counts
        if now + last >= end:
            break
        begin = now
        net.eval()
        player = MuZeroPlayer(NetModel(net), settings.simulations, settings.considered, noise=True)
        first = seed + g * settings.games
        data, played = self_play(game, player, range(first, first + settings.games))
        buffer = data if buffer is None else Trajectories.join([buffer, data])
        sizes = np.diff(buffer.bounds)
        keep = np.cumsum(sizes[::-1])[::-1] <= settings.buffer  # whole trajectories, latest
        first = int(np.argmax(keep)) if keep.any() else buffer.n_trajectories - 1
        buffer = buffer.since(first)
        played_at = time.perf_counter()
        losses = fit(
            net,
            buffer,
            settings.fit,
            seconds=max(end - played_at, 1e-3),
            steps=settings.steps,
            seed=seed + g,
            optimizer=optimizer,
        )
        done = time.perf_counter()
        yield Generation(
            g + 1,
            tuple(played),
            len(data),
            len(buffer),
            losses,
            played_at - begin,
            done - played_at,
        )
        g += 1


def _policies(data: Trajectories, rows: NDArray[np.int64]) -> NDArray[np.float32]:
    """Return the target policies of decisions, one after another."""
    lengths = data.start[rows + 1] - data.start[rows]
    begin = np.repeat(data.start[rows] - np.cumsum(lengths) + lengths, lengths)
    return data.policy[begin + np.arange(lengths.sum())]


def _halved(x: Tensor) -> Tensor:
    """Return ``x``, with the gradient through it halved (MuZero's scaling of the dynamics)."""
    return x * 0.5 + x.detach() * 0.5
