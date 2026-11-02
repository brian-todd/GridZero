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
representation. With a weight on it, EfficientZero's consistency loss also holds each unrolled
latent state close to the one the representation makes of the observation that step reaches,
the latter's gradient stopped (SimSiam, through `WorldNet.project`), so that the states a
search reaches stay like those it starts from.

A network with codes (Stochastic MuZero) unrolls each step in two: the action taken leads to
an afterstate, whose ``sigma`` learns to predict the code the encoder gives what happened (from
the player's observations at the decision and the next), and whose ``Q`` learns the outcome; the
encoder's code, one-hot, with its gradient passed straight through its softmax, leads on to the
next state. A commitment loss holds the encoder's softmax near its one-hot code (VQ-VAE), and a
diversity loss, the KL divergence of the batch's mean softmax from uniform, keeps the encoder
using its codes: without it a new encoder, whose scores barely vary with what it reads, sends
every decision to one code within a generation, its softmax saturates, and no gradient reaches
it again. The data needs nothing new: the observations are those kept.

`train` closes MuZero's loop: generations of self-play (`agents.muzero.self_play`, the
network on its device judging every game's leaves at once), each added to a buffer of the
latest trajectories, then optionally the stalest policy targets in the buffer searched again
by the network as it is now (`agents.muzero.reanalyze`, MuZero Reanalyze), then `fit` on the
buffer, within a time box.

This module imports Torch, so the package does not import it: import it as
``gridzero.agents.muzero_train``.
"""

import math
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
from gridzero.agents.muzero import MuZeroPlayer, Trajectories, reanalyze, self_play
from gridzero.agents.muzero_net import WorldNet, world_model
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
    consistency_weight : float, default 0.0
        Weight of the consistency loss against the policy's, at least 0 (EfficientZero's is 2).
    chance_weight : float, default 1.0
        Weight of the chance loss (``sigma`` against the encoder's code) against the policy's,
        at least 0; for networks with codes. Their afterstates' ``Q`` weighs ``value_weight``.
    commitment_weight : float, default 0.25
        Weight of the encoder's commitment loss, at least 0 (VQ-VAE's 0.25); for networks with
        codes.
    diversity_weight : float, default 1.0
        Weight of the encoder's diversity loss, at least 0; for networks with codes.
    """

    unroll: int = 0
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    belief_weight: float = 1.0
    consistency_weight: float = 0.0
    chance_weight: float = 1.0
    commitment_weight: float = 0.25
    diversity_weight: float = 1.0

    def __post_init__(self) -> None:
        check_int(self.unroll, "unroll", minimum=0)
        check_int(self.batch_size, "batch_size", minimum=1)
        if not check_real(self.lr, "lr") > 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        check_real(self.weight_decay, "weight_decay", 0.0)
        check_real(self.value_weight, "value_weight", 0.0)
        check_real(self.belief_weight, "belief_weight", 0.0)
        check_real(self.consistency_weight, "consistency_weight", 0.0)
        check_real(self.chance_weight, "chance_weight", 0.0)
        check_real(self.commitment_weight, "commitment_weight", 0.0)
        check_real(self.diversity_weight, "diversity_weight", 0.0)


class Losses(NamedTuple):
    """Mean losses over the decisions and unrolled steps that count, per step of the unroll."""

    policy: NDArray[np.float64]
    """The policy's cross-entropy against its target, shape (unroll + 1,)."""
    value: NDArray[np.float64]
    """The value's squared error against the outcome, shape (unroll + 1,)."""
    belief: NDArray[np.float64]
    """The belief's binary cross-entropy against the revealed planes, shape (unroll + 1,)."""
    consistency: NDArray[np.float64]
    """One minus the cosine similarity of each unrolled latent state's projection and the
    represented one's, shape (unroll + 1,): 0 at the first step, where nothing is unrolled."""
    chance: NDArray[np.float64]
    """The cross-entropy of ``sigma`` against the encoder's code of the step reached, shape
    (unroll + 1,): 0 at the first step, and without codes."""
    afterstate: NDArray[np.float64]
    """``Q``'s squared error against the outcome, by the step reached, shape (unroll + 1,): 0 at
    the first step, and without codes."""
    commitment: NDArray[np.float64]
    """The squared distance of the encoder's softmax from its one-hot code, by the step
    reached, shape (unroll + 1,): 0 at the first step, and without codes."""
    diversity: NDArray[np.float64]
    """The KL divergence from uniform of the encoder's mean softmax over a batch's decisions
    within their trajectories, by the step reached, shape (unroll + 1,): 0 at the first step,
    and without codes; from 0 (every code used alike) to ``log(codes)`` (one code)."""


class Unrolled(NamedTuple):
    """Each decision's losses at each step unrolled from it, as `losses` gives them.

    Each is float32 of shape (unroll + 1, n), but ``live``, bool. The policy's, the belief's,
    the consistency's, the chance's and the commitment's are 0 past a trajectory's end; the
    value's and the afterstate's count there, against the outcome. Those of a transition (the
    consistency's and the four of codes) go by the step it reaches, 0 at the first.
    """

    policy: Tensor
    """The cross-entropy of the target against the policy over every action."""
    value: Tensor
    """The value's squared error against the outcome."""
    belief: Tensor
    """The belief's binary cross-entropy, the mean over its planes and cells."""
    consistency: Tensor
    """``1 - cos`` of the unrolled state's projection and the represented state's."""
    chance: Tensor
    """The cross-entropy of the afterstate's ``sigma`` against the encoder's code."""
    afterstate: Tensor
    """The afterstate's ``Q``'s squared error against the outcome."""
    commitment: Tensor
    """The squared distance of the encoder's softmax from its one-hot code."""
    diversity: Tensor
    """The KL divergence from uniform of the encoder's mean softmax over the batch's live
    decisions, the same for each."""
    live: Tensor
    """Whether each step lies within its trajectory."""


def losses(
    net: WorldNet,
    data: Trajectories,
    rows: NDArray[np.int64],
    unroll: int,
    device: torch.device,
    *,
    consistency: bool = True,
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
    consistency : bool, default True
        Whether to compute the consistency loss, which costs a pass of the representation over
        every unrolled step's observation; without it, it is 0.

    Returns
    -------
    Unrolled
        Each decision's losses at each step. The policy's is over every action, legal or not,
        so that it learns to put nothing on illegal ones, which a search below the root relies
        on. The consistency's is ``1 - cos`` of the unrolled state's projection with the head
        and the represented state's without, over channels and cells, the latter computed
        without gradients. Without codes, the chance's, the afterstate's and the commitment's
        are 0. With codes, the step before step ``k`` reads the observations at ``k - 1`` and
        ``k`` (past the end, the last twice) into the encoder's code.
    """
    remaining = data.remaining()[rows]
    n_actions = net.n_verbs * net.n_cells
    obs = torch.from_numpy(data.obs[rows]).to(device).float() / 255
    latent = net.represent(obs)
    targets: tuple[Tensor, ...] = ()
    if (
        consistency and unroll > 0
    ):  # every unrolled step's target at once, from the decisions it reaches
        ahead = (rows + np.minimum(np.arange(1, unroll + 1)[:, None], remaining)).ravel()
        with torch.no_grad():
            seen = torch.from_numpy(data.obs[ahead]).to(device).float() / 255
            targets = net.project(net.represent(seen), head=False).flatten(1).split(len(rows))
    zero = torch.zeros(len(rows), device=device)
    out: list[tuple[Tensor, ...]] = []
    live = []
    for k in range(unroll + 1):
        alive = remaining >= k
        at = rows + np.minimum(k, remaining)  # past the end: the last decision, masked
        coded = (zero, zero, zero, zero)
        if k > 0:  # the action taken before step k, or past the end the last one, repeated
            before = rows + np.minimum(k - 1, remaining)
            action = torch.from_numpy(data.action[before]).to(device)
            if net.codes == 0:
                latent = _halved(net.dynamics(latent, action))
            else:
                latent, coded = _through_chance(
                    net, data, latent, action, before, at, alive, device
                )
        pred = net.predict(latent)
        legal = unpack(torch.from_numpy(data.legal[at]).to(device), n_actions)
        log_policy = torch.log_softmax(pred.logits.flatten(1), dim=1)  # over every action
        target = torch.zeros(legal.shape, device=device)
        target[legal] = torch.from_numpy(_policies(data, at)).to(device)  # row by row
        outcome = torch.from_numpy(data.outcome[at]).to(device)
        revealed = torch.from_numpy(data.revealed[at]).to(device).float() / 255
        belief = F.binary_cross_entropy_with_logits(pred.belief, revealed, reduction="none")
        if targets and k > 0:
            online = net.project(latent).flatten(1)
            held = 1 - F.cosine_similarity(online, targets[k - 1], dim=1)
        else:
            held = torch.zeros(len(rows), device=device)
        chance, afterstate, commitment, diversity = coded
        out.append(
            (
                -(target * log_policy).sum(dim=1),
                (pred.value - outcome) ** 2,
                belief.mean(dim=(1, 2)),
                held,
                chance,
                (afterstate - outcome) ** 2 if k > 0 and net.codes > 0 else zero,
                commitment,
                diversity,
            )
        )
        live.append(torch.from_numpy(alive).to(device))
    mask = torch.stack(live)
    policy, value, belief, held, chance, afterstate, commitment, diversity = (
        torch.stack(x) for x in zip(*out, strict=True)
    )
    return Unrolled(
        policy * mask,
        value,
        belief * mask,
        held * mask,
        chance * mask,
        afterstate,
        commitment * mask,
        diversity * mask,
        mask,
    )


def _through_chance(
    net: WorldNet,
    data: Trajectories,
    latent: Tensor,
    action: Tensor,
    before: NDArray[np.int64],
    at: NDArray[np.int64],
    alive: NDArray[np.bool_],
    device: torch.device,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Return the next latent states through afterstates and the encoder's codes.

    Also the chance loss, ``Q`` (whose error `losses` takes against the outcome), the
    commitment loss and the diversity loss over the decisions ``alive``, each of shape (n,).
    The encoder reads the observations at the decisions ``before`` and ``at``; its code is
    one-hot, the gradient passed straight through its softmax.
    """
    after = net.afterstate(latent, action)
    pred = net.predict_afterstate(after)
    seen = torch.from_numpy(data.obs[before]).to(device).float() / 255
    then = torch.from_numpy(data.obs[at]).to(device).float() / 255
    probs = torch.softmax(net.encode(seen, then), dim=1)
    code = probs.argmax(dim=1)
    one_hot = F.one_hot(code, net.codes).to(probs.dtype)
    chance = F.cross_entropy(pred.sigma, code, reduction="none")
    commitment = ((probs - one_hot) ** 2).sum(dim=1)
    through = one_hot + probs - probs.detach()  # the code, its gradient through the softmax
    live = torch.from_numpy(alive).to(device)
    usage = (through * live[:, None]).sum(dim=0) / live.sum().clamp(min=1)
    spread = math.log(net.codes) + (usage * torch.log(usage.clamp(min=1e-12))).sum()
    latent = _halved(net.chance(after, through))
    return latent, (chance, pred.value, commitment, spread.expand(len(at)))


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
        The sum over steps and decisions of ``policy + value_weight * (value + afterstate) +
        belief_weight * belief + consistency_weight * consistency + chance_weight * chance +
        commitment_weight * commitment + diversity_weight * diversity``, the unrolled steps
        weighing ``1 / unroll`` of the first, divided by ``n``.
    """
    steps, n = unrolled.policy.shape
    weight = torch.ones(steps, device=unrolled.policy.device)
    weight[1:] = 1.0 / max(steps - 1, 1)
    per_step = (
        unrolled.policy
        + settings.value_weight * (unrolled.value + unrolled.afterstate)
        + settings.belief_weight * unrolled.belief
        + settings.consistency_weight * unrolled.consistency
        + settings.chance_weight * unrolled.chance
        + settings.commitment_weight * unrolled.commitment
        + settings.diversity_weight * unrolled.diversity
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
        unrolled = losses(
            net, data, rows, settings.unroll, device, consistency=settings.consistency_weight > 0
        )
        total = weighted(unrolled, settings)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        each = unrolled[:-1]  # every loss, in `Losses`' order
        counts = torch.stack([unrolled.live.sum(dim=1).clamp(min=1)] * len(each))
        counts[[1, 5]] = settings.batch_size  # the value and Q count at every step
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
    reanalyze : int, default 0
        Decisions of the buffer whose policy targets are searched again each generation, after
        self-play and before `fit`, at least 0: those whose targets are oldest, among decisions
        with more than one legal action and targets older than the generation's.
    """

    games: int = 128
    simulations: int = 32
    considered: int = 16
    buffer: int = 100_000
    steps: int = 300
    fit: Fit = field(default_factory=lambda: Fit(unroll=3))
    reanalyze: int = 0

    def __post_init__(self) -> None:
        for name in ("games", "simulations", "considered", "buffer", "steps"):
            check_int(getattr(self, name), name, minimum=1)
        check_int(self.reanalyze, "reanalyze", minimum=0)
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
    reanalyzed: int = 0
    """Decisions whose policy targets were searched again."""
    reanalyze_seconds: float = 0.0


def train[P: Position[State, object]](
    net: WorldNet,
    game: Revealable[P, Any],
    settings: MuZero | None = None,
    *,
    seconds: float,
    generations: int | None = None,
    seed: int = 0,
    rival: MuZeroPlayer[Any] | None = None,
) -> Iterator[Generation[P]]:
    """Train a network by MuZero's loop: self-play, then learning from the latest games.

    Each generation plays ``settings.games`` games of `MuZeroPlayer` (with Gumbel noise) on the
    network against itself, from seeds ``seed + generation * games`` on, adds their
    trajectories to the buffer, drops the oldest whole trajectories past ``settings.buffer``
    decisions, searches again the ``settings.reanalyze`` stalest policy targets in it (with the
    self-play player, seeded ``seed + generation``), and takes ``settings.steps`` steps of `fit`
    on it, with one optimizer for the whole run. No generation starts that the last one's
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
    rival : MuZeroPlayer, optional
        A fixed player in every seat but the network's, which takes each seat in turn
        (`agents.muzero.self_play`): the network then learns to beat it, as an exploiter does.

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
    made = np.zeros(0, dtype=np.int64)  # the generation each target in the buffer comes from
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
        model = world_model(net)
        player = MuZeroPlayer(model, settings.simulations, settings.considered, noise=True)
        first = seed + g * settings.games
        data, played = self_play(game, player, range(first, first + settings.games), rival=rival)
        buffer = data if buffer is None else Trajectories.join([buffer, data])
        made = np.concatenate([made, np.full(len(data), g)])
        sizes = np.diff(buffer.bounds)
        keep = np.cumsum(sizes[::-1])[::-1] <= settings.buffer  # whole trajectories, latest
        first = int(np.argmax(keep)) if keep.any() else buffer.n_trajectories - 1
        made = made[int(buffer.bounds[first]) :]
        buffer = buffer.since(first)
        played_at = time.perf_counter()
        rows = stalest(buffer, made, g, settings.reanalyze)
        if rows.size:
            buffer = reanalyze(game, player, buffer, rows, seed=seed + g)
            made[rows] = g
        reanalyzed_at = time.perf_counter()
        losses = fit(
            net,
            buffer,
            settings.fit,
            seconds=max(end - reanalyzed_at, 1e-3),
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
            done - reanalyzed_at,
            int(rows.size),
            reanalyzed_at - played_at,
        )
        g += 1


def stalest(
    data: Trajectories, made: NDArray[np.int64], generation: int, k: int
) -> NDArray[np.int64]:
    """Return the decisions whose policy targets are oldest, to search again.

    Parameters
    ----------
    data : Trajectories
        The buffer.
    made : ndarray of int64, shape (len(data),)
        The generation each decision's target was made in.
    generation : int
        The generation now: targets made in it are not stale.
    k : int
        At most this many, at least 0.

    Returns
    -------
    ndarray of int64, shape (at most k,)
        Decisions with more than one legal action and targets made before ``generation``,
        oldest first, the earlier row first among targets of one age.
    """
    choices = np.diff(data.start)
    eligible = np.flatnonzero((choices > 1) & (made < generation))
    order = np.argsort(made[eligible], kind="stable")
    return eligible[order[:k]]


def _policies(data: Trajectories, rows: NDArray[np.int64]) -> NDArray[np.float32]:
    """Return the target policies of decisions, one after another."""
    lengths = data.start[rows + 1] - data.start[rows]
    begin = np.repeat(data.start[rows] - np.cumsum(lengths) + lengths, lengths)
    return data.policy[begin + np.arange(lengths.sum())]


def _halved(x: Tensor) -> Tensor:
    """Return ``x``, with the gradient through it halved (MuZero's scaling of the dynamics)."""
    return x * 0.5 + x.detach() * 0.5
