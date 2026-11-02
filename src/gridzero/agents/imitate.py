"""Teaching a `PolicyNet` to play like another player, in any learnable game.

`collect` plays a teacher against itself and keeps each decision that had a choice as
`zero.Experience`: the observation, the legal actions, and the teacher's action as a policy
with all its weight on it, in the deciding player's frame, with how the game ended for that
player. `imitate` trains a network toward those policies and outcomes with the same losses as
AlphaZero's training (`zero_train.losses`), reporting after each pass. A network taught so can
go on to `zero_train.train`, and imitation can learn from a search's own decisions just as
well.

Written for Tactics (T7, stage 1), where a network imitating the greedy player matched 79% of
its held-out decisions.

This module imports Torch, so the package does not import it.
"""

import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from gridzero.agents.frame import Learnable
from gridzero.agents.net import PolicyNet, channels_last, outputs
from gridzero.agents.zero import Experience
from gridzero.agents.zero_train import _losses_of, losses
from gridzero.core import State
from gridzero.core._arrays import check_int, check_real
from gridzero.game import Action, Player, Position, play


def collect[G: Learnable[Any], P: Position[State, object]](
    game: G, teacher: Player[G, P], games: int, seed: int = 0, *, workers: int = 1
) -> Experience:
    """Play a teacher against itself and keep each decision with more than one legal action.

    Parameters
    ----------
    game : Learnable
        The rules.
    teacher : Player
        The player to learn from, on both sides.
    games : int
        How many games, at least 1. Game ``i`` starts from ``game.start(seed + i)``.
    seed : int, default 0
        The seed of the first game, at least 0.
    workers : int, default 1
        How many games to play at once, each in a process of its own, as in `game.series`.
        The decisions are the same, in game order.

    Returns
    -------
    Experience
        Every decision with a choice, in order of games and plies, each policy all on the
        teacher's action.

    Raises
    ------
    TypeError, ValueError
        If ``games``, ``seed`` or ``workers`` is not an integer in range.
    """
    games = check_int(games, "games", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    workers = check_int(workers, "workers", minimum=1)
    jobs = [_Collect(game, teacher, seed + i) for i in range(games)]
    if workers == 1 or games == 1:
        return Experience.join([job() for job in jobs])
    context = multiprocessing.get_context("spawn")  # as in series
    with ProcessPoolExecutor(min(workers, games), mp_context=context) as pool:
        return Experience.join(list(pool.map(_Collect.__call__, jobs)))


@dataclass(frozen=True)
class _Collect[G: Learnable[Any], P: Position[State, object]]:
    """One game of a teacher against itself, to play here or in another process."""

    game: G
    teacher: Player[G, P]
    seed: int

    def __call__(self) -> Experience:
        """Play the game and return its decisions with a choice."""
        recorder = _Recorder(self.teacher)
        record = play(self.game, (recorder, recorder), self.game.start(self.seed))
        n = len(recorder.at)
        player = np.array(recorder.players, dtype=np.int64)
        winner = record.final.winner
        outcome = np.zeros(n, dtype=np.float32)
        if winner is not None:
            outcome = np.where(player == winner, 1.0, -1.0).astype(np.float32)
        lengths = np.array(recorder.lengths, dtype=np.int64)
        start = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=start[1:])
        policy = np.zeros(start[-1], dtype=np.float32)
        policy[start[:-1] + np.array(recorder.at, dtype=np.int64)] = 1.0
        n_planes, n_cells = len(self.game.planes), self.game.space.n_cells
        return Experience(
            obs=np.array(recorder.obs, dtype=np.uint8).reshape(n, n_planes, n_cells),
            legal=np.array(recorder.legal, dtype=np.uint8).reshape(n, -1),
            policy=policy,
            start=start,
            outcome=outcome,
            game=np.full(n, self.seed, dtype=np.int64),
            ply=np.array(recorder.plies, dtype=np.int64),
        )


@dataclass(eq=False)
class _Recorder[G: Learnable[Any], P: Position[State, object]]:
    """A player that plays as its teacher and keeps each decision with a choice."""

    teacher: Player[G, P]
    obs: list[NDArray[np.uint8]] = field(default_factory=list)
    legal: list[NDArray[np.uint8]] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    at: list[int] = field(default_factory=list)
    """Where the teacher's action lies among the legal ones, in flat order."""
    players: list[int] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    def __call__(
        self,
        game: G,
        player: int,
        position: P,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return the teacher's action, keeping the decision if there was a choice."""
        action = self.teacher(game, player, position, legal, rng)
        if np.count_nonzero(legal) > 1:
            player = game.player(position)
            frame = game.frame(player)
            obs = game.observe(position, player)
            self.obs.append(np.rint(obs * 255).astype(np.uint8))
            mine = frame.into(legal).ravel()
            self.legal.append(np.packbits(mine))
            verb, cell = frame.action_into(action)
            self.lengths.append(int(np.count_nonzero(mine)))
            self.at.append(int(np.count_nonzero(mine[: verb * legal.shape[1] + cell])))
            self.players.append(player)
            self.plies.append(position.ply)
        return action


@dataclass(frozen=True)
class Epoch:
    """How training went in one pass over the decisions.

    Parameters
    ----------
    epoch : int
        Passes done, from 1.
    policy_loss, value_loss : float
        Mean losses over the training decisions in this pass (`zero_train.losses`): the
        cross-entropy of the target policy against the network's over the legal actions, and
        the squared error of the value.
    holdout_policy_loss, holdout_value_loss : float
        The same losses on the held-out decisions after the pass, NaN if none are held out.
    holdout_accuracy : float
        The share of held-out decisions where the network's best legal action is the target's
        (the teacher's action, for `collect`), NaN if none are held out.
    """

    epoch: int
    policy_loss: float
    value_loss: float
    holdout_policy_loss: float
    holdout_value_loss: float
    holdout_accuracy: float


def imitate(
    net: PolicyNet,
    experience: Experience,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    value_weight: float = 0.5,
    holdout: float = 0.1,
    seed: int = 0,
) -> Iterator[Epoch]:
    """Train a network on decisions toward their policies and outcomes.

    The loss is `zero_train.losses`'s: the cross-entropy of each decision's policy against
    the network's over the legal actions, plus ``value_weight`` times the squared error of the
    value against the outcome. For a teacher's decisions (`collect`) the policy is all on its
    action. The last ``holdout`` share of the games (by seed) is held out, so that held-out
    decisions come from games the network never saw. The network trains on the device its
    parameters are on, with AdamW, one batch at a time sent from host memory.

    Parameters
    ----------
    net : PolicyNet
        The network, trained in place.
    experience : Experience
        Decisions, from `collect` (or `zero.self_play`) with the same rules as the network's.
    epochs : int, default 10
        Passes over the training decisions, at least 1.
    batch_size : int, default 256
        Decisions per step, at least 1.
    lr, weight_decay : float, default 1e-3 and 1e-4
        AdamW's learning rate (above 0) and weight decay (at least 0).
    value_weight : float, default 0.5
        Weight of the value's loss, at least 0.
    holdout : float, default 0.1
        Share of the games held out, in [0, 1). At least one game is held out if it is above
        0 and there are two games or more.
    seed : int, default 0
        Seed of the order of decisions in each pass, at least 0.

    Yields
    ------
    Epoch
        The losses after each pass. Training runs as the epochs are taken.

    Raises
    ------
    TypeError, ValueError
        If a parameter is not a number in range, or there are no decisions to train on.
    """
    epochs = check_int(epochs, "epochs", minimum=1)
    batch_size = check_int(batch_size, "batch_size", minimum=1)
    check_real(lr, "lr", 0.0)
    if lr == 0:
        raise ValueError("lr must be above 0")
    check_real(weight_decay, "weight_decay", 0.0)
    check_real(value_weight, "value_weight", 0.0)
    check_real(holdout, "holdout", 0.0, 1.0)
    if holdout == 1:
        raise ValueError("holdout must be below 1")
    seed = check_int(seed, "seed", minimum=0)
    seeds = np.unique(experience.game)
    n_held = 0
    if holdout > 0 and seeds.size > 1:
        n_held = min(max(round(holdout * seeds.size), 1), seeds.size - 1)
    held = np.isin(experience.game, seeds[seeds.size - n_held :])
    train, test = experience.take(~held), experience.take(held)
    if len(train) == 0:
        raise ValueError("no decisions to train on")
    return _imitate(net, train, test, epochs, batch_size, lr, weight_decay, value_weight, seed)


def _imitate(
    net: PolicyNet,
    train: Experience,
    test: Experience,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    value_weight: float,
    seed: int,
) -> Iterator[Epoch]:
    """Run the passes of `imitate` once its arguments are checked."""
    device = channels_last(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(train))
        sums = np.zeros(2)
        for start in range(0, order.size, batch_size):
            index = np.sort(order[start : start + batch_size])  # sorted reads are faster
            policy, value = losses(net, train.take(index), device)
            loss = policy.mean() + value_weight * value.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += [policy.sum().item(), value.sum().item()]
        policy_loss, value_loss = sums / len(train)
        yield Epoch(
            epoch, float(policy_loss), float(value_loss), *_evaluate(net, test, batch_size, device)
        )


def _evaluate(
    net: PolicyNet, experience: Experience, batch_size: int, device: torch.device
) -> tuple[float, float, float]:
    """Return the mean policy loss, value loss and accuracy on decisions, NaN for none."""
    if len(experience) == 0:
        return (np.nan, np.nan, np.nan)
    net.eval()
    sums = np.zeros(3)
    with torch.no_grad():
        for start in range(0, len(experience), batch_size):
            batch = experience.take(np.arange(start, min(start + batch_size, len(experience))))
            scores, values, legal = outputs(net, batch.obs, batch.legal, device)
            policy, value = _losses_of(scores, values, legal, batch)  # one forward pass for both
            best = torch.from_numpy(_best(batch)).to(device)
            hits = scores.argmax(dim=1) == best
            sums += [policy.sum().item(), value.sum().item(), hits.sum().item()]
    policy, value, hits = sums / len(experience)
    return float(policy), float(value), float(hits)


def _best(batch: Experience) -> NDArray[np.int64]:
    """Return each decision's most likely action under its policy, as ``verb * n_cells + cell``.

    Ties go to the first legal action in flat order. Shape (n,).
    """
    legal = np.unpackbits(batch.legal, axis=1).astype(bool)
    flat = np.nonzero(legal)[1]  # row by row, in flat order: the order of the policy
    rows = np.repeat(np.arange(len(batch)), np.diff(batch.start))
    order = np.lexsort((-batch.policy, rows))  # each row's most likely first, ties stable
    return flat[order[batch.start[:-1]]]
