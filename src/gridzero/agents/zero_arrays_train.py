"""AlphaZero's loop wholly on the GPU, for a game whose rules are arrays.

`train` plays self-play games of the search in arrays (`agents.zero_arrays.ArraySearch`) on a
game's array rules (`gridzero.batched.ArrayRules`), many at once, and trains a `PolicyNet` on
what they decided, as `agents.zero_train.train` does with `agents.zero`'s search in processes on
the CPU. The rules, the searches and the network stay on the device: each step searches every
running game's decision in one batch, steps every real state, and replaces each game that ended
with a new start. A game that hides something is played whole: each decision is searched from
what its mover is shown (`ArrayRules.view`), with rules for positions that hide nothing.

Every decision with more than one legal action is kept, in a replay buffer of the latest on the
host, in its mover's frame: the observation, the legal actions, the improved policy, the
search's value of the position (`agents.zero.Experience.root`) and, once its game has ended,
the outcome. Only decisions whose game has ended are trained on: every `ArrayZero.every` steps,
`ArrayZero.steps` batches drawn at random from them, toward the improved policy and the value's
target (`ArrayZero.root_value`), by `agents.zero_train.losses`.

Use it for a game with array rules, and `agents.zero_train` for one without. On Tactics under
fog (9x9, five units a side, a network of 64 channels and 6 blocks, RTX 3080, 2026-11-16) it
played about 2,700 decisions a second at 64 simulations, 1,024 games at once.

This module imports JAX and Torch, so the package does not import it.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpy.typing import NDArray

from gridzero import batched
from gridzero.agents.net import NetArrayEvaluator, PolicyNet, channels_last
from gridzero.agents.zero import Experience
from gridzero.agents.zero_arrays import ArrayEvaluator, ArraySearch
from gridzero.agents.zero_train import _entropy, losses
from gridzero.batched import ArrayRules
from gridzero.core._arrays import check_int, check_real


@dataclass(frozen=True)
class ArrayZero:
    """The settings of AlphaZero training in arrays.

    Parameters
    ----------
    games : int, default 1024
        Self-play games played at once, at least 1: each step searches one decision of each.
    simulations, considered : int, default 64 and 16
        The search's simulations a decision and actions considered at the root
        (`agents.zero_arrays.ArraySearch`), which plays with Gumbel noise.
    buffer : int, default 400_000
        Decisions the replay buffer holds, at least 1: the latest.
    every : int, default 16
        Self-play steps between trainings, at least 1.
    steps : int, default 48
        Training steps each time, at least 0.
    batch_size : int, default 1024
        Decisions per step, drawn at random from those whose game has ended, at least 1.
    warmup : int, default 20_000
        Decisions whose game has ended that the buffer must hold before training starts, at
        least 0, so that the first batches are not all from the first few games.
    rounds : int, default 10
        Trainings per generation, at least 1: a generation is ``rounds * every`` steps.
    lr, weight_decay : float, default 1e-3 and 1e-4
        AdamW's learning rate (above 0) and weight decay (at least 0).
    value_weight : float, default 1.0
        Weight of the value's squared error beside the policy's cross-entropy, at least 0.
    root_value : float, default 0.5
        The share of the value's target taken from the search's value of the decision's
        position, the rest from the game's outcome (`agents.zero.Experience.target`), in
        [0, 1], as `agents.zero_train.Zero.root_value`.

    Raises
    ------
    TypeError, ValueError
        If a setting is not a number in its range.
    """

    games: int = 1024
    simulations: int = 64
    considered: int = 16
    buffer: int = 400_000
    every: int = 16
    steps: int = 48
    batch_size: int = 1024
    warmup: int = 20_000
    rounds: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    root_value: float = 0.5

    def __post_init__(self) -> None:
        for name in ("games", "simulations", "considered", "buffer", "every", "batch_size"):
            check_int(getattr(self, name), name, minimum=1)
        check_int(self.rounds, "rounds", minimum=1)
        check_int(self.steps, "steps", minimum=0)
        check_int(self.warmup, "warmup", minimum=0)
        if check_real(self.lr, "lr", 0.0) == 0:
            raise ValueError("lr must be above 0")
        check_real(self.weight_decay, "weight_decay", 0.0)
        check_real(self.value_weight, "value_weight", 0.0)
        check_real(self.root_value, "root_value", 0.0, 1.0)


@dataclass(frozen=True)
class ArrayGeneration:
    """What one generation of AlphaZero training in arrays did.

    Parameters
    ----------
    generation : int
        Generations done, from 1.
    decisions : int
        Decisions searched: one per game a step.
    kept : int
        Of those, the ones with more than one legal action, kept in the buffer.
    games : int
        Games that ended.
    wins : tuple of int
        Of those, the games each player won, by player.
    draws : int
        Of those, the games drawn.
    plies : float
        Mean plies of the games that ended, NaN if none did.
    buffer, ready : int
        Decisions in the buffer, and of those the ones whose game has ended.
    trained : int
        Training steps taken.
    policy_loss, value_loss : float
        Mean losses over the training steps (NaN without steps): the cross-entropy of the
        improved policy and the squared error of the value against its target.
    target_entropy : float
        The mean entropy of the improved policies trained on (the floor of the policy's
        cross-entropy), NaN without steps.
    seconds : float
        Wall time of the generation.
    next_seed : int
        The first seed no game of the run has started from, nor will (a start is made for
        each game under way and one more ahead for each, to replace it): a run continued from
        it plays only new games.
    """

    generation: int
    decisions: int
    kept: int
    games: int
    wins: tuple[int, ...]
    draws: int
    plies: float
    buffer: int
    ready: int
    trained: int
    policy_loss: float
    value_loss: float
    target_entropy: float
    seconds: float
    next_seed: int


def train[S](
    net: PolicyNet,
    rules: ArrayRules[S, Any],
    starts: Callable[[int], S],
    settings: ArrayZero | None = None,
    *,
    searched: ArrayRules[S, Any] | None = None,
    generations: int = 10,
    seed: int = 0,
    seconds: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Iterator[ArrayGeneration]:
    """Train a network by AlphaZero's loop on a game's array rules, one generation at a time.

    `ArrayZero.games` games run at once. At each step every game's mover is shown its view of
    the real state (`ArrayRules.view`), the search decides from it with the network as it
    stands (`agents.net.NetArrayEvaluator`), and the real state steps; a game that ended
    starts again from the next seed. Every `ArrayZero.every` steps, once the buffer holds
    `ArrayZero.warmup` decisions whose game has ended, the network trains on
    `ArrayZero.steps` batches of them, with AdamW kept across generations.

    Games start from seeds ``seed, seed + 1, ...`` in the order they are needed. The search
    draws its noise from `gridzero.batched.root` of ``seed``, stream 1, one stream a step;
    batches are drawn by ``numpy.random.default_rng(seed)``. On the CPU the same arguments give
    the same network; on a GPU the network runs in bfloat16.

    Parameters
    ----------
    net : PolicyNet
        The network, trained in place, on the device its parameters are on (where JAX's
        arrays lie too, so that observations pass between them without a copy).
    rules : ArrayRules
        The game's rules as arrays, which play the real states.
    starts : callable
        ``starts(seed)``: the start of game ``seed``, as the game's state of arrays (NumPy
        arrays), such as ``arrays.to_arrays(game.start(seed))``.
    settings : ArrayZero, optional
        The settings; `ArrayZero()` by default.
    searched : ArrayRules, optional
        The rules the search plays the views on, which hide nothing: rules built without the
        bookkeeping of what is hidden cost a search less (`ArrayRules.view`). ``rules`` by
        default.
    generations : int, default 10
        The most generations to run, at least 1.
    seed : int, default 0
        Seed of the first game, at least 0, and of the search's noise and the batches.
    seconds : float, optional
        A time box: no generation starts that would end after ``seconds`` from the start, as
        estimated from the last generation's length. The first always runs. None for no box.
        As in `agents.zero_train.train`, a caller's work between generations counts.
    optimizer : torch.optim.Optimizer, optional
        The optimizer of the network's parameters, to continue a run with its state; AdamW
        with the settings' rate and decay by default.

    Yields
    ------
    ArrayGeneration
        What each generation did. Training runs as the generations are taken.

    Raises
    ------
    TypeError, ValueError
        If ``generations``, ``seed`` or ``seconds`` is not a number in range, or two legal
        actions of a decision are scored at one place of the network's output.
    """
    settings = ArrayZero() if settings is None else settings
    generations = check_int(generations, "generations", minimum=1)
    seed = check_int(seed, "seed", minimum=0)
    if seconds is not None:
        check_real(seconds, "seconds", 0.0)
    return _train(net, rules, starts, settings, searched, generations, seed, seconds, optimizer)


def _train[S](
    net: PolicyNet,
    rules: ArrayRules[S, Any],
    starts: Callable[[int], S],
    settings: ArrayZero,
    searched: ArrayRules[S, Any] | None,
    generations: int,
    seed: int,
    seconds: float | None,
    optimizer: torch.optim.Optimizer | None,
) -> Iterator[ArrayGeneration]:
    """Run the generations of `train` once its arguments are checked."""
    device = channels_last(net)
    adam = optimizer
    if adam is None:
        adam = torch.optim.AdamW(
            net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
        )
    search = ArraySearch(
        rules if searched is None else searched,
        simulations=settings.simulations,
        considered=settings.considered,
        noise=True,
    )
    evaluate = NetArrayEvaluator(net)
    games = _Games(rules, starts, settings.games, seed)
    buffer = _Buffer(settings.buffer)
    rng = np.random.default_rng(seed)
    key = batched.stream(batched.root(seed), 1)
    step = 0
    begin = time.perf_counter()
    last = 0.0
    for g in range(generations):
        if g > 0 and not (seconds is None or time.perf_counter() - begin + last <= seconds):
            return
        start = time.perf_counter()
        tally = _Tally()
        sums = np.zeros(3)
        trained = 0
        for _ in range(settings.rounds):
            for _ in range(settings.every):
                decided = games.decide(search, evaluate, batched.stream(key, step))
                step += 1
                buffer.add(decided)
                ended = games.play(decided.action)
                buffer.end(ended)
                tally.add(decided, ended)
            ready = buffer.ready()
            if ready.size < max(settings.warmup, 1):
                continue
            net.train()
            for _ in range(settings.steps):
                rows = ready[np.sort(rng.integers(ready.size, size=settings.batch_size))]
                batch = buffer.experience(rows, net.n_verbs)
                policy, value = losses(net, batch, device, root_value=settings.root_value)
                loss = policy.mean() + settings.value_weight * value.mean()
                adam.zero_grad(set_to_none=True)
                loss.backward()
                adam.step()
                sums += [policy.mean().item(), value.mean().item(), _entropy(batch)]
                trained += 1
            net.eval()
        means = sums / trained if trained else np.full(3, np.nan)
        last = time.perf_counter() - start
        yield ArrayGeneration(
            generation=g + 1,
            decisions=tally.decisions,
            kept=tally.kept,
            games=tally.games,
            wins=tuple(tally.wins),
            draws=tally.draws,
            plies=tally.plies / tally.games if tally.games else np.nan,
            buffer=buffer.size,
            ready=int(buffer.ready().size),
            trained=trained,
            policy_loss=float(means[0]),
            value_loss=float(means[1]),
            target_entropy=float(means[2]),
            seconds=last,
            next_seed=games.next_seed,
        )


@dataclass(frozen=True)
class _Decided:
    """One step's decisions, one per game, on the host: what the buffer keeps of them."""

    action: jax.Array
    """The local action each search chose, on the device: int32, shape (games,)."""
    obs: NDArray[np.uint8]
    """The movers' observations: shape (games, n_planes, n_cells)."""
    legal: NDArray[np.bool_]
    """Legal local actions: shape (games, n_actions)."""
    framed: NDArray[np.int32]
    """Where the network scores each local action: shape (games, n_actions)."""
    policy: NDArray[np.float32]
    """The improved policies over local actions: shape (games, n_actions)."""
    root: NDArray[np.float32]
    """The search's value of each position for its mover, in [-1, 1]: shape (games,)."""
    mover: NDArray[np.int32]
    """Each game's mover: shape (games,)."""
    seed: NDArray[np.int64]
    """Each game's seed: shape (games,)."""
    ply: NDArray[np.int64]
    """Each decision's ply in its game: shape (games,)."""


@dataclass(frozen=True)
class _Ended:
    """The games that ended at one step."""

    slots: NDArray[np.int64]
    """Which games: shape (k,)."""
    winner: NDArray[np.int32]
    """Who won each, -1 for a draw: shape (k,)."""
    plies: NDArray[np.int64]
    """How long each lasted: shape (k,)."""


class _Games[S]:
    """The games under way, on the device, and the starts that replace those that end."""

    def __init__(
        self, rules: ArrayRules[S, Any], starts: Callable[[int], S], n: int, seed: int
    ) -> None:
        self.starts = starts
        self.n = n
        self.next_seed = seed
        self.seed = np.zeros(n, dtype=np.int64)
        self.ply = np.zeros(n, dtype=np.int64)
        first, self.seed[:] = self._made(n)
        self.states = jax.device_put(first)
        self.fresh, self.fresh_seed = self._made(n)

        def shown(states: S) -> tuple[Any, ...]:
            movers = jax.vmap(rules.player)(states)
            views = jax.vmap(rules.view)(states, movers)
            obs = jax.vmap(rules.observe)(views, movers)
            obs = jnp.rint(obs * 255).astype(jnp.uint8)
            return views, movers, obs, jax.vmap(rules.legal)(views), jax.vmap(rules.framed)(views)

        def played(states: S, action: jax.Array, fresh: S) -> tuple[S, jax.Array, jax.Array]:
            after = jax.vmap(rules.step)(states, action)
            over = jax.vmap(rules.over)(after)
            winner = jax.vmap(rules.winner)(after)
            nth = jnp.cumsum(over) - 1  # the k-th game to end takes the k-th fresh start
            taken = jax.tree.map(lambda f: f[jnp.maximum(nth, 0)], fresh)
            refilled = jax.tree.map(
                lambda a, b: jnp.where(over.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
                taken,
                after,
            )
            return refilled, over, winner

        self._shown = jax.jit(shown)
        self._played = jax.jit(played)

    def _made(self, k: int) -> tuple[Any, NDArray[np.int64]]:
        """Return the next ``k`` starts, stacked, and their seeds."""
        seeds = np.arange(self.next_seed, self.next_seed + k, dtype=np.int64)
        self.next_seed += k
        made = [self.starts(int(s)) for s in seeds]
        return jax.tree.map(lambda *xs: np.stack(xs), *made), seeds

    def decide(self, search: ArraySearch[S], evaluate: ArrayEvaluator, key: jax.Array) -> _Decided:
        """Search every game's decision from what its mover is shown."""
        views, movers, obs, legal, framed = self._shown(self.states)
        found = search.search(views, evaluate, key)
        mover = np.asarray(movers)
        value = np.asarray(found.value, dtype=np.float32)
        return _Decided(
            action=found.action,
            obs=np.asarray(obs),
            legal=np.asarray(legal),
            framed=np.asarray(framed),
            policy=np.asarray(found.policy, dtype=np.float32),
            root=np.where(mover == 0, 1, -1).astype(np.float32) * (2 * value - 1),
            mover=mover.astype(np.int32),
            seed=self.seed.copy(),
            ply=self.ply.copy(),
        )

    def play(self, action: jax.Array) -> _Ended:
        """Step every game, start new ones where games ended, and return those that did."""
        self.states, over, winner = self._played(self.states, action, self.fresh)
        self.ply += 1
        ended = np.flatnonzero(np.asarray(over))
        result = _Ended(ended, np.asarray(winner)[ended].astype(np.int32), self.ply[ended].copy())
        k = ended.size
        if k:
            self.seed[ended] = self.fresh_seed[:k]
            self.ply[ended] = 0
            more, seeds = self._made(k)
            self.fresh = jax.tree.map(lambda f, m: np.concatenate([f[k:], m]), self.fresh, more)
            self.fresh_seed = np.concatenate([self.fresh_seed[k:], seeds])
        return result


class _Buffer:
    """The replay buffer: the latest decisions kept, on the host, filled as a ring."""

    def __init__(self, size: int) -> None:
        self.capacity = size
        self.size = 0
        self.head = 0
        self.arrays: dict[str, NDArray[Any]] = {}
        self.written = np.zeros(size, dtype=np.int64)  # writes to each row, to know it again
        self.pending: dict[int, list[tuple[int, int, int]]] = {}  # game slot: (row, write, mover)

    def add(self, decided: _Decided) -> None:
        """Keep each decision with more than one legal action; its outcome comes later."""
        slots = np.flatnonzero(decided.legal.sum(axis=1) > 1)
        if not self.arrays:
            n = self.capacity
            shapes = {
                "obs": decided.obs,
                "legal": decided.legal,
                "framed": decided.framed,
                "policy": decided.policy,
            }
            self.arrays = {k: np.zeros((n, *v.shape[1:]), v.dtype) for k, v in shapes.items()}
            self.arrays |= {
                "root": np.zeros(n, np.float32),
                "outcome": np.zeros(n, np.float32),
                "done": np.zeros(n, bool),
                "game": np.zeros(n, np.int64),
                "ply": np.zeros(n, np.int64),
            }
        rows = (self.head + np.arange(slots.size)) % self.capacity
        a = self.arrays
        for name in ("obs", "legal", "framed", "policy", "root"):
            a[name][rows] = getattr(decided, name)[slots]
        a["game"][rows], a["ply"][rows] = decided.seed[slots], decided.ply[slots]
        a["outcome"][rows], a["done"][rows] = 0.0, False
        self.written[rows] += 1
        for slot, row in zip(slots.tolist(), rows.tolist(), strict=True):
            entry = (row, int(self.written[row]), int(decided.mover[slot]))
            self.pending.setdefault(slot, []).append(entry)
        self.head = int((self.head + slots.size) % self.capacity)
        self.size = min(self.capacity, self.size + int(slots.size))

    def end(self, ended: _Ended) -> None:
        """Give the decisions of each game that ended its outcome, for their movers."""
        a = self.arrays
        for slot, winner in zip(ended.slots.tolist(), ended.winner.tolist(), strict=True):
            for row, write, mover in self.pending.pop(slot, []):
                if self.written[row] != write:  # overwritten while its game went on
                    continue
                a["outcome"][row] = 0.0 if winner < 0 else (1.0 if mover == winner else -1.0)
                a["done"][row] = True

    def ready(self) -> NDArray[np.int64]:
        """Return the rows whose game has ended: shape (k,)."""
        if not self.arrays:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.arrays["done"][: self.size])

    def experience(self, rows: NDArray[np.int64], n_verbs: int) -> Experience:
        """Return decisions as `agents.zero.Experience`: actions where the network scores them."""
        a = self.arrays
        obs, legal, framed = a["obs"][rows], a["legal"][rows], a["framed"][rows]
        n_moves = n_verbs * obs.shape[-1]
        r, local = np.nonzero(legal)
        at = framed[r, local].astype(np.int64)
        board = np.zeros((rows.size, n_moves), dtype=bool)
        board[r, at] = True
        lengths = board.sum(axis=1)
        if not np.array_equal(lengths, legal.sum(axis=1)):
            raise ValueError("two legal actions of a decision are scored at one place")
        dense = np.zeros((rows.size, n_moves), dtype=np.float32)
        dense[r, at] = a["policy"][rows][r, local]
        start = np.zeros(rows.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=start[1:])
        return Experience(
            obs,
            np.packbits(board, axis=1),
            dense[board],
            start,
            a["outcome"][rows],
            a["root"][rows],
            a["game"][rows],
            a["ply"][rows],
        )


class _Tally:
    """What a generation's self-play did."""

    def __init__(self) -> None:
        self.decisions = self.kept = self.games = self.draws = self.plies = 0
        self.wins = [0, 0]  # by player: the search is for two

    def add(self, decided: _Decided, ended: _Ended) -> None:
        """Count a step's decisions and the games it ended."""
        self.decisions += decided.legal.shape[0]
        self.kept += int((decided.legal.sum(axis=1) > 1).sum())
        self.games += ended.slots.size
        self.plies += int(ended.plies.sum())
        for w in ended.winner.tolist():
            if w < 0:
                self.draws += 1
            else:
                self.wins[w] += 1
