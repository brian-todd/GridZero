"""An AlphaZero player for any game of two players: Gumbel tree search, judged by an evaluator.

`AlphaZeroPlayer` searches each decision of a `game.Game` whose view is its position (perfect
information, two players, a result from ``Position.winner``) with the search of Gumbel
AlphaZero (Danihelka et al., "Policy improvement by planning with Gumbel", ICLR 2022), as
DeepMind's ``mctx`` implements it. An `Evaluator`, such as a network
(`agents.net.NetEvaluator`), gives the prior over the legal actions of each new position and
its value. At the root, the search draws Gumbel noise (when ``noise`` is set), considers the
best ``considered`` actions by noise plus prior, and splits the simulations between them by
sequential halving; below the root, it visits actions to bring their visits toward the policy
improved by their values. The result carries that improved policy, the target a network learns
from.

The tree holds exact positions. A step that drew nothing from its stream (``position.drew``)
gives the same position under any seed, so its child is kept. A step that drew is a chance
edge: each visit steps again with a fresh seed drawn from the player's generator, and outcomes
that leave the same position (``Position.key``) share a child, so their visits sample the
chance. The search never reads the game's seed, which fixes every future draw. Positions below
the root with one legal action are passed through without asking the evaluator.

The search runs one leaf at a time (`_Search.leaf` and `_Search.expand`), so that many searches
can share an evaluator's batches: `self_play` plays many games at once that way, one batch of
leaves per call of the evaluator, and keeps decisions to learn from as `Experience`, in each
deciding player's frame (`agents.frame.Learnable`). The search was first written for Tactics
(`examples.games.tactics`).
"""

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from gridzero.agents.frame import Learnable
from gridzero.core import State
from gridzero.core._arrays import check_int, check_real
from gridzero.game import Action, Game, Position


class Evaluator[G, P](Protocol):
    """Judges positions of game ``G`` for the search: a prior over legal actions, and who wins."""

    def __call__(
        self, game: G, positions: Sequence[P], legal: Sequence[NDArray[np.bool_]], /
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the prior's logits and the value of each position.

        Parameters
        ----------
        game : G
            The rules.
        positions : sequence of P
            Positions not over.
        legal : sequence of ndarray of bool, shape (n_verbs, n_cells)
            ``game.legal(position)`` for each position.

        Returns
        -------
        logits : list of ndarray of float64, shape (n_legal,)
            For each position, finite logits of the acting player's prior over its legal
            actions, in the order of ``np.flatnonzero(legal)`` (action ``verb * n_cells +
            cell``). Only their differences matter.
        values : ndarray of float64, shape (n_positions,)
            Player 0's value of each position, in [0, 1]: 1 won, 0 lost.
        """
        ...


class SearchResult(NamedTuple):
    """What a search of one decision found."""

    action: Action
    """The action chosen."""
    policy: NDArray[np.float64]
    """The improved policy, shape (n_verbs, n_cells): the softmax of the prior's logits plus
    the scaled completed values, 0 where illegal, summing to 1. The target to learn from."""
    visits: NDArray[np.int64]
    """Simulations through each action at the root, shape (n_verbs, n_cells)."""
    q: NDArray[np.float64]
    """Player 0's mean value of each action at the root, shape (n_verbs, n_cells); NaN where
    unvisited or illegal."""
    value: float
    """Player 0's value of the root: its evaluation and every simulation's, averaged."""


@dataclass(frozen=True)
class AlphaZeroPlayer[G: Game[Any, Any], P: Position[State, object]]:
    """Picks each decision by Gumbel AlphaZero tree search, judged by an evaluator.

    The root is evaluated, then each of ``simulations`` simulations walks down the tree:
    at the root, among the actions sequential halving still considers, the one with the
    highest Gumbel noise plus logit plus scaled completed value (``sigma``); below it, the one
    whose share of the visits falls furthest short of the improved policy. A simulation stops
    at a position not yet evaluated, which the evaluator judges, or at the end of the game,
    scored by its result (1, 0, or 0.5 for a draw), and adds the value to every action on its
    way. The action played is, among those the last round of sequential halving still
    considered, the one with the highest noise plus logit plus ``sigma``. After a whole round
    they are the actions visited most, as in ``mctx``; when the simulations end inside a round,
    the ones it had yet to visit count too, so that an action is never played for an extra
    visit alone.

    ``sigma`` follows ``mctx``'s completed values by mixed value: an action's mean value for
    the acting player where visited, elsewhere a mix of the node's own value and the visited
    actions' means weighted by the prior; multiplied by ``(c_visit + most visits) * c_scale``.
    At the root they are first rescaled to [0, 1] over the node's actions, as ``mctx`` does, for
    them to weigh against the noise. Below it they aren't, unlike ``mctx`` (see `sigma`).

    ``G`` is the game's own class and ``P`` its position. ``G`` is bounded by ``Game[Any,
    Any]`` only: Python can't bound a type parameter by another (``Game[P, P]``).

    Parameters
    ----------
    evaluator : Evaluator
        Gives priors and values.
    simulations : int, default 32
        Simulations per decision, at least 1.
    considered : int, default 16
        Actions sequential halving considers at the root, at least 1.
    noise : bool, default False
        Draw Gumbel noise at the root, which samples the actions considered and played: on for
        self-play, off to play for strength (the best by prior and value).
    c_visit : float, default 50.0
        Visits added to the most visits in ``sigma``'s scale, at least 0.
    c_scale : float, default 0.1
        ``sigma``'s scale per visit, at least 0.

    Raises
    ------
    TypeError, ValueError
        If a parameter has the wrong type or is out of range.
    """

    evaluator: Evaluator[G, P]
    simulations: int = 32
    considered: int = 16
    noise: bool = False
    c_visit: float = 50.0
    c_scale: float = 0.1

    def __post_init__(self) -> None:
        check_int(self.simulations, "simulations", minimum=1)
        check_int(self.considered, "considered", minimum=1)
        if not isinstance(self.noise, bool):
            raise TypeError(f"noise must be a bool, got {type(self.noise).__name__}")
        check_real(self.c_visit, "c_visit", 0.0)
        check_real(self.c_scale, "c_scale", 0.0)
        if not callable(self.evaluator):
            raise TypeError(f"evaluator must be callable, got {type(self.evaluator).__name__}")

    def __call__(
        self,
        game: G,
        player: int,
        position: P,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return the action the search prefers, or the only legal one (see `Player`)."""
        if np.count_nonzero(legal) == 1:
            verb, cell = np.argwhere(legal)[0]
            return Action(int(verb), int(cell))
        return self._search(game, position, rng, legal).action  # legal: no second game.legal

    def search(self, game: G, position: P, rng: np.random.Generator) -> SearchResult:
        """Search the decision at ``position``.

        Parameters
        ----------
        game : G
            The rules: a game of two players whose view is its position.
        position : P
            A position not over.
        rng : numpy.random.Generator
            The source of the noise, every seed and every tie the search draws.

        Returns
        -------
        SearchResult
            The action, the improved policy, the visits and values at the root.

        Raises
        ------
        ValueError
            If the game is over or doesn't have two players, or the evaluator gives logits of
            the wrong length, logits not finite, or a value outside [0, 1].
        """
        return self._search(game, position, rng, None)

    def _search(
        self,
        game: G,
        position: P,
        rng: np.random.Generator,
        legal: NDArray[np.bool_] | None,
    ) -> SearchResult:
        """Run `search`, given the root's legal actions, shape (n_verbs, n_cells), or None."""
        search = _Search(self, game, position, rng, legal)
        while (leaf := search.leaf()) is not None:
            logits, values = self.evaluator(game, [leaf.node.position], [leaf.legal])
            search.expand(leaf, logits[0], float(values[0]))
        return search.result()


@dataclass(frozen=True, eq=False)
class Experience:
    """Decisions of self-play games kept to learn from, each in its deciding player's frame.

    Parameters
    ----------
    obs : ndarray of uint8, shape (n, n_planes, n_cells)
        Observations, ``round(255 * game.observe(...))``.
    legal : ndarray of uint8, shape (n, ceil(n_verbs * n_cells / 8))
        The legal actions, flattened over verbs and cells and packed by `numpy.packbits`.
    policy : ndarray of float32, shape (m,)
        The search's improved policy over each decision's legal actions, in the order of
        their flat index: decision ``i``'s are ``policy[start[i]:start[i + 1]]``.
    start : ndarray of int64, shape (n + 1,)
        Where each decision's policy starts in ``policy``, and its end.
    outcome : ndarray of float32, shape (n,)
        1 if the deciding player went on to win, -1 if it lost, 0 for a draw.
    game : ndarray of int64, shape (n,)
        The seed of the game each decision was taken in.
    ply : ndarray of int64, shape (n,)
        The ply of each decision, to tell the stages of a game apart.
    """

    obs: NDArray[np.uint8]
    legal: NDArray[np.uint8]
    policy: NDArray[np.float32]
    start: NDArray[np.int64]
    outcome: NDArray[np.float32]
    game: NDArray[np.int64]
    ply: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.outcome.size)

    def take(self, index: NDArray[np.integer] | NDArray[np.bool_]) -> Experience:
        """Return the decisions at ``index``, an array of positions or a boolean mask.

        Parameters
        ----------
        index : ndarray of int, shape (k,), or ndarray of bool, shape (n,)
            Which decisions, in the order wanted (repeats allowed), or a mask over all ``n``.

        Returns
        -------
        Experience
            Those decisions, each with its own policy, ``start`` counted afresh from 0.
        """
        index = np.flatnonzero(index) if index.dtype == np.bool_ else np.asarray(index)
        lengths = np.diff(self.start)[index]
        start = np.zeros(index.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=start[1:])
        shift = np.repeat(self.start[:-1][index] - start[:-1], lengths)
        return Experience(
            self.obs[index],
            self.legal[index],
            self.policy[shift + np.arange(start[-1])],
            start,
            self.outcome[index],
            self.game[index],
            self.ply[index],
        )

    @staticmethod
    def join(parts: Sequence[Experience]) -> Experience:
        """Return decisions one after another, in order.

        Parameters
        ----------
        parts : sequence of Experience
            At least one, of the same game (the same planes, verbs and cells).

        Returns
        -------
        Experience
            Every part's decisions, the first part's first, ``start`` running on across them.
        """
        sizes = np.cumsum([0] + [p.policy.size for p in parts[:-1]])
        start = [parts[0].start[:1]] + [p.start[1:] + s for p, s in zip(parts, sizes, strict=True)]
        return Experience(
            np.concatenate([p.obs for p in parts]),
            np.concatenate([p.legal for p in parts]),
            np.concatenate([p.policy for p in parts]),
            np.concatenate(start),
            np.concatenate([p.outcome for p in parts]),
            np.concatenate([p.game for p in parts]),
            np.concatenate([p.ply for p in parts]),
        )


class Played[P](NamedTuple):
    """How one self-play game went."""

    seed: int
    winner: int | None
    """The player that won, or None for a draw."""
    plies: int
    searched: int
    """Decisions searched: those with more than one legal action."""
    kept: int
    """Decisions kept as experience."""
    final: P
    """The last position, for what is the game's own (such as Tactics' rounds)."""


def self_play[G: Learnable[Any], P: Position[State, object]](
    game: G, player: AlphaZeroPlayer[G, P], seeds: Sequence[int], *, keep: float = 0.25
) -> tuple[Experience, list[Played[P]]]:
    """Play games of a player against itself, all at once, and keep decisions to learn from.

    Each game searches its decisions in turn, as ``player.search`` does; the games' searches
    take a leaf each, and the player's evaluator judges those leaves in one call, so a batched
    evaluator (`net.NetEvaluator` on a GPU) sees as many positions per call as games still
    running. Each searched decision is kept with probability ``keep``: the observation, the
    legal actions and the improved policy in the deciding player's frame (`Learnable.frame`),
    and later how the game ended for that player. Single legal actions are played without a
    search and never kept.

    Game ``seed`` starts from ``game.start(seed)`` and searches with its own generator (a
    stream of the seed), so a game plays the same whatever other games run beside it, given an
    evaluator whose results do not depend on the batch.

    Parameters
    ----------
    game : Learnable
        The rules: a game of two players that a learner can play.
    player : AlphaZeroPlayer
        The player on both sides, with its evaluator; set ``noise`` for exploration.
    seeds : sequence of int
        One game per seed, each at least 0.
    keep : float, default 0.25
        The share of searched decisions kept, in [0, 1].

    Returns
    -------
    experience : Experience
        The decisions kept, in order of the seeds given and of plies.
    played : list of Played
        How each game went, in order of the seeds given.

    Raises
    ------
    TypeError, ValueError
        If a seed or ``keep`` is not a number in range, or the evaluator gives bad results.
    """
    keep = check_real(keep, "keep", 0.0, 1.0)
    running: list[_Running[P]] = [
        _Running(s, game.start(s), np.random.default_rng(np.random.SeedSequence(s, spawn_key=(4,))))
        for s in (check_int(seed, "seed", minimum=0) for seed in seeds)
    ]
    games = list(running)
    while running:
        leaves = [(g, g.advance(game, player, keep)) for g in running]
        waiting = [(g, leaf) for g, leaf in leaves if leaf is not None]
        running = [g for g, _ in waiting]
        if waiting:
            logits, values = player.evaluator(
                game,
                [leaf.node.position for _, leaf in waiting],
                [leaf.legal for _, leaf in waiting],
            )
            for (g, leaf), row, value in zip(waiting, logits, values, strict=True):
                assert g.search is not None
                g.search.expand(leaf, row, float(value))
    parts = [g.experience(game) for g in games]
    return Experience.join(parts) if parts else _empty(game), [g.played() for g in games]


def _empty(game: Learnable[Any]) -> Experience:
    """Return experience with no decisions, shaped for ``game``."""
    n_cells = game.space.n_cells
    n_planes = len(game.planes)
    return Experience(
        np.zeros((0, n_planes, n_cells), dtype=np.uint8),
        np.zeros((0, -(-game.n_verbs * n_cells // 8)), dtype=np.uint8),
        np.zeros(0, dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )


@dataclass(eq=False)
class _Running[P: Position[State, object]]:
    """A self-play game under way: its position, its search, and the decisions kept."""

    seed: int
    position: P
    rng: np.random.Generator
    search: _Search[Any, P] | None = None
    searched: int = 0
    obs: list[NDArray[np.uint8]] = field(default_factory=list)
    legal: list[NDArray[np.uint8]] = field(default_factory=list)
    policy: list[NDArray[np.float32]] = field(default_factory=list)
    movers: list[int] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    def advance(
        self, game: Learnable[Any], player: AlphaZeroPlayer[Any, P], keep: float
    ) -> _Leaf[P] | None:
        """Play on until a search needs a leaf judged; return it, or None once the game ends."""
        while True:
            if self.search is not None:
                leaf = self.search.leaf()
                if leaf is not None:
                    return leaf
                result = self.search.result()
                if self.rng.random() < keep:
                    self._keep(game, self.search.legal, result.policy)
                self.search = None
                self.position = game.step(self.position, result.action)
            elif self.position.over:
                return None
            else:
                legal = game.legal(self.position)
                if np.count_nonzero(legal) == 1:
                    verb, cell = np.argwhere(legal)[0]
                    self.position = game.step(self.position, (int(verb), int(cell)))
                else:
                    self.search = _Search(player, game, self.position, self.rng, legal)
                    self.searched += 1

    def _keep(
        self, game: Learnable[Any], legal: NDArray[np.bool_], policy: NDArray[np.float64]
    ) -> None:
        """Keep the decision at the current position, in the deciding player's frame."""
        mover = game.player(self.position)
        own = game.frame(mover).into(legal)
        self.obs.append(np.rint(game.observe(self.position, mover) * 255).astype(np.uint8))
        self.legal.append(np.packbits(own.ravel()))
        self.policy.append(game.frame(mover).into(policy)[own].astype(np.float32))
        self.movers.append(mover)
        self.plies.append(self.position.ply)

    def experience(self, game: Learnable[Any]) -> Experience:
        """Return the decisions kept, with how the game ended for each deciding player."""
        if not self.movers:
            return _empty(game)
        mover = np.array(self.movers)
        winner = self.position.winner
        outcome = np.zeros(mover.size, dtype=np.float32)
        if winner is not None:
            outcome = np.where(mover == winner, 1.0, -1.0).astype(np.float32)
        start = np.zeros(mover.size + 1, dtype=np.int64)
        np.cumsum([p.size for p in self.policy], out=start[1:])
        return Experience(
            np.stack(self.obs),
            np.stack(self.legal),
            np.concatenate(self.policy),
            start,
            outcome,
            np.full(mover.size, self.seed, dtype=np.int64),
            np.array(self.plies, dtype=np.int64),
        )

    def played(self) -> Played[P]:
        """Return how the game went."""
        p = self.position
        return Played(self.seed, p.winner, p.ply, self.searched, len(self.movers), p)


@functools.cache
def considered_visits(considered: int, simulations: int) -> tuple[int, ...]:
    """Return the visits an action must have to be chosen at each simulation of the root.

    Sequential halving as ``mctx`` schedules it: rounds visit each considered action once,
    repeated so that the simulations spread over about ``log2(considered)`` phases, and each
    phase keeps the better half (at least 2). At simulation ``i``, the root chooses among the
    actions with exactly ``visits[i]`` visits.

    Parameters
    ----------
    considered : int
        Actions considered at first, at least 1.
    simulations : int
        Simulations, at least 0.

    Returns
    -------
    tuple of int
        ``simulations`` visit counts, one per simulation.
    """
    return _rounds(considered, simulations)[:simulations]


@functools.cache
def _rounds(considered: int, simulations: int) -> tuple[int, ...]:
    """Return `considered_visits` up to the end of the round its last simulation is in."""
    if considered <= 1:
        return tuple(range(simulations))
    log2max = math.ceil(math.log2(considered))
    sequence: list[int] = []
    visits = [0] * considered
    alive = considered
    while len(sequence) < simulations:
        extra = max(1, int(simulations / (log2max * alive)))
        for _ in range(extra):
            sequence.extend(visits[:alive])
            for i in range(alive):
                visits[i] += 1
        alive = max(2, alive // 2)
    return tuple(sequence)


def finalists(n: NDArray[np.int64], considered: int, simulations: int) -> NDArray[np.bool_]:
    """Return which root actions the last round of sequential halving still considered.

    After a whole round, those it visited, which have the most visits. When the simulations end
    inside a round, those with the visits it chose among are finalists too: the score ranks the
    actions it had yet to visit as it ranked the ones it visited, so the best of them is among
    those the round would have kept.

    Parameters
    ----------
    n : ndarray of int64, shape (n_actions,)
        Visits of each root action after every simulation, at least 1 simulation.
    considered : int
        Actions sequential halving considered at first, at least 1.
    simulations : int
        Simulations, every one through the root, at least 1.

    Returns
    -------
    ndarray of bool, shape (n_actions,)
        The finalists.
    """
    considered = min(considered, n.size)
    rounds = _rounds(considered, simulations)
    last = rounds[simulations - 1]
    cut_short = len(rounds) > simulations and rounds[simulations] == last
    return n >= last if cut_short else n > last


def sigma(
    q: NDArray[np.float64],
    raw: float,
    n: NDArray[np.int64],
    logits: NDArray[np.float64],
    c_visit: float,
    c_scale: float,
    *,
    rescale: bool,
) -> NDArray[np.float64]:
    """Return the scaled completed values of a node's actions, as ``mctx`` completes them.

    An action's mean value where visited; elsewhere the node's own value mixed with the visited
    actions' means weighted by the prior; if asked, rescaled to [0, 1] over the node's actions;
    then multiplied by ``(c_visit + most visits) * c_scale``.

    ``mctx`` rescales at every node. Below the root that locks a node onto the first action it
    tries, if that looks better than the node's own value: the mix tends to the action's mean,
    so the gap between them vanishes, and rescaling stretches any gap to the whole range. The
    action keeps the whole bonus and the others none, so they are never tried, however many
    simulations follow. With a fit on Connect Four at 512 simulations, 68% of the nodes below
    the root with 20 visits or more had tried one action only, and 1% without rescaling there.

    Parameters
    ----------
    q : ndarray of float64, shape (n_actions,)
        Each action's mean value for the acting player, any value where unvisited.
    raw : float
        The node's own value for the acting player.
    n : ndarray of int64, shape (n_actions,)
        Visits of each action.
    logits : ndarray of float64, shape (n_actions,)
        The prior's log-probabilities.
    c_visit, c_scale : float
        The scale's constants.
    rescale : bool
        Rescale to [0, 1] over the node's actions: at the root, not below it.

    Returns
    -------
    ndarray of float64, shape (n_actions,)
        Scaled completed values.
    """
    seen = n > 0
    if seen.any():
        prior = np.maximum(np.exp(logits), np.finfo(np.float64).tiny)
        weighted = float((prior[seen] * q[seen]).sum() / prior[seen].sum())
        total = int(n.sum())
        raw = (raw + total * weighted) / (total + 1)
    completed = np.where(seen, q, raw)
    if rescale:
        low, high = completed.min(), completed.max()
        completed = (completed - low) / max(high - low, 1e-8)
    return (c_visit + int(n.max())) * c_scale * completed


def _result(position: Position[State, object]) -> float:
    """Return player 0's value of a finished game: 1 won, 0 lost, 0.5 drawn."""
    return 0.5 if position.winner is None else float(position.winner == 0)


def _argmax(x: NDArray[np.float64], rng: np.random.Generator) -> int:
    """Return the index of the largest value, ties broken at random."""
    best = np.flatnonzero(x == x.max())
    return int(best[0]) if best.size == 1 else int(best[rng.integers(best.size)])


def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the softmax of finite values, shape of ``x``."""
    z = np.exp(x - x.max())
    return z / z.sum()


@dataclass(eq=False)
class _Node[P: Position[State, object]]:
    """A position of the search, and, once expanded, its actions and their statistics.

    ``actions`` holds the legal actions (``verb * n_cells + cell``, increasing), ``logits``
    the prior's log-probabilities over them, ``value`` player 0's value from the evaluator (NaN
    for a position passed through), ``n`` and ``w`` the visits of each action and the sum of
    player 0's values over them. A child is a node for a step without dice, a dict of nodes by
    outcome for a step that rolled, or None before the first visit.
    """

    position: P
    mover: int = -1
    actions: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    logits: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    value: float = math.nan
    n: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    w: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    children: list[_Node[P] | dict[bytes, _Node[P]] | None] = field(default_factory=list)

    @property
    def expanded(self) -> bool:
        """Whether the node has its actions."""
        return self.actions.size > 0


class _Leaf[P: Position[State, object]](NamedTuple):
    """A position a simulation reached that the evaluator must judge."""

    path: list[tuple[_Node[P], int]]
    """The nodes and actions taken from the root to it."""
    node: _Node[P]
    legal: NDArray[np.bool_]
    """``game.legal`` of its position, shape (n_verbs, n_cells)."""


class _Search[G: Game[Any, Any], P: Position[State, object]]:
    """One search of one decision: leaves out, evaluations in, a result at the end."""

    def __init__(
        self,
        player: AlphaZeroPlayer[G, P],
        game: G,
        position: P,
        rng: np.random.Generator,
        legal: NDArray[np.bool_] | None = None,
    ) -> None:
        if position.over:
            raise ValueError("search needs a position not over")
        if game.n_players != 2:
            raise ValueError(f"the search plays games of two players, not {game.n_players}")
        self.legal = game.legal(position) if legal is None else legal
        """The root's legal actions, shape (n_verbs, n_cells)."""
        self.player = player
        self.game = game
        self.rng = rng
        self.root: _Node[P] = _Node(position)
        self.left = player.simulations
        self.gumbel = np.zeros(0)
        self.n_cells = position.state.n_cells

    def leaf(self) -> _Leaf[P] | None:
        """Return the next position to evaluate, or None once every simulation is done.

        Simulations that end the game are backed up here, without an evaluation.
        """
        if not self.root.expanded:
            return _Leaf([], self.root, self.legal)
        while self.left > 0:
            self.left -= 1
            found = self._descend()
            if found is not None:
                return found
        return None

    def expand(self, leaf: _Leaf[P], logits: NDArray[np.float64], value: float) -> None:
        """Give ``leaf`` its prior and value, and back the value up its path."""
        node = leaf.node
        actions = np.flatnonzero(leaf.legal)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != actions.shape:
            raise ValueError(
                f"the evaluator gave {logits.shape} logits for {actions.size} legal actions"
            )
        if not np.isfinite(logits).all():
            raise ValueError("the evaluator gave logits that are not finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"the evaluator gave a value outside [0, 1]: {value}")
        top = logits.max()
        self._open(node, leaf.legal, logits - top - math.log(np.exp(logits - top).sum()))
        node.value = value
        if node is self.root:
            size = node.actions.size
            self.gumbel = self.rng.gumbel(size=size) if self.player.noise else np.zeros(size)
        self._backup(leaf.path, value)

    def result(self) -> SearchResult:
        """Return the action to play and what the search found at the root."""
        root = self.root
        sigma = self._sigma(root)
        score = self.gumbel + root.logits + sigma
        i = _argmax(np.where(self._finalists(root), score, -np.inf), self.rng)
        shape = (self.game.n_verbs, self.n_cells)
        policy = np.zeros(shape)
        policy.flat[root.actions] = _softmax(root.logits + sigma)
        visits = np.zeros(shape, dtype=np.int64)
        visits.flat[root.actions] = root.n
        q = np.full(shape, np.nan)
        seen = root.n > 0
        q.flat[root.actions[seen]] = root.w[seen] / root.n[seen]
        value = (root.value + root.w.sum()) / (1 + root.n.sum())
        verb, cell = divmod(int(root.actions[i]), self.n_cells)
        return SearchResult(Action(verb, cell), policy, visits, q, float(value))

    def _open(self, node: _Node[P], legal: NDArray[np.bool_], logits: NDArray[np.float64]) -> None:
        """Give ``node`` its actions, with these log-priors and no visits."""
        node.mover = self.game.player(node.position)
        node.actions = np.flatnonzero(legal)
        node.logits = logits
        node.n = np.zeros(node.actions.size, dtype=np.int64)
        node.w = np.zeros(node.actions.size)
        node.children = [None] * node.actions.size

    def _descend(self) -> _Leaf[P] | None:
        """Walk down once: return the leaf reached, or None if the game ended (backed up)."""
        path: list[tuple[_Node[P], int]] = []
        node = self.root
        while True:
            if node.position.over:
                self._backup(path, _result(node.position))
                return None
            if not node.expanded:
                legal = self.game.legal(node.position)
                if np.count_nonzero(legal) > 1:
                    return _Leaf(path, node, legal)
                self._open(node, legal, np.zeros(1))  # one action: passed through
            if node is self.root:
                i = self._at_root(node)
            elif node.actions.size == 1:
                i = 0
            else:
                i = self._below(node)
            path.append((node, i))
            node = self._child(node, i)

    def _at_root(self, node: _Node[P]) -> int:
        """Return the root's action for this simulation, by sequential halving."""
        considered = min(self.player.considered, node.actions.size)
        target = considered_visits(considered, self.player.simulations)[int(node.n.sum())]
        score = np.maximum(-1e9, self.gumbel + node.logits + self._sigma(node))
        return _argmax(np.where(node.n == target, score, -np.inf), self.rng)

    def _finalists(self, root: _Node[P]) -> NDArray[np.bool_]:
        """Return which root actions the last round of sequential halving still considered."""
        return finalists(root.n, self.player.considered, self.player.simulations)

    def _below(self, node: _Node[P]) -> int:
        """Return the action whose visits fall furthest short of the improved policy."""
        probs = _softmax(node.logits + self._sigma(node))
        return _argmax(probs - node.n / (1 + node.n.sum()), self.rng)

    def _sigma(self, node: _Node[P]) -> NDArray[np.float64]:
        """Return the scaled completed values of the node's actions, for its acting player."""
        n = node.n
        q = np.divide(node.w, n, out=np.zeros(n.size), where=n > 0)
        raw = node.value
        if node.mover == 1:
            q, raw = 1.0 - q, 1.0 - raw
        return sigma(
            q,
            raw,
            n,
            node.logits,
            self.player.c_visit,
            self.player.c_scale,
            rescale=node is self.root,
        )

    def _child(self, node: _Node[P], i: int) -> _Node[P]:
        """Return the position after action ``i``: kept if its step drew no dice, else sampled."""
        child = node.children[i]
        if isinstance(child, _Node):
            return child
        seed = int(self.rng.integers(2**63))
        verb, cell = divmod(int(node.actions[i]), self.n_cells)
        after = self.game.step(replace(node.position, seed=seed), (verb, cell))
        if child is None and not after.drew:
            made = _Node(after)
            node.children[i] = made
            return made
        outcomes: dict[bytes, _Node[P]] = {} if child is None else child
        node.children[i] = outcomes
        key = after.key()
        found = outcomes.get(key)
        if found is None:
            found = outcomes[key] = _Node(after)
        return found

    @staticmethod
    def _backup(path: list[tuple[_Node[Any], int]], value: float) -> None:
        """Add player 0's ``value`` to every action on ``path``."""
        for node, i in path:
            node.n[i] += 1
            node.w[i] += value
