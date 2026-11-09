"""A player with a world model: MuZero, learning a game from what it is shown.

A MuZero player (Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a
learned model", Nature 2020) never steps the game's rules: it learns a model of them, from its
own views, and plans with that. So it plays games whose positions it can't see, such as
Colonies under fog, without peeking. From one seat, a decision is followed by whatever the
world does until the player's next one (the other players' hidden orders and the dice), which
the model learns as one step: the tree holds only the player's own decisions.

`MuZeroPlayer` searches each decision with the Gumbel search of `agents.zero` (the same root
and the same completed values, `zero.sigma` and `zero.finalists`), over latent states of a
`WorldModel` instead of positions: the root is the model's reading of the player's
observation, and each action leads to the model's next latent state. It reads only its view,
so it plays fairly whatever the game hides. Below the root, where the player can't know what
will be legal, it considers every action, and the network's policy, trained over every action
with nothing on the illegal ones, steers it toward those likely to be legal (``below="all"``,
as MuZero does); ``below="root"`` keeps the actions legal at the root instead, the way of
networks trained before 2026-10-30, whose scores of illegal actions mean nothing.

With a `StochasticModel` (Stochastic MuZero; Antonoglou et al., "Planning in stochastic
environments with a learned model", ICLR 2022), each action leads to a chance node, the model's
afterstate, and each of its codes, what the world may do, to the next decision. The search then
takes, at a chance node, the code whose visits fall furthest short of the model's probabilities,
as `mctx` does, and each simulation expands one node of either kind.

`Trajectories` are what it learns from: each player's decisions through a game, in order, with
the whole position as that player would see it with nothing hidden (`agents.Revealable`), which
the model's belief learns to guess. `trajectories` makes them from records of any players'
games, and `self_play` from games of a MuZero player against itself, many at once, the model
judging a leaf of each per call. `reanalyze` searches decisions already kept again, with the
model as it is now, for fresher policy targets (MuZero Reanalyze). The network
(`agents.muzero_net`, whose `NetModel` is a `WorldModel`) and its training
(`agents.muzero_train`) import Torch; this module needs only NumPy.
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from gridzero.agents.frame import Observable, Revealable
from gridzero.agents.zero import (
    Played,
    SearchResult,
    _argmax,
    _softmax,
    considered_visits,
    finalists,
    sigma,
)
from gridzero.core import State, check_int, check_real
from gridzero.core._arrays import int_array
from gridzero.game import Action, Position, Record, replay


class WorldModel[L](Protocol):
    """A learned model of a game, from one player's seat: latent states ``L`` and their futures.

    Values are the deciding player's, from -1 (lost) to 1 (won), and scores cover every action
    in the player's frame (action ``verb * n_cells + cell``); the search masks them.
    """

    def initial(
        self, obs: Sequence[NDArray[np.float32]], /
    ) -> tuple[list[L], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states of observations, with their scores and values.

        Parameters
        ----------
        obs : sequence of ndarray of float32, shape (n_planes, n_cells)
            Observations (`agents.Observable.observe`), each in its player's frame.

        Returns
        -------
        latents : list of L
            One per observation.
        logits : ndarray of float64, shape (n, n_verbs * n_cells)
            Finite scores of every action.
        values : ndarray of float64, shape (n,)
            Values in [-1, 1].
        """
        ...

    def recurrent(
        self, latents: Sequence[L], actions: NDArray[np.int64], /
    ) -> tuple[list[L], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states at the player's next decisions, after its actions.

        Parameters
        ----------
        latents : sequence of L
            Latent states.
        actions : ndarray of int64, shape (n,)
            The action taken in each, in the player's frame.

        Returns
        -------
        latents, logits, values
            As `initial` returns them, for the states after.
        """
        ...


@runtime_checkable
class StochasticModel[L](Protocol):
    """A learned model whose dynamics go through afterstates and chance codes.

    As `WorldModel`, from one player's seat: values are the deciding player's, from -1 to 1,
    and scores cover every action in the player's frame. A player's action leads to an
    afterstate, before the world answers; one of ``codes`` outcomes then leads to the latent
    state at the player's next decision.
    """

    def initial(
        self, obs: Sequence[NDArray[np.float32]], /
    ) -> tuple[list[L], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states of observations, with their scores and values.

        As `WorldModel.initial`.
        """
        ...

    def afterstates(
        self, latents: Sequence[L], actions: NDArray[np.int64], /
    ) -> tuple[list[L], NDArray[np.float64], NDArray[np.float64]]:
        """Return the afterstates of the player's actions, what may follow them, and their values.

        Parameters
        ----------
        latents : sequence of L
            Latent states.
        actions : ndarray of int64, shape (n,)
            The action taken in each, in the player's frame.

        Returns
        -------
        afterstates : list of L
            One per action.
        probs : ndarray of float64, shape (n, codes)
            The probability of each code, each row summing to 1.
        values : ndarray of float64, shape (n,)
            Values in [-1, 1].
        """
        ...

    def chance(
        self, afterstates: Sequence[L], codes: NDArray[np.int64], /
    ) -> tuple[list[L], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states at the player's next decisions, after the world's answers.

        Parameters
        ----------
        afterstates : sequence of L
            Afterstates.
        codes : ndarray of int64, shape (n,)
            What the world did after each, in ``[0, codes)``.

        Returns
        -------
        latents, logits, values
            As `WorldModel.initial` returns them, for the states after.
        """
        ...


@dataclass(frozen=True)
class MuZeroPlayer[L]:
    """Picks each decision by Gumbel search over a learned model's latent states.

    The search is `agents.zero.AlphaZeroPlayer`'s (sequential halving at the root over the
    best ``considered`` actions by noise plus prior, completed values below it), with the
    model's latent states in place of positions and every value the deciding player's. With
    ``simulations`` 0 it plays the prior's best legal action (plus noise, with ``noise``): the
    network alone, for comparison.

    Parameters
    ----------
    model : WorldModel or StochasticModel
        The learned model, such as `agents.muzero_net.NetModel` or
        `agents.muzero_net.StochasticNetModel` (`agents.muzero_net.world_model` gives either).
    simulations : int, default 32
        Simulations per decision, at least 0.
    considered : int, default 16
        Actions sequential halving considers at the root, at least 1.
    noise : bool, default False
        Draw Gumbel noise at the root: on for self-play, off to play for strength.
    c_visit : float, default 50.0
        Visits added to the most visits in ``sigma``'s scale, at least 0.
    c_scale : float, default 0.03
        ``sigma``'s scale per visit, at least 0. ``mctx`` uses 0.1, but the root's values are
        rescaled to [0, 1] (`sigma`), so at 0.1 any gap between them outweighs the prior, and a
        network's noisy values decide: on 15x15 Tactics (2026-11-09), a network taught by a code
        player beat greedy 22-74 at 0.1 and 42-54 at 0.03.
    below : {"all", "root"}, default "all"
        The actions below the root: every action, or those legal at the root (see the module).

    Raises
    ------
    TypeError, ValueError
        If a parameter has the wrong type or is out of range.
    """

    model: WorldModel[L] | StochasticModel[L]
    simulations: int = 32
    considered: int = 16
    noise: bool = False
    c_visit: float = 50.0
    c_scale: float = 0.03
    below: Literal["all", "root"] = "all"

    def __post_init__(self) -> None:
        check_int(self.simulations, "simulations", minimum=0)
        check_int(self.considered, "considered", minimum=1)
        if not isinstance(self.noise, bool):
            raise TypeError(f"noise must be a bool, got {type(self.noise).__name__}")
        check_real(self.c_visit, "c_visit", 0.0)
        check_real(self.c_scale, "c_scale", 0.0)
        if self.below not in ("all", "root"):
            raise ValueError(f'below must be "all" or "root", got {self.below!r}')

    def __call__(
        self,
        game: Observable[Any, Any],
        player: int,
        view: object,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return the action the search prefers, or the only legal one (see `game.Player`)."""
        if np.count_nonzero(legal) == 1:
            verb, cell = np.argwhere(legal)[0]
            return Action(int(verb), int(cell))
        frame = game.frame(player)
        result = self.search(game.observe(view, player), frame.into(legal), rng)
        return frame.action_back(result.action)

    def search(
        self, obs: NDArray[np.float32], legal: NDArray[np.bool_], rng: np.random.Generator
    ) -> SearchResult:
        """Search one decision, all in the player's frame.

        Parameters
        ----------
        obs : ndarray of float32, shape (n_planes, n_cells)
            The player's observation.
        legal : ndarray of bool, shape (n_verbs, n_cells)
            Its legal actions, at least one.
        rng : numpy.random.Generator
            The source of the noise and of every tie.

        Returns
        -------
        SearchResult
            The action, the improved policy, the visits, and the values of the root's actions
            and of the root, the deciding player's, in [0, 1].

        Raises
        ------
        ValueError
            If no action is legal, or the model gives scores not finite or values outside
            [-1, 1].
        """
        search = LatentSearch(self, obs, legal, rng)
        while (leaf := search.leaf()) is not None:
            _judge(self.model, [(search, leaf)])
        return search.result()


@dataclass(eq=False)
class _LatentNode[L]:
    """A latent state of the search, its actions and their statistics.

    ``actions`` are those legal at the root, or every action below it (``below="all"``).
    ``logits`` holds the prior's log-probabilities over ``actions``, ``value`` the model's value
    in [0, 1] for the deciding player, ``n`` and ``w`` the visits of each action and the sum of
    values over them.
    """

    latent: L
    actions: NDArray[np.int64]
    logits: NDArray[np.float64]
    value: float
    n: NDArray[np.int64]
    w: NDArray[np.float64]
    children: list[_LatentNode[L] | _ChanceNode[L] | None] = field(default_factory=list)


@dataclass(eq=False)
class _ChanceNode[L]:
    """An afterstate of the search, what may follow it, and their statistics.

    ``probs`` holds the model's probability of each code, ``value`` its value in [0, 1] for the
    deciding player, ``n`` and ``w`` the visits of each code and the sum of values over them.
    """

    latent: L
    probs: NDArray[np.float64]
    value: float
    n: NDArray[np.int64]
    w: NDArray[np.float64]
    children: list[_LatentNode[L] | None] = field(default_factory=list)


type _Node[L] = _LatentNode[L] | _ChanceNode[L]


class LatentLeaf[L](NamedTuple):
    """What a search needs the model for: the root's observation, or a step from a node."""

    path: list[tuple[_Node[L], int]]
    """The nodes and steps taken from the root, the last one to the state wanted."""
    parent: _Node[L] | None
    """The node the step leaves, or None for the root: from a decision an action, to a state
    (`WorldModel.recurrent`) or an afterstate (`StochasticModel.afterstates`); from a chance
    node a code, to a state (`StochasticModel.chance`)."""
    index: int
    """The step's place among the parent's actions or codes (0 for the root)."""
    obs: NDArray[np.float32]
    """The root's observation, quantized as stored for training."""


class LatentSearch[L]:
    """One search of one decision over a model's latent states: leaves out, states in.

    It runs a leaf at a time (`leaf`, then `expand` with the model's answer), so that many
    searches can share a model's batches, as `agents.zero` does for positions.

    Parameters
    ----------
    player : MuZeroPlayer
        Its settings.
    obs : ndarray of float32, shape (n_planes, n_cells)
        The deciding player's observation.
    legal : ndarray of bool, shape (n_verbs, n_cells)
        Its legal actions in its frame, at least one.
    rng : numpy.random.Generator
        The source of the noise and of every tie.
    """

    def __init__(
        self,
        player: MuZeroPlayer[L],
        obs: NDArray[np.float32],
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> None:
        self.actions = np.flatnonzero(legal)
        """The root's actions: those legal."""
        self.every = np.arange(legal.size)
        """Every action, for nodes below the root with ``below="all"``."""
        if self.actions.size == 0:
            raise ValueError("search needs at least one legal action")
        self.player = player
        self.legal = legal
        self.obs = (np.rint(obs * 255) / 255).astype(np.float32)
        self.rng = rng
        self.root: _LatentNode[L] | None = None
        self.stochastic = isinstance(player.model, StochasticModel)
        """Whether actions lead to chance nodes."""
        self.left = player.simulations
        self.gumbel = np.zeros(0)

    def leaf(self) -> LatentLeaf[L] | None:
        """Return the next latent state the search needs, or None once it is done."""
        if self.root is None:
            return LatentLeaf([], None, 0, self.obs)
        if self.left == 0:
            return None
        self.left -= 1
        path: list[tuple[_Node[L], int]] = []
        node: _Node[L] = self.root
        while True:
            if isinstance(node, _ChanceNode):
                i = _argmax(node.probs - node.n / (1 + node.n.sum()), self.rng)
            else:
                i = self._at_root() if node is self.root else self._below(node)
            path.append((node, i))
            child = node.children[i]
            if child is None:
                return LatentLeaf(path, node, i, self.obs)
            node = child

    def expand(
        self, leaf: LatentLeaf[L], latent: L, logits: NDArray[np.float64], value: float
    ) -> None:
        """Give the state the leaf asked for its model's answer, and back its value up."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (self.legal.size,):
            raise ValueError(f"the model gave {logits.shape} scores for {self.legal.size} actions")
        if not np.isfinite(logits).all():
            raise ValueError("the model gave scores that are not finite")
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"the model gave a value outside [-1, 1]: {value}")
        root = leaf.parent is None
        actions = self.actions if root or self.player.below == "root" else self.every
        own = logits[actions]
        top = own.max()
        size = actions.size
        node = _LatentNode(
            latent,
            actions,
            own - top - math.log(np.exp(own - top).sum()),
            (value + 1) / 2,
            np.zeros(size, dtype=np.int64),
            np.zeros(size),
            [None] * size,
        )
        if root:
            self.root = node
            self.gumbel = self.rng.gumbel(size=size) if self.player.noise else np.zeros(size)
            return
        if isinstance(leaf.parent, _ChanceNode) != self.stochastic:
            raise ValueError("a state follows an afterstate's code, or an action without codes")
        leaf.parent.children[leaf.index] = node
        _backed_up(leaf.path, node.value)

    def expand_chance(
        self, leaf: LatentLeaf[L], afterstate: L, probs: NDArray[np.float64], value: float
    ) -> None:
        """Give the afterstate the leaf asked for its model's answer, and back its value up.

        Parameters
        ----------
        leaf : LatentLeaf
            A leaf of this search that steps from a decision, with a stochastic model.
        afterstate : L
            The model's afterstate.
        probs : ndarray of float64, shape (codes,)
            The probability of each code: finite, at least 0, summing to 1.
        value : float
            Its value for the deciding player, in [-1, 1].

        Raises
        ------
        ValueError
            If the leaf doesn't step from a decision with a stochastic model, or the model's
            answer is out of range.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError(f"the model gave probabilities of shape {probs.shape}")
        if not (np.isfinite(probs).all() and (probs >= 0).all()):
            raise ValueError("the model gave probabilities that are not finite and at least 0")
        if not math.isclose(probs.sum(), 1.0, abs_tol=1e-6):
            raise ValueError(f"the model gave probabilities summing to {probs.sum()}")
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"the model gave a value outside [-1, 1]: {value}")
        parent = leaf.parent
        if not self.stochastic or not isinstance(parent, _LatentNode):
            raise ValueError("an afterstate follows an action, with a stochastic model")
        size = probs.size
        node = _ChanceNode(
            afterstate,
            probs,
            (value + 1) / 2,
            np.zeros(size, dtype=np.int64),
            np.zeros(size),
            [None] * size,
        )
        parent.children[leaf.index] = node
        _backed_up(leaf.path, node.value)

    def result(self) -> SearchResult:
        """Return the action to play and what the search found at the root."""
        root = self.root
        assert root is not None, "the root is evaluated before any result"
        shape = self.legal.shape
        n_cells = shape[1]
        score = self.gumbel + root.logits
        if self.player.simulations == 0:
            i = _argmax(score, self.rng)
            policy_logits = root.logits
        else:
            s = self._sigma(root)
            chosen = finalists(root.n, self.player.considered, self.player.simulations)
            i = _argmax(np.where(chosen, score + s, -np.inf), self.rng)
            policy_logits = root.logits + s
        policy = np.zeros(shape)
        policy.flat[root.actions] = _softmax(policy_logits)
        visits = np.zeros(shape, dtype=np.int64)
        visits.flat[root.actions] = root.n
        q = np.full(shape, np.nan)
        seen = root.n > 0
        q.flat[root.actions[seen]] = root.w[seen] / root.n[seen]
        value = (root.value + root.w.sum()) / (1 + root.n.sum())
        verb, cell = divmod(int(root.actions[i]), n_cells)
        return SearchResult(Action(verb, cell), policy, visits, q, float(value))

    def _sigma(self, node: _LatentNode[L]) -> NDArray[np.float64]:
        """Return the scaled completed values of the node's actions."""
        q = np.divide(node.w, node.n, out=np.zeros(node.n.size), where=node.n > 0)
        p = self.player
        return sigma(
            q, node.value, node.n, node.logits, p.c_visit, p.c_scale, rescale=node is self.root
        )

    def _at_root(self) -> int:
        """Return the root's action for this simulation, by sequential halving."""
        root = self.root
        assert root is not None
        considered = min(self.player.considered, root.actions.size)
        target = considered_visits(considered, self.player.simulations)[int(root.n.sum())]
        score = np.maximum(-1e9, self.gumbel + root.logits + self._sigma(root))
        return _argmax(np.where(root.n == target, score, -np.inf), self.rng)

    def _below(self, node: _LatentNode[L]) -> int:
        """Return the action whose visits fall furthest short of the improved policy."""
        probs = _softmax(node.logits + self._sigma(node))
        return _argmax(probs - node.n / (1 + node.n.sum()), self.rng)


@dataclass(frozen=True, eq=False)
class Trajectories:
    """Each player's decisions through games, in order, one trajectory after another.

    Row ``i`` is one decision. A trajectory's rows are consecutive, from ``bounds[t]`` to
    ``bounds[t + 1]``: every decision of one player in one game, single legal actions too,
    since a model learns what follows each of them.

    Parameters
    ----------
    obs : ndarray of uint8, shape (n, n_planes, n_cells)
        Observations of the player's view, ``round(255 * game.observe(view, player))``.
    revealed : ndarray of uint8, shape (n, n_revealed, n_cells)
        The whole position as the player would see it, ``round(255 * game.reveal(...))``.
    legal : ndarray of uint8, shape (n, ceil(n_verbs * n_cells / 8))
        The legal actions in the player's frame, flattened over verbs and cells and packed by
        `numpy.packbits`.
    action : ndarray of int64, shape (n,)
        The action taken, in the player's frame: ``verb * n_cells + cell``.
    policy : ndarray of float32, shape (m,)
        The target policy over each decision's legal actions, in the order of their flat
        index: decision ``i``'s are ``policy[start[i]:start[i + 1]]``. All on the action taken
        for decisions of players that don't search.
    start : ndarray of int64, shape (n + 1,)
        Where each decision's policy starts in ``policy``, and its end.
    outcome : ndarray of float32, shape (n,)
        1 if the player went on to win, -1 if another player won, 0 for a draw.
    round : ndarray of int64, shape (n,)
        The decision's place in its trajectory: 0 for the player's first.
    bounds : ndarray of int64, shape (n_trajectories + 1,)
        Where each trajectory's rows start, and the end of the last.
    """

    obs: NDArray[np.uint8]
    revealed: NDArray[np.uint8]
    legal: NDArray[np.uint8]
    action: NDArray[np.int64]
    policy: NDArray[np.float32]
    start: NDArray[np.int64]
    outcome: NDArray[np.float32]
    round: NDArray[np.int64]
    bounds: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.action.size)

    @property
    def n_trajectories(self) -> int:
        """How many trajectories."""
        return int(self.bounds.size - 1)

    def remaining(self) -> NDArray[np.int64]:
        """Return how many decisions follow each one in its trajectory.

        Returns
        -------
        ndarray of int64, shape (n,)
            0 for each trajectory's last decision.
        """
        lengths = np.diff(self.bounds)
        last = np.repeat(self.bounds[1:] - 1, lengths)
        return last - np.arange(len(self))

    def since(self, first: int) -> Trajectories:
        """Return the trajectories from number ``first`` on, rows renumbered from 0.

        Parameters
        ----------
        first : int
            The first trajectory kept, from 0 to `n_trajectories`.

        Returns
        -------
        Trajectories
            The later trajectories, sharing this one's arrays where they can.
        """
        first = check_int(first, "first", minimum=0)
        if first > self.n_trajectories:
            raise ValueError(f"first must be at most {self.n_trajectories}, got {first}")
        row = int(self.bounds[first])
        at = int(self.start[row])
        return Trajectories(
            self.obs[row:],
            self.revealed[row:],
            self.legal[row:],
            self.action[row:],
            self.policy[at:],
            self.start[row:] - at,
            self.outcome[row:],
            self.round[row:],
            self.bounds[first:] - row,
        )

    @staticmethod
    def join(parts: Sequence[Trajectories]) -> Trajectories:
        """Return trajectories one after another, in order.

        Parameters
        ----------
        parts : sequence of Trajectories
            At least one, of the same game.

        Returns
        -------
        Trajectories
            Every part's trajectories, the first part's first.
        """
        policy_sizes = np.cumsum([0] + [p.policy.size for p in parts[:-1]])
        row_sizes = np.cumsum([0] + [len(p) for p in parts[:-1]])
        start = [parts[0].start[:1]]
        bounds = [parts[0].bounds[:1]]
        for p, ps, rs in zip(parts, policy_sizes, row_sizes, strict=True):
            start.append(p.start[1:] + ps)
            bounds.append(p.bounds[1:] + rs)
        return Trajectories(
            np.concatenate([p.obs for p in parts]),
            np.concatenate([p.revealed for p in parts]),
            np.concatenate([p.legal for p in parts]),
            np.concatenate([p.action for p in parts]),
            np.concatenate([p.policy for p in parts]),
            np.concatenate(start),
            np.concatenate([p.outcome for p in parts]),
            np.concatenate([p.round for p in parts]),
            np.concatenate(bounds),
        )


def trajectories[P: Position[State, object]](
    game: Revealable[P, Any], records: Sequence[Record[P]]
) -> Trajectories:
    """Return every player's decisions in games already played, as trajectories to learn from.

    Each record is replayed (`game.replay`). Every decision goes to its player's trajectory,
    with the observation of its view, the whole position revealed to it, the legal actions and
    the action taken, all in its frame, and a policy entirely on that action: what a network
    learns by imitating the players. Trajectories go by record, then by player.

    Parameters
    ----------
    game : Revealable
        The rules the games were played by.
    records : sequence of Record
        Games played to their end, at least one.

    Returns
    -------
    Trajectories
        One trajectory per player per game who decided at least once.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("trajectories need at least one record")
    games: list[_Played] = []
    for record in records:
        decisions = _Decisions()
        for position, played in zip(replay(game, record), record.actions, strict=False):
            p = game.player(position)
            frame = game.frame(p)
            mask = frame.into(game.legal(position)).ravel()
            verb, cell = frame.action_into(played)
            flat = verb * game.space.n_cells + cell
            target = np.zeros(np.count_nonzero(mask), dtype=np.float32)
            target[np.count_nonzero(mask[:flat])] = 1.0  # the action's place among the legal
            decisions.add(game, position, p, mask, flat, target)
        games.append((decisions, record.final.winner))
    return _assembled(games)


def self_play[P: Position[State, object], L](
    game: Revealable[P, Any],
    player: MuZeroPlayer[L],
    seeds: Sequence[int],
    *,
    rival: MuZeroPlayer[Any] | None = None,
) -> tuple[Trajectories, list[Played[P]]]:
    """Play games of a MuZero player in every seat, all at once, and keep every decision.

    With a ``rival``, the player takes one seat of each game, seat ``i % n_players`` in the
    ``i``-th, and the rival every other, and only the player's decisions are kept: the games
    an exploiter learns from against a fixed player.

    Each game searches its decisions in turn, as ``player.search`` does; the games' searches
    take a leaf each, and the model judges them in two calls (`WorldModel.initial` for new
    roots, `WorldModel.recurrent` for the rest), so a batched model (`muzero_net.NetModel` on a
    GPU) sees as many states per call as games still running. Every decision is kept, those
    with a single legal action too (with its policy all on it), since the model learns what
    follows each: the observation, the revealed planes, the legal actions, the action and the
    improved policy, in the deciding player's frame.

    Game ``seed`` starts from ``game.start(seed)`` and searches with its own generator, a stream
    of the seed, so it plays the same whatever other games run beside it, given a model whose
    answers don't depend on the batch. A network on a GPU is not such a model: its sums may run
    in another order at another batch size, so self-play there replays only roughly.

    Parameters
    ----------
    game : Revealable
        The rules.
    player : MuZeroPlayer
        The player in every seat, with its model; set ``noise`` for exploration.
    seeds : sequence of int
        One game per seed, each at least 0, at least one.
    rival : MuZeroPlayer, optional
        The player in the other seats, with a model of its own; its leaves are judged in
        batches of their own.

    Returns
    -------
    trajectories : Trajectories
        Every decision of the player, by game in the order of the seeds, then by seat.
    played : list of Played
        How each game went, in the order of the seeds (``searched`` and ``kept`` count the
        decisions searched and kept).

    Raises
    ------
    TypeError, ValueError
        If a seed is not a non-negative integer, there are none, or the model answers badly.
    """
    if not seeds:
        raise ValueError("self-play needs at least one seed")
    games = [
        _SelfPlay(
            s,
            game.start(s),
            np.random.default_rng(np.random.SeedSequence(s, spawn_key=(4,))),
            None if rival is None else i % game.n_players,
        )
        for i, s in enumerate(check_int(seed, "seed", minimum=0) for seed in seeds)
    ]
    running = list(games)
    while running:
        waiting = [
            (g, leaf) for g in running if (leaf := g.advance(game, player, rival)) is not None
        ]
        running = [g for g, _ in waiting]
        for searcher in (player, rival):
            if searcher is not None:
                mine = [(g.search, leaf) for g, leaf in waiting if g.searcher is searcher]
                _judge(searcher.model, [(s, leaf) for s, leaf in mine if s is not None])
    data = _assembled([(g.decisions, g.position.winner) for g in games])
    return data, [g.played() for g in games]


def reanalyze[L](
    game: Observable[Any, Any],
    player: MuZeroPlayer[L],
    data: Trajectories,
    rows: Sequence[int] | NDArray[np.integer],
    *,
    seed: int = 0,
    batch: int = 512,
) -> Trajectories:
    """Return trajectories whose policy targets at some decisions are searched again.

    MuZero Reanalyze: a target kept from an earlier search is as good as the model was then.
    Each decision in ``rows`` is searched again from its stored observation and legal actions,
    as ``player.search`` does, and its policy target becomes the new improved policy. What
    happened stays: the observations, the actions taken and the outcomes. A decision with one
    legal action keeps its target, all on that action. No game is stepped, so it works for any
    game, hidden or not: the search reads only what the player saw.

    The searches run ``batch`` at a time in lockstep, the model judging a leaf of each per call,
    as in `self_play`; a model holds every latent state of a batch's trees until it is done, so
    ``batch`` bounds the memory. All draw from one generator seeded by ``seed``.

    Parameters
    ----------
    game : Observable
        The rules the decisions were taken under: they fix the number of verbs and cells.
    player : MuZeroPlayer
        The searcher, with its model, at least one simulation.
    data : Trajectories
        Decisions of the game.
    rows : sequence of int or ndarray of int, shape (k,)
        The decisions searched again, each in ``[0, len(data))``.
    seed : int, default 0
        The seed of the noise and of every tie, non-negative.
    batch : int, default 512
        Searches at once, at least 1.

    Returns
    -------
    Trajectories
        ``data`` with a new policy array; every other array shared.

    Raises
    ------
    TypeError, ValueError
        If an argument has the wrong type or is out of range, or the model answers badly.
    IndexError
        If a row is outside the trajectories.
    """
    rows = int_array(rows, "rows").ravel()
    if np.any((rows < 0) | (rows >= len(data))):
        raise IndexError(f"rows must be in [0, {len(data)})")
    if player.simulations == 0:
        raise ValueError("reanalyzing needs a player that searches, with simulations")
    batch = check_int(batch, "batch", minimum=1)
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    shape = (game.n_verbs, game.space.n_cells)
    rows = rows[data.start[rows + 1] - data.start[rows] > 1]  # more than one legal action
    policy = data.policy.copy()
    for chunk in (rows[i : i + batch] for i in range(0, rows.size, batch)):
        legal = np.unpackbits(data.legal[chunk], axis=1)[:, : shape[0] * shape[1]].astype(bool)
        searches = [
            LatentSearch(player, (data.obs[r] / 255).astype(np.float32), m.reshape(shape), rng)
            for r, m in zip(chunk, legal, strict=True)
        ]
        while waiting := [(s, leaf) for s in searches if (leaf := s.leaf()) is not None]:
            _judge(player.model, waiting)
        for r, m, search in zip(chunk, legal, searches, strict=True):
            policy[data.start[r] : data.start[r + 1]] = search.result().policy.ravel()[m]
    return dataclasses.replace(data, policy=policy)


def _judge[L](
    model: WorldModel[L] | StochasticModel[L],
    waiting: Sequence[tuple[LatentSearch[L], LatentLeaf[L]]],
) -> None:
    """Give each search's leaf the model's answer, a call per kind of leaf.

    New roots go to ``initial``; steps from decisions to ``recurrent``, or with a
    `StochasticModel` to ``afterstates``; steps from chance nodes to ``chance``.
    """
    new = [(s, leaf) for s, leaf in waiting if leaf.parent is None]
    acted = [(s, leaf) for s, leaf in waiting if isinstance(leaf.parent, _LatentNode)]
    coded = [(s, leaf) for s, leaf in waiting if isinstance(leaf.parent, _ChanceNode)]
    if new:
        _expand(new, *model.initial([leaf.obs for _, leaf in new]))
    if acted:
        nodes = [leaf.parent for _, leaf in acted if isinstance(leaf.parent, _LatentNode)]
        latents = [p.latent for p in nodes]
        actions = np.array(
            [p.actions[leaf.index] for p, (_, leaf) in zip(nodes, acted, strict=True)]
        )
        if isinstance(model, StochasticModel):
            afterstates, probs, values = model.afterstates(latents, actions)
            for i, ((search, leaf), after) in enumerate(zip(acted, afterstates, strict=True)):
                search.expand_chance(leaf, after, probs[i], float(values[i]))
        else:
            _expand(acted, *model.recurrent(latents, actions))
    if coded:
        if not isinstance(model, StochasticModel):
            raise ValueError("only a stochastic model has chance nodes")
        chances = [leaf.parent for _, leaf in coded if isinstance(leaf.parent, _ChanceNode)]
        codes = np.array([leaf.index for _, leaf in coded], dtype=np.int64)
        _expand(coded, *model.chance([c.latent for c in chances], codes))


def _expand[L](
    waiting: Sequence[tuple[LatentSearch[L], LatentLeaf[L]]],
    latents: Sequence[L],
    logits: NDArray[np.float64],
    values: NDArray[np.float64],
) -> None:
    """Give each search's leaf the model's answer."""
    for i, ((search, leaf), latent) in enumerate(zip(waiting, latents, strict=True)):
        search.expand(leaf, latent, logits[i], float(values[i]))


def _backed_up[L](path: Sequence[tuple[_Node[L], int]], value: float) -> None:
    """Count a visit of each step along a path, with the value found at its end."""
    for node, i in path:
        node.n[i] += 1
        node.w[i] += value


class _SelfPlay[P: Position[State, object], L]:
    """A self-play game under way: its position, its search, and its decisions."""

    def __init__(self, seed: int, position: P, rng: np.random.Generator, seat: int | None) -> None:
        self.seed = seed
        self.position = position
        self.rng = rng
        self.seat = seat
        """The learner's seat, or None if it sits in every seat."""
        self.search: LatentSearch[L] | None = None
        self.searcher: MuZeroPlayer[Any] | None = None
        """Whose search is under way."""
        self.decisions = _Decisions()
        self.searched = 0
        self.kept = 0

    def advance(
        self,
        game: Revealable[P, Any],
        player: MuZeroPlayer[L],
        rival: MuZeroPlayer[Any] | None,
    ) -> LatentLeaf[L] | None:
        """Play on until a search needs its model; return its leaf, or None once over."""
        while True:
            if self.search is not None:
                leaf = self.search.leaf()
                if leaf is not None:
                    return leaf
                result = self.search.result()
                self.search = None
                legal = self._mask(game)
                flat = int(result.action.verb * legal.shape[1] + result.action.cell)
                policy = result.policy.ravel()[legal.ravel()].astype(np.float32)
                self._take(game, legal.ravel(), flat, policy)
            elif self.position.over:
                return None
            else:
                legal = self._mask(game)
                if np.count_nonzero(legal) == 1:
                    flat = int(np.flatnonzero(legal)[0])
                    self._take(game, legal.ravel(), flat, np.ones(1, dtype=np.float32))
                else:
                    mover = game.player(self.position)
                    obs = game.observe(game.view(self.position, mover), mover)
                    learns = self.seat is None or mover == self.seat
                    self.searcher = player if learns or rival is None else rival
                    self.search = LatentSearch(self.searcher, obs, legal, self.rng)
                    self.searched += learns

    def _mask(self, game: Revealable[P, Any]) -> NDArray[np.bool_]:
        """Return the mover's legal actions in its frame, shape (n_verbs, n_cells)."""
        return game.frame(game.player(self.position)).into(game.legal(self.position))

    def _take(
        self,
        game: Revealable[P, Any],
        legal: NDArray[np.bool_],
        flat: int,
        policy: NDArray[np.float32],
    ) -> None:
        """Keep the mover's decision if it is the learner's, then play its action."""
        mover = game.player(self.position)
        frame = game.frame(mover)
        if self.seat is None or mover == self.seat:
            self.decisions.add(game, self.position, mover, legal, flat, policy)
            self.kept += 1
        n_cells = legal.size // game.n_verbs
        self.position = game.step(self.position, frame.action_back(divmod(flat, n_cells)))

    def played(self) -> Played[P]:
        """Return how the game went."""
        p = self.position
        return Played(self.seed, p.winner, p.ply, self.searched, self.kept, p)


Row = tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], int, NDArray[np.float32]]
"""One decision: observation, revealed planes, packed legal actions, action and policy."""


class _Decisions:
    """One game's decisions, each player's in order, as they are taken."""

    def __init__(self) -> None:
        self.by_player: dict[int, list[Row]] = {}

    def add(
        self,
        game: Revealable[Any, Any],
        position: Position[State, object],
        player: int,
        mask: NDArray[np.bool_],
        action: int,
        policy: NDArray[np.float32],
    ) -> None:
        """Keep a decision of ``player`` at ``position``: its legal ``mask`` (flat) and choice."""
        obs = np.rint(game.observe(game.view(position, player), player) * 255).astype(np.uint8)
        revealed = np.rint(game.reveal(position, player) * 255).astype(np.uint8)
        row = (obs, revealed, np.packbits(mask), action, policy)
        self.by_player.setdefault(player, []).append(row)


type _Played = tuple[_Decisions, int | None]
"""A game's decisions and its winner, or None for a draw."""


def _assembled(games: Sequence[_Played]) -> Trajectories:
    """Return the trajectories of games, by game, then by player."""
    rows: list[Row] = []
    outcome: list[float] = []
    rounds: list[int] = []
    lengths: list[int] = []
    for decisions, winner in games:
        for p, kept in decisions.by_player.items():
            rows += kept
            outcome += [0.0 if winner is None else (1.0 if winner == p else -1.0)] * len(kept)
            rounds += range(len(kept))
            lengths.append(len(kept))
    policy = [r[4] for r in rows]
    start = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([t.size for t in policy], out=start[1:])
    bounds = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=bounds[1:])
    return Trajectories(
        np.stack([r[0] for r in rows]),
        np.stack([r[1] for r in rows]),
        np.stack([r[2] for r in rows]),
        np.array([r[3] for r in rows], dtype=np.int64),
        np.concatenate(policy),
        start,
        np.array(outcome, dtype=np.float32),
        np.array(rounds, dtype=np.int64),
        bounds,
    )
