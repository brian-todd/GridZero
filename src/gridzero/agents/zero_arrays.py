"""The AlphaZero search for a game's rules as arrays: many decisions at once on the GPU.

`ArraySearch` is the search of `agents.zero.AlphaZeroPlayer` (Gumbel AlphaZero: sequential
halving at the root, the improved policy below it, completed values rescaled at the root only)
written with ``jax.numpy`` on a `gridzero.batched.ArrayRules`, so that thousands of decisions
are searched together. The tree of one decision is a `Tree` of arrays of fixed shape, and
`ArraySearch.plant`, `ArraySearch.descend`, `ArraySearch.expand` and `ArraySearch.found` are
functions of one tree: a batch is those under ``jax.vmap``, which `ArraySearch.search` runs,
one batch of leaves per call of an evaluator.

The evaluator is not JAX's: a network in Torch (`agents.net.NetArrayEvaluator`) takes each
batch of observations where it lies on the GPU and gives its scores and values back, without
a copy.

Where it differs from `agents.zero`:

- **Dice are weighed, not rolled.** A step whose outcome the dice decide (`ArrayRules.outcomes`)
  leads to every outcome it has visited, each a position searched further, and visits go to
  the outcomes in proportion to their exact chances: to the one whose share of the visits
  falls furthest short of its chance. The value of an action is then the expectation over its
  outcomes, without the noise of sampling them. `agents.zero` rolls anew on each visit.
- **A position with one legal action is not a node.** The step it forces is taken when the
  position is reached, with dice drawn once, and the node holds the position after it.
  `agents.zero` keeps such positions and passes through them; neither asks the evaluator.
- Floats are float32, and ties go to the lowest action, not to a random one.

This module imports JAX, so the package does not import it: import it as
``gridzero.agents.zero_arrays``.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from gridzero import batched
from gridzero.agents.zero import _rounds, considered_visits
from gridzero.batched import ArrayRules
from gridzero.core._arrays import check_int, check_real


class Exported(Protocol):
    """An array of any library that shares its memory by DLPack: Torch's, JAX's, NumPy's."""

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # the array API's
        ...

    def __dlpack_device__(self) -> tuple[Any, int]: ...


class ArrayEvaluator(Protocol):
    """Judges a batch of observations for `ArraySearch`: scores of actions, and who wins."""

    def __call__(self, obs: jax.Array, /) -> tuple[Exported, Exported]:
        """Return the scores of every action and the value of each observation.

        Parameters
        ----------
        obs : Array of uint8, shape (batch, n_planes, n_cells)
            Positions as their players to move see them, ``round(255 *
            ArrayRules.observe(...))``, as self-play stores them.

        Returns
        -------
        scores : array of float32, shape (batch, n_verbs * n_cells) or (batch, n_verbs, n_cells)
            A finite score per action, in the frame of each position's player to move, as
            `agents.net.PolicyNet` gives them. Only their differences matter.
        values : array of float32, shape (batch,)
            Each position's value for its player to move, in [-1, 1]: 1 won, -1 lost.
        """
        ...


class Tree[S](NamedTuple):
    """The search of one decision, as arrays: ``N = simulations + 1`` nodes of ``A`` actions.

    Node 0 is the root. A node is a position where a player has a choice, or the end of a
    game. An action that leads to one position has that node as its child. An action whose
    outcome the dice decide has a record of its ``O`` outcomes instead: their chances, the
    node each leads to once visited, and their visits. Unused rows hold what `plant` put
    there.
    """

    state: S
    """The position of each node: the game's state with a leading axis of ``N``."""
    mover: jax.Array
    """The player to move at each node: int32, shape (N,)."""
    over: jax.Array
    """Whether the game is over at each node: bool, shape (N,)."""
    logits: jax.Array
    """The prior's log-probabilities over each node's actions, -inf where illegal: float32,
    shape (N, A). Before a node is judged, 0 where legal."""
    value: jax.Array
    """Player 0's value of each node, in [0, 1]: the evaluator's, or the result of a game that
    is over (0.5 for a draw). float32, shape (N,)."""
    n: jax.Array
    """Simulations through each action of each node: int32, shape (N, A)."""
    w: jax.Array
    """The sum of player 0's values over those simulations: float32, shape (N, A)."""
    child: jax.Array
    """Where each action leads: -1 before its first visit, a node below ``N``, or ``N + r``
    for the record ``r`` of an action the dice decide. int32, shape (N, A)."""
    chances: jax.Array
    """The probability of each outcome of each record: float32, shape (N, O)."""
    reached: jax.Array
    """The node each outcome leads to, -1 before its first visit: int32, shape (N, O)."""
    seen: jax.Array
    """Simulations through each outcome: int32, shape (N, O)."""
    gumbel: jax.Array
    """The noise on the root's actions: float32, shape (A,)."""
    nodes: jax.Array
    """How many nodes there are: int32, shape ()."""
    records: jax.Array
    """How many records there are: int32, shape ()."""
    leaf: jax.Array
    """The node the last simulation ended on: int32, shape ()."""
    path: jax.Array
    """The way it took there, a row per step: the node it left, the action it took, the
    record of that action if the dice decided (else -1), and the outcome. int32, shape
    (N, 4)."""
    depth: jax.Array
    """How many steps that was, 0 once `ArraySearch.expand` has counted them: int32,
    shape ()."""
    fresh: jax.Array
    """Whether that node waits for the evaluator's prior and value: bool, shape ()."""


class Found(NamedTuple):
    """What a search of one decision found, over the game's local actions."""

    action: jax.Array
    """The local action chosen: int32, shape ()."""
    policy: jax.Array
    """The improved policy: the softmax of the prior's logits plus the scaled completed
    values, 0 where illegal. The target to learn from. float32, shape (A,)."""
    visits: jax.Array
    """Simulations through each action at the root: int32, shape (A,)."""
    q: jax.Array
    """Player 0's mean value of each action at the root, NaN where unvisited: float32,
    shape (A,)."""
    value: jax.Array
    """Player 0's value of the root: its evaluation and every simulation's, averaged.
    float32, shape ()."""


@dataclass(frozen=True, eq=False)
class ArraySearch[S]:
    """Gumbel AlphaZero tree search on a game's rules as arrays, for two players.

    The settings are those of `agents.zero.AlphaZeroPlayer`, and so is the search, but for
    the differences this module names. `search` runs a batch of decisions. The functions it
    is made of take one tree, for ``jax.vmap`` and ``jax.jit``: `plant` a root, then for each
    simulation `expand` the leaf with the evaluator's judgement of `observation` and `descend`
    to the next, and read what it `found`.

    Parameters
    ----------
    game : ArrayRules
        The rules as arrays, of a game of two players without hidden information.
    simulations : int, default 32
        Simulations per decision, at least 1.
    considered : int, default 16
        Actions sequential halving considers at the root, at least 1.
    noise : bool, default False
        Whether `search` draws Gumbel noise at the root, which samples the actions considered
        and played: on for self-play, off to play for strength.
    c_visit : float, default 50.0
        Visits added to the most visits in the scale of the completed values, at least 0.
    c_scale : float, default 0.03
        That scale per visit, at least 0.

    Raises
    ------
    TypeError, ValueError
        If a parameter has the wrong type or is out of range.
    """

    game: ArrayRules[S, Any]
    simulations: int = 32
    considered: int = 16
    noise: bool = False
    c_visit: float = 50.0
    c_scale: float = 0.03

    def __post_init__(self) -> None:
        check_int(self.simulations, "simulations", minimum=1)
        check_int(self.considered, "considered", minimum=1)
        if not isinstance(self.noise, bool):
            raise TypeError(f"noise must be a bool, got {type(self.noise).__name__}")
        check_real(self.c_visit, "c_visit", 0.0)
        check_real(self.c_scale, "c_scale", 0.0)

    # A batch of decisions.

    def search(self, states: S, evaluate: ArrayEvaluator, key: ArrayLike) -> Found:
        """Search a batch of decisions, one batch of leaves per call of the evaluator.

        Parameters
        ----------
        states : S
            The positions to decide in, none over: the game's state with a leading axis of
            the batch.
        evaluate : ArrayEvaluator
            Judges the leaves: called ``simulations + 1`` times, each with one observation per
            decision. A decision whose simulation ended the game sends an observation whose
            judgement is not read.
        key : Array of uint32, shape (2,)
            The key of this search's draws (`gridzero.batched.root`): decision ``i`` draws its
            noise and its dice from stream ``i`` of it, so a decision is searched the same
            whatever others are beside it, given an evaluator that judges each alone.

        Returns
        -------
        Found
            What each search found: every array with a leading axis of the batch.
        """
        tree, obs = self._planted(states, jnp.asarray(key, jnp.uint32))
        for simulation in range(self.simulations):
            scores, values = evaluate(obs)
            scores, values = jnp.from_dlpack(scores), jnp.from_dlpack(values)
            tree, obs = self._grown(tree, scores, values, jnp.int32(simulation))
        scores, values = evaluate(obs)
        return self._read(self._judged(tree, jnp.from_dlpack(scores), jnp.from_dlpack(values)))

    @cached_property
    def _planted(self) -> Any:  # noqa: ANN401  # a compiled function
        """Compiled: the trees of a batch of states, and their roots' observations."""

        def planted(states: S, key: jax.Array) -> tuple[_Keyed[S], jax.Array]:
            batch = jax.tree.leaves(states)[0].shape[0]
            keys = jax.vmap(lambda i: batched.stream(key, i))(jnp.arange(batch))
            noise = jax.vmap(self.gumbel)(keys) * self.noise
            trees = jax.vmap(self.plant)(states, noise)
            return _Keyed(trees, keys), jax.vmap(self.observation)(trees)

        return jax.jit(planted)

    @cached_property
    def _grown(self) -> Any:  # noqa: ANN401  # a compiled function
        """Compiled: judge a batch of leaves, run the next simulation, observe its leaves."""

        def grown(
            trees: _Keyed[S], scores: jax.Array, values: jax.Array, simulation: jax.Array
        ) -> tuple[_Keyed[S], jax.Array]:
            def one(tree: Tree[S], key: jax.Array, score: jax.Array, value: jax.Array) -> Any:  # noqa: ANN401
                tree = self.expand(tree, score.reshape(-1), value)
                tree = self.descend(tree, batched.stream(key, 1, simulation))
                return tree, self.observation(tree)

            tree, obs = jax.vmap(one)(trees.tree, trees.key, scores, values)
            return _Keyed(tree, trees.key), obs

        return jax.jit(grown, donate_argnums=0)

    @cached_property
    def _judged(self) -> Any:  # noqa: ANN401  # a compiled function
        """Compiled: judge the last batch of leaves."""

        def judged(trees: _Keyed[S], scores: jax.Array, values: jax.Array) -> Tree[S]:
            def one(tree: Tree[S], score: jax.Array, value: jax.Array) -> Tree[S]:
                return self.expand(tree, score.reshape(-1), value)

            return jax.vmap(one)(trees.tree, scores, values)

        return jax.jit(judged, donate_argnums=0)

    @cached_property
    def _read(self) -> Any:  # noqa: ANN401  # a compiled function
        """Compiled: what a batch of searches found."""
        return jax.jit(jax.vmap(self.found))

    # One decision.

    def gumbel(self, key: ArrayLike) -> jax.Array:
        """Draw the noise of one decision's root.

        Parameters
        ----------
        key : Array of uint32, shape (2,)
            The decision's key.

        Returns
        -------
        Array of float32, shape (n_actions,)
            Gumbel noise, one draw per local action, from stream 0 of the key.
        """
        wrapped = jax.random.wrap_key_data(batched.stream(key, 0))
        return jax.random.gumbel(wrapped, (self.game.n_actions,), jnp.float32)

    def plant(self, state: S, gumbel: ArrayLike) -> Tree[S]:
        """Return the tree of a decision before any simulation: its root, not yet judged.

        Parameters
        ----------
        state : S
            The position to decide in, not over.
        gumbel : Array of float, shape (n_actions,)
            The noise on the root's actions: `gumbel`'s, or zeros to search without noise.

        Returns
        -------
        Tree
            A tree whose leaf is the root, waiting for `expand`.
        """
        game, size = self.game, self.simulations + 1
        actions, outcomes = game.n_actions, game.n_outcomes
        legal = game.legal(state)
        return Tree(
            state=jax.tree.map(lambda x: jnp.zeros((size, *x.shape), x.dtype).at[0].set(x), state),
            mover=jnp.zeros(size, jnp.int32).at[0].set(game.player(state)),
            over=jnp.zeros(size, jnp.bool_),
            logits=jnp.zeros((size, actions), jnp.float32).at[0].set(_waiting(legal)),
            value=jnp.zeros(size, jnp.float32),
            n=jnp.zeros((size, actions), jnp.int32),
            w=jnp.zeros((size, actions), jnp.float32),
            child=jnp.full((size, actions), -1, jnp.int32),
            chances=jnp.zeros((size, outcomes), jnp.float32),
            reached=jnp.full((size, outcomes), -1, jnp.int32),
            seen=jnp.zeros((size, outcomes), jnp.int32),
            gumbel=jnp.asarray(gumbel, jnp.float32),
            nodes=jnp.int32(1),
            records=jnp.int32(0),
            leaf=jnp.int32(0),
            path=jnp.zeros((size, 4), jnp.int32),
            depth=jnp.int32(0),
            fresh=jnp.bool_(True),
        )

    def observation(self, tree: Tree[S]) -> jax.Array:
        """Return what the evaluator reads of a tree's leaf.

        Parameters
        ----------
        tree : Tree
            A tree after `plant` or `descend`.

        Returns
        -------
        Array of uint8, shape (n_planes, n_cells)
            The leaf's position as its player to move sees it, ``round(255 *
            game.observe(...))``. It means nothing where the leaf is the end of a game.
        """
        state = jax.tree.map(lambda x: x[tree.leaf], tree.state)
        planes = self.game.observe(state, tree.mover[tree.leaf])
        return jnp.rint(planes * 255).astype(jnp.uint8)

    def expand(self, tree: Tree[S], scores: ArrayLike, value: ArrayLike) -> Tree[S]:
        """Give a tree's leaf its prior and value, and back the value up to the root.

        Parameters
        ----------
        tree : Tree
            A tree after `plant` or `descend`.
        scores : Array of float, shape (n_verbs * n_cells,)
            The evaluator's scores of the leaf's `observation`, finite, in the frame of its
            player to move (`ArrayRules.framed` finds each local action's).
        value : Array of float, shape ()
            Its value for that player, in [-1, 1].

        Returns
        -------
        Tree
            The tree with the simulation that reached the leaf counted along its way. Where
            the leaf is the end of a game, its result is backed up, and ``scores`` and
            ``value`` are not read. Nothing is checked: scores that are not finite, or a
            value out of range, give a tree that means nothing.
        """
        leaf = tree.leaf
        state = jax.tree.map(lambda x: x[leaf], tree.state)
        legal = tree.logits[leaf] > -jnp.inf
        prior = jnp.asarray(scores, jnp.float32)[self.game.framed(state)]
        logits = jax.nn.log_softmax(jnp.where(legal, prior, -jnp.inf))
        sign = jnp.where(tree.mover[leaf] == 0, 1.0, -1.0)
        judged = (1.0 + sign * jnp.asarray(value, jnp.float32)) / 2.0
        value = jnp.where(tree.fresh, judged, tree.value[leaf])
        # Count the simulation at each step of its way. Rows one past the end are dropped.
        size = self.simulations + 1
        left, action, record, outcome = tree.path.T
        taken = jnp.arange(size) < tree.depth
        nodes = jnp.where(taken, left, size)
        rows = jnp.where(taken & (record >= 0), record, size)
        return tree._replace(
            logits=tree.logits.at[leaf].set(jnp.where(tree.fresh, logits, tree.logits[leaf])),
            value=tree.value.at[leaf].set(value),
            n=tree.n.at[nodes, action].add(1, mode="drop"),
            w=tree.w.at[nodes, action].add(value, mode="drop"),
            seen=tree.seen.at[rows, outcome].add(1, mode="drop"),
            depth=jnp.int32(0),
            fresh=jnp.bool_(False),
        )

    def descend(self, tree: Tree[S], key: ArrayLike) -> Tree[S]:
        """Run one simulation down a tree: to a position not yet in it, or the end of a game.

        From the root it takes, at each node, the action the search prefers there (`found`
        tells how), and where the dice decide, the outcome whose share of the visits falls
        furthest short of its chance. It stops at an action or an outcome not yet visited,
        makes the node of the position that follows (after any steps that are forced, see the
        module), and leaves it as the tree's leaf for `observation` and `expand`.

        Parameters
        ----------
        tree : Tree
            A tree whose last leaf `expand` has judged, with a simulation left to run.
        key : Array of uint32, shape (2,)
            The key of this simulation's draws: the dice of forced steps, and of what
            `ArrayRules.outcomes` does not tell apart.

        Returns
        -------
        Tree
            The tree with the new node, if any, as its leaf. Where the simulation reached the
            end of a game, that node is the leaf, new or not.
        """
        game, size = self.game, self.simulations + 1

        def down(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            node, depth, path = carry[0], carry[5], carry[6]
            action = self._choice(tree, node)
            target = tree.child[node, action]
            record = jnp.where(target >= size, target - size, -1)
            outcome = _owed(tree.chances[record], tree.seen[record])
            below = jnp.where(record >= 0, tree.reached[record, outcome], target)
            on = (below >= 0) & ~tree.over[below]
            path = path.at[depth].set(jnp.stack([node, action, record, outcome]))
            return jnp.where(on, below, node), action, record, outcome, below, depth + 1, path, on

        start = (jnp.int32(0), jnp.int32(0), jnp.int32(-1), jnp.int32(0), jnp.int32(-1))
        node, action, record, outcome, below, depth, path, _ = jax.lax.while_loop(
            lambda carry: carry[7], down, (*start, jnp.int32(0), tree.path, jnp.bool_(True))
        )

        # The position after the action: one of its outcomes, then any forced steps.
        before = jax.tree.map(lambda x: x[node], tree.state)
        ways, chances = game.outcomes(before, action, game.roll(batched.stream(key, 0)))
        rolled = (record < 0) & ((chances > 0).sum() > 1)  # the dice decide: a new record
        outcome = jnp.where(record >= 0, outcome, jnp.argmax(chances).astype(jnp.int32))
        after = jax.tree.map(lambda x: x[outcome], ways)

        def forced(carry: tuple[S, jax.Array, jax.Array]) -> tuple[S, jax.Array, jax.Array]:
            state, legal, count = carry
            dice = game.roll(batched.stream(key, 1, count))
            state = game.apply(state, jnp.argmax(legal), dice)
            return state, game.legal(state), count + 1

        after, legal, _ = jax.lax.while_loop(
            lambda carry: carry[1].sum() == 1, forced, (after, game.legal(after), jnp.int32(0))
        )
        over, winner = game.over(after), game.winner(after)
        result = jnp.where(winner == 0, 1.0, jnp.where(winner < 0, 0.5, 0.0))

        # Its node, where the simulation left the tree. Rows one past the end are dropped.
        made = below < 0
        new = jnp.where(made, tree.nodes, size)
        record = jnp.where(rolled, tree.records, record)
        row = jnp.where(made & (record >= 0), record, size)
        return tree._replace(
            state=jax.tree.map(lambda all_, x: all_.at[new].set(x, mode="drop"), tree.state, after),
            mover=tree.mover.at[new].set(jnp.where(over, 0, game.player(after)), mode="drop"),
            over=tree.over.at[new].set(over, mode="drop"),
            logits=tree.logits.at[new].set(_waiting(legal), mode="drop"),
            value=tree.value.at[new].set(result, mode="drop"),
            child=tree.child.at[jnp.where(made, node, size), action].set(
                jnp.where(record >= 0, size + record, new), mode="drop"
            ),
            chances=tree.chances.at[jnp.where(made & rolled, record, size)].set(
                chances, mode="drop"
            ),
            reached=tree.reached.at[row, outcome].set(tree.nodes, mode="drop"),
            nodes=tree.nodes + made,
            records=tree.records + (made & rolled),
            leaf=jnp.where(made, tree.nodes, below),
            path=path.at[depth - 1, 2:].set(jnp.stack([record, outcome])),
            depth=depth,
            fresh=made & ~over,
        )

    def found(self, tree: Tree[S]) -> Found:
        """Return the action to play and what a search found at the root.

        Each simulation chose, at the root, among the actions sequential halving still
        considers, the one with the highest noise plus logit plus scaled completed value;
        below it, the action whose share of the visits falls furthest short of the improved
        policy (`agents.zero.AlphaZeroPlayer`). The action played is the best by that score
        among those the last round still considered (`agents.zero.finalists`).

        Parameters
        ----------
        tree : Tree
            A tree whose last leaf `expand` has judged.

        Returns
        -------
        Found
            The action, the improved policy, the visits and the values at the root.
        """
        logits, n, w = tree.logits[0], tree.n[0], tree.w[0]
        legal = logits > -jnp.inf
        sigma = self._sigma(tree, jnp.int32(0), rescale=True)
        score = tree.gumbel + logits + sigma
        most = jnp.minimum(self.considered, legal.sum())
        _, last, cut_short = (jnp.asarray(table) for table in self._schedule)
        last, cut_short = last[most], cut_short[most]
        finalists = legal & jnp.where(cut_short, n >= last, n > last)
        return Found(
            action=jnp.argmax(jnp.where(finalists, score, -jnp.inf)).astype(jnp.int32),
            policy=jax.nn.softmax(jnp.where(legal, logits + sigma, -jnp.inf)),
            visits=n,
            q=jnp.where(n > 0, w / jnp.maximum(n, 1), jnp.nan),
            value=(tree.value[0] + w.sum()) / (1 + n.sum()),
        )

    def _choice(self, tree: Tree[S], node: jax.Array) -> jax.Array:
        """Return the action a simulation takes at a node: int32, shape ()."""
        logits, n = tree.logits[node], tree.n[node]
        legal = logits > -jnp.inf
        # At the root, sequential halving: the best among those with the visits now due.
        most = jnp.minimum(self.considered, legal.sum())
        done = jnp.minimum(tree.n[0].sum(), self.simulations - 1)
        due = jnp.asarray(self._schedule[0])[most, done]
        score = jnp.maximum(-1e9, tree.gumbel + logits + self._sigma(tree, node, rescale=True))
        at_root = jnp.argmax(jnp.where(legal & (n == due), score, -jnp.inf))
        # Below it, the action whose visits fall furthest short of the improved policy.
        improved = jax.nn.softmax(logits + self._sigma(tree, node, rescale=False))
        short = improved - n / (1 + n.sum())
        below = jnp.argmax(jnp.where(legal, short, -jnp.inf))
        return jnp.where(node == 0, at_root, below).astype(jnp.int32)

    def _sigma(self, tree: Tree[S], node: jax.Array, *, rescale: bool) -> jax.Array:
        """Return the scaled completed values of a node's actions: `agents.zero.sigma`.

        For the node's player to move: float32, shape (A,). What it gives illegal actions
        means nothing.
        """
        logits, n = tree.logits[node], tree.n[node]
        legal = logits > -jnp.inf
        ours = tree.mover[node] == 0
        q = tree.w[node] / jnp.maximum(n, 1)
        q, raw = (
            jnp.where(ours, q, 1.0 - q),
            jnp.where(ours, tree.value[node], 1 - tree.value[node]),
        )
        visited = n > 0
        prior = jnp.where(visited, jnp.maximum(jnp.exp(logits), jnp.finfo(jnp.float32).tiny), 0.0)
        weighted = (prior * q).sum() / jnp.maximum(prior.sum(), jnp.finfo(jnp.float32).tiny)
        total = n.sum()
        raw = jnp.where(visited.any(), (raw + total * weighted) / (total + 1), raw)
        completed = jnp.where(visited, q, raw)
        if rescale:
            low = jnp.where(legal, completed, jnp.inf).min()
            high = jnp.where(legal, completed, -jnp.inf).max()
            completed = (completed - low) / jnp.maximum(high - low, 1e-8)
        return (self.c_visit + n.max()) * self.c_scale * completed

    @cached_property
    def _schedule(self) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.bool_]]:
        """Sequential halving's schedule for each number of actions considered.

        Row ``m`` is for ``m`` actions, up to ``considered``: the visits an action must have
        to be chosen at each simulation, int32 (considered + 1, simulations); the visits the
        last simulation chose among, int32 (considered + 1,); and whether the simulations
        ended inside a round, bool (considered + 1,).
        """
        sims = self.simulations
        due = np.zeros((self.considered + 1, sims), dtype=np.int32)
        last = np.zeros(self.considered + 1, dtype=np.int32)
        cut_short = np.zeros(self.considered + 1, dtype=np.bool_)
        for most in range(1, self.considered + 1):
            due[most] = considered_visits(most, sims)
            rounds = _rounds(most, sims)
            last[most] = rounds[sims - 1]
            cut_short[most] = len(rounds) > sims and rounds[sims] == last[most]
        return due, last, cut_short  # NumPy's: made into constants wherever they are traced


class _Keyed[S](NamedTuple):
    """The trees of a batch of decisions, and each decision's key."""

    tree: Tree[S]
    key: jax.Array


def _waiting(legal: jax.Array) -> jax.Array:
    """Return the logits of a node not yet judged: 0 where legal, -inf elsewhere."""
    return jnp.where(legal, 0.0, -jnp.inf).astype(jnp.float32)


def _owed(chances: jax.Array, seen: jax.Array) -> jax.Array:
    """Return the outcome whose share of the visits falls furthest short of its chance.

    Among outcomes that can happen, the lowest on a tie: int32, shape ().
    """
    short = chances - seen / (1 + seen.sum())
    return jnp.argmax(jnp.where(chances > 0, short, -jnp.inf)).astype(jnp.int32)
