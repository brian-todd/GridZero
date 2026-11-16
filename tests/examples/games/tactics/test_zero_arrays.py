"""The search in arrays against `agents.zero` on Tactics: the same trees, given the same judge."""

import dataclasses
import functools
from typing import Any

import jax
import numpy as np
import pytest
import torch

from examples.games.tactics import GreedyPlayer, Verb, small_rules
from examples.games.tactics.arrays import Dice, State, TacticsArrays
from gridzero import batched
from gridzero.agents.net import NetArrayEvaluator, NetEvaluator, PolicyNet
from gridzero.agents.zero import AlphaZeroPlayer, _Node, _Search
from gridzero.agents.zero_arrays import ArraySearch

OPEN = dataclasses.replace(small_rules(), fog=False)


@functools.cache
def arrays_of(fog: bool) -> TacticsArrays:
    """The array rules of the small game, with or without fog, for searches of beliefs."""
    return TacticsArrays(small_rules() if fog else OPEN, hidden=False)


@functools.cache
def net_of(fog: bool) -> PolicyNet:
    """A small network of random weights: a judge with an opinion on every position."""
    torch.manual_seed(3)
    net = PolicyNet(arrays_of(fog).game, channels=8, blocks=1)
    with torch.no_grad():  # a new network's values are 0: give them a spread
        net.value_out.weight.mul_(30.0)
        net.policy_out.weight.mul_(4.0)
        net.policy_bias.bias[Verb.ATTACK] += 4.0  # and a liking for what rolls dice
    return net


@functools.cache
def searching(fog: bool, simulations: int, noise: bool) -> ArraySearch[State]:
    """A search in arrays, compiled once a session for each setting."""
    return ArraySearch(arrays_of(fog), simulations=simulations, noise=noise)


def positions(game, plies, seeds):
    """Positions of games greedy played for a number of plies: what the mover is shown.

    From ply 20 on, a game is played further until its mover may attack, for searches that
    weigh dice at once.
    """
    greedy = GreedyPlayer()
    out = []
    for seed, count in zip(seeds, plies, strict=True):
        rng = np.random.default_rng(seed)
        position = game.start(seed)
        while position.ply < count or (count >= 20 and not shown(game, position)[1].any()):
            me = game.player(position)
            action = greedy(game, me, game.view(position, me), game.legal(position), rng)
            position = game.step(position, action)
        assert not position.over
        out.append(shown(game, position)[0])
    return out


def shown(game, position):
    """What a position's mover is shown, and the attacks legal in it: bool, (n_cells,)."""
    view = game.view(position, game.player(position))
    return view, game.legal(view)[Verb.ATTACK]


def stacked(arrays, positions_):
    """Positions as one batch of states."""
    return jax.tree.map(lambda *xs: np.stack(xs), *(arrays.to_arrays(p) for p in positions_))


class Weighed(_Search[Any, Any]):
    """`agents.zero`'s search with the dice weighed as the search in arrays weighs them.

    Its tree, its choices and its values are `agents.zero`'s. Only what a step that rolls
    leads to differs: every outcome the array rules list, visited in proportion to its
    chance, instead of a fresh roll each visit.
    """

    def __init__(self, arrays, player, game, position, rng):
        super().__init__(player, game, position, rng)
        self.arrays = arrays
        self.outcomes = jax.jit(arrays.outcomes)
        self.seen: dict[tuple[int, int], np.ndarray] = {}
        self.rolled = 0
        idle = np.arange(arrays.n_units, dtype=np.int32)
        self.dice = Dice(np.ones(arrays.n_rolls, np.int32), np.int32(0), idle)  # ty: ignore[invalid-argument-type]  # NumPy arrays stand in for JAX's

    def _child(self, node, i):
        child = node.children[i]
        if isinstance(child, _Node):
            return child
        arrays, position = self.arrays, node.position
        action = divmod(int(node.actions[i]), self.n_cells)
        local = arrays.local(position, action)
        ways, chances = self.outcomes(arrays.to_arrays(position), local, self.dice)
        chances = np.asarray(chances)
        if np.count_nonzero(chances) == 1:
            made = _Node(arrays.to_position(jax.tree.map(lambda x: x[chances.argmax()], ways)))
            node.children[i] = made
            return made
        outcomes = {} if child is None else child
        node.children[i] = outcomes
        seen = self.seen.setdefault((id(node), i), np.zeros(chances.size, dtype=np.float32))
        short = chances - seen / np.float32(1 + seen.sum())  # float32, as the arrays compute it
        which = int(np.argmax(np.where(chances > 0, short, -np.inf)))
        seen[which] += 1
        self.rolled += 1
        if which not in outcomes:
            outcomes[which] = _Node(arrays.to_position(jax.tree.map(lambda x: x[which], ways)))
        return outcomes[which]


def reference(arrays, player, position, gumbel=None):
    """Search one decision with `Weighed`; return its result and the visits to chance edges."""
    game = arrays.game
    search = Weighed(arrays, player, game, position, np.random.default_rng(0))
    while (leaf := search.leaf()) is not None:
        logits, values = player.evaluator(game, [leaf.node.position], [leaf.legal])
        search.expand(leaf, logits[0], float(values[0]))
        if not leaf.path and gumbel is not None:  # the root: the noise the arrays drew
            on_board = [divmod(int(a), search.n_cells) for a in search.root.actions]
            search.gumbel = gumbel[[arrays.local(position, a) for a in on_board]].astype(float)
    return search.result(), search.rolled


def compare(found, i, arrays, position, result):
    """Assert that decision ``i`` of the arrays' search is ``result``, `agents.zero`'s."""
    legal = np.argwhere(arrays.game.legal(position))
    local = np.array([arrays.local(position, (int(v), int(c))) for v, c in legal])
    verbs, cells = legal.T
    visits = np.asarray(found.visits[i])
    assert visits[local].tolist() == result.visits[verbs, cells].tolist()
    assert visits.sum() == visits[local].sum()
    assert tuple(arrays.action(position, int(found.action[i]))) == tuple(result.action)
    policy = np.asarray(found.policy[i])
    np.testing.assert_allclose(policy[local], result.policy[verbs, cells], atol=2e-5)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(np.asarray(found.q[i])[local], result.q[verbs, cells], atol=2e-5)
    assert float(found.value[i]) == pytest.approx(result.value, abs=2e-5)


def check_equal(fog, simulations, noise, plies, seeds):
    """Search positions both ways and compare; return how often `agents.zero` weighed dice."""
    arrays, net = arrays_of(fog), net_of(fog)
    search = searching(fog, simulations, noise)
    roots = positions(arrays.game, plies, seeds)
    key = batched.root(11)
    found = search.search(stacked(arrays, roots), NetArrayEvaluator(net), key)
    player = AlphaZeroPlayer(NetEvaluator(net), simulations=simulations, noise=noise)
    rolled = 0
    for i, position in enumerate(roots):
        gumbel = np.asarray(search.gumbel(batched.stream(key, i))) if noise else None
        result, count = reference(arrays, player, position, gumbel)
        compare(found, i, arrays, position, result)
        rolled += count
    return rolled


def test_the_search_in_arrays_is_zeros_with_the_dice_weighed():
    # Battles of every stage, where attacks roll, and deployments, where nothing does.
    plies = [0, 3, 9, 20, 30, 45, 60, 80, 100, 120]
    assert check_equal(False, 32, False, plies, range(len(plies))) > 50


def test_the_search_in_arrays_is_zeros_under_gumbel_noise():
    # 37 simulations end inside a round of sequential halving: the actions it had yet to
    # visit are finalists too.
    plies = [2, 25, 50, 75, 110]
    assert check_equal(False, 37, True, plies, range(20, 25)) > 20


def test_the_search_in_arrays_is_zeros_on_a_belief_under_fog():
    plies = [40, 90]
    assert check_equal(True, 32, False, plies, range(30, 32)) > 10


@pytest.mark.slow
def test_the_search_in_arrays_is_zeros_at_more_simulations():
    plies = [5, 22, 38, 52, 66, 84, 96, 130]
    assert check_equal(False, 200, True, plies, range(40, 48)) > 500
