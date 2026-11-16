from dataclasses import dataclass

import jax
import numpy as np
import pytest
import torch

from batched_demo import Wager
from gridzero import batched
from gridzero.agents.net import PolicyNet
from gridzero.agents.zero_arrays import ArraySearch
from gridzero.agents.zero_arrays_train import ArrayZero, _Buffer, _Decided, _Ended, _Games, train
from gridzero.core import LatticeSpace

GAME = Wager(faces=14, waits=2)
QUICK = ArrayZero(
    games=8,
    simulations=4,
    considered=3,
    buffer=1_000,
    every=4,
    steps=2,
    batch_size=16,
    warmup=0,
    rounds=2,
)
"""Settings for generations of a fraction of a second, once compiled."""


@dataclass(frozen=True)
class WagerShape:
    """What a network reads of a game, for `Wager`: one cell, one plane, a verb per action."""

    space = LatticeSpace((1, 1), periodic=False)
    planes = ("ply",)
    n_verbs = 3


def network():
    torch.manual_seed(0)
    return PolicyNet(WagerShape(), channels=4, blocks=1)  # ty: ignore[invalid-argument-type]  # all a network reads of a game


def start(seed):
    """Every game of `Wager` starts alike: its dice are its plies'."""
    return jax.tree.map(np.asarray, GAME.start())


def test_a_generation_plays_keeps_and_trains():
    net = network()
    before = [p.detach().clone() for p in net.parameters()]
    [generation] = list(train(net, GAME, start, QUICK, generations=1))
    steps = QUICK.rounds * QUICK.every
    assert generation.decisions == QUICK.games * steps
    assert 0 < generation.kept < generation.decisions  # a wait while the other gambles is forced
    assert generation.games == sum(generation.wins) + generation.draws > 0
    assert generation.trained == QUICK.rounds * QUICK.steps
    assert generation.buffer == generation.kept
    assert 0 < generation.ready <= generation.buffer
    assert np.isfinite([generation.policy_loss, generation.value_loss]).all()
    assert generation.next_seed == 2 * QUICK.games + generation.games  # a start ahead for each
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_the_same_seed_trains_the_same_network():
    def trained(seed):
        net = network()
        list(train(net, GAME, start, QUICK, generations=2, seed=seed))
        return [p.detach() for p in net.parameters()]

    first, again, other = trained(3), trained(3), trained(4)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_the_value_trains_toward_the_share_of_the_search_s_value_set():
    def trained(share):
        net = network()
        settings = ArrayZero(**{**QUICK.__dict__, "root_value": share})
        list(train(net, GAME, start, settings, generations=1))
        return [p.detach() for p in net.parameters()]

    results, roots = trained(0.0), trained(1.0)
    assert any(not torch.equal(a, b) for a, b in zip(results, roots, strict=True))


class Even:
    """An evaluator with no opinion: even scores, and every position worth half to its mover."""

    def __call__(self, obs):
        return torch.zeros(obs.shape[0], 3), torch.zeros(obs.shape[0])


def test_the_search_s_value_is_kept_for_its_mover():
    # Player 0 decides at ply 0 and player 1 at ply 3. To each, a draw is worth 0.5 and a
    # gamble 0.7 (14 faces of 20), so a search that tries both values its position above 0.5:
    # above 0 on [-1, 1], for player 1 as for player 0.
    plies = {0: 0, 1: GAME.waits + 1}
    games = _Games(GAME, lambda seed: jax.tree.map(np.asarray, GAME.start(plies[seed % 2])), 2, 0)
    search = ArraySearch(GAME, simulations=8, considered=2)
    decided = games.decide(search, Even(), batched.root(0))
    np.testing.assert_array_equal(decided.mover, [0, 1])
    assert (decided.root > 0).all()
    assert (decided.root < 0.4 + 1e-6).all()  # no more than the gamble's own worth


def test_no_training_before_the_warmup_is_done():
    settings = ArrayZero(**{**QUICK.__dict__, "warmup": 10**6})
    [generation] = list(train(network(), GAME, start, settings, generations=1))
    assert generation.trained == 0
    assert np.isnan([generation.policy_loss, generation.value_loss]).all()


def test_a_time_box_stops_after_the_first_generation():
    generations = list(train(network(), GAME, start, QUICK, generations=3, seconds=0.0))
    assert [g.generation for g in generations] == [1]


def test_a_game_that_ends_is_replaced_by_the_next_start_under_its_seed():
    def start_of(seed):  # even seeds start at player 0's decision, odd ones at player 1's
        return jax.tree.map(np.asarray, GAME.start(0 if seed % 2 == 0 else GAME.waits + 1))

    games = _Games(GAME, start_of, 8, 0)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(12):
        fresh = games.ply == 0
        wanted = np.where(games.seed % 2 == 0, 0, GAME.waits + 1)
        np.testing.assert_array_equal(np.asarray(games.states.ply)[fresh], wanted[fresh])
        seen |= set(games.seed.tolist())
        legal = np.asarray(jax.vmap(GAME.legal)(games.states))
        action = [rng.choice(np.flatnonzero(m)) for m in legal]
        games.play(jax.numpy.asarray(action, dtype=jax.numpy.int32))
    assert len(seen) > 2 * 8  # games ended, and new seeds came in


def decided(movers, legal, framed=None, policy=None):
    """A step's decisions in ``len(movers)`` games of one plane and one cell."""
    n = len(movers)
    legal = np.array(legal, dtype=bool)
    return _Decided(
        action=None,  # ty: ignore[invalid-argument-type]  # the buffer never reads it
        obs=np.zeros((n, 1, 1), dtype=np.uint8),
        legal=legal,
        framed=np.tile(np.arange(legal.shape[1], dtype=np.int32), (n, 1))
        if framed is None
        else np.array(framed, dtype=np.int32),
        policy=np.full(legal.shape, 0.5, np.float32)
        if policy is None
        else np.array(policy, dtype=np.float32),
        root=np.linspace(-0.5, 0.5, n).astype(np.float32),
        mover=np.array(movers, dtype=np.int32),
        seed=np.arange(n, dtype=np.int64),
        ply=np.zeros(n, dtype=np.int64),
    )


def ended(slots, winners):
    return _Ended(np.array(slots), np.array(winners, dtype=np.int32), np.ones(len(slots), int))


def test_each_decision_gets_its_game_s_outcome_for_its_mover_and_a_draw_none():
    buffer = _Buffer(10)
    buffer.add(decided([0, 1, 0], [[1, 1], [1, 1], [1, 1]]))
    buffer.add(decided([1, 0, 1], [[1, 1], [1, 1], [1, 1]]))
    assert buffer.ready().size == 0
    buffer.end(ended([0, 1], [1, -1]))  # game 0 won by player 1, game 1 drawn; game 2 goes on
    np.testing.assert_array_equal(buffer.ready(), [0, 1, 3, 4])
    np.testing.assert_array_equal(buffer.arrays["outcome"][[0, 3, 1, 4]], [-1.0, 1.0, 0.0, 0.0])


def test_forced_decisions_are_not_kept():
    buffer = _Buffer(10)
    buffer.add(decided([0, 1, 0], [[1, 1], [0, 1], [1, 0]]))
    assert buffer.size == 1
    np.testing.assert_array_equal(buffer.arrays["game"][:1], [0])


def test_a_decision_overwritten_while_its_game_goes_on_gets_no_outcome():
    buffer = _Buffer(2)
    buffer.add(decided([0, 0], [[1, 1], [1, 1]]))  # rows 0 and 1: games 0 and 1
    buffer.add(decided([1, 1], [[0, 1], [1, 1]]))  # game 1's next decision takes row 0
    buffer.end(ended([0], [0]))  # game 0 ends: its decision is gone, row 0 is game 1's
    assert buffer.ready().size == 0
    buffer.end(ended([1], [1]))
    np.testing.assert_array_equal(buffer.ready(), [0, 1])
    np.testing.assert_array_equal(buffer.arrays["outcome"][:2], [1.0, -1.0])  # by mover


def test_decisions_become_experience_where_the_network_scores_them():
    buffer = _Buffer(4)
    # Local actions 0 and 2 legal, scored at places 2 and 1 of the network's 3.
    buffer.add(decided([0], [[1, 0, 1]], framed=[[2, 0, 1]], policy=[[0.25, 0.0, 0.75]]))
    experience = buffer.experience(np.array([0]), n_verbs=3)
    board = np.unpackbits(experience.legal, axis=1)[:, :3]
    np.testing.assert_array_equal(board, [[0, 1, 1]])
    np.testing.assert_array_equal(
        experience.policy, np.array([0.75, 0.25], dtype=np.float32)
    )  # in place order
    np.testing.assert_array_equal(experience.start, [0, 2])


def test_two_legal_actions_scored_at_one_place_raise():
    buffer = _Buffer(4)
    buffer.add(decided([0], [[1, 1, 0]], framed=[[1, 1, 0]]))
    with pytest.raises(ValueError, match="one place"):
        buffer.experience(np.array([0]), n_verbs=3)


@pytest.mark.parametrize(
    "name", ["games", "simulations", "considered", "buffer", "every", "batch_size", "rounds"]
)
def test_counts_must_be_positive_integers(name, not_an_integer):
    with pytest.raises((TypeError, ValueError), match=name):
        ArrayZero(**{name: not_an_integer})
    with pytest.raises(ValueError, match=name):
        ArrayZero(**{name: 0})


def test_rates_and_shares_must_be_in_range():
    with pytest.raises(ValueError, match="lr"):
        ArrayZero(lr=0.0)
    with pytest.raises(ValueError, match="root_value"):
        ArrayZero(root_value=1.5)
    with pytest.raises(ValueError, match="warmup"):
        ArrayZero(warmup=-1)
