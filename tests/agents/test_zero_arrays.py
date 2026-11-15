"""The search in arrays on a fake game whose values are known (`batched_demo.Wager`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_demo import Wager, WagerState
from gridzero import batched
from gridzero.agents.zero import considered_visits, finalists, sigma
from gridzero.agents.zero_arrays import ArraySearch

GAME = Wager(faces=14, waits=2)
WINS = 0.7  # a gamble's chance: 14 faces of 20


@functools.cache
def searching(simulations: int, noise: bool = False) -> ArraySearch[WagerState]:
    """A search of `GAME`, compiled once a session for each setting."""
    return ArraySearch(GAME, simulations=simulations, noise=noise)


class Judge:
    """An evaluator with the same scores everywhere, and a value for positions past the first.

    The first position is worth 0 to its player to move, even chances. It keeps the
    observations it was sent in ``seen``.
    """

    def __init__(self, scores=(0.0, 0.0, 0.0), later=0.0):
        self.scores = jnp.asarray(scores, jnp.float32)
        self.later = later
        self.seen: list[np.ndarray] = []

    def __call__(self, obs):
        self.seen.append(np.asarray(obs))
        rows = jnp.broadcast_to(self.scores, (obs.shape[0], 3))
        return rows, jnp.where(obs[:, 0, 0] > 0, self.later, 0.0).astype(jnp.float32)


def batch(*plies):
    """Positions at some plies, as one batch."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *(GAME.start(ply) for ply in plies))


def owed(visits, wins=WINS):
    """How many of some visits to a gamble go to the outcome where it is won.

    Each visit goes to the outcome whose share falls furthest short of its chance, the loss
    on a tie, in float32 as the search computes it.
    """
    chances = np.array([np.float32(1) - np.float32(wins), np.float32(wins)])
    seen = np.zeros(2, dtype=np.float32)
    for _ in range(visits):
        seen[np.argmax(chances - seen / np.float32(1 + seen.sum()))] += 1
    return int(seen[1])


def test_a_gamble_is_worth_its_chance_and_a_draw_a_half():
    found = searching(64).search(batch(0), Judge(), batched.root(0))
    visits = np.asarray(found.visits[0])
    assert visits.sum() == 64
    assert visits[Wager.GAMBLE] >= 16
    won = owed(int(visits[Wager.GAMBLE]))
    assert float(found.q[0, Wager.GAMBLE]) == pytest.approx(won / visits[Wager.GAMBLE], abs=1e-6)
    assert abs(float(found.q[0, Wager.GAMBLE]) - WINS) <= 1 / visits[Wager.GAMBLE]
    assert float(found.q[0, Wager.SETTLE]) == 0.5
    assert int(found.action[0]) == Wager.GAMBLE


def test_visits_to_outcomes_follow_their_chances_from_the_first():
    assert [owed(k) for k in range(1, 5)] == [1, 1, 2, 3]  # won, lost, won, won
    assert all(abs(owed(k) - WINS * k) <= 1 for k in range(1, 300))
    assert owed(1, wins=0.25) == 0  # one visit sees the likelier outcome
    assert owed(2, wins=0.5) == 1  # a tie goes to the first outcome, then to the other


def test_the_other_players_gamble_is_worth_what_it_leaves():
    # Player 1 to move at the last decision: its gamble wins 0.7, so player 0 keeps 0.3.
    found = searching(64).search(batch(3), Judge(), batched.root(0))
    visits = int(found.visits[0, Wager.GAMBLE])
    assert float(found.q[0, Wager.GAMBLE]) == pytest.approx(1 - owed(visits) / visits, abs=1e-6)
    assert int(found.action[0]) == Wager.GAMBLE
    assert not found.visits[0, Wager.WAIT]


def test_a_value_is_its_movers():
    # Player 1 is sure to win after the wait: the wait is worth nothing to player 0.
    search = searching(64)
    tree = search.plant(GAME.start(0), jnp.zeros(3))
    tree = search.expand(tree, jnp.array([0.0, 0.0, 9.0]), 0.0)
    tree = search.expand(search.descend(tree, batched.root(1)), jnp.zeros(3), 1.0)
    assert float(tree.w[0, Wager.WAIT]) == 0.0
    assert int(tree.n[0, Wager.WAIT]) == 1


def test_forced_steps_are_taken_without_a_node_or_a_judgement():
    search = searching(64)
    tree = search.plant(GAME.start(0), jnp.zeros(3))
    tree = search.expand(tree, jnp.array([0.0, 0.0, 9.0]), 0.0)  # the prior likes to wait
    tree = search.descend(tree, batched.root(1))
    assert int(tree.nodes) == 2
    assert int(tree.child[0, Wager.WAIT]) == 1
    assert int(tree.state.ply[1]) == 3  # past the two plies where player 1 can only wait
    assert bool(tree.fresh)
    assert int(tree.mover[1]) == 1
    assert np.asarray(tree.logits[1] > -np.inf).tolist() == [True, True, False]
    assert float(search.observation(tree)[0, 0]) == 255


def test_the_end_of_a_game_is_a_node_judged_by_its_result():
    search = searching(64)
    tree = search.plant(GAME.start(0), jnp.zeros(3))
    tree = search.expand(tree, jnp.array([9.0, 0.0, 0.0]), 0.0)  # the prior likes to settle
    tree = search.descend(tree, batched.root(1))
    assert int(tree.child[0, Wager.SETTLE]) == 1
    assert bool(tree.over[1])
    assert not bool(tree.fresh)
    tree = search.expand(tree, jnp.full(3, jnp.nan), jnp.nan)  # not read
    assert float(tree.w[0, Wager.SETTLE]) == 0.5
    assert np.isfinite(np.asarray(tree.logits[0])).all()


def test_a_gamble_is_a_record_of_its_outcomes():
    search = searching(64)
    tree = search.plant(GAME.start(0), jnp.zeros(3))
    tree = search.expand(tree, jnp.array([0.0, 9.0, 0.0]), 0.0)  # the prior likes to gamble
    for simulation in range(4):
        tree = search.descend(tree, batched.root(simulation))
        tree = search.expand(tree, jnp.zeros(3), 0.0)
    assert int(tree.child[0, Wager.GAMBLE]) == 65  # the first record, past the 65 nodes
    assert int(tree.records) == 1
    np.testing.assert_allclose(np.asarray(tree.chances[0]), [0.3, 0.7], atol=1e-6)
    gambles = int(tree.n[0, Wager.GAMBLE])
    assert gambles >= 2
    assert np.asarray(tree.seen[0]).tolist() == [gambles - owed(gambles), owed(gambles)]
    lost, won = (int(node) for node in tree.reached[0])
    assert np.asarray(tree.state.winner)[[lost, won]].tolist() == [1, 0]
    assert float(tree.w[0, Wager.GAMBLE]) == owed(gambles)
    assert int(tree.nodes) <= 5


def test_the_root_follows_sequential_halving():
    search = searching(24)
    found = search.search(batch(0), Judge(), batched.root(0))
    schedule = considered_visits(3, 24)
    most = max(schedule) + 1
    assert sorted(np.asarray(found.visits[0]).tolist())[-1] == most
    assert int(found.visits[0].sum()) == 24


def test_the_action_played_is_the_best_finalist_when_a_round_is_cut_short():
    # 63 simulations end inside a round: the action it had yet to visit is a finalist too.
    search = searching(63, noise=True)
    key = batched.root(9)
    found = search.search(batch(*[0] * 64), Judge(), key)
    told_apart = 0
    for i in range(64):
        n = np.asarray(found.visits[i]).astype(np.int64)
        q = np.nan_to_num(np.asarray(found.q[i]).astype(np.float64))
        logits = np.full(3, -np.log(3.0))
        score = np.asarray(search.gumbel(batched.stream(key, i))) + logits
        score = score + sigma(q, 0.5, n, logits, 50.0, 0.03, rescale=True)
        final = finalists(n, 16, 63)
        assert int(found.action[i]) == int(np.argmax(np.where(final, score, -np.inf)))
        told_apart += int(found.action[i]) != int(np.argmax(np.where(n == n.max(), score, -np.inf)))
    assert told_apart > 0  # some play an action with one visit less than the most


def test_one_legal_action_at_the_root_takes_every_simulation():
    found = searching(64).search(batch(1), Judge(), batched.root(0))
    assert np.asarray(found.visits[0]).tolist() == [0, 0, 64]
    assert int(found.action[0]) == Wager.WAIT
    assert np.asarray(found.policy[0]).tolist() == [0.0, 0.0, 1.0]
    assert np.isnan(np.asarray(found.q[0])).tolist() == [True, True, False]


def test_the_evaluator_is_asked_once_for_the_root_and_once_a_simulation():
    evaluate = Judge()
    searching(64).search(batch(0, 3, 0), evaluate, batched.root(0))
    assert len(evaluate.seen) == 65
    assert all(obs.shape == (3, 1, 1) and obs.dtype == np.uint8 for obs in evaluate.seen)
    assert evaluate.seen[0][:, 0, 0].tolist() == [0, 255, 0]


def test_a_decision_is_searched_the_same_whatever_is_beside_it():
    search = searching(64, noise=True)
    key = batched.root(5)
    alone = search.search(batch(0), Judge(), key)
    beside = search.search(batch(0, 3, 0, 0), Judge(), key)
    assert int(alone.action[0]) == int(beside.action[0])
    np.testing.assert_array_equal(np.asarray(alone.visits[0]), np.asarray(beside.visits[0]))
    for one, many in zip(alone, beside, strict=True):  # floats: to float32's precision
        np.testing.assert_allclose(np.asarray(one[0]), np.asarray(many[0]), rtol=1e-6)
    # The others drew their own noise: the same position, searched another way.
    visits = np.asarray(beside.visits)
    assert visits[2].tolist() != visits[0].tolist() or visits[3].tolist() != visits[0].tolist()


def test_noise_is_the_keys():
    search = searching(64, noise=True)
    states = batch(*[0] * 16)
    first = search.search(states, Judge(), batched.root(5))
    again = search.search(states, Judge(), batched.root(5))
    other = search.search(states, Judge(), batched.root(6))
    np.testing.assert_array_equal(np.asarray(first.visits), np.asarray(again.visits))
    assert np.asarray(first.visits).tolist() != np.asarray(other.visits).tolist()
    assert len({tuple(row) for row in np.asarray(first.visits).tolist()}) > 1
    quiet = searching(64).search(states, Judge(), batched.root(5))
    assert len({tuple(row) for row in np.asarray(quiet.visits).tolist()}) == 1


def test_gumbel_noise_is_a_draw_per_action():
    search = searching(64)
    noise = np.asarray(jax.vmap(search.gumbel)(jax.vmap(batched.root)(jnp.zeros((2000, 2), int))))
    assert noise.shape == (2000, 3)
    draws = np.asarray(
        jax.vmap(lambda i: search.gumbel(batched.stream(batched.root(0), i)))(jnp.arange(4000))
    )
    assert draws.mean() == pytest.approx(0.5772, abs=0.03)  # Euler's constant
    assert draws.var() == pytest.approx(np.pi**2 / 6, abs=0.1)
    assert np.unique(noise, axis=0).shape[0] == 1  # the same key, the same noise


def test_bad_settings_raise(not_an_integer):
    with pytest.raises(TypeError, match="simulations"):
        ArraySearch(GAME, simulations=not_an_integer)
    with pytest.raises(TypeError, match="considered"):
        ArraySearch(GAME, considered=not_an_integer)


def test_settings_out_of_range_raise():
    with pytest.raises(ValueError, match="simulations"):
        ArraySearch(GAME, simulations=0)
    with pytest.raises(ValueError, match="considered"):
        ArraySearch(GAME, considered=0)
    with pytest.raises(TypeError, match="noise"):
        ArraySearch(GAME, noise=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="c_visit"):
        ArraySearch(GAME, c_visit=-1.0)
    with pytest.raises(ValueError, match="c_scale"):
        ArraySearch(GAME, c_scale=-0.1)
