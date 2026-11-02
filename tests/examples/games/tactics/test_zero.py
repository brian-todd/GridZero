import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from examples.games.tactics import (
    GreedyPlayer,
    TacticsPosition,
    Verb,
    board_space,
    homes,
)
from gridzero.agents.zero import (
    AlphaZeroPlayer,
    Experience,
    considered_visits,
    self_play,
)
from gridzero.game import play
from tactics_demo import CAPTAIN, FOOT, SMALL, TINY, mirrored, piece, play_random


@dataclasses.dataclass
class FakeEvaluator:
    """An evaluator with priors from ``favor`` (uniform by default) and values from ``value``.

    It checks what the search asks of it, and keeps each position it judged in ``calls``.
    """

    favor: Callable[[TacticsPosition, np.ndarray], np.ndarray] | None = None
    value: Callable[[TacticsPosition], float] = lambda p: 0.5
    calls: list[TacticsPosition] = dataclasses.field(default_factory=list)

    def __call__(self, game, positions, legal):
        logits = []
        for position, mask in zip(positions, legal, strict=True):
            assert not position.over
            np.testing.assert_array_equal(mask, game.legal(position))
            self.calls.append(position)
            prior = np.zeros(mask.shape) if self.favor is None else self.favor(position, mask)
            logits.append(prior[mask])
        return logits, np.array([self.value(p) for p in positions])


def favoring(verb, cell=None, weight=10.0):
    """Return a ``favor`` that adds ``weight`` to the logit of a verb, or of one action."""

    def favor(position, mask):
        prior = np.zeros(mask.shape)
        if cell is None:
            prior[verb] = weight
        else:
            prior[verb, cell] = weight
        return prior

    return favor


def search(player, game, position, seed=0):
    return player.search(game, position, np.random.default_rng(seed))


def battles(game, seed, share):
    """Yield some positions not over from a random game, deployment included."""
    rng = np.random.default_rng(seed)
    for position, _ in play_random(game, seed, rng):
        if not position.over and rng.random() < share:
            yield position


def test_considered_visits_halve_the_actions_considered():
    assert considered_visits(4, 8) == (0, 0, 0, 0, 1, 1, 2, 2)
    assert considered_visits(16, 32) == (0,) * 16 + (1,) * 8 + (2,) * 4 + (3,) * 4
    assert considered_visits(1, 3) == (0, 1, 2)
    assert considered_visits(2, 5) == (0, 0, 1, 1, 2)
    for k, n in ((3, 7), (16, 32), (16, 100), (5, 2)):
        assert len(considered_visits(k, n)) == n


def finishing_blow(make_battle, game=SMALL):
    """Return a battle that one attack wins, and the target's cell: a captain beside the last
    enemy, which has 1 hp left, hitting 95% of the time."""
    position = make_battle([piece(5, 4, CAPTAIN, 0), piece(4, 4, FOOT, 1, hp=1)], game=game)
    return position, int(np.flatnonzero(position.state.units.team == 1)[0])


def test_the_search_plays_a_winning_action_for_either_team(make_battle):
    # Attacking now wins 95% of the time, and so does moving first and attacking after.
    position, _ = finishing_blow(make_battle)
    player = AlphaZeroPlayer(FakeEvaluator(), simulations=64, considered=64)
    for battle in (position, mirrored(SMALL, position)):
        found = search(player, SMALL, battle)
        team0 = found.q[found.action]
        assert (team0 if SMALL.player(battle) == 0 else 1 - team0) > 0.9
        assert found.policy[found.action] == found.policy.max()


def test_with_noise_the_search_mostly_plays_winning_actions(make_battle):
    position, _ = finishing_blow(make_battle)
    player = AlphaZeroPlayer(FakeEvaluator(), simulations=64, considered=64, noise=True)
    q = [search(player, SMALL, position, seed) for seed in range(20)]
    assert sum(found.q[found.action] > 0.9 for found in q) >= 15


def test_the_improved_policy_favors_the_winning_attack_over_the_prior(make_battle):
    position, target = finishing_blow(make_battle)
    found = search(AlphaZeroPlayer(FakeEvaluator(), simulations=64, considered=64), SMALL, position)
    prior = 1.0 / np.count_nonzero(SMALL.legal(position))
    assert found.policy[Verb.ATTACK, target] > 2 * prior
    assert found.policy[Verb.ATTACK, target] == found.policy.max()


def test_results_are_distributions_over_legal_actions():
    player = AlphaZeroPlayer(FakeEvaluator(), simulations=12, considered=4, noise=True)
    checked = 0
    for position in battles(TINY, 1, 0.1):
        legal = TINY.legal(position)
        found = search(player, TINY, position, checked)
        assert legal[found.action]
        assert found.policy.shape == legal.shape
        assert found.policy.sum() == pytest.approx(1.0)
        assert np.all(found.policy[~legal] == 0)
        assert np.all(found.policy[legal] > 0)
        assert found.visits.sum() == 12
        assert np.all(found.visits[~legal] == 0)
        np.testing.assert_array_equal(np.isnan(found.q), found.visits == 0)
        assert 0.0 <= found.value <= 1.0
        checked += 1
    assert checked >= 10


def test_with_even_values_the_policy_is_the_prior_and_its_best_is_played(make_battle, at):
    # Every action worth the same leaves nothing to improve: completed values, visited or not,
    # are all the node's value.
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    favor = favoring(Verb.MOVE, at(2, 4), weight=3.0)
    found = search(AlphaZeroPlayer(FakeEvaluator(favor), simulations=16), SMALL, position)
    legal = SMALL.legal(position)
    logits = favor(position, legal)[legal]
    prior = np.exp(logits - logits.max())
    np.testing.assert_allclose(found.policy[legal], prior / prior.sum(), rtol=1e-12)
    assert found.action == (Verb.MOVE, at(2, 4))


def test_gumbel_noise_samples_the_action_played(make_battle, at):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    favor = favoring(Verb.MOVE, at(2, 4), weight=2.0)
    played = [
        search(
            AlphaZeroPlayer(FakeEvaluator(favor), simulations=4, noise=noise), SMALL, position, s
        ).action
        for noise in (False, True)
        for s in range(30)
    ]
    assert set(played[:30]) == {(Verb.MOVE, at(2, 4))}
    assert len(set(played[30:])) > 3


def test_completed_values_mix_the_root_value_with_the_visited_means(make_battle):
    # The root is worth 0.9 to the evaluator, and what follows it less, by where the unit
    # stands: the improved policy, the value and the action follow mctx's formulas.
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])

    def value(p):
        return (
            0.9
            if p.ply == position.ply
            else 0.1 + 0.1 * (int(np.flatnonzero(p.state.units.team == 0)[0]) % 5)
        )

    player = AlphaZeroPlayer(FakeEvaluator(value=value), simulations=6, considered=4)
    found = search(player, SMALL, position)
    legal = SMALL.legal(position)
    n, q = found.visits[legal].astype(float), found.q[legal]
    seen = n > 0
    assert len({float(x) for x in q[seen]}) > 1
    prior = np.full(n.size, 1.0 / n.size)
    weighted = (prior[seen] * q[seen]).sum() / prior[seen].sum()
    mixed = (0.9 + n.sum() * weighted) / (n.sum() + 1)
    completed = np.where(seen, np.nan_to_num(q), mixed)
    completed = (completed - completed.min()) / (completed.max() - completed.min())
    sigma = (50 + n.max()) * 0.1 * completed
    expected = np.exp(np.log(prior) + sigma - sigma.max())
    np.testing.assert_allclose(found.policy[legal], expected / expected.sum(), rtol=1e-9)
    assert found.value == pytest.approx((0.9 + (n[seen] * q[seen]).sum()) / (1 + n.sum()))
    assert found.visits[found.action] == found.visits.max()


def test_below_the_root_visits_follow_the_improved_policy(make_battle, at):
    # All visits go to the favored move; after it the unit's own turn goes on, where ending is
    # favored about 1 to 1 against all else. A search that always took the most likely
    # action there would judge only one position two plies down.
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])

    def favor(p, mask):
        prior = favoring(Verb.MOVE, at(2, 4), weight=5.0)(p, mask)
        prior[Verb.END] = 2.0
        return prior

    evaluator = FakeEvaluator(favor)
    found = search(AlphaZeroPlayer(evaluator, simulations=20, considered=1), SMALL, position)
    assert found.visits[Verb.MOVE, at(2, 4)] == 20
    assert len([p for p in evaluator.calls if p.ply == position.ply + 2]) >= 3


def test_chance_edges_sample_the_dice(make_battle):
    # One round only, the enemy done: after a miss nothing can change before the draw at the
    # cap, so the attack's value is the chance of a hit plus half the chance of a miss.
    game = dataclasses.replace(SMALL, max_rounds=1)
    position = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1, hp=1, done=True)], game=game)
    target = int(np.flatnonzero(position.state.units.team == 1)[0])
    chance = game.odds(position, game.active(position), target).chance
    evaluator = FakeEvaluator(favoring(Verb.ATTACK, target, weight=20.0))
    player = AlphaZeroPlayer(evaluator, simulations=400, considered=1)
    found = search(player, game, position)
    assert found.visits[Verb.ATTACK, target] == 400
    expected = chance + (1 - chance) / 2
    spread = 0.5 * np.sqrt(chance * (1 - chance) / 400)
    assert abs(found.q[Verb.ATTACK, target] - expected) < 4 * spread


def test_outcomes_that_leave_the_same_state_share_a_child(make_battle):
    # The same battle with 3 hp left: a hit and a miss each always leave the same state, so
    # after both have happened, every visit goes deeper and the evaluator sees each once.
    game = dataclasses.replace(SMALL, max_rounds=1)
    position = make_battle([piece(5, 4, FOOT, 0), piece(4, 4, FOOT, 1, hp=3, done=True)], game=game)
    target = int(np.flatnonzero(position.state.units.team == 1)[0])
    evaluator = FakeEvaluator(favoring(Verb.ATTACK, target, weight=20.0))
    search(AlphaZeroPlayer(evaluator, simulations=60, considered=1), game, position)
    after = [p for p in evaluator.calls if p.ply == position.ply + 1]
    assert len(after) == 2
    assert {p.state.units.hp[target] for p in after} == {1, 3}


def test_the_search_never_reads_the_games_seed():
    # The seed fixes every future roll: a search that read it would know which attacks hit.
    player = AlphaZeroPlayer(FakeEvaluator(value=lambda p: float(p.ply % 7) / 7), simulations=10)
    checked = 0
    for position in battles(TINY, 4, 0.2):
        found = [
            search(player, TINY, dataclasses.replace(position, seed=seed), 7) for seed in (1, 2)
        ]
        assert found[0].action == found[1].action
        np.testing.assert_array_equal(found[0].policy, found[1].policy)
        np.testing.assert_array_equal(found[0].visits, found[1].visits)
        checked += 1
    assert checked >= 10


def test_the_same_generator_gives_the_same_result():
    player = AlphaZeroPlayer(FakeEvaluator(), simulations=10, noise=True)
    position = next(p for p in battles(TINY, 5, 0.3) if not p.deploying)
    first, again = search(player, TINY, position, 3), search(player, TINY, position, 3)
    assert first.action == again.action
    np.testing.assert_array_equal(first.policy, again.policy)
    np.testing.assert_array_equal(first.q, again.q)


def test_the_evaluator_judges_each_new_position_once_and_skips_forced_ones():
    for position in battles(TINY, 6, 0.1):
        evaluator = FakeEvaluator()
        search(AlphaZeroPlayer(evaluator, simulations=10), TINY, position)
        assert evaluator.calls[0] is position
        assert len(evaluator.calls) <= 11
        assert all(np.count_nonzero(TINY.legal(p)) > 1 for p in evaluator.calls[1:])


def test_the_player_plays_its_only_legal_action_without_searching(make_battle):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    legal = np.zeros_like(SMALL.legal(position))
    legal[Verb.END, SMALL.active(position)] = True
    evaluator = FakeEvaluator()
    action = AlphaZeroPlayer(evaluator)(
        SMALL, SMALL.player(position), position, legal, np.random.default_rng(0)
    )
    assert action == (Verb.END, SMALL.active(position))
    assert evaluator.calls == []


@pytest.mark.parametrize("noise", [False, True])
def test_the_player_plays_whole_games_from_deployment(noise):
    player = AlphaZeroPlayer(FakeEvaluator(), simulations=6, considered=4, noise=noise)
    for players in ((player, GreedyPlayer()), (GreedyPlayer(), player)):
        assert play(TINY, players, TINY.start(2)).final.over


@pytest.mark.parametrize(
    ("evaluator", "message"),
    [
        (lambda game, ps, legal: ([np.zeros(1) for _ in ps], np.full(len(ps), 0.5)), "logits"),
        (
            lambda game, ps, legal: (
                [np.full(np.count_nonzero(m), np.nan) for m in legal],
                np.full(len(ps), 0.5),
            ),
            "finite",
        ),
        (
            lambda game, ps, legal: (
                [np.zeros(np.count_nonzero(m)) for m in legal],
                np.full(len(ps), 1.5),
            ),
            r"\[0, 1\]",
        ),
    ],
)
def test_bad_evaluations_raise(make_battle, evaluator, message):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    with pytest.raises(ValueError, match=message):
        search(AlphaZeroPlayer(evaluator), SMALL, position)


def test_a_finished_game_cannot_be_searched(make_battle):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    with pytest.raises(ValueError, match="over"):
        search(AlphaZeroPlayer(FakeEvaluator()), SMALL, dataclasses.replace(position, over=True))


def test_parameters_are_checked(not_an_integer):
    player = AlphaZeroPlayer(FakeEvaluator())
    for name in ("simulations", "considered"):
        with pytest.raises(TypeError, match=name):
            dataclasses.replace(player, **{name: not_an_integer})
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(player, **{name: 0})
    for name in ("c_visit", "c_scale"):
        for bad in (-1.0, float("nan")):
            with pytest.raises(ValueError, match=name):
                dataclasses.replace(player, **{name: bad})
    with pytest.raises(TypeError, match="noise"):
        dataclasses.replace(player, noise=1)
    with pytest.raises(TypeError, match="evaluator"):
        AlphaZeroPlayer("network")  # ty: ignore[invalid-argument-type]  # not callable on purpose


DECIDED = dataclasses.replace(TINY, decide_at_cap=True)
"""TINY with a winner at the round cap: the fake evaluator's games all reach it."""


def quick(evaluator=None, noise=True):
    return AlphaZeroPlayer(evaluator or FakeEvaluator(), simulations=4, considered=4, noise=noise)


def test_self_play_plays_every_game_to_its_end():
    experience, played = self_play(TINY, quick(), [3, 4, 5])
    assert [p.seed for p in played] == [3, 4, 5]
    for p in played:
        assert p.plies > 0
        assert p.winner in (0, 1, None)
        assert 0 <= p.kept <= p.searched < p.plies
    assert len(experience) == sum(p.kept for p in played)
    np.testing.assert_array_equal(np.unique(experience.game), [3, 4, 5])


def test_a_self_play_game_plays_the_same_beside_others():
    alone, [one] = self_play(TINY, quick(), [6], keep=0.5)
    together, played = self_play(TINY, quick(), [5, 6, 7], keep=0.5)
    assert played[1][:-1] == one[:-1]
    assert played[1].final.key() == one.final.key()  # positions compare by identity
    mine = together.take(together.game == 6)
    for name in ("obs", "legal", "policy", "start", "outcome", "game", "ply"):
        np.testing.assert_array_equal(getattr(mine, name), getattr(alone, name))


def test_the_evaluator_judges_one_leaf_of_every_game_at_once():
    sizes = []

    def evaluator(game, positions, legal):
        sizes.append(len(positions))
        return [np.zeros(np.count_nonzero(m)) for m in legal], np.full(len(positions), 0.5)

    self_play(TINY, quick(evaluator), [1, 2, 3, 4])
    assert sizes[0] == 4
    assert max(sizes) == 4
    assert sizes == sorted(sizes, reverse=True)


def test_kept_decisions_hold_policies_over_their_legal_actions_and_outcomes():
    experience, played = self_play(DECIDED, quick(), [8, 9], keep=1.0)
    assert all(p.winner is not None for p in played)
    assert len(experience) == sum(p.searched for p in played)
    n_cells = TINY.shape[0] * TINY.shape[1]
    n_bits = TINY.n_verbs * n_cells
    legal = np.unpackbits(experience.legal, axis=1)[:, :n_bits].astype(bool)
    lengths = np.diff(experience.start)
    np.testing.assert_array_equal(lengths, legal.sum(axis=1))
    assert np.all(lengths > 1)
    sums = np.add.reduceat(experience.policy.astype(np.float64), experience.start[:-1])
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
    assert np.all(experience.policy > 0)
    for p in played:
        outcome = experience.outcome[experience.game == p.seed]
        assert set(outcome.tolist()) == ({0.0} if p.winner is None else {1.0, -1.0})
    assert np.all(np.diff(experience.ply[experience.game == 8]) > 0)


def test_self_play_keeps_what_searching_alone_decides_in_the_deciding_teams_frame():
    # The oracle plays the game with `search` and the game's own stream, which self-play
    # also draws from to keep a decision, right after each search.
    def ramp(position, mask):  # uneven priors, so that a policy's order shows
        return np.arange(mask.size, dtype=np.float64).reshape(mask.shape) / mask.size

    player, seed, keep = quick(FakeEvaluator(favor=ramp)), 15, 0.5
    experience, [played] = self_play(DECIDED, player, [seed], keep=keep)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
    position, kept = DECIDED.start(seed), []
    while not position.over:
        legal = DECIDED.legal(position)
        if np.count_nonzero(legal) == 1:
            verb, cell = np.argwhere(legal)[0]
            action = (int(verb), int(cell))
        else:
            result = player.search(DECIDED, position, rng)
            if rng.random() < keep:
                kept.append((position, legal, result.policy))
            action = result.action
        position = DECIDED.step(position, action)
    assert position.winner is not None
    assert {DECIDED.player(p) for p, _, _ in kept} == {0, 1}
    assert len(experience) == len(kept) == played.kept
    assert np.ptp(experience.policy) > 0
    assert played.plies == position.ply
    for i, (p, legal, policy) in enumerate(kept):
        team = DECIDED.player(p)
        own = DECIDED.frame(team).into(legal)
        np.testing.assert_array_equal(experience.obs[i], np.rint(DECIDED.observe(p, team) * 255))
        np.testing.assert_array_equal(experience.legal[i], np.packbits(own.ravel()))
        np.testing.assert_array_equal(
            experience.policy[experience.start[i] : experience.start[i + 1]],
            DECIDED.frame(team).into(policy)[own].astype(np.float32),
        )
        assert experience.ply[i] == p.ply
        assert experience.outcome[i] == (1.0 if team == position.winner else -1.0)
        assert experience.game[i] == seed


def test_kept_decisions_are_in_the_deciding_teams_frame():
    # In its own frame, each team deploys on team 0's home.
    experience, _ = self_play(TINY, quick(), [10, 11], keep=1.0)
    n_cells = TINY.shape[0] * TINY.shape[1]
    legal = np.unpackbits(experience.legal, axis=1)[:, : TINY.n_verbs * n_cells]
    place = legal.reshape(len(experience), TINY.n_verbs, n_cells)[:, Verb.PLACE :].any(axis=1)
    deploying = place.any(axis=1)
    assert deploying.sum() >= 2 * TINY.team_size - 2
    home = homes(board_space(TINY.shape), TINY.home_depth)
    assert np.all(home[np.flatnonzero(place[deploying].any(axis=0))] == 0)


def test_experience_takes_and_joins_by_decision():
    experience, _ = self_play(TINY, quick(), [12], keep=0.5)
    n = len(experience)
    order = np.random.default_rng(0).permutation(n)
    shuffled = experience.take(order)
    back = shuffled.take(np.argsort(order))
    halves = Experience.join(
        [experience.take(np.arange(n // 2)), experience.take(np.arange(n // 2, n))]
    )
    for again in (back, halves):
        for name in ("obs", "legal", "policy", "start", "outcome", "game", "ply"):
            np.testing.assert_array_equal(getattr(again, name), getattr(experience, name))
    first = experience.take(np.arange(n) == 1)
    np.testing.assert_array_equal(
        first.policy, experience.policy[experience.start[1] : experience.start[2]]
    )


def test_self_play_keeps_nothing_when_told_to_or_without_games():
    experience, played = self_play(TINY, quick(), [13], keep=0.0)
    assert len(experience) == 0
    assert played[0].kept == 0
    assert experience.obs.shape[1:] == (len(TINY.planes), TINY.shape[0] * TINY.shape[1])
    none, nothing = self_play(TINY, quick(), [])
    assert len(none) == 0
    assert nothing == []


def test_self_play_checks_its_arguments(not_an_integer):
    with pytest.raises(ValueError, match="keep"):
        self_play(TINY, quick(), [1], keep=1.5)
    with pytest.raises(ValueError, match="seed"):
        self_play(TINY, quick(), [-1])
    with pytest.raises(TypeError, match="seed"):
        self_play(TINY, quick(), [not_an_integer])
