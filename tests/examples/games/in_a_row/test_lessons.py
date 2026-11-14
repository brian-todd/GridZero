"""Tests of lessons and of the fit of a code player's constants to them."""

import numpy as np
import pytest

from examples.games.in_a_row import (
    MOVE_FEATURES,
    PLACE,
    VALUE_FEATURES,
    InARow,
    Lessons,
    LineEvaluator,
    distinct_positions,
    judge,
    move_features,
    position_of,
    value_features,
)
from examples.games.in_a_row.fit import fit
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.game import replay, series

GAME = InARow()


def games(starts=4, seed=0):
    return series(GAME, [("a", UniformPlayer()), ("b", UniformPlayer())], starts, seed)


def judged(records, simulations=8, workers=1):
    owner, ply, numbers = distinct_positions(GAME, records)
    return judge(
        GAME,
        LineEvaluator(),
        owner,
        ply,
        numbers,
        simulations=simulations,
        seed=3,
        workers=workers,
    )


def assert_same(one, other):
    for name in ("owner", "ply", "game", "source", "value", "q", "policy", "move"):
        np.testing.assert_array_equal(getattr(one, name), getattr(other, name))


def test_distinct_positions_keep_each_once_and_none_over():
    records = games()
    owner, ply, numbers = distinct_positions(GAME, records)
    assert len({row.tobytes() for row in owner}) == len(owner)
    assert owner.shape == (len(ply), GAME.space.n_cells)
    assert numbers.max() < len(records)
    assert ((owner >= 0).sum(axis=1) == ply).all()
    every = {p.state.owner.tobytes() for r in records for p in replay(GAME, r) if not p.over}
    assert {row.tobytes() for row in owner} == every


def test_a_position_rebuilt_from_its_owners_and_ply_plays_on_alike():
    record = games(1)[0]
    original = list(replay(GAME, record))[5]
    rebuilt = position_of(GAME, original.state.owner, original.ply)
    assert GAME.player(rebuilt) == GAME.player(original)
    np.testing.assert_array_equal(GAME.legal(rebuilt), GAME.legal(original))


def test_judging_gives_one_row_per_position_of_values_and_policies():
    lessons = judged(games(2))
    n = len(lessons)
    assert lessons.q.shape == lessons.policy.shape == (n, GAME.shape[0])
    assert ((lessons.value >= 0.0) & (lessons.value <= 1.0)).all()
    np.testing.assert_allclose(lessons.policy.sum(axis=1), 1.0)


def test_q_is_nan_only_where_a_column_is_full_or_unvisited():
    lessons = judged(games(2), simulations=32)
    full = np.array(
        [
            np.flatnonzero(GAME.legal(position_of(GAME, o, int(p)))[PLACE]).size
            for o, p in zip(lessons.owner, lessons.ply, strict=True)
        ]
    )
    assert (np.isfinite(lessons.q).sum(axis=1) <= full).all()
    assert (lessons.policy > 0.0).sum(axis=1).max() <= GAME.shape[0]


def test_judging_doesnt_depend_on_the_workers():
    records = games(2)
    assert_same(judged(records, workers=1), judged(records, workers=2))


def test_lessons_add_only_new_positions_and_renumber_their_games():
    first = judged(games(2, seed=0))
    second = judged(games(2, seed=100))
    together = first + second
    assert len({row.tobytes() for row in together.owner}) == len(together)
    assert len(together) >= len(first)
    np.testing.assert_array_equal(together.owner[: len(first)], first.owner)
    assert together.game[len(first) :].min(initial=first.game.max() + 1) > first.game.max()
    assert_same(first + first, first)


def test_lessons_remember_their_round_when_added():
    first = judged(games(2, seed=0))
    owner, ply, numbers = distinct_positions(GAME, games(2, seed=100))
    second = judge(GAME, LineEvaluator(), owner, ply, numbers, simulations=8, seed=3, source=1)
    together = first + second
    assert (together.source[: len(first)] == 0).all()
    assert (together.source[len(first) :] == 1).all()


def test_lessons_survive_a_file(tmp_path):
    lessons = judged(games(1))
    lessons.save(tmp_path / "lessons.npz")
    assert_same(Lessons.load(tmp_path / "lessons.npz"), lessons)


def synthetic(value_weights, intercept, prior_weights, starts=120):
    """Return lessons a known evaluator's constants would give, with no search."""
    owner, ply, numbers = distinct_positions(GAME, games(starts))
    features = value_features(GAME, owner, ply)
    value = 0.5 + 0.5 * np.tanh(0.5 * (intercept + features @ value_weights))
    q = np.full((len(ply), GAME.shape[0]), 0.5)
    policy = np.zeros((len(ply), GAME.shape[0]))
    for i in range(len(ply)):
        cells, moves = move_features(GAME, owner[i], int(ply[i]))
        logits = moves @ prior_weights
        logits -= logits.max()
        chance = np.exp(logits) / np.exp(logits).sum()
        policy[i, GAME.space.unravel(cells)[:, 0]] = chance
    source = np.zeros(len(ply), dtype=np.int64)
    return Lessons(owner, ply, numbers, source, value, q, policy, np.argmax(policy, axis=1))


def test_a_fit_recovers_constants_that_made_the_lessons():
    rng = np.random.default_rng(0)
    value_weights = rng.normal(0.0, 0.3, len(VALUE_FEATURES))
    prior_weights = rng.normal(0.0, 0.3, len(MOVE_FEATURES))
    lessons = synthetic(value_weights, 0.2, prior_weights)
    evaluator, score = fit(GAME, lessons, seed=0)
    assert score.value_correlation > 0.99
    assert score.agreement > 0.95
    assert score.trained + score.held_out == len(lessons)
    used = np.abs(prior_weights) > 0.0
    assert used.all()
    np.testing.assert_allclose(evaluator.prior, prior_weights, atol=0.1)
    moved = value_features(GAME, lessons.owner, lessons.ply).std(axis=0) > 0.3
    np.testing.assert_allclose(np.array(evaluator.value)[moved], value_weights[moved], atol=0.2)


@pytest.mark.parametrize(("toward_center", "regret"), [(5.0, 0.0), (-5.0, 0.4)])
def test_regret_is_what_the_priors_favorite_loses_for_the_player_to_move(toward_center, regret):
    prior = np.zeros(len(MOVE_FEATURES))
    prior[MOVE_FEATURES.index("center")] = toward_center
    rng = np.random.default_rng(0)
    lessons = synthetic(rng.normal(0.0, 0.3, len(VALUE_FEATURES)), 0.0, prior, starts=30)
    free = lessons.owner.reshape(-1, *GAME.shape)[:, 3, -1] == -1  # column 3 isn't full
    edge = lessons.owner.reshape(-1, *GAME.shape)[:, [0, 6], -1] == -1  # nor both edges
    keep = free & edge.all(axis=1)
    best_for_mover = np.full((keep.sum(), GAME.shape[0]), 0.5)  # q of 0.5 but the center's
    best_for_mover[:, 3] = 0.9
    ply = lessons.ply[keep]
    planted = np.where((ply % 2 == 0)[:, None], best_for_mover, 1.0 - best_for_mover)
    kept = Lessons(
        lessons.owner[keep],
        ply,
        lessons.game[keep],
        lessons.source[keep],
        lessons.value[keep],
        planted,
        lessons.policy[keep],
        lessons.move[keep],
    )
    _, score = fit(GAME, kept, seed=0)
    assert score.regret == pytest.approx(regret)


def test_a_constant_teacher_has_no_correlation():
    lessons = synthetic(np.zeros(len(VALUE_FEATURES)), 0.0, np.zeros(len(MOVE_FEATURES)), starts=8)
    _, score = fit(GAME, lessons, seed=0)
    assert score.value_correlation == 0.0


def test_judging_files_each_columns_values_under_that_column():
    position = list(replay(GAME, games(1)[0]))[7]  # stones make the columns differ
    one = judge(
        GAME,
        LineEvaluator(),
        position.state.owner[None],
        np.array([position.ply]),
        np.zeros(1, dtype=np.int64),
        simulations=32,
        seed=3,
    )
    agent = AlphaZeroPlayer(LineEvaluator(), simulations=32, considered=GAME.shape[0])
    result = agent.search(GAME, position, np.random.default_rng(3))
    cells = np.flatnonzero(GAME.legal(position)[PLACE])
    columns = GAME.space.unravel(cells)[:, 0]
    assert len(np.unique(result.q[PLACE][cells])) > 1  # or the test can't tell columns apart
    np.testing.assert_array_equal(one.q[0, columns], result.q[PLACE][cells])
    np.testing.assert_array_equal(one.policy[0, columns], result.policy[PLACE][cells])
    assert one.value[0] == result.value
    assert one.move[0] == GAME.space.unravel(result.action.cell)[0]


def test_a_fit_keeps_about_one_position_in_five_out():
    rng = np.random.default_rng(1)
    lessons = synthetic(
        rng.normal(0.0, 0.3, len(VALUE_FEATURES)),
        0.0,
        rng.normal(0.0, 0.3, len(MOVE_FEATURES)),
        starts=40,
    )
    _, score = fit(GAME, lessons, seed=0)
    assert 0.1 < score.held_out / len(lessons) < 0.3


def test_distinct_positions_of_no_games_are_empty_with_the_usual_shapes():
    owner, ply, numbers = distinct_positions(GAME, [])
    assert owner.shape == (0, GAME.space.n_cells)
    assert owner.dtype == np.int8
    assert ply.shape == numbers.shape == (0,)
    assert ply.dtype == numbers.dtype == np.int64
