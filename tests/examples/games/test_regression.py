import numpy as np
import pytest

from examples.games.regression import HELD_OUT, correlation, fit_choices, fit_logistic, held_out


def test_held_out_keeps_whole_games_out():
    games = np.repeat(np.arange(50, dtype=np.int64), 4)
    out = held_out(games, seed=0)
    for game in range(50):
        assert out[games == game].all() or not out[games == game].any()


def test_held_out_keeps_one_game_in_so_many_out_set_by_the_seed():
    games = np.arange(100, dtype=np.int64)
    assert held_out(games, seed=0).sum() == 100 // HELD_OUT
    assert np.array_equal(held_out(games, seed=0), held_out(games, seed=0))
    assert not np.array_equal(held_out(games, seed=0), held_out(games, seed=1))


def test_held_out_keeps_a_game_out_of_even_the_fewest():
    assert held_out(np.arange(2, dtype=np.int64), seed=0).sum() == 1


@pytest.mark.parametrize(
    ("a", "b"),
    [([1.0], [1.0]), ([1.0, 1.0], [1.0, 2.0]), ([1.0, 2.0], [3.0, 3.0])],
)
def test_correlation_is_zero_with_none_to_measure(a, b):
    assert correlation(np.array(a), np.array(b)) == 0.0


def test_correlation_of_series_that_move_together_and_apart():
    a = np.array([1.0, 2.0, 4.0])
    assert correlation(a, 3.0 * a + 1.0) == pytest.approx(1.0)
    assert correlation(a, -a) == pytest.approx(-1.0)


def test_fit_logistic_recovers_soft_targets():
    features = np.random.default_rng(0).normal(size=(2000, 2))
    target = 1.0 / (1.0 + np.exp(-(0.5 + features @ np.array([2.0, -1.0]))))
    model = fit_logistic(features, target)
    assert model.coef_[0] == pytest.approx([2.0, -1.0], abs=0.02)
    assert model.intercept_[0] == pytest.approx(0.5, abs=0.02)


def test_fit_choices_recovers_weights_with_soft_targets_and_positions_of_any_size():
    rng = np.random.default_rng(0)
    truth = np.array([1.5, -0.5, 0.25])
    sizes = rng.integers(2, 9, size=2000)
    first = np.cumsum(np.r_[0, sizes[:-1]], dtype=np.int64)
    features = rng.normal(size=(sizes.sum(), 3))
    logits = features @ truth
    normal = np.repeat(np.logaddexp.reduceat(logits, first), sizes)
    # The penalty pulls a little toward 0, and nothing else could: the targets are exact.
    assert fit_choices(features, np.exp(logits - normal), first) == pytest.approx(truth, abs=0.05)


def test_fit_choices_needs_nothing_of_a_feature_that_never_differs_within_a_position():
    features = np.array([[1.0, 2.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    target = np.array([1.0, 0.0, 1.0, 0.0])
    weights = fit_choices(features, target, np.array([0, 2], dtype=np.int64))
    assert weights[0] == pytest.approx(0.0, abs=1e-3)
    assert weights[1] > 1.0
