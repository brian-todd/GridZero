import numpy as np
import pytest

from examples.games.fitting import sigmoid, weighted


def test_weighted_adds_the_weighted_features_to_the_start():
    features = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert weighted(features, (0.5, 2.0), 1.0).tolist() == [5.5, 0.5]


def test_weighted_is_the_same_alone_as_in_a_batch():
    features = np.random.default_rng(0).normal(size=(9, 7)) * 1e8
    weights = tuple(np.random.default_rng(1).normal(size=7).tolist())
    together = weighted(features, weights, 0.1)
    alone = [weighted(row[None], weights, 0.1)[0] for row in features]
    assert together.tolist() == alone


def test_weighted_raises_when_a_sum_overflows():
    with pytest.raises(ValueError, match="overflows"):
        weighted(np.array([[1e308]]), (1e308,), 0.0)


@pytest.mark.parametrize("score", [0.0, 1.0, -3.0, 1e308, -1e308])
def test_sigmoid_stays_in_bounds_and_turns_over_at_any_size(score):
    value = sigmoid(np.array([score, -score]))
    assert 0.0 <= value[0] <= 1.0
    assert value[0] + value[1] == pytest.approx(1.0)


def test_sigmoid_is_a_half_at_0():
    assert sigmoid(np.array([0.0]))[0] == 0.5
