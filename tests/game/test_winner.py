import numpy as np
import pytest

from gridzero.game import winner


def test_the_highest_score_alone_wins():
    assert winner([3, 5, 4]) == 1
    assert winner(np.array([2.5, 0.5])) == 0
    assert winner([7]) == 0  # one player always wins


def test_a_tie_for_the_highest_is_a_draw_whatever_the_others():
    assert winner([5, 5]) is None
    assert winner([5, 1, 5]) is None
    assert winner([1, 5, 1, 1]) == 1  # a tie below the highest doesn't matter


def test_scores_within_the_tolerance_of_the_highest_tie_it():
    assert winner([1.0, 1.0 + 1e-12]) == 1
    assert winner([1.0, 1.0 + 1e-12], tolerance=1e-9) is None
    assert winner([1.0, 1.1], tolerance=1e-9) == 1
    assert winner([1.0, 1.1, 1.1 - 1e-12], tolerance=1e-9) is None


def test_integer_scores_compare_exactly_past_float_precision():
    big = 2**60
    assert winner(np.array([big, big + 1], dtype=np.int64)) == 1
    assert winner(np.array([big + 1, big + 1], dtype=np.int64)) is None


@pytest.mark.parametrize("dtype", [np.int8, np.uint64, np.float16, np.float32, np.float64])
def test_every_real_dtype_is_scored(dtype):
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    assert winner(np.array([info.max, 0], dtype=dtype)) == 0
    assert winner(np.array([info.max, info.max], dtype=dtype)) is None


def test_winner_refuses_bad_input(not_a_real):
    with pytest.raises(TypeError, match="real numbers"):
        winner([True, False])
    with pytest.raises(TypeError, match="real numbers"):
        winner(["a", "b"])
    with pytest.raises(TypeError, match="real numbers"):
        winner([1j, 2j])
    with pytest.raises(ValueError, match="shape"):
        winner([])
    with pytest.raises(ValueError, match="shape"):
        winner([[1, 2]])
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            winner([1.0, bad])
    with pytest.raises(TypeError, match="tolerance"):
        winner([1, 2], tolerance=not_a_real)
    with pytest.raises(ValueError, match="tolerance"):
        winner([1, 2], tolerance=-1.0)
