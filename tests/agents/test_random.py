import numpy as np
import pytest

from gridzero.agents.random import UniformPlayer


def test_the_uniform_player_picks_each_legal_action_about_as_often(rng):
    legal = np.zeros((3, 5), dtype=bool)
    legal[0, 1] = legal[2, 4] = legal[1, 0] = True
    picks = [tuple(UniformPlayer()(None, 0, None, legal, rng)) for _ in range(3000)]
    counts = {a: picks.count(a) for a in set(picks)}
    assert set(counts) == {(0, 1), (2, 4), (1, 0)}
    assert all(900 < n < 1100 for n in counts.values())


def test_the_uniform_player_follows_its_generator():
    legal = np.ones((2, 7), dtype=bool)
    first = [UniformPlayer()(None, 0, None, legal, np.random.default_rng(s)) for s in range(20)]
    again = [UniformPlayer()(None, 0, None, legal, np.random.default_rng(s)) for s in range(20)]
    assert first == again
    assert len(set(first)) > 5


def test_the_uniform_player_needs_a_legal_action():
    with pytest.raises(ValueError, match="no action"):
        UniformPlayer()(None, 0, None, np.zeros((2, 3), dtype=bool), np.random.default_rng(0))
