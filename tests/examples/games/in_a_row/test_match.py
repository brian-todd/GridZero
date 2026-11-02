"""Tests of what an evaluator makes of the opening of k in a row."""

import numpy as np
import pytest

from examples.games.in_a_row import (
    PLACE,
    InARow,
    opening,
)

GAME = InARow()


class Favors:
    """Gives every legal action of the start a logit that rises toward one column."""

    def __init__(self, column):
        self.column = column

    def __call__(self, game, positions, legal):
        columns = game.space.unravel(np.flatnonzero(legal[0][PLACE]))[:, 0]
        logits = -np.abs(columns - self.column).astype(np.float64) + 7.0  # shifted: only
        return [logits], np.array([0.75])  # differences matter


@pytest.mark.parametrize("column", [0, 3, 6])
def test_the_opening_reads_the_favored_column_its_priors_and_value(column):
    first = opening(GAME, Favors(column))
    assert first.column == column
    assert len(first.prior) == 7
    assert np.isclose(sum(first.prior), 1.0)
    assert first.prior[column] == max(first.prior)
    assert first.value == 0.75


def test_the_opening_needs_gravity():
    with pytest.raises(ValueError, match="gravity"):
        opening(InARow((3, 3), 3, gravity=False), Favors(0))
