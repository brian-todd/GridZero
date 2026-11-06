"""The rules about the point as pure functions; `test_game.py` plays them through `step`."""

import numpy as np
import pytest

from examples.games.tactics import Hold
from examples.games.tactics.objective import DEADLINE_LOSER, after_round, ending, wanted
from examples.games.tactics.state import NOBODY

POINT = np.array([False, True, False])
TIMES = {"hold_rounds": 5, "capture_by": 10, "max_rounds": 20}


def team_on_point(team):
    return np.array([-1, team, -1], dtype=np.int8)


@pytest.mark.parametrize(
    ("hold", "team", "after"),
    [
        (NOBODY, -1, NOBODY),
        (NOBODY, 0, Hold(0, 1)),
        (NOBODY, 1, Hold(1, 1)),
        (Hold(0, 2), 0, Hold(0, 3)),
        (Hold(0, 2), -1, Hold(0, 3)),
        (Hold(0, 4), 1, Hold(1, 1)),
        (Hold(1, 4), 0, Hold(0, 1)),
    ],
)
def test_after_round_captures_keeps_and_flips(hold, team, after):
    assert after_round(hold, POINT, team_on_point(team)) == after


def test_after_round_with_no_point_on_the_board_changes_no_owner():
    nowhere = np.zeros(3, dtype=np.bool_)
    assert after_round(NOBODY, nowhere, team_on_point(1)) == NOBODY
    assert after_round(Hold(0, 1), nowhere, team_on_point(1)) == Hold(0, 2)


@pytest.mark.parametrize(
    ("hold", "round_", "result"),
    [
        (NOBODY, 1, None),
        (NOBODY, 9, None),
        (NOBODY, 10, (0, "deadline")),
        (NOBODY, 20, (0, "deadline")),
        (Hold(1, 1), 10, None),
        (Hold(1, 4), 10, None),
        (Hold(1, 5), 10, (1, "held")),
        (Hold(0, 6), 10, (0, "held")),
        (Hold(0, 1), 19, None),
        (Hold(0, 1), 20, (0, "cap")),
        (Hold(1, 1), 20, (1, "cap")),
        (Hold(1, 5), 20, (1, "held")),
    ],
)
def test_ending_says_who_wins_and_why(hold, round_, result):
    assert ending(hold, round_, **TIMES) == result


def test_the_deadline_goes_against_the_seat_named_as_its_loser():
    assert ending(NOBODY, 10, **TIMES) == (1 - DEADLINE_LOSER, "deadline")


def test_ending_never_gives_a_draw_when_the_deadline_is_within_the_cap():
    for owner in (-1, 0, 1):
        for clock in range(8):
            for round_ in range(1, 21):
                hold = Hold(owner, clock if owner >= 0 else 0)
                result = ending(hold, round_, **TIMES)
                assert result is None or result[0] in (0, 1)
                if round_ == 20:
                    assert result is not None


@pytest.mark.parametrize(
    ("hold", "on_point", "side", "cell", "wants"),
    [
        (NOBODY, -1, 0, 0, True),  # nobody has it
        (Hold(1, 2), -1, 0, 0, True),  # the enemy has it
        (Hold(1, 2), 1, 0, 0, True),  # an enemy stands there
        (Hold(0, 2), -1, 0, 0, False),  # its side has it
        (Hold(0, 2), 1, 0, 0, True),  # its side has it, but an enemy stands there
        (NOBODY, 0, 0, 0, False),  # a friend is on it
        (NOBODY, 0, 0, 1, True),  # it is on it: stay
        (Hold(0, 2), 0, 0, 1, True),  # it holds it: stay
    ],
)
def test_wanted_is_the_point_unless_the_side_has_it_covered(hold, on_point, side, cell, wants):
    tiles = wanted(POINT, hold, team_on_point(on_point), side, cell)
    assert tiles.tolist() == ([1] if wants else [])
