import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    DEFAULT_CAPS,
    DEFAULT_UNITS,
    UnitStats,
    default_roster,
    hit_chance,
    hits,
    roll_needed,
)


def test_default_units_follow_the_design_table():
    table = [(u.name, u.hp, u.atk, u.defense, u.move, u.reach) for u in DEFAULT_UNITS]
    assert table == [
        ("footsoldier", 5, 4, 2, 2, 1),  # Atk 2 until T6 (2026-10-25), then 3, then 4
        ("archer", 3, 2, 1, 2, 3),
        ("knight", 7, 3, 2, 3, 1),  # move 4 until T6 (2026-10-25)
        ("captain", 9, 4, 3, 2, 1),
        ("engineer", 3, 2, 1, 2, 3),
    ]
    passives = np.array(
        [(u.formation, u.defers, u.double_move, u.aura, u.builds) for u in DEFAULT_UNITS]
    )
    np.testing.assert_array_equal(passives, np.eye(5, dtype=bool))  # one passive each
    assert DEFAULT_CAPS == (6, 3, 2, 1, 1)


def test_roll_needed_is_eight_less_two_per_point_of_atk_over_def():
    assert roll_needed(2, 2) == 8
    assert roll_needed(5, 2) == 2
    assert roll_needed(1, 3) == 12
    assert roll_needed(3, 1, base=10, step=3) == 4


def test_a_one_always_misses_and_a_twenty_always_hits():
    assert not hits(1, -5)
    assert hits(20, 30)
    assert hits(8, 8)
    assert not hits(7, 8)


@pytest.mark.parametrize("gap", range(-12, 13))
def test_hit_chance_is_65_percent_plus_10_per_point_clamped(gap):
    expected = float(np.clip(0.65 + 0.1 * gap, 0.05, 0.95))
    assert hit_chance(3 + gap, 3) == pytest.approx(expected)


def test_unit_stats_reject_numbers_out_of_range():
    base = DEFAULT_UNITS[0]
    for name, value in [("hp", 0), ("atk", -1), ("defense", -1), ("move", -1), ("reach", 0.0)]:
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(base, **{name: value})


def test_unit_stats_reject_values_of_the_wrong_type(not_an_integer):
    with pytest.raises(TypeError, match="hp"):
        dataclasses.replace(DEFAULT_UNITS[0], hp=not_an_integer)


def test_unit_stats_reject_flags_that_are_not_bools():
    with pytest.raises(TypeError, match="aura"):
        UnitStats("x", 1, 1, 1, 1, 1, aura=1)  # ty: ignore[invalid-argument-type]: a wrong type on purpose
    with pytest.raises(TypeError, match="name"):
        UnitStats(3, 1, 1, 1, 1, 1)  # ty: ignore[invalid-argument-type]: a wrong type on purpose


@pytest.mark.parametrize(
    ("caps", "team_size", "roster"),
    [
        ((6, 3, 2, 1, 1), 8, (3, 2, 1, 1, 1)),
        ((6, 3, 2, 1), 8, (4, 2, 1, 1)),
        ((6, 3, 2, 1), 12, (6, 3, 2, 1)),
        ((2, 1, 1, 1), 5, (2, 1, 1, 1)),
        ((6, 0, 2, 1), 5, (3, 0, 1, 1)),
        ((3, 3, 3), 2, (1, 1, 0)),
        ((0, 0), 0, (0, 0)),
    ],
)
def test_default_roster_fields_each_kind_then_follows_the_caps(caps, team_size, roster):
    assert default_roster(caps, team_size) == roster


@pytest.mark.parametrize("team_size", range(1, 14))
def test_default_roster_always_fills_the_team_within_the_caps(team_size):
    caps = (6, 3, 2, 1, 1)
    roster = default_roster(caps, team_size)
    assert sum(roster) == team_size
    assert all(n <= cap for n, cap in zip(roster, caps, strict=True))
