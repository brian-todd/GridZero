import dataclasses

import pytest

from examples.games.tactics import GreedyPlayer, RandomPlayer
from examples.games.tactics.experiments.rules import (
    duel,
    engineer_swap,
    figures,
    rosters,
    styles,
)
from gridzero.game import series
from tactics_demo import TINY, WORKS

GAME = dataclasses.replace(TINY, caps=(3, 1, 1, 1), team_size=5)


def test_figures_are_shares_and_means_of_the_games():
    players = [("a", GreedyPlayer()), ("b", RandomPlayer())]
    records = series(GAME, players, 2, seed=0)
    f = figures(GAME, records)
    assert f["games"] == 4
    assert sum(f["endings"].values()) == pytest.approx(1)
    assert f["rounds_p50"] <= f["rounds_p99"] <= f["rounds_max"] <= GAME.max_rounds
    assert 0 <= f["first_capturer_won"] <= 1
    assert 0 <= f["team_0"] <= 1


def test_a_duel_counts_every_game_for_a_side():
    result = duel(GAME, GreedyPlayer(), RandomPlayer(), 1, 1)
    assert result["won"] + result["lost"] == result["games"] == 2


def test_styles_play_each_match():
    results = styles(GAME, 1, 1)
    assert len(results) == 6
    assert results["sitter v sitter"]["endings"]["deadline"] == 1


def test_rosters_plays_each_other_roster_and_fits_a_value_per_kind():
    report = rosters(GAME, 1, 1, base=(2, 1, 1, 1))
    played = [tuple(r) for r, _ in report["scores"]]
    assert (2, 1, 1, 1) not in played
    assert len(played) == len(set(played)) == 3  # (3, 1, 1, 0), (3, 1, 0, 1), (3, 0, 1, 1)
    assert set(report["per_slot"]) == {s.name for s in GAME.units[1:]}


def test_the_engineer_swap_plays_a_roster_against_itself_with_an_archer_instead():
    result = engineer_swap(WORKS, 1, 1)
    assert result["with_engineer"] == [1, 1, 1, 1, 1]
    assert result["with_archer"] == [1, 2, 1, 1, 0]
    assert result["won"] + result["lost"] == 2


def test_the_engineer_swap_needs_a_kind_that_builds():
    assert not any(stats.builds for stats in GAME.units)
    with pytest.raises(ValueError, match="builds"):
        engineer_swap(GAME, 1, 1)


def test_rosters_with_no_other_roster_to_play_score_none():
    only = dataclasses.replace(GAME, caps=(2, 1, 1, 1), team_size=5)
    report = rosters(only, 1, 1)
    assert report["against"] == [2, 1, 1, 1]
    assert report["scores"] == []
    assert set(report["per_slot"].values()) == {0}


def test_the_engineer_swap_needs_an_engineer_to_swap():
    no_engineer = dataclasses.replace(WORKS, caps=(6, 3, 2, 1, 0))
    with pytest.raises(ValueError, match="no engineer"):
        engineer_swap(no_engineer, 1, 1)
