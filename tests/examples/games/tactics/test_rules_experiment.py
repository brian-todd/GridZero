import dataclasses

import pytest

from examples.games.tactics import Fielded, GreedyPlayer, RandomPlayer, summarize
from examples.games.tactics.experiments.rules import (
    duel,
    engineer_swap,
    figures,
    greedy_fielding,
    own_picks,
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


def test_figures_say_what_each_kind_did_and_agree_with_the_totals():
    players = [("a", GreedyPlayer()), ("b", RandomPlayer())]
    f = figures(GAME, series(GAME, players, 2, seed=0))
    assert set(f["kinds"]) == {stats.name for stats in GAME.units}
    assert sum(k["placed"] for k in f["kinds"].values()) == 2 * GAME.team_size
    assert sum(k["losses"] for k in f["kinds"].values()) == pytest.approx(f["dead"])
    assert sum(k["dealt"] for k in f["kinds"].values()) == pytest.approx(
        sum(k["taken"] for k in f["kinds"].values())
    )


def test_each_kinds_figures_are_its_own_columns_of_the_summary_a_game():
    records = series(GAME, [("a", GreedyPlayer()), ("b", RandomPlayer())], 2, seed=0)
    summary = summarize(GAME, records)
    for k, stats in enumerate(GAME.units):
        assert figures(GAME, records)["kinds"][stats.name] == {
            "placed": summary.placed[k] / 4,
            "dealt": summary.damage_dealt[k] / 4,
            "taken": summary.damage_taken[k] / 4,
            "kills": summary.kills[k] / 4,
            "losses": summary.losses[k] / 4,
        }


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


def test_the_engineer_swap_fields_its_rosters_through_the_player_it_is_given():
    made = []

    def make(roster):
        made.append(roster)
        return Fielded(RandomPlayer(), GreedyPlayer(roster=roster))

    result = engineer_swap(WORKS, 1, 1, make)
    assert made == [(1, 1, 1, 1, 1), (1, 2, 1, 1, 0)]
    placed = {kind: k["placed"] for kind, k in result["kinds"].items()}
    assert placed["archer"] == 3
    assert placed["engineer"] == 1


def test_rosters_plays_every_roster_through_the_player_it_is_given():
    made = []

    def make(roster):
        made.append(roster)
        return Fielded(RandomPlayer(), GreedyPlayer(roster=roster))

    report = rosters(GAME, 1, 1, base=(2, 1, 1, 1), make=make)
    assert set(made) == {(2, 1, 1, 1), *(tuple(r) for r, _ in report["scores"])}


def test_own_picks_counts_each_team_of_the_self_play_games_by_the_roster_it_fielded():
    own = own_picks(WORKS, GreedyPlayer(roster=(0, 2, 1, 1, 1)), greedy_fielding, 1, 1)
    assert own["picked"] == {"[0, 2, 1, 1, 1]": 4}
    assert own["self_play"]["games"] == 2
    assert own["against_default"]["kinds"]["footsoldier"]["placed"] == 1
    assert own["against_default"]["won"] + own["against_default"]["lost"] == 2
