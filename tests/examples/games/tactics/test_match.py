import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    Attacked,
    Balance,
    Died,
    GameOver,
    GreedyPlayer,
    RandomPlayer,
    SearchPlayer,
    Summary,
    TacticsPosition,
    balance,
    summarize,
)
from gridzero.game import Record, Tally, play, series
from tactics_demo import TINY

PLAYERS = [RandomPlayer(), RandomPlayer()]


def test_a_game_replays_exactly_from_its_seed():
    a = play(TINY, PLAYERS, TINY.start(2))
    b = play(TINY, PLAYERS, TINY.start(2))
    assert a.events == b.events
    assert a.final.ply == b.final.ply
    assert a.events != play(TINY, PLAYERS, TINY.start(3)).events


def test_a_record_names_its_winner():
    record = play(TINY, [GreedyPlayer(), RandomPlayer()], TINY.start(1), names=("g", "r"))
    assert record.final.over
    expected = None if record.final.winner is None else ("g", "r")[record.final.winner]
    assert record.winner == expected


def test_a_series_plays_each_board_from_both_sides():
    records = series(TINY, [("a", PLAYERS[0]), ("b", PLAYERS[1])], 2, seed=7)
    assert [r.names for r in records] == [("a", "b"), ("b", "a")] * 2
    assert [r.final.seed for r in records] == [7, 7, 8, 8]
    np.testing.assert_array_equal(records[0].final.state.wall, records[1].final.state.wall)


def test_a_series_in_parallel_gives_the_same_records_in_order():
    pair = ("search", SearchPlayer(simulations=3, plans=2, depth=2)), ("random", RandomPlayer())
    alone = series(TINY, pair, 2, seed=5)
    together = series(TINY, pair, 2, seed=5, workers=3)
    assert [r.names for r in together] == [r.names for r in alone]
    assert [r.events for r in together] == [r.events for r in alone]
    for a, b in zip(alone, together, strict=True):
        assert (a.final.ply, a.final.winner) == (b.final.ply, b.final.winner)
        np.testing.assert_array_equal(a.final.state.units.hp, b.final.state.units.hp)


def test_a_series_with_dice_of_their_own_keeps_each_pairs_board():
    pair = ("a", PLAYERS[0]), ("b", PLAYERS[1])
    shared = series(TINY, pair, 2, seed=7)
    own = series(TINY, pair, 2, seed=7, same_dice=False)
    seeds = [r.final.seed for r in own]
    assert len(set(seeds)) == 4
    assert not set(seeds) & {7, 8}
    assert seeds == [r.final.seed for r in series(TINY, pair, 2, seed=7, same_dice=False)]
    for a, b in zip(shared, own, strict=True):
        assert a.names == b.names
        np.testing.assert_array_equal(a.final.state.wall, b.final.state.wall)
        np.testing.assert_array_equal(a.final.state.home, b.final.state.home)
    assert [r.events for r in own] != [r.events for r in shared]


def test_a_series_with_dice_of_their_own_gives_the_same_records_in_parallel():
    pair = ("a", PLAYERS[0]), ("b", PLAYERS[1])
    alone = series(TINY, pair, 2, seed=2, same_dice=False)
    together = series(TINY, pair, 2, seed=2, same_dice=False, workers=2)
    assert [r.events for r in together] == [r.events for r in alone]


def test_series_refuse_bad_counts(not_an_integer):
    pair = ("a", PLAYERS[0]), ("b", PLAYERS[1])
    with pytest.raises(TypeError, match="workers"):
        series(TINY, pair, 1, workers=not_an_integer)
    with pytest.raises(ValueError, match="workers"):
        series(TINY, pair, 1, workers=0)
    with pytest.raises(TypeError, match="same_dice"):
        series(TINY, pair, 1, same_dice=1)  # ty: ignore[invalid-argument-type]


def test_a_summary_counts_add_up():
    records = series(TINY, [("greedy", GreedyPlayer()), ("random", RandomPlayer())], 2, seed=3)
    summary = summarize(TINY, records)
    counts = summary.tally
    assert counts.games == 4
    assert sum(counts.wins.values()) + counts.draws == 4
    assert counts.seat_wins.sum() == 4 - counts.draws
    assert set(counts.wins) == {"greedy", "random"}
    assert summary.damage_dealt.sum() == summary.damage_taken.sum() > 0
    np.testing.assert_array_equal(summary.kills.sum(), summary.losses.sum())
    assert counts.plies == np.mean([r.final.ply for r in records])
    assert summary.rounds == np.mean([r.final.round for r in records])


def test_summaries_refuse_no_games():
    with pytest.raises(ValueError, match="at least one"):
        summarize(TINY, [])


def test_a_summary_credits_damage_to_each_side_and_kills_to_the_killing_blow():
    def attack(attacker_kind, target, target_kind, damage):
        return Attacked(
            attacker=9,
            attacker_kind=attacker_kind,
            attacker_team=0,
            target=target,
            target_kind=target_kind,
            cell=0,
            roll=15,
            needed=8,
            atk=damage,
            defense=1,
            flank=False,
            hit=damage > 0,
            damage=damage,
            deferred=False,
        )

    events = (
        attack(2, 5, 1, 2),  # a knight wounds archer 5
        attack(3, 5, 1, 1),  # the captain finishes it
        Died(5, 1, 1, 0),
        attack(1, 6, 0, 0),  # an archer misses footsoldier 6
        GameOver(0),
    )
    final = TacticsPosition(TINY.start(0).state, 0, ply=10, round=3, winner=0, over=True)
    summary = summarize(TINY, [Record(("a", "b"), TINY.start(0), (), final, events)])
    np.testing.assert_array_equal(summary.damage_dealt, [0, 0, 2, 1])
    np.testing.assert_array_equal(summary.damage_taken, [0, 3, 0, 0])
    np.testing.assert_array_equal(summary.kills, [0, 0, 0, 1])
    np.testing.assert_array_equal(summary.losses, [0, 1, 0, 0])
    assert summary.tally.wins == {"a": 1, "b": 0}
    assert (summary.tally.plies, summary.rounds) == (10, 3)


def made_summary(games, team_wins, dealt):
    """A summary of ``games`` games with the given wins by team and damage dealt by kind."""
    dealt = np.array(dealt, dtype=np.int64)
    zeros = np.zeros_like(dealt)
    seat_wins = np.array(team_wins, dtype=np.int64)
    return Summary(
        tally=Tally(games, {"a": 0, "b": 0}, games - sum(team_wins), seat_wins, 1.0),
        rounds=1.0,
        damage_dealt=dealt,
        damage_taken=zeros,
        kills=zeros,
        losses=zeros,
    )


def test_balance_scores_sides_draws_and_damage_per_unit_deployed():
    # TINY deploys (2, 1, 1, 1) a side: over 5 games, 20, 10, 10 and 10 units of each kind.
    scores = balance(TINY, made_summary(5, [3, 1], [10, 3, 6, 4]))
    assert isinstance(scores, Balance)
    fair = scores.fairness
    assert (fair.decided, list(fair.seat_edges), fair.draws) == (4, [0.25, -0.25], 0.2)
    np.testing.assert_array_equal(scores.impact, [0.5, 0.3, 0.6, 0.4])
    assert scores.evenness == 0.5


def test_balance_gives_team_1_a_negative_edge():
    assert balance(TINY, made_summary(4, [1, 3], [1, 1, 1, 1])).fairness.seat_edges[0] == -0.25


def test_balance_without_a_decided_game_has_no_edge():
    scores = balance(TINY, made_summary(3, [0, 0], [1, 1, 1, 1]))
    fair = scores.fairness
    assert (fair.decided, list(fair.seat_edges), fair.draws) == (0, [0.0, 0.0], 1.0)


def test_balance_leaves_kinds_nobody_deploys_out_of_evenness():
    game = dataclasses.replace(TINY, counts=(2, 0, 1, 1))
    scores = balance(game, made_summary(1, [1, 0], [4, 0, 2, 1]))
    np.testing.assert_array_equal(scores.impact, [1.0, np.nan, 1.0, 0.5])
    assert scores.evenness == 0.5


def test_balance_of_games_without_damage_is_even():
    assert balance(TINY, made_summary(2, [0, 0], [0, 0, 0, 0])).evenness == 1.0


def test_balance_of_one_kind_doing_nothing_is_uneven():
    assert balance(TINY, made_summary(2, [1, 1], [0, 3, 3, 3])).evenness == 0.0


def test_balance_refuses_a_summary_of_other_rules():
    with pytest.raises(ValueError, match="3 kinds of unit"):
        balance(TINY, made_summary(1, [1, 0], [1, 1, 1]))


def test_balance_of_played_games_agrees_with_their_summary():
    records = series(TINY, [("greedy", GreedyPlayer()), ("random", RandomPlayer())], 2, seed=3)
    summary = summarize(TINY, records)
    scores = balance(TINY, summary)
    assert scores.fairness.decided == 4 - summary.tally.draws
    deployed = 2 * 4 * np.array(TINY.counts)
    np.testing.assert_array_equal(scores.impact, summary.damage_dealt / deployed)
    assert scores.evenness == scores.impact.min() / scores.impact.max()
