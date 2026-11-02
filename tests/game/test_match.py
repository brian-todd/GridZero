import dataclasses

import numpy as np
import pytest

from game_demo import CLAIM, FLIP, HIDDEN, Claims, FirstFree, RandomClaimer
from gridzero.game import (
    Action,
    Fairness,
    Score,
    Tally,
    fairness,
    ladder,
    play,
    replay,
    series,
    tally,
)

GAME = Claims()
THREE = Claims(length=7, players=3)
RANDOM = RandomClaimer()
PLAYERS = [RANDOM, RANDOM]


@dataclasses.dataclass(frozen=True)
class Scripted:
    """Plays the given actions in turn, whatever it is shown, and keeps what it saw."""

    actions: tuple[tuple[object, object], ...]
    seen: list[tuple[object, ...]] = dataclasses.field(default_factory=list, compare=False)

    def __call__(self, game, player, view, legal, rng):
        self.seen.append((view.copy(), legal.copy(), rng))
        return self.actions[len(self.seen) - 1]


@dataclasses.dataclass(frozen=True)
class Drawing:
    """Claims the first free cell, and keeps one draw of each generator it is given."""

    draws: list[float] = dataclasses.field(default_factory=list, compare=False)

    def __call__(self, game, player, view, legal, rng):
        self.draws.append(rng.random())
        return Action(CLAIM, int(np.flatnonzero(legal[CLAIM])[0]))


# Playing one game.


def test_a_game_replays_exactly_from_its_start():
    a = play(GAME, PLAYERS, GAME.start(2))
    b = play(GAME, PLAYERS, GAME.start(2))
    assert a.actions == b.actions
    assert a.events == b.events
    assert a.final.key() == b.final.key()
    assert {play(GAME, PLAYERS, GAME.start(s)).actions for s in range(8)} != {a.actions}


def test_a_record_keeps_the_start_the_actions_and_every_event():
    start = GAME.start(1)
    record = play(GAME, PLAYERS, start)
    assert record.start is start
    assert len(record.actions) == record.final.ply == GAME.length
    assert len(record.events) == GAME.length
    assert record.final.over


def test_a_record_names_its_winner():
    record = play(GAME, [FirstFree(), FirstFree()], GAME.start(0), names=("a", "b"))
    assert record.final.winner == 0  # three cells of five
    assert record.winner == "a"
    draw = play(Claims(length=4), [FirstFree(), FirstFree()], Claims(length=4).start(0))
    assert draw.final.winner is None
    assert draw.winner is None


def test_names_default_to_player_numbers():
    assert play(GAME, PLAYERS, GAME.start(0)).names == ("player 0", "player 1")


def test_players_are_shown_the_view_not_the_position():
    a = Scripted(((FLIP, 0), (CLAIM, 2), (CLAIM, 4)))
    b = Scripted(((CLAIM, 1), (CLAIM, 3)))
    record = play(GAME, [a, b], GAME.start(0))
    view, legal, _ = b.seen[0]
    np.testing.assert_array_equal(view, [HIDDEN, -1, -1, -1, -1])
    np.testing.assert_array_equal(legal, GAME.legal(record.start)[:, :] & [0, 1, 1, 1, 1])
    assert record.final.state.owner[0] in (0, 1)


@dataclasses.dataclass(frozen=True)
class Seated:
    """Claims the first free cell, and keeps which player it was told it is."""

    told: list[int] = dataclasses.field(default_factory=list, compare=False)

    def __call__(self, game, player, view, legal, rng):
        self.told.append(player)
        return Action(CLAIM, int(np.flatnonzero(legal[CLAIM])[0]))


def test_each_player_is_told_which_player_it_is():
    # Claims' view is an array that hides the ply: a player couldn't ask the rules.
    player = Seated()
    record = play(THREE, [player] * 3, THREE.start(0))
    assert player.told == [ply % 3 for ply in range(record.final.ply)]


def test_each_decision_gets_the_stream_of_its_ply():
    player = Drawing()
    record = play(GAME, [player, player], GAME.start(7))
    expected = [
        np.random.default_rng(np.random.SeedSequence(7, spawn_key=(2, ply))).random()
        for ply in range(record.final.ply)
    ]
    assert player.draws == expected


def test_actions_are_recorded_as_python_ints():
    player = Scripted(tuple((np.int64(CLAIM), np.int32(c)) for c in range(5)))
    record = play(GAME, [player, player], GAME.start(0))
    assert record.actions == tuple(Action(CLAIM, c) for c in range(5))
    assert all(type(a.verb) is int and type(a.cell) is int for a in record.actions)


def test_a_game_plays_from_any_start_not_over():
    start = GAME.step(GAME.step(GAME.start(3), (CLAIM, 0)), (CLAIM, 4))
    record = play(GAME, PLAYERS, start)
    assert len(record.actions) == GAME.length - 2
    assert record.final.state.owner[0] == 0


def test_an_illegal_action_raises_with_the_player_and_ply():
    cheat = Scripted(((CLAIM, 0), (CLAIM, 0)))
    with pytest.raises(ValueError, match="illegal") as error:
        play(GAME, [FirstFree(), cheat], GAME.start(0), names=("fair", "cheat"))
    assert "cheat (player 1) chose (0, 0) at ply 1" in error.value.__notes__


def test_play_refuses_wrong_counts_and_a_start_over():
    with pytest.raises(ValueError, match="2 players, got 1 players"):
        play(GAME, [RANDOM], GAME.start(0))
    with pytest.raises(ValueError, match="2 players, got 3 names"):
        play(GAME, PLAYERS, GAME.start(0), names=("a", "b", "c"))
    over = play(GAME, PLAYERS, GAME.start(0)).final
    with pytest.raises(ValueError, match="over"):
        play(GAME, PLAYERS, over)


# Replays.


def test_a_replay_gives_the_start_then_each_position():
    record = play(GAME, PLAYERS, GAME.start(4))
    positions = list(replay(GAME, record))
    assert positions[0] is record.start
    assert [p.ply for p in positions] == list(range(len(record.actions) + 1))
    assert positions[-1].key() == record.final.key()
    assert tuple(e for p in positions for e in p.events) == record.events


# Series.


def test_a_series_plays_each_start_from_both_sides():
    records = series(GAME, [("a", RANDOM), ("b", FirstFree())], 2, seed=7)
    assert [r.names for r in records] == [("a", "b"), ("b", "a")] * 2
    assert [r.start.seed for r in records] == [7, 7, 8, 8]


def test_a_series_rotates_the_seats_so_each_player_sits_in_each_once():
    players = [("a", RANDOM), ("b", FirstFree()), ("c", RANDOM)]
    records = series(THREE, players, 2, seed=7)
    rotations = [("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")]
    assert [r.names for r in records] == rotations * 2
    assert [r.start.seed for r in records] == [7, 7, 7, 8, 8, 8]


def test_a_series_with_dice_of_their_own_keeps_each_start():
    players = [("a", RANDOM), ("b", RANDOM), ("c", RANDOM)]
    own = series(THREE, players, 2, seed=7, same_dice=False)
    seeds = [r.start.seed for r in own]
    assert len(set(seeds)) == 6
    assert not set(seeds) & {7, 8}
    assert seeds == [r.start.seed for r in series(THREE, players, 2, seed=7, same_dice=False)]
    for record, start_seed in zip(own, [7, 7, 7, 8, 8, 8], strict=True):
        assert record.start.key() == THREE.start(start_seed).key()


def test_a_series_seed_of_its_own_comes_from_the_starts_seed_and_rotation():
    for game in (GAME, THREE):
        players = [(str(s), RANDOM) for s in range(game.n_players)]
        own = series(game, players, 2, seed=7, same_dice=False)
        n = game.n_players
        for i, record in enumerate(own):
            spawn = np.random.SeedSequence(7 + i // n, spawn_key=(3, i % n))
            assert record.start.seed == int(spawn.generate_state(1, np.uint64)[0]) >> 1


def test_a_series_in_parallel_gives_the_same_records_in_order():
    for game in (GAME, THREE):
        players = [("a", RANDOM), ("b", FirstFree()), ("c", RANDOM)][: game.n_players]
        for same_dice in (True, False):
            alone = series(game, players, 2, seed=5, same_dice=same_dice)
            together = series(game, players, 2, seed=5, same_dice=same_dice, workers=3)
            assert [r.names for r in together] == [r.names for r in alone]
            assert [r.actions for r in together] == [r.actions for r in alone]
            assert [r.final.key() for r in together] == [r.final.key() for r in alone]


def test_series_refuse_bad_input(not_an_integer):
    pair = ("a", RANDOM), ("b", RANDOM)
    with pytest.raises(TypeError, match="starts"):
        series(GAME, pair, not_an_integer)
    with pytest.raises(TypeError, match="workers"):
        series(GAME, pair, 1, workers=not_an_integer)
    with pytest.raises(TypeError, match="seed"):
        series(GAME, pair, 1, seed=not_an_integer)
    with pytest.raises(ValueError, match="starts"):
        series(GAME, pair, 0)
    with pytest.raises(ValueError, match="workers"):
        series(GAME, pair, 1, workers=0)
    with pytest.raises(ValueError, match="seed"):
        series(GAME, pair, 1, seed=-1)
    with pytest.raises(TypeError, match="same_dice"):
        series(GAME, pair, 1, same_dice=1)  # ty: ignore[invalid-argument-type]


def test_a_series_needs_one_player_per_seat():
    with pytest.raises(ValueError, match="3 players, got 2 players"):
        series(THREE, [("a", RANDOM), ("b", RANDOM)], 1)
    with pytest.raises(ValueError, match="2 players, got 3 players"):
        series(GAME, [("a", RANDOM), ("b", RANDOM), ("c", RANDOM)], 1)


# Tallies.


def test_a_tally_counts_wins_by_name_and_by_seat():
    for game in (GAME, THREE):
        players = [("first", FirstFree()), ("random", RANDOM), ("other", RANDOM)]
        players = players[: game.n_players]
        records = series(game, players, 10, seed=3)
        result = tally(records)
        games = 10 * game.n_players
        assert result.games == games
        assert set(result.wins) == {name for name, _ in players}
        assert sum(result.wins.values()) + result.draws == games
        assert result.seat_wins.dtype == np.int64
        assert result.seat_wins.sum() == sum(result.wins.values())
        assert result.wins == {name: sum(r.winner == name for r in records) for name, _ in players}
        seats = range(game.n_players)
        assert list(result.seat_wins) == [sum(r.final.winner == s for r in records) for s in seats]
        assert result.plies == game.length


def test_a_tally_counts_players_of_one_name_as_one():
    records = series(THREE, [("first", FirstFree())] + [("random", RANDOM)] * 2, 10, seed=3)
    result = tally(records)
    assert set(result.wins) == {"first", "random"}
    assert result.wins["random"] == sum(r.winner == "random" for r in records)
    assert result.score("first").lost == result.wins["random"]


def test_a_tally_counts_draws():
    small = Claims(length=4)
    records = series(small, [("a", FirstFree()), ("b", FirstFree())], 1)
    result = tally(records)
    assert result.draws == 2
    assert result.wins == {"a": 0, "b": 0}


def test_a_tally_refuses_no_games_and_mixed_seats():
    with pytest.raises(ValueError, match="at least one"):
        tally([])
    record = play(GAME, PLAYERS, GAME.start(0))
    three = dataclasses.replace(record, names=("a", "b", "c"))
    with pytest.raises(ValueError, match="numbers of players"):
        tally([record, three])


def test_a_score_is_a_players_wins_losses_and_draws():
    records = series(Claims(length=4), [("a", RANDOM), ("b", RANDOM)], 10, same_dice=False)
    result = tally(records)
    assert result.wins == {"a": 2, "b": 4}  # unequal, so a loss can't pass for a win
    assert result.draws == 14
    for name, other in (("a", "b"), ("b", "a")):
        score = result.score(name)
        assert isinstance(score, Score)
        assert score == (result.wins[name], result.wins[other], result.draws)


def test_a_score_needs_a_player_who_played():
    result = tally(series(GAME, [("a", RANDOM), ("b", RANDOM)], 1))
    with pytest.raises(KeyError, match="c"):
        result.score("c")


def made_tally(games, seat_wins):
    """A tally of ``games`` games with the given wins by seat."""
    wins = dict(zip("abcd", seat_wins, strict=False))
    return Tally(games, wins, games - sum(seat_wins), np.array(seat_wins, dtype=np.int64), 1.0)


def test_fairness_is_each_seats_edge_over_the_decided_games_and_the_draws():
    scores = fairness(made_tally(5, [3, 1]))
    assert (scores.decided, scores.draws) == (4, 0.2)
    assert scores.seat_edges.dtype == np.float64
    assert list(scores.seat_edges) == [0.25, -0.25]
    assert list(fairness(made_tally(4, [1, 3])).seat_edges) == [-0.25, 0.25]
    three = fairness(made_tally(10, [4, 2, 2]))
    np.testing.assert_allclose(three.seat_edges, [0.5 - 1 / 3, 0.25 - 1 / 3, 0.25 - 1 / 3])
    assert three.seat_edges.sum() == pytest.approx(0.0)


def test_fairness_without_a_decided_game_has_no_edge():
    for seat_wins in ([0, 0], [0, 0, 0]):
        scores = fairness(made_tally(3, seat_wins))
        assert (scores.decided, scores.draws) == (0, 1.0)
        assert list(scores.seat_edges) == [0.0] * len(seat_wins)


def test_fairness_of_real_games_counts_their_tally():
    for game in (GAME, THREE):
        players = [("first", FirstFree()), ("random", RANDOM), ("other", RANDOM)]
        records = series(game, players[: game.n_players], 10, seed=3, same_dice=False)
        counts = tally(records)
        scores = fairness(counts)
        assert scores.decided == counts.games - counts.draws
        n = game.n_players
        assert list(scores.seat_edges) == list(counts.seat_wins / scores.decided - 1 / n)


def edges(decided, *seat_edges):
    return Fairness(decided, np.array(seat_edges), 0.0)


def test_the_excess_edge_is_what_lies_beyond_the_noise_of_fair_seats():
    # 100 decided games of two seats: a fair coin's noise is 0.1.
    assert edges(100, -0.3, 0.3).excess_edge() == pytest.approx(0.2)
    assert edges(100, 0.1, -0.1).excess_edge() == 0.0
    assert edges(100, 0.05, -0.05).excess_edge() == 0.0
    assert Fairness(0, np.zeros(2), 1.0).excess_edge() == 0.0
    # 72 decided games of three seats: the noise is 2 * sqrt(2 / 72) / 3 = 1 / 9.
    assert edges(72, 0.0, -0.2, 0.2).excess_edge() == pytest.approx(0.2 - 1 / 9)
    assert edges(72, 0.1, -0.05, -0.05).excess_edge() == 0.0


# Ladders.


def test_a_ladder_scores_the_player_against_each_rival_in_order():
    rivals = {"random": RANDOM, "first": FirstFree()}
    scores = ladder(GAME, FirstFree(), rivals, 5, seed=3)
    assert list(scores) == ["random", "first"]
    for name, rival in rivals.items():
        records = series(GAME, [("player", FirstFree()), (name, rival)], 5, seed=3)
        assert scores[name] == tally(records).score("player")
        assert sum(scores[name]) == 10


def test_a_ladders_rival_takes_every_other_seat():
    scores = ladder(THREE, FirstFree(), {"random": RANDOM}, 5, seed=3)
    records = series(THREE, [("player", FirstFree())] + [("random", RANDOM)] * 2, 5, seed=3)
    assert scores["random"] == tally(records).score("player")
    assert sum(scores["random"]) == 15


def test_a_ladder_passes_its_dice_and_workers_to_each_series():
    rivals = {"random": RANDOM}
    own = ladder(GAME, RANDOM, rivals, 6, seed=1, same_dice=False)
    records = series(GAME, [("player", RANDOM), ("random", RANDOM)], 6, seed=1, same_dice=False)
    assert own["random"] == tally(records).score("player")
    assert ladder(GAME, RANDOM, rivals, 6, seed=1, same_dice=False, workers=2) == own
    assert ladder(GAME, RANDOM, rivals, 6, seed=1) != own  # shared dice play other games


def test_a_ladder_refuses_bad_input(not_an_integer):
    with pytest.raises(TypeError, match="starts"):
        ladder(GAME, RANDOM, {"b": RANDOM}, not_an_integer)
    with pytest.raises(ValueError, match="starts"):
        ladder(GAME, RANDOM, {"b": RANDOM}, 0)
    with pytest.raises(ValueError, match="named"):
        ladder(GAME, RANDOM, {"player": RANDOM}, 1)
    with pytest.raises(TypeError, match="same_dice"):
        ladder(GAME, RANDOM, {"b": RANDOM}, 1, same_dice=1)  # ty: ignore[invalid-argument-type]
