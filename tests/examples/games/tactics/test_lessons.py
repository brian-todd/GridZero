from dataclasses import fields

import numpy as np
import pytest

from examples.games.tactics import Lessons, SearchPlayer, lessons_of, positions_of, teach
from gridzero.game import Action, play
from tactics_demo import TINY, WORKS

TEACHER = SearchPlayer(simulations=4, plans=3, depth=2, rollout_turns=2)
WORKING = SearchPlayer(simulations=4, plans=2, depth=2, rollout_turns=2, work_plans=3)


def game_of(seed, game=TINY, teacher=TEACHER):
    return lessons_of(game, teacher, seed)


def same(a, b):
    return all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(Lessons))


def test_a_game_replays_to_the_winner_and_the_round_it_kept():
    lessons = game_of(1)
    position = None
    action = None
    count = 0
    for position, action in positions_of(TINY, lessons, 0):
        assert TINY.legal(position)[action]
        count += 1
    assert position is not None
    assert action is not None
    final = TINY.step(position, action)
    assert final.over
    assert final.winner == lessons.winner[0]
    assert final.round == lessons.rounds[0]
    assert count == lessons.verb.size


def test_a_search_is_kept_where_a_turn_with_a_choice_begins():
    lessons = game_of(2)
    assert lessons.searched.size > 0
    assert (np.diff(lessons.searched) > 0).all()
    positions = list(positions_of(TINY, lessons, 0))
    previous = None
    for at in lessons.searched:
        position, _ = positions[at]
        assert not position.deploying
        assert np.count_nonzero(TINY.legal(position)) > 1
        turn = (position.round, position.state.units.uid[TINY.active(position)])
        assert turn != previous  # one search a turn
        previous = turn


def test_each_search_keeps_every_simulation_and_its_value_averages_its_plans():
    lessons = game_of(3)
    for s in range(lessons.searched.size):
        first, last = lessons.plan_start[s], lessons.plan_start[s + 1]
        visits = lessons.plan_visits[first:last]
        values = lessons.plan_value[first:last]
        assert visits.sum() == TEACHER.simulations
        assert (values >= 0).all()
        assert (values <= 1).all()
        assert lessons.value[s] == pytest.approx((visits * values).sum() / visits.sum())


def test_a_game_plays_as_the_teacher_would_alone():
    lessons = game_of(4)
    record = play(TINY, [TEACHER, TEACHER], TINY.start(4))
    assert [(a.verb, a.cell) for a in record.actions] == list(
        zip(lessons.verb.tolist(), lessons.cell.tolist(), strict=True)
    )


def test_the_same_seed_gives_the_same_game_and_another_seed_another():
    assert same(game_of(5), game_of(5))
    assert not np.array_equal(game_of(5).cell, game_of(6).cell)


def test_plans_to_work_are_kept_with_their_tiles_and_others_are_marked_none():
    found = 0
    for seed in range(4):
        lessons = game_of(seed, WORKS, WORKING)
        work = lessons.plan_work_verb >= 0
        assert ((lessons.plan_work_cell >= 0) == work).all()
        found += work.sum()
    assert found > 0


def test_join_runs_the_indices_on_and_each_game_replays_as_it_did_alone():
    parts = [game_of(seed) for seed in (7, 8, 9)]
    joined = Lessons.join(parts)
    assert len(joined) == 3
    assert joined.seed.tolist() == [7, 8, 9]
    assert joined.game_start[-1] == joined.verb.size
    assert joined.plan_start[-1] == joined.plan_tile.size
    for i, part in enumerate(parts):
        first, last = joined.game_start[i], joined.game_start[i + 1]
        assert np.array_equal(joined.verb[first:last], part.verb)
        assert np.array_equal(joined.cell[first:last], part.cell)
        mine = (joined.searched >= first) & (joined.searched < last)
        assert np.array_equal(joined.searched[mine] - first, part.searched)
        assert np.array_equal(joined.value[mine], part.value)
        s = np.flatnonzero(mine)[0]
        begin, end = joined.plan_start[s], joined.plan_start[s + part.searched.size]
        assert np.array_equal(joined.plan_visits[begin:end], part.plan_visits)
        assert np.array_equal(joined.plan_tile[begin:end], part.plan_tile)


def test_join_sorts_parts_by_seed_whatever_their_order():
    a, b = game_of(7), game_of(8)
    assert same(Lessons.join([b, a]), Lessons.join([a, b]))


def test_join_refuses_a_seed_twice():
    a = game_of(7)
    with pytest.raises(ValueError, match="seeds of their own"):
        Lessons.join([a, a])


def test_saved_lessons_load_the_same_and_leave_no_partial_file(tmp_path):
    lessons = Lessons.join([game_of(7), game_of(8)])
    path = tmp_path / "x_lessons.npz"
    lessons.save(path)
    assert same(Lessons.load(path), lessons)
    assert [p.name for p in tmp_path.iterdir()] == ["x_lessons.npz"]
    for f in fields(Lessons):
        assert getattr(Lessons.load(path), f.name).dtype == getattr(lessons, f.name).dtype


def test_decide_is_what_the_player_does_and_judges_only_where_it_searches():
    position = TINY.start(1)
    rng = np.random.default_rng(1)
    while position.deploying:
        legal = TINY.legal(position)
        action, judgment = TEACHER.decide(TINY, TINY.player(position), position, legal, rng)
        assert judgment is None
        position = TINY.step(position, action)
    legal = TINY.legal(position)
    player = SearchPlayer(simulations=4, plans=3, depth=2, rollout_turns=2)
    first = player.decide(TINY, TINY.player(position), position, legal, np.random.default_rng(2))
    other = SearchPlayer(simulations=4, plans=3, depth=2, rollout_turns=2)
    assert first[0] == other(TINY, TINY.player(position), position, legal, np.random.default_rng(2))
    assert first[1] is not None
    assert sum(first[1].visits) == 4
    after = TINY.step(position, first[0])
    if not after.over and TINY.active(after) == TINY.active(position):
        legal = TINY.legal(after)
        assert (
            player.decide(TINY, TINY.player(after), after, legal, np.random.default_rng(3))[1]
            is None
        )


def test_teach_keeps_every_game_in_seed_order_and_saves_them(tmp_path):
    path = tmp_path / "t_lessons.npz"
    lessons = teach(TINY, TEACHER, [3, 1, 2], path, seconds=300.0, workers=2)
    assert lessons.seed.tolist() == [1, 2, 3]
    assert same(lessons, Lessons.join([game_of(s) for s in (1, 2, 3)]))
    assert same(Lessons.load(path), lessons)


def test_teach_stops_at_its_time_limit_and_keeps_what_finished(tmp_path):
    path = tmp_path / "t_lessons.npz"
    lessons = teach(TINY, TEACHER, list(range(500)), path, seconds=2.0, workers=2)
    assert 0 < len(lessons) < 500
    assert same(Lessons.load(path), lessons)


def test_an_action_of_lessons_is_an_action():
    _, action = next(positions_of(TINY, game_of(1), 0))
    assert isinstance(action, Action)
