import math
from dataclasses import replace

import numpy as np
import pytest

from examples.games.regression import held_out
from examples.games.tactics import (
    HAND_PICKED,
    MOVE_FEATURES,
    VALUE_FEATURES,
    GreedyPlayer,
    Lessons,
    SearchPlayer,
    lessons_of,
    move_features,
    positions_of,
    value_features,
)
from examples.games.tactics.fit import (
    LOOKAHEAD,
    Cases,
    PriorScore,
    cases_of,
    fit,
    fit_prior,
    fit_value,
    game_cases,
    join_cases,
    overruled,
    playing,
    prior_score,
    value_score,
)
from gridzero.game import Action
from tactics_demo import WORKS

TEACHER = SearchPlayer(simulations=4, plans=3, depth=2, rollout_turns=2)
N_VALUE, N_MOVE = len(VALUE_FEATURES), len(MOVE_FEATURES)
ALL_VALUE, ALL_MOVE = np.arange(N_VALUE), np.arange(N_MOVE)


def real_game(seed):
    return lessons_of(WORKS, TEACHER, seed)


def made_cases(
    games=40, decisions=30, moves=4, truth=None, value=None, seed=0, chosen=None, sizes=None
):
    """Cases of made-up games: moves drawn from a logit of ``truth``, values from ``value``."""
    rng = np.random.default_rng(seed)
    truth = np.zeros(N_MOVE) if truth is None else truth
    value = np.full(N_VALUE, 0.5) if value is None else value
    d = games * decisions
    sizes = np.full(d, moves) if sizes is None else sizes
    start = np.cumsum(np.r_[0, sizes], dtype=np.int64)
    x = rng.normal(size=(start[-1], N_MOVE))
    logits = x @ truth + rng.gumbel(size=start[-1])
    first = start[:-1]
    best = np.repeat(np.maximum.reduceat(logits, first), sizes)
    picked = np.flatnonzero(logits == best) if chosen is None else first + chosen
    values = rng.normal(size=(d, N_VALUE))
    target = 1.0 / (1.0 + np.exp(-(values @ value)))
    game = np.repeat(np.arange(games, dtype=np.int64), decisions)
    return Cases(
        games,
        game,
        x,
        start,
        picked.astype(np.int64),
        first.astype(np.int64),
        game,
        values,
        target,
        np.ones(d, dtype=np.bool_),
        (target > 0.5).astype(np.float64),
    )


def test_each_decisions_moves_are_its_legal_moves_and_chosen_is_the_teachers():
    lessons = real_game(1)
    cases = game_cases(WORKS, lessons)
    assert cases.chosen.size == lessons.verb.size
    for d, (position, action) in enumerate(positions_of(WORKS, lessons, 0)):
        legal = WORKS.legal(position)
        first, last = cases.start[d], cases.start[d + 1]
        assert np.array_equal(cases.moves[first:last], move_features(WORKS, position, legal))
        verbs, cells = legal.nonzero()
        at = cases.chosen[d] - first
        assert Action(int(verbs[at]), int(cells[at])) == action


def test_greedy_is_what_greedy_does_there_with_the_games_draws():
    lessons = real_game(2)
    cases = game_cases(WORKS, lessons)
    rng = np.random.default_rng(int(lessons.seed[0]))
    for d, (position, _) in enumerate(positions_of(WORKS, lessons, 0)):
        legal = WORKS.legal(position)
        action = GreedyPlayer()(WORKS, WORKS.player(position), position, legal, rng)
        verbs, cells = legal.nonzero()
        at = cases.greedy[d] - cases.start[d]
        assert Action(int(verbs[at]), int(cells[at])) == action


def test_searched_positions_carry_the_teachers_value_and_deployment_its_first():
    lessons = real_game(3)
    cases = game_cases(WORKS, lessons)
    positions = [position for position, _ in positions_of(WORKS, lessons, 0)]
    deploying = [position for position in positions if position.deploying]
    searched = [positions[at] for at in lessons.searched]
    assert cases.searched.sum() == len(searched)
    assert (~cases.searched).sum() == len(deploying)
    assert (cases.target[~cases.searched] == lessons.value[0]).all()
    assert np.array_equal(cases.target[cases.searched], lessons.value)
    assert np.array_equal(cases.values[cases.searched], value_features(WORKS, searched))
    assert np.array_equal(cases.values[~cases.searched], value_features(WORKS, deploying))


@pytest.mark.parametrize("winner", [0, 1])
def test_won_says_whether_team_zero_went_on_to_win(winner):
    lessons = real_game(4)
    changed = replace(lessons, winner=np.array([winner], dtype=np.int8))
    assert (game_cases(WORKS, changed).won == float(winner == 0)).all()


def test_a_game_with_no_search_judges_nothing():
    lessons = real_game(5)
    empty = replace(
        lessons,
        searched=np.zeros(0, dtype=np.int64),
        value=np.zeros(0),
        plan_start=np.zeros(1, dtype=np.int64),
        plan_tile=lessons.plan_tile[:0],
        plan_attack_first=lessons.plan_attack_first[:0],
        plan_work_verb=lessons.plan_work_verb[:0],
        plan_work_cell=lessons.plan_work_cell[:0],
        plan_visits=lessons.plan_visits[:0],
        plan_value=lessons.plan_value[:0],
    )
    cases = game_cases(WORKS, empty)
    assert cases.values.shape == (0, N_VALUE)
    assert cases.chosen.size == lessons.verb.size


def test_join_numbers_games_and_shifts_rows_to_their_own_moves():
    a, b = game_cases(WORKS, real_game(6)), game_cases(WORKS, real_game(7))
    both = join_cases([a, b])
    assert both.games == 2
    assert both.game.tolist() == [0] * a.chosen.size + [1] * b.chosen.size
    assert both.value_game.tolist() == [0] * a.target.size + [1] * b.target.size
    assert np.array_equal(both.moves[both.chosen[: a.chosen.size]], a.moves[a.chosen])
    assert np.array_equal(both.moves[both.chosen[a.chosen.size :]], b.moves[b.chosen])
    assert np.array_equal(both.moves[both.greedy[a.chosen.size :]], b.moves[b.greedy])
    assert both.start[0] == 0
    assert both.start[-1] == both.moves.shape[0]
    assert (np.diff(both.start) > 0).all()


def test_cases_of_is_each_game_in_order_on_any_number_of_processes():
    lessons = Lessons.join([real_game(8), real_game(9)])
    joined = join_cases([game_cases(WORKS, lessons.only(i)) for i in range(2)])
    together = cases_of(WORKS, lessons, workers=2)
    for a, b in zip(together, joined, strict=True):
        assert np.array_equal(a, b)


def test_a_fit_recovers_the_weights_that_made_the_cases():
    truth = np.zeros(N_MOVE)
    truth[:3] = 1.5, -1.0, 0.5
    value = np.zeros(N_VALUE)
    value[:2] = 1.0, -2.0
    evaluator, score = fit(made_cases(truth=truth, value=value), 0)
    assert np.array(evaluator.prior) == pytest.approx(truth, abs=0.25)
    assert np.array(evaluator.value) == pytest.approx(value, abs=0.1)
    assert evaluator.intercept == pytest.approx(0.0, abs=0.05)
    assert score.value["fit"].correlation > 0.99
    assert score.prior["fit"].agreement > score.prior["uniform"].agreement
    assert (score.games, score.games_out) == (32, 8)


def test_fit_reads_only_the_games_it_keeps_in():
    cases = made_cases(games=10, decisions=8)
    out = np.zeros(10, dtype=np.bool_)
    out[3] = True
    kept = fit_value(cases, ~out, ALL_VALUE), fit_prior(cases, ~out, ALL_MOVE)
    changed_value = cases.target.copy()
    changed_value[cases.value_game == 3] = 1.0
    changed_chosen = cases.chosen.copy()
    changed_chosen[cases.game == 3] = cases.start[:-1][cases.game == 3] + 1
    changed = cases._replace(target=changed_value, chosen=changed_chosen)
    assert fit_value(changed, ~out, ALL_VALUE) == kept[0]
    assert fit_prior(changed, ~out, ALL_MOVE) == kept[1]
    inside = np.array(cases.target)
    inside[cases.value_game == 4] = 1.0
    assert fit_value(cases._replace(target=inside), ~out, ALL_VALUE) != kept[0]


def test_fits_give_features_left_out_no_weight():
    cases = made_cases(games=6, decisions=6)
    games = np.ones(6, dtype=np.bool_)
    columns = np.array([1, 3])
    value, _ = fit_value(cases, games, columns)
    prior = fit_prior(cases, games, columns)
    assert [j for j, w in enumerate(value) if w != 0.0] == [1, 3]
    assert [j for j, w in enumerate(prior) if w != 0.0] == [1, 3]


def test_a_fit_of_the_prior_counts_no_decision_without_a_choice():
    sizes = np.tile([3, 1], 30)
    cases = made_cases(games=6, decisions=10, sizes=sizes, chosen=np.zeros(60, dtype=np.int64))
    games = np.ones(6, dtype=np.bool_)
    rows = np.repeat(sizes > 1, sizes)
    only_choices = cases._replace(
        game=cases.game[sizes > 1],
        moves=cases.moves[rows],
        start=np.arange(31, dtype=np.int64) * 3,
        chosen=np.arange(30, dtype=np.int64) * 3,
    )
    assert fit_prior(cases, games, ALL_MOVE) == fit_prior(only_choices, games, ALL_MOVE)


def test_the_constants_of_a_fit_read_nothing_of_games_kept_out():
    cases = made_cases(games=10, decisions=8)
    out = held_out(np.arange(10, dtype=np.int64), 0)
    target = cases.target.copy()
    target[out[cases.value_game]] = 1.0
    chosen = cases.chosen.copy()
    chosen[out[cases.game]] = cases.start[:-1][out[cases.game]] + 1
    evaluator, _ = fit(cases, 0)
    assert fit(cases._replace(target=target, chosen=chosen), 0)[0] == evaluator


def test_a_fit_of_one_game_has_none_to_keep_out():
    with pytest.raises(ValueError, match="at least 2 games"):
        fit(made_cases(games=1), 0)


def test_baselines_are_named_and_the_teachers_value_is_its_own():
    _, score = fit(made_cases(games=10, decisions=8), 0)
    assert list(score.value) == ["mean", "material", "hand", "fit", "teacher"]
    assert list(score.prior) == ["uniform", "verbs", "hand", "greedy", "fit"]
    assert score.value["mean"].correlation == 0.0
    assert score.value["teacher"].correlation == pytest.approx(1.0)
    assert score.value["teacher"].error == 0.0
    assert math.isnan(score.prior["greedy"].loss)


def test_material_and_verbs_baselines_read_only_their_columns():
    """A value written in a column they can't read leaves them where a fit on it would not be."""
    value = np.zeros(N_VALUE)
    value[VALUE_FEATURES.index("clock")] = 3.0
    truth = np.zeros(N_MOVE)
    truth[MOVE_FEATURES.index("damage")] = 3.0
    _, score = fit(made_cases(value=value, truth=truth), 0)
    assert abs(score.value["material"].correlation) < 0.2
    assert score.value["fit"].correlation > 0.99
    assert score.prior["verbs"].agreement < 0.35
    assert score.prior["fit"].agreement > 0.6


def test_value_scores_read_only_positions_searched():
    cases = made_cases(games=10, decisions=8)
    some = cases._replace(searched=cases.value_game % 2 == 0)
    loss = fit(cases, 0)[1].value["teacher"].winner_loss
    assert fit(some, 0)[1].value["teacher"].winner_loss != loss


def test_scores_read_only_games_kept_out():
    cases = made_cases(games=10, decisions=8)
    out = held_out(np.arange(10, dtype=np.int64), 0)
    won = cases.won.copy()
    won[~out[cases.value_game]] = 1.0 - won[~out[cases.value_game]]
    chosen = cases.chosen.copy()
    chosen[~out[cases.game]] = cases.start[:-1][~out[cases.game]]
    _, score = fit(cases, 0)
    assert fit(cases._replace(won=won), 0)[1].value["teacher"] == score.value["teacher"]
    assert fit(cases._replace(chosen=chosen), 0)[1].prior["hand"] == score.prior["hand"]


def hand_cases():
    """Two decisions by hand: the first's moves score 2, 1, 0, the second's tie at 1."""
    moves = np.zeros((6, N_MOVE))
    moves[:6, 0] = 2.0, 1.0, 0.0, 1.0, 1.0, 1.0
    weights = (1.0,) + (0.0,) * (N_MOVE - 1)
    cases = Cases(
        2,
        np.array([0, 1], dtype=np.int64),
        moves,
        np.array([0, 3, 6], dtype=np.int64),
        np.array([0, 4], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, N_VALUE)),
        np.zeros(0),
        np.zeros(0, dtype=np.bool_),
        np.zeros(0),
    )
    return cases, weights


def test_prior_score_by_hand():
    cases, weights = hand_cases()
    first = np.array([True, False])
    second = np.array([False, True])
    p = np.exp(2.0) / (np.exp(2.0) + np.exp(1.0) + 1.0)
    assert prior_score(cases, weights, first) == pytest.approx(PriorScore(1.0, -math.log(p)))
    # Three moves tied: the favorite is the teacher's a third of the time.
    assert prior_score(cases, weights, second) == pytest.approx(PriorScore(1 / 3, math.log(3.0)))


def test_prior_score_with_the_teacher_away_from_the_favorite():
    cases, weights = hand_cases()
    other = cases._replace(chosen=np.array([2, 4], dtype=np.int64))
    p = 1.0 / (np.exp(2.0) + np.exp(1.0) + 1.0)
    score = prior_score(other, weights, np.array([True, False]))
    assert score == pytest.approx(PriorScore(0.0, -math.log(p)))


def test_prior_score_with_no_decisions_has_nothing_to_say():
    cases, weights = hand_cases()
    score = prior_score(cases, weights, np.zeros(2, dtype=np.bool_))
    assert score.agreement == 0.0
    assert math.isnan(score.loss)


def test_value_score_by_hand():
    predicted = np.array([0.75, 0.25, 0.5])
    target = np.array([0.5, 0.25, 0.5])
    won = np.array([1.0, 1.0, 0.0])
    score = value_score(predicted, target, won)
    assert score.error == pytest.approx(0.25 / math.sqrt(3.0))
    loss = -(math.log(0.75) + math.log(0.25) + math.log(0.5)) / 3.0
    assert score.winner_loss == pytest.approx(loss)
    assert score.correlation == pytest.approx(np.corrcoef(predicted, target)[0, 1])


def test_value_score_is_finite_for_values_of_exactly_0_and_1_that_are_wrong():
    score = value_score(np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert math.isfinite(score.winner_loss)
    assert score.winner_loss == pytest.approx(-math.log(1e-6), rel=1e-3)


def test_value_score_with_no_positions_has_nothing_to_say():
    score = value_score(np.zeros(0), np.zeros(0), np.zeros(0))
    assert score.correlation == 0.0
    assert math.isnan(score.error)
    assert math.isnan(score.winner_loss)


def test_overruled_takes_only_the_knights_weight_from_the_hand_picked_constants():
    fitted, _ = fit(made_cases(games=6, decisions=6), 0)
    played = overruled(fitted)
    knight = VALUE_FEATURES.index("hp_knight")
    assert played.value[knight] == HAND_PICKED.value[knight]
    assert played.value[:knight] == fitted.value[:knight]
    assert played.value[knight + 1 :] == fitted.value[knight + 1 :]
    assert (played.intercept, played.prior) == (fitted.intercept, fitted.prior)
    assert played.lookahead == fitted.lookahead == 0


def test_the_constants_that_play_are_the_fit_overruled_and_looking_ahead():
    fitted, _ = fit(made_cases(games=6, decisions=6), 0)
    assert playing(fitted) == replace(overruled(fitted), lookahead=LOOKAHEAD)
    assert LOOKAHEAD > 0
