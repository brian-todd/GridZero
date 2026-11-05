"""Tests of the code players of k in a row: evaluators written as code."""

import dataclasses

import numpy as np
import pytest

from examples.games.in_a_row import (
    MOVE_FEATURES,
    PLACE,
    SEARCH_ALONE,
    VALUE_FEATURES,
    FittedEvaluator,
    InARow,
    InARowPosition,
    InARowState,
    LineEvaluator,
    clean_threats,
    move_features,
    open_lines,
    value_features,
)
from gridzero.agents import AlphaZeroPlayer

GAME = InARow()


def drop(position, column):
    """Return the position after a stone falls into a column."""
    cells = np.flatnonzero(GAME.legal(position)[PLACE])
    cell = cells[GAME.space.unravel(cells)[:, 0] == column][0]
    return GAME.step(position, (PLACE, int(cell)))


def random_positions(game, n, seed=0):
    """Return ``n`` positions not over, each after a random number of uniform moves."""
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < n:
        position = game.start(0)
        for _ in range(rng.integers(0, game.space.n_cells)):
            cells = np.flatnonzero(game.legal(position)[PLACE])
            after = game.step(position, (PLACE, int(rng.choice(cells))))
            if after.over:
                break
            position = after
        positions.append(position)
    return positions


def swapped(game, position):
    """Return the position with every stone's owner swapped, at the same ply."""
    owner = position.state.owner.copy()
    owner[owner >= 0] = 1 - owner[owner >= 0]
    state = InARowState(game.space, {InARowState.owner: owner})
    return InARowPosition(state, 0, ply=position.ply)


def counted_by_hand(game, position):
    counts = np.zeros((2, game.length + 1), dtype=np.int64)
    for line in game.lines:
        owners = [int(position.state.owner[cell]) for cell in line]
        for player in (0, 1):
            if 1 - player not in owners:
                counts[player, owners.count(player)] += 1
    return counts


@pytest.mark.parametrize(
    "game", [GAME, InARow((3, 3), 3, gravity=False), InARow((5, 4), 3)], ids=str
)
def test_open_lines_match_a_count_line_by_line(game):
    positions = random_positions(game, 12)
    counts = open_lines(game, positions)
    assert counts.shape == (12, 2, game.length + 1)
    assert counts.dtype == np.int64
    for position, count in zip(positions, counts, strict=True):
        np.testing.assert_array_equal(count, counted_by_hand(game, position))


def test_on_the_empty_board_every_line_is_open_and_empty_for_both():
    counts = open_lines(GAME, [GAME.start(0)])[0]
    assert (counts[:, 0] == len(GAME.lines)).all()  # 69 lines of four on 7x6
    assert (counts[:, 1:] == 0).all()


def test_a_corner_stone_opens_three_lines_for_its_player_and_closes_them_for_the_other():
    position = GAME.step(GAME.start(0), (PLACE, 0))  # bottom of column 0
    counts = open_lines(GAME, [position])[0]
    assert counts[0, 1] == 3  # along its row, its column and one diagonal
    assert counts[1, 0] == len(GAME.lines) - 3


def test_the_values_are_the_same_in_a_batch_and_one_at_a_time():
    positions = random_positions(GAME, 10, seed=1)
    legal = [GAME.legal(p) for p in positions]
    logits, values = LineEvaluator()(GAME, positions, legal)
    for i, position in enumerate(positions):
        one_logits, one_value = LineEvaluator()(GAME, [position], [legal[i]])
        np.testing.assert_array_equal(one_logits[0], logits[i])
        assert one_value[0] == values[i]


def test_swapping_every_stone_mirrors_the_value():
    positions = random_positions(GAME, 10, seed=2)
    mirrored = [swapped(GAME, p) for p in positions]
    evaluator = LineEvaluator()
    _, values = evaluator(GAME, positions, [GAME.legal(p) for p in positions])
    _, flipped = evaluator(GAME, mirrored, [GAME.legal(p) for p in mirrored])
    np.testing.assert_allclose(values + flipped, 1.0, atol=1e-15)


def test_the_value_rises_with_a_players_open_lines():
    one = GAME.step(GAME.start(0), (PLACE, int(GAME.space.ravel([3, 0]))))  # player 0, center
    _, values = LineEvaluator()(GAME, [GAME.start(0), one], [GAME.legal(GAME.start(0))] * 2)
    assert values[0] == 0.5
    assert values[1] > 0.5


@pytest.mark.parametrize("scale", [1e-310, 1e300])
def test_values_stay_in_bounds_at_extreme_weights(scale):
    positions = random_positions(GAME, 10, seed=3)
    evaluator = LineEvaluator((scale, scale, scale))
    _, values = evaluator(GAME, positions, [GAME.legal(p) for p in positions])
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_a_score_that_overflows_raises():
    positions = random_positions(GAME, 10, seed=3)
    evaluator = LineEvaluator((1e308, 1e308, 1e308))
    with pytest.raises(ValueError, match="overflows"):
        evaluator(GAME, positions, [GAME.legal(p) for p in positions])


def test_the_prior_prefers_the_middle_column_in_the_order_of_the_legal_actions():
    start = GAME.start(0)
    legal = GAME.legal(start)
    logits, _ = LineEvaluator(center=0.5)(GAME, [start], [legal])
    columns = GAME.space.unravel(np.flatnonzero(legal[PLACE]))[:, 0]
    np.testing.assert_array_equal(logits[0], -0.5 * np.abs(columns - 3.0))


def test_the_search_alone_has_a_flat_prior_and_even_values():
    positions = random_positions(GAME, 5, seed=4)
    legal = [GAME.legal(p) for p in positions]
    logits, values = SEARCH_ALONE(GAME, positions, legal)
    for lg, mask in zip(logits, legal, strict=True):
        assert len(lg) == np.count_nonzero(mask)
        assert (lg == 0.0).all()
    assert (values == 0.5).all()


def test_the_weights_must_match_the_length_of_a_line():
    start = GAME.start(0)
    with pytest.raises(ValueError, match="needs 3 weights, got 2"):
        LineEvaluator((1.0, 2.0))(GAME, [start], [GAME.legal(start)])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"weights": [1.0, 2.0, 3.0]}, TypeError),  # a list, not a tuple
        ({"weights": (1.0, True, 3.0)}, TypeError),  # booleans are never numbers
        ({"weights": (1.0, np.nan, 3.0)}, ValueError),
        ({"center": -0.1}, ValueError),
        ({"center": np.inf}, ValueError),
    ],
)
def test_bad_parameters_raise(kwargs, error):
    with pytest.raises(error):
        LineEvaluator(**kwargs)


def test_a_code_player_takes_a_win_in_one():
    position = GAME.start(0)
    for column in (0, 6, 1, 6, 2, 5):  # player 0 has three on the bottom row: 3 wins
        position = drop(position, column)
    player = AlphaZeroPlayer(LineEvaluator(), simulations=16, considered=7)
    action = player(GAME, 0, position, GAME.legal(position), np.random.default_rng(0))
    assert int(GAME.space.unravel(action.cell)[0]) == 3


BOARDS = [GAME, InARow((5, 4), 4)]  # columns differ from rows: a swapped axis shows


def values_of(game, positions):
    owner = np.stack([p.state.owner for p in positions])
    ply = np.array([p.ply for p in positions])
    return value_features(game, owner, ply)


def wins_now(game, position, player):
    """Return the columns where ``player`` would complete a line, by the rules themselves."""
    passed = position if game.player(position) == player else _passed(position)
    cells = np.flatnonzero(game.legal(position)[PLACE])
    return {int(c) for c in cells if game.step(passed, (PLACE, int(c))).winner == player}


def _passed(position):
    return dataclasses.replace(position, ply=position.ply + 1)


def threats_by_hand(game, position, player):
    """Return the free cells that complete a line of ``player``, one line at a time."""
    found = set()
    for line in game.lines:
        owners = position.state.owner[line]
        if (owners == player).sum() == game.length - 1 and (owners == 1 - player).sum() == 0:
            found.add(int(line[np.argmax(owners == -1)]))
    return found


@pytest.mark.parametrize("game", BOARDS, ids=str)
def test_the_threat_features_count_free_cells_by_row_parity(game):
    positions = random_positions(game, 40, seed=5)
    features = values_of(game, positions)
    rows = game.space.unravel(np.arange(game.space.n_cells))[:, 1]
    for position, f in zip(positions, features, strict=True):
        for player, (odd, even) in ((0, (2, 3)), (1, (4, 5))):
            cells = threats_by_hand(game, position, player)
            assert f[odd] == sum(rows[c] % 2 == 0 for c in cells)
            assert f[even] == sum(rows[c] % 2 == 1 for c in cells)


@pytest.mark.parametrize("game", BOARDS, ids=str)
def test_win_now_and_lost_now_agree_with_the_rules(game):
    positions = random_positions(game, 60, seed=6)
    features = values_of(game, positions)
    for position, f in zip(positions, features, strict=True):
        mover = game.player(position)
        toward = 1.0 if mover == 0 else -1.0
        mine, theirs = wins_now(game, position, mover), wins_now(game, position, 1 - mover)
        assert f[VALUE_FEATURES.index("win_now")] == (toward if mine else 0.0)
        lost = len(theirs) >= 2 and not mine
        assert f[VALUE_FEATURES.index("lost_now")] == (-toward if lost else 0.0)
        assert f[VALUE_FEATURES.index("to_move")] == toward


def test_the_features_of_a_known_board():
    position = GAME.start(0)
    for column in (0, 6, 1, 6, 2):  # player 0 on columns 0-2 of the bottom row; player 1 to move
        position = drop(position, column)
    f = dict(zip(VALUE_FEATURES, values_of(GAME, [position])[0], strict=True))
    assert f["first_threats_odd"] == 1  # column 3, bottom row: row 1, odd
    assert f["first_threats_even"] == 0
    assert f["second_threats_odd"] == 0
    assert f["second_threats_even"] == 0
    assert f["center"] == 0
    assert f["beside_center"] == 1  # column 2 is player 0's
    assert f["win_now"] == 0  # player 1 moves, and has no win
    assert f["lost_now"] == 0  # player 0 has one win ready, not two
    assert f["to_move"] == -1


def clean_features(rows):
    """Return the clean features of a board from rows drawn top down, any heights."""
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    for height, marks in enumerate(reversed(rows)):
        for column, mark in enumerate(marks):
            if mark != ".":
                owner[GAME.space.ravel([column, height])] = "XO".index(mark)
    f = value_features(GAME, owner[None], np.array([int((owner >= 0).sum())]))[0]
    named = dict(zip(VALUE_FEATURES, f, strict=True))
    return named["first_clean_odd"], named["second_clean_even"]


def test_a_first_players_threat_on_an_odd_row_is_clean():
    assert clean_features(["XXX....", "OOX....", "XOO...."]) == (1.0, 0.0)  # column 3, 3rd row


def test_a_first_players_threat_on_an_even_row_isnt_counted():
    assert clean_features(["OOX....", "XXX....", "OOO...."]) == (0.0, 0.0)  # column 3, 2nd row


def test_a_second_players_threat_on_an_even_row_is_clean():
    assert clean_features(["OOO....", "XXO...."]) == (0.0, 1.0)  # column 3, 2nd row


def test_a_rival_threat_below_takes_the_clean_away():
    # O's win is on column 3's 2nd row, X's directly above it on the 3rd.
    assert clean_features(["XXX....", "OXO.OOO", "XOX.XOX"]) == (0.0, 1.0)


def test_a_rival_threat_above_doesnt_take_the_clean_away():
    # X's win is on column 3's 3rd row, O's above it on the 4th: an even row, not clean.
    assert clean_features(["OOO....", "XXX....", "OOX....", "XOO...."]) == (1.0, 0.0)


def test_clean_threats_keep_only_those_with_no_rival_lower_in_the_column():
    mine = np.zeros((1, 2, 4), dtype=bool)
    other = np.zeros((1, 2, 4), dtype=bool)
    mine[0, 0, [0, 2]] = True
    other[0, 0, 1] = True
    mine[0, 1, 3] = True
    expected = np.zeros((1, 2, 4), dtype=bool)
    expected[0, 0, 0] = True  # below the rival's; the one above it is taken away
    expected[0, 1, 3] = True  # another column's rival doesn't matter
    np.testing.assert_array_equal(clean_threats(mine, other), expected)


def test_a_rival_threat_any_distance_below_takes_the_clean_away():
    mine = np.zeros((1, 1, 5), dtype=bool)
    other = np.zeros((1, 1, 5), dtype=bool)
    mine[0, 0, 4] = other[0, 0, 0] = True
    assert not clean_threats(mine, other).any()


def test_a_threat_doesnt_stop_being_clean_for_a_rival_on_the_same_cell():
    mine = np.zeros((1, 1, 3), dtype=bool)
    other = np.zeros((1, 1, 3), dtype=bool)
    mine[0, 0, 1] = other[0, 0, 1] = True
    assert clean_threats(mine, other)[0, 0, 1]


def test_the_features_are_the_same_in_a_batch_and_one_at_a_time():
    positions = random_positions(GAME, 10, seed=7)
    together = values_of(GAME, positions)
    for i, position in enumerate(positions):
        np.testing.assert_array_equal(values_of(GAME, [position])[0], together[i])


@pytest.mark.parametrize("game", BOARDS, ids=str)
def test_the_stones_are_the_legal_ones_in_their_order(game):
    for position in random_positions(game, 30, seed=8):
        cells, features = move_features(game, position.state.owner, position.ply)
        np.testing.assert_array_equal(cells, np.flatnonzero(game.legal(position)[PLACE]))
        assert features.shape == (len(cells), len(MOVE_FEATURES))


@pytest.mark.parametrize("game", BOARDS, ids=str)
def test_the_move_features_agree_with_the_rules(game):
    above = {"gives_win": 1, "under_own_win": 0}  # whose win the cell above is, relative to me
    for position in random_positions(game, 40, seed=9):
        me = game.player(position)
        cells, features = move_features(game, position.state.owner, position.ply)
        named = {name: features[:, j] for j, name in enumerate(MOVE_FEATURES)}
        wins, blocks = wins_now(game, position, me), wins_now(game, position, 1 - me)
        threats = (threats_by_hand(game, position, me), threats_by_hand(game, position, 1 - me))
        for i, cell in enumerate(cells):
            assert (named["win"][i] > 0) == (int(cell) in wins)
            assert (named["block"][i] > 0) == (int(cell) in blocks)
            column, row = game.space.unravel(cell)
            for name, whose in above.items():
                upper = int(cell) + 1 if row + 1 < game.shape[1] else None
                assert named[name][i] == (upper in threats[whose])
            assert named["own_parity"][i] == (row % 2 == me)
            assert named["center"][i] == -abs(column - (game.shape[0] - 1) / 2)


def test_the_line_counts_of_a_stone_match_a_count_line_by_line():
    for position in random_positions(GAME, 20, seed=10):
        me = GAME.player(position)
        cells, features = move_features(GAME, position.state.owner, position.ply)
        for i, cell in enumerate(cells):
            mine, theirs = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
            for line in GAME.lines[(GAME.lines == cell).any(axis=1)]:
                owners = position.state.owner[line]
                a, b = int((owners == me).sum()), int((owners == 1 - me).sum())
                if b == 0:
                    mine[a] += 1
                if a == 0:
                    theirs[b] += 1
            np.testing.assert_array_equal(features[i, 1:5], mine)
            np.testing.assert_array_equal(features[i, 5:8], theirs[1:])


def test_the_features_refuse_other_games():
    tic = InARow((3, 3), 3, gravity=False)
    with pytest.raises(ValueError, match="four in a row with gravity"):
        value_features(tic, np.full((1, 9), -1, dtype=np.int8), np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError, match="four in a row with gravity"):
        move_features(InARow((4, 4), 3), np.full(16, -1, dtype=np.int8), 0)


def fitted(value=None, intercept=0.0, prior=None):
    return FittedEvaluator(
        tuple(value or [0.0] * len(VALUE_FEATURES)),
        intercept,
        tuple(prior or [0.0] * len(MOVE_FEATURES)),
    )


def test_a_fitted_evaluator_with_no_constants_is_the_search_alone():
    positions = random_positions(GAME, 5, seed=11)
    legal = [GAME.legal(p) for p in positions]
    logits, values = fitted()(GAME, positions, legal)
    expected, even = SEARCH_ALONE(GAME, positions, legal)
    np.testing.assert_array_equal(values, even)
    for one, other in zip(logits, expected, strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_fitted_evaluator_adds_its_weighted_features_in_a_batch_or_alone():
    positions = random_positions(GAME, 10, seed=12)
    legal = [GAME.legal(p) for p in positions]
    rng = np.random.default_rng(0)
    evaluator = fitted(
        rng.normal(size=len(VALUE_FEATURES)).tolist(),
        0.3,
        rng.normal(size=len(MOVE_FEATURES)).tolist(),
    )
    logits, values = evaluator(GAME, positions, legal)
    for i, position in enumerate(positions):
        one_logits, one_value = evaluator(GAME, [position], [legal[i]])
        np.testing.assert_array_equal(one_logits[0], logits[i])
        assert one_value[0] == values[i]
        assert len(logits[i]) == np.count_nonzero(legal[i])


def test_a_fit_survives_a_file_by_name(tmp_path):
    rng = np.random.default_rng(1)
    evaluator = fitted(
        rng.normal(size=len(VALUE_FEATURES)).tolist(),
        -0.25,
        rng.normal(size=len(MOVE_FEATURES)).tolist(),
    )
    evaluator.save(tmp_path / "fit.json")
    assert FittedEvaluator.load(tmp_path / "fit.json") == evaluator


@pytest.mark.parametrize(
    ("value", "intercept", "prior", "error"),
    [
        ((0.0,), 0.0, (0.0,) * len(MOVE_FEATURES), ValueError),
        ((0.0,) * len(VALUE_FEATURES), 0.0, (0.0,), ValueError),
        ([0.0] * len(VALUE_FEATURES), 0.0, (0.0,) * len(MOVE_FEATURES), TypeError),
        ((np.nan,) * len(VALUE_FEATURES), 0.0, (0.0,) * len(MOVE_FEATURES), ValueError),
        ((0.0,) * len(VALUE_FEATURES), np.inf, (0.0,) * len(MOVE_FEATURES), ValueError),
        ((0.0,) * len(VALUE_FEATURES), True, (0.0,) * len(MOVE_FEATURES), TypeError),
    ],
)
def test_a_fitted_evaluator_checks_its_constants(value, intercept, prior, error):
    with pytest.raises(error):
        FittedEvaluator(value, intercept, prior)


def test_gives_win_sees_a_win_on_the_top_row():
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    for column in range(3):  # player 1 holds the top row of columns 0-2, the rest filler
        for row in range(5):
            owner[GAME.space.ravel([column, row])] = (column + row) % 2
        owner[GAME.space.ravel([column, 5])] = 1
    for row in range(4):  # column 3 is two short of its top: its next stone is under the win
        owner[GAME.space.ravel([3, row])] = row % 2
    cells, features = move_features(GAME, owner, 22)  # player 0 to move
    gives_win = features[:, MOVE_FEATURES.index("gives_win")]
    columns = GAME.space.unravel(cells)[:, 0]
    assert gives_win[columns == 3] == 1.0
    assert gives_win[columns != 3].sum() == 0.0


def test_the_intercept_moves_every_value():
    start = GAME.start(0)
    _, values = fitted(intercept=0.3)(GAME, [start], [GAME.legal(start)])
    assert values[0] == pytest.approx(1.0 / (1.0 + np.exp(-0.3)))


@pytest.mark.parametrize("intercept", [0, np.float32(0.25)])
def test_an_intercept_of_any_real_type_gives_float64_values(intercept):
    # An integer intercept once made the sum an integer array, which the features couldn't enter.
    n_value, n_move = len(VALUE_FEATURES), len(MOVE_FEATURES)
    fit = FittedEvaluator((0.1,) * n_value, intercept, (0.0,) * n_move)
    position = GAME.start(0)
    _, values = fit(GAME, [position], [GAME.legal(position)])
    assert values.dtype == np.float64
    same = FittedEvaluator((0.1,) * n_value, float(intercept), (0.0,) * n_move)
    assert values[0] == same(GAME, [position], [GAME.legal(position)])[1][0]
