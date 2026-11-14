"""Tests of the report on a code player."""

from dataclasses import replace
from html.parser import HTMLParser

import numpy as np
import pytest

from examples.games.in_a_row import (
    MOVE_FEATURES,
    PLACE,
    VALUE_FEATURES,
    FittedEvaluator,
    InARow,
    Lessons,
    LineEvaluator,
    distinct_positions,
    judge,
    position_of,
)
from examples.games.in_a_row.play import start_of
from examples.games.in_a_row.report import (
    CANDIDATES,
    PLAY_PORT,
    _contested,
    _keeps_value,
    _on_side,
    _pick,
    _value_cards,
    analyze,
    board_key,
    board_shapes,
    board_svg,
    exact_values,
    late_threats,
    one_ready,
    over_ready,
    page,
    parity_holder,
    stacked,
    steering,
)
from gridzero.agents import UniformPlayer
from gridzero.game import series

GAME = InARow()
TRUTH_FROM_HERE = 34  # `agents.exact` has no pruning: 8 free cells at most, milliseconds


def board(rows):
    """Return owners from rows drawn top down ("X" first, "O" second, "." free), any heights."""
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    for height, marks in enumerate(reversed(rows)):
        for column, mark in enumerate(marks):
            if mark != ".":
                owner[GAME.space.ravel([column, height])] = "XO".index(mark)
    return owner


def measure(candidate, rows, ply=None):
    owner = board(rows)
    plies = np.array([(owner >= 0).sum() if ply is None else ply])
    return float(candidate(GAME, owner[None], plies)[0])


# X holds columns 0-2 of the 2nd row from the bottom, under fillers: its win there is on an even
# row. X holds columns 0-2 of the 3rd: its win there is on an odd row.
EVEN_FOR_X = ["OOX....", "XXX....", "OOO...."]  # a threat at (3, row index 1)
ODD_FOR_X = ["XXX....", "OOX....", "XOO...."]  # a threat at (3, row index 2)


def test_only_the_holder_of_a_clean_parity_threat_is_named():
    assert measure(parity_holder, ODD_FOR_X) == 1.0
    assert measure(parity_holder, ["XXO.OOO", "OOX.OXO"]) == -1.0  # O's win, on an even row
    assert measure(parity_holder, EVEN_FOR_X) == 0.0


def test_two_threats_in_a_column_count_as_stacked():
    # Column 3: X wins on row index 1 (columns 0-2) and on row index 2 (columns 4-6).
    rows = ["....XXX", "XXX.OOO", "OOX.XOX"]
    assert measure(stacked, rows) >= 1.0
    assert measure(stacked, EVEN_FOR_X) == 0.0


def test_late_threats_grow_with_the_board_filled():
    assert measure(late_threats, ODD_FOR_X, ply=21) == pytest.approx(0.5)
    assert measure(late_threats, ODD_FOR_X, ply=42) == pytest.approx(1.0)


@pytest.mark.parametrize(("ply", "expected"), [(5, 1.0), (6, 0.0)])
def test_one_ready_names_a_forced_reply_when_the_mover_has_no_win(ply, expected):
    rows = ["XXX...."]  # X wins at column 3 of the bottom row: ready
    assert measure(one_ready, rows, ply=ply) == expected  # X to move has the win itself at 6


def test_one_ready_favors_the_second_player_when_it_is_the_other():
    assert measure(one_ready, ["OOO...."], ply=6) == -1.0


def test_over_ready_counts_a_threat_just_above_a_playable_cell():
    assert measure(over_ready, EVEN_FOR_X) == 1.0  # column 3 is empty: row 0 plays, row 1 is above
    assert measure(over_ready, ODD_FOR_X) == 0.0  # two above the playable cell


def test_over_ready_counts_a_threat_on_the_top_row():
    # X's win is on the top of column 3, which has room for two: only features read this board.
    rows = ["XXX....", ".......", "...X...", "...O...", "...X...", "...O..."]
    assert measure(over_ready, rows) == 1.0


def test_a_gallery_takes_at_most_two_cards_from_a_game():
    owner = np.stack([board([f"{'.' * i}X{'.' * (6 - i)}"]) for i in range(3)])
    lessons = Lessons(
        owner,
        np.array([5, 5, 5]),
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.full(3, 0.5),
        np.full((3, 7), 0.5),
        np.full((3, 7), 1 / 7),
        np.zeros(3, dtype=np.int64),
    )
    assert _pick(np.array([3.0, 2.0, 1.0]), lessons, GAME, shown=3) == [0, 1]


def test_every_candidate_gives_one_number_per_position():
    owner = np.stack([board(EVEN_FOR_X), board(ODD_FOR_X), board(["......."])])
    ply = np.array([9, 9, 0])
    for function, _ in CANDIDATES.values():
        assert function(GAME, owner, ply).shape == (3,)


def nearly_full(free_on_top):
    """Return owners of a full board with no line, and a ply: the top of some columns free.

    ``free_on_top[c]`` cells are left free at the top of column ``c``.
    """
    up, down = [0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]
    grid = np.array([up, up, down, down, up, up, down], dtype=np.int8)
    for column, free in enumerate(free_on_top):
        if free:
            grid[column, -free:] = -1
    owner = grid.ravel()
    return owner, int((owner >= 0).sum())


def minimax(position):
    """Return player 0's value under perfect play, by the rules alone."""
    if position.over:
        return 0.5 if position.winner is None else float(position.winner == 0)
    values = [
        minimax(GAME.step(position, (PLACE, int(cell))))
        for cell in np.flatnonzero(GAME.legal(position)[PLACE])
    ]
    return max(values) if GAME.player(position) == 0 else min(values)


@pytest.mark.parametrize(
    "free", [(0, 2, 2, 0, 2, 0, 0), (1, 1, 1, 1, 1, 1, 0), (0, 0, 3, 0, 3, 0, 0)]
)
def test_exact_values_match_a_minimax_by_the_rules(free):
    owner, ply = nearly_full(free)
    values, by_column = exact_values(GAME, owner[None], np.array([ply]))
    position = position_of(GAME, owner, ply)
    assert values[0] == minimax(position)
    for cell in np.flatnonzero(GAME.legal(position)[PLACE]):
        after = GAME.step(position, (PLACE, int(cell)))
        assert by_column[0, GAME.space.unravel(cell)[0]] == minimax(after)


def test_exact_values_mark_a_full_column_nan():
    owner, ply = nearly_full((0, 1, 1, 0, 0, 0, 0))
    _, by_column = exact_values(GAME, owner[None], np.array([ply]))
    assert np.isnan(by_column[0, [0, 3, 4, 5, 6]]).all()
    assert np.isfinite(by_column[0, [1, 2]]).all()


def test_exact_values_dont_depend_on_the_workers():
    boards = [nearly_full((0, 2, 2, 0, 0, 0, 0)), nearly_full((1, 1, 1, 1, 0, 0, 0))]
    owner = np.stack([b[0] for b in boards])
    ply = np.array([b[1] for b in boards])
    one = exact_values(GAME, owner, ply, workers=1)
    two = exact_values(GAME, owner, ply, workers=2)
    np.testing.assert_array_equal(one[0], two[0])
    np.testing.assert_array_equal(one[1], two[1])


def test_no_positions_have_no_exact_values():
    values, by_column = exact_values(
        GAME, np.zeros((0, GAME.space.n_cells), dtype=np.int8), np.zeros(0, dtype=np.int64)
    )
    assert values.shape == (0,)
    assert by_column.shape == (0, GAME.shape[0])


def lessons_of(q, ply, move):
    n = len(ply)
    return Lessons(
        np.full((n, GAME.space.n_cells), -1, dtype=np.int8),
        np.asarray(ply),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
        np.full(n, 0.5),
        np.asarray(q, dtype=np.float64),
        np.full((n, 7), 1.0 / 7.0),
        np.asarray(move),
    )


def test_steering_reads_the_picked_move_for_the_mover():
    columns = np.full(7, np.nan)
    taught = columns.copy()
    taught[[0, 1]] = (0.5, 0.5)
    even = columns.copy()
    even[[0, 1]] = (0.8, 0.6)  # at an even ply q is the mover's: +0.3 chosen, +0.2 on average
    odd = columns.copy()
    odd[[0, 1]] = (0.2, 0.4)  # at an odd ply the mover's q is 1 - q: the same differences
    test = steering("x", lessons_of([even, odd], [0, 1], [0, 0]), np.stack([taught, taught]))
    assert test.n == 2
    assert test.chosen == pytest.approx(0.3)
    assert test.every == pytest.approx(0.2)
    assert test.extra == pytest.approx(0.1)
    assert test.error == pytest.approx(0.0)


def test_steering_leaves_out_a_pick_the_teacher_never_visited():
    q = np.full((1, 7), np.nan)
    q[0, [0, 1, 2]] = (0.8, 0.6, 0.6)
    taught = np.full((1, 7), np.nan)
    taught[0, [1, 2]] = 0.5  # two moves to compare, but not the one picked
    assert steering("x", lessons_of(q, [0], [0]), taught).n == 0


def test_a_move_keeps_the_value_only_if_nothing_is_better_for_the_mover():
    by_column = np.array([[1.0, 0.5, np.nan, 0.0, 0.0, 0.0, 0.0]] * 2)
    assert _keeps_value(by_column, np.array([0, 0]), np.array([0, 1])) == 0.5
    assert _keeps_value(by_column, np.array([1, 1]), np.array([3, 0])) == 0.5  # 0's rival moves


def test_a_position_with_no_wrong_move_doesnt_count_for_keeping_the_value():
    contested = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    open_and_shut = [1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0]
    by_column = np.array([contested, open_and_shut])
    assert _keeps_value(by_column, np.array([0, 0]), np.array([1, 1])) == 0.0
    assert _contested(by_column).tolist() == [True, False]
    assert _keeps_value(by_column[1:], np.array([0]), np.array([1])) == 0.0  # none contested


def test_the_side_counts_only_decisive_positions():
    value = np.array([0.9, 0.2, 0.9, 0.9])
    truth = np.array([1.0, 1.0, 0.0, 0.5])
    assert _on_side(value, truth) == pytest.approx(1 / 3)
    assert _on_side(value[3:], truth[3:]) == 0.0


def test_a_gallery_takes_the_highest_per_bucket_without_mirrors_or_too_many_of_a_game():
    left, mirrored, other = board(["X......"]), board(["......X"]), board(["...X..."])
    owner = np.stack([left, mirrored, other, other * 0 - 1, other])
    ply = np.array([5, 5, 5, 5, 15])
    lessons = Lessons(
        owner,
        ply,
        np.array([0, 1, 2, 2, 3]),
        np.zeros(5, dtype=np.int64),
        np.full(5, 0.5),
        np.full((5, 7), 0.5),
        np.full((5, 7), 1 / 7),
        np.zeros(5, dtype=np.int64),
    )
    score = np.array([9.0, 8.0, 7.0, np.nan, 1.0])
    assert _pick(score, lessons, GAME, shown=2) == [0, 2, 4]  # the mirror and the NaN skipped
    assert _pick(score, lessons, GAME, shown=1) == [0, 4]


def test_a_board_is_drawn_with_its_cells_stones_and_marks_row_zero_at_the_bottom():
    owner = board(["X......", "O......"])  # O at row 0, X at row 1
    svg = board_svg(GAME, owner, [(3 * GAME.shape[1], "played")], scale=10.0)
    assert svg.count("<polygon") == GAME.space.n_cells
    assert svg.count('class="first"') == 1
    assert svg.count('class="second"') == 1
    assert svg.count('class="played"') == 1
    assert svg.index('class="second"') < svg.index('class="first"')  # by cell id
    assert 'cy="55.0"' in svg  # row 0: 10 per row, 60 high
    assert 'cy="45.0"' in svg  # row 1


class Strict(HTMLParser):
    """Checks that tags close in order, and notes what the page loads."""

    VOID = frozenset({"meta", "br", "polygon", "circle"})

    def __init__(self):
        super().__init__()
        self.open = []
        self.loads = []
        self.links = []
        self.svgs = 0

    def handle_starttag(self, tag, attrs):
        self.svgs += tag == "svg"
        found = [v for k, v in attrs if k in ("src", "href")]
        if tag == "a":
            self.links += found
        else:
            self.loads += found
        if tag not in self.VOID:
            self.open.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.loads += [v for k, v in attrs if k in ("src", "href")]

    def handle_endtag(self, tag):
        assert self.open.pop() == tag


@pytest.fixture(scope="module")
def small():
    """Two rounds of a few games judged by a small search, with a fit per round."""
    uniform = [("a", UniformPlayer()), ("b", UniformPlayer())]
    parts = []
    for source, seed in enumerate((0, 100)):
        records = series(GAME, uniform, 3, seed)
        owner, ply, numbers = distinct_positions(GAME, records)
        parts.append(
            judge(
                GAME, LineEvaluator(), owner, ply, numbers, simulations=8, seed=seed, source=source
            )
        )
    lessons = parts[0] + parts[1]
    value = (0.1,) * len(VALUE_FEATURES)
    prior = (0.1,) * len(MOVE_FEATURES)
    fits = [FittedEvaluator(value, 0.0, prior), FittedEvaluator(value, 0.2, prior)]
    report = analyze(
        GAME,
        LineEvaluator(),
        fits,
        lessons,
        shown=2,
        seed=0,
        referee_at=16,
        truth_from=TRUTH_FROM_HERE,
        workers=1,
    )
    return report, lessons


def test_the_report_has_a_row_per_fit_and_round(small):
    report, lessons = small
    assert len(report.by_source) == 2
    assert all(len(row) == 2 for row in report.by_source)
    assert sum(report.positions) == len(lessons)
    assert len(report.labels) == len(report.truth) == 2


def test_the_steering_test_marks_the_positions_a_fit_chose_and_has_a_baseline(small):
    report, _ = small
    labels = [t.label for t in report.steering]
    assert len(labels) == 2 * 2 + 1
    assert "fit 0 on round 1 (the positions it chose)" in labels
    assert labels[-1].startswith("teacher at 64")


def test_the_candidates_come_most_useful_first(small):
    report, _ = small
    gains = [c.gain for c in report.candidates]
    assert gains == sorted(gains, reverse=True)
    assert {c.name for c in report.candidates} == set(CANDIDATES)


def test_cards_are_within_a_bucket_limit_and_carry_their_judges(small):
    report, _ = small
    assert 0 < len(report.value_cards) <= 2 * 4
    assert 0 < len(report.move_cards) <= 2 * 4
    for card in report.move_cards:
        cell = GAME.legal(position_of(GAME, card.owner, card.ply))[PLACE]
        columns = set(GAME.space.unravel(np.flatnonzero(cell))[:, 0].tolist())
        assert card.played in columns
        assert card.best in columns
        assert card.regret >= 0.0


def test_a_late_position_gets_the_truth_and_an_early_one_doesnt(small):
    owner, ply = nearly_full((0, 2, 2, 0, 0, 0, 0))
    early = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    lessons = judge(
        GAME,
        LineEvaluator(),
        np.stack([early, owner]),
        np.array([0, ply]),
        np.array([0, 1]),
        simulations=8,
        seed=0,
    )
    fit = small[0].fits[-1]
    cards = _value_cards(GAME, LineEvaluator(), fit, lessons, [0, 1], 0, 8, TRUTH_FROM_HERE, 1)
    assert np.isnan(cards[0].truth)
    assert cards[1].truth == minimax(position_of(GAME, owner, ply))


def test_the_page_is_well_formed_and_needs_nothing_else(small):
    report, _ = small
    text = page(GAME, report)
    parser = Strict()
    parser.feed(text)
    assert parser.open == []
    assert parser.loads == []
    assert parser.svgs == len(report.value_cards) + len(report.move_cards)
    assert len(parser.links) == parser.svgs
    assert "<title>Connect Four Code Player</title>" in text
    assert "prefers-color-scheme: dark" in text
    assert "nan" not in text.replace("&ndash;", "")


def test_each_card_links_to_its_board_on_the_play_page(small):
    report, _ = small
    parser = Strict()
    parser.feed(page(GAME, report))
    cards = (*report.value_cards, *report.move_cards)
    keys = {board_key(card.owner) for card in cards}
    linked = {link.removeprefix(f"http://localhost:{PLAY_PORT}/#board=") for link in parser.links}
    assert linked == keys
    for key in keys:  # each opens as a game still going
        assert board_key(start_of(GAME, key).state.owner) == key


def test_a_boards_key_has_a_character_per_cell_in_id_order():
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    owner[0], owner[1], owner[GAME.shape[1]] = 0, 1, 1
    key = board_key(owner)
    assert len(key) == GAME.space.n_cells
    assert key[:2] == "01"
    assert key[GAME.shape[1]] == "1"
    assert key.count(".") == GAME.space.n_cells - 3


def test_board_shapes_fill_the_picture_with_row_0_at_the_bottom():
    shapes = board_shapes(GAME, 10.0)
    assert (shapes.width, shapes.height) == (10.0 * GAME.shape[0], 10.0 * GAME.shape[1])
    assert shapes.polygons.min() == 0.0
    assert shapes.polygons[:, :, 0].max() == shapes.width
    assert shapes.polygons[:, :, 1].max() == shapes.height
    assert shapes.centers.shape == (GAME.space.n_cells, 2)
    assert shapes.centers[0, 1] > shapes.centers[GAME.shape[1] - 1, 1]
    assert shapes.centers[GAME.shape[1], 0] > shapes.centers[0, 0]


def test_a_move_card_numbers_columns_from_one_as_the_play_page_does(small):
    report, _ = small
    card = report.move_cards[0]
    text = page(GAME, report)
    assert f"Played column <b>{card.played + 1}</b>" in text
    assert f"teacher's best <b>{card.best + 1}</b>" in text
    assert "<th>1</th>" in text
    assert "<th>7</th>" in text
    assert "<th>0</th>" not in text


def test_the_page_states_the_settings_it_measured_at(small):
    report, _ = small
    text = page(GAME, replace(report, referee_at=11, truth_from=29))
    assert "teacher searching 11 simulations" in text
    assert "from ply 29." in text.replace("\n", " ")
