"""Tests of the play page's table and server. A browser runs the page itself by hand."""

import http.client
import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from examples.games.in_a_row import (
    MOVE_FEATURES,
    PLACE,
    SEARCH_ALONE,
    VALUE_FEATURES,
    FittedEvaluator,
    InARow,
    move_features,
    position_of,
)
from examples.games.in_a_row.experiments.play import fingerprint
from examples.games.in_a_row.play import (
    MOVE_TEXT,
    VALUE_TEXT,
    Rung,
    Table,
    explain,
    page,
    rungs,
    serve,
    start_of,
)
from examples.games.in_a_row.report import board_key
from gridzero.agents import AlphaZeroPlayer, UniformPlayer
from gridzero.game import Action

GAME = InARow()
ROWS = GAME.shape[1]
# Every constant differs, so a contribution read from the wrong feature shows.
FIT = FittedEvaluator(
    tuple(0.1 + 0.07 * j for j in range(len(VALUE_FEATURES))),
    -0.2,
    tuple(0.3 - 0.05 * j for j in range(len(MOVE_FEATURES))),
)
SEARCH = AlphaZeroPlayer(FIT, simulations=16, considered=7)
LADDER = (
    Rung("random", "Any column.", UniformPlayer()),
    Rung("fit", "A small search.", SEARCH),
)


def board(*stones):
    """Return owners, shape (n_cells,), from (column, row, player) stones."""
    owner = np.full(GAME.space.n_cells, -1, dtype=np.int8)
    for column, row, player in stones:
        owner[column * ROWS + row] = player
    return owner


def text_of(*stones):
    return board_key(board(*stones))


def searched(position, seed=0):
    return SEARCH.search(GAME, position, np.random.default_rng(seed))


def sigmoid_logit(value):
    return np.log(value / (1.0 - value))


@pytest.fixture
def table(tmp_path):
    return Table(GAME, LADDER, tmp_path / "deep" / "me_games.json", seed=3)


def test_every_feature_says_what_it_counts():
    assert set(VALUE_TEXT) == set(VALUE_FEATURES)
    assert set(MOVE_TEXT) == set(MOVE_FEATURES)


def test_rungs_follow_the_round_robin_with_the_teacher_only_when_given():
    names = [rung.name for rung in rungs(FIT)]
    assert names == [
        "uniform",
        "search 64",
        "threat",
        "lines 64",
        "fit 16",
        "fit 64",
        "fit 256",
        "fit 1024",
    ]
    with_teacher = [rung.name for rung in rungs(FIT, SEARCH_ALONE)]
    assert [name for name in with_teacher if not name.startswith("teacher")] == names
    assert with_teacher[6:8] == ["teacher 64", "teacher 256"]
    assert with_teacher[-1] == "teacher 1024"
    assert len(with_teacher) == len(names) + 3


def test_every_search_rung_considers_every_column_without_noise_until_a_game_asks():
    for rung in rungs(FIT, SEARCH_ALONE):
        if isinstance(rung.player, AlphaZeroPlayer):
            assert not rung.player.noise
            assert rung.player.considered == GAME.shape[0]


@pytest.mark.parametrize("noise", [False, True])
def test_a_game_plays_with_the_noise_it_asked_for_and_records_it(tmp_path, noise):
    # Noise changes what a search does with one seed; the same seed without it repeats itself.
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=1)
    replies = {
        table.reply(table.new("fit", 1, noise=noise)["game"])["moves"][0]["column"]
        for _ in range(30)
    }
    assert (len(replies) > 1) == noise
    state = table.new("fit", 1, noise=noise)
    assert state["noise"] is noise


def test_noise_is_kept_in_the_file(tmp_path):
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=1)
    text = text_of((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1))
    table.reply(table.new("fit", 1, text, noise=True)["game"])
    assert json.loads(table.path.read_text())[0]["noise"] is True


def test_a_game_with_noise_leaves_the_rung_as_it_was(table):
    table.new("fit", 1, noise=True)
    assert not table.ladder["fit"].player.noise


def test_a_player_with_no_search_records_no_noise(table):
    assert table.new("random", 0, noise=True)["noise"] is False


@pytest.mark.parametrize("mover", [0, 1])
def test_a_stones_reasons_add_up_to_the_priors_difference(mover):
    # Player 1 moves on odd plies: a stone more of player 0's.
    stones = [(3, 0, 0), (3, 1, 1), (4, 0, 0), (2, 0, 1), (5, 0, 0)]
    position = position_of(GAME, board(*stones[: 4 + mover]), 4 + mover)
    result = searched(position)
    why = explain(GAME, FIT, position, result)
    cells, _ = move_features(GAME, position.state.owner, position.ply)
    logits = FIT(GAME, [position], [GAME.legal(position)])[0][0]
    by_column = dict(zip((cells // ROWS).tolist(), logits, strict=True))
    assert sum(r.logits for r in why.stone) == pytest.approx(
        by_column[why.column] - by_column[why.rival]
    )


@pytest.mark.parametrize("mover", [0, 1])
def test_a_boards_reasons_add_up_to_the_values_difference_for_the_player_moving(mover):
    stones = [(3, 0, 0), (3, 1, 1), (4, 0, 0), (2, 0, 1), (5, 0, 0)]
    position = position_of(GAME, board(*stones[: 4 + mover]), 4 + mover)
    why = explain(GAME, FIT, position, searched(position))
    assert why.board
    after = {
        column: GAME.step(position, (PLACE, int(np.flatnonzero(GAME.legal(position)[PLACE])[i])))
        for i, column in enumerate(c.column for c in why.columns)
    }
    values = {column: FIT(GAME, [a], [GAME.legal(a)])[1][0] for column, a in after.items()}
    assert why.rival is not None
    gap = sigmoid_logit(values[why.column]) - sigmoid_logit(values[why.rival])
    assert sum(r.logits for r in why.board) == pytest.approx(gap if mover == 0 else -gap)


def test_reasons_are_largest_first_and_leave_out_what_adds_nothing():
    position = position_of(GAME, board((3, 0, 0), (3, 1, 1), (4, 0, 0), (2, 0, 1)), 4)
    why = explain(GAME, FIT, position, searched(position))
    for reasons in (why.stone, why.board):
        sizes = [abs(r.logits) for r in reasons]
        assert sizes == sorted(sizes, reverse=True)
        assert all(r.logits != 0.0 for r in reasons)
    assert all(r.chosen != r.rival for r in why.stone)
    assert "to_move" not in {r.feature for r in why.board}  # the same ply after either move


def test_a_reason_names_the_feature_its_weight_belongs_to():
    # A stone taking a win against any other: the win is the fit's "win" constant, times one.
    position = position_of(
        GAME, board((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1)), 6
    )
    why = explain(GAME, FIT, position, searched(position))
    assert why.column == 0
    win = next(r for r in why.stone if r.feature == "win")
    assert win.logits == pytest.approx(FIT.prior[MOVE_FEATURES.index("win")])
    assert (win.chosen, win.rival) == (1.0, 0.0)
    assert why.board == ()  # the move ends the game: no value is asked of the board after


def test_values_in_an_explanation_are_the_movers():
    # Player 1 can win at once: its own numbers should say so, though q is player 0's.
    position = position_of(
        GAME, board((0, 0, 0), (2, 0, 0), (4, 0, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1), (6, 0, 0)), 7
    )
    why = explain(GAME, FIT, position, searched(position))
    assert why.column == 1
    played = next(c for c in why.columns if c.column == 1)
    assert played.value == pytest.approx(1.0)
    assert why.expects == pytest.approx(1.0)


def test_the_rival_is_the_other_column_most_visited():
    position = position_of(GAME, board((3, 0, 0), (3, 1, 1)), 2)
    result = searched(position)
    why = explain(GAME, FIT, position, result)
    others = [c for c in why.columns if c.column != why.column]
    assert why.rival is not None
    rival = next(c for c in others if c.column == why.rival)
    assert rival.visits == max(c.visits for c in others)


def test_the_glance_is_the_priors_favorite():
    position = position_of(GAME, board((0, 0, 0), (0, 1, 0), (0, 2, 0)), 3)
    why = explain(GAME, FIT, position, searched(position))
    logits = FIT(GAME, [position], [GAME.legal(position)])[0][0]
    cells = np.flatnonzero(GAME.legal(position)[PLACE])
    assert why.glance == cells[np.argmax(logits)] // ROWS


def test_an_evaluator_without_features_gives_counts_but_no_reasons():
    player = replace(SEARCH, evaluator=SEARCH_ALONE)
    position = GAME.start(0)
    result = player.search(GAME, position, np.random.default_rng(0))
    why = explain(GAME, SEARCH_ALONE, position, result)
    assert not why.named
    assert why.stone == ()
    assert why.board == ()
    assert sum(c.visits for c in why.columns) == 16
    assert explain(GAME, FIT, position, searched(position)).named


def test_a_column_the_search_never_visited_has_no_value():
    player = replace(SEARCH, simulations=2)
    position = GAME.start(0)
    result = player.search(GAME, position, np.random.default_rng(0))
    why = explain(GAME, FIT, position, result)
    assert any(c.visits == 0 and c.value is None for c in why.columns)
    json.dumps([c.value for c in why.columns], allow_nan=False)


def test_a_position_with_one_legal_column_has_no_rival():
    # Every column full but the last, with no line: only the last has a free cell.
    columns = [
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
    ]
    stones = [(c, r, p) for c, col in enumerate(columns) for r, p in enumerate(col)]
    position = position_of(GAME, board(*stones), 36)
    why = explain(GAME, FIT, position, searched(position))
    assert why.column == 6
    assert why.rival is None
    assert why.stone == ()
    assert why.board == ()


@pytest.mark.parametrize(
    ("text", "ply"),
    [
        (text_of(), 0),
        (text_of((3, 0, 0)), 1),
        (text_of((3, 0, 0), (3, 1, 1)), 2),
    ],
)
def test_a_boards_text_gives_back_its_owners_and_the_ply_it_implies(text, ply):
    position = start_of(GAME, text)
    assert board_key(position.state.owner) == text
    assert position.state.owner.dtype == np.int8
    assert position.ply == ply
    assert not position.over


@pytest.mark.parametrize(
    "text",
    [
        "",
        "." * (GAME.space.n_cells - 1),
        "." * (GAME.space.n_cells + 1),
        "x" + "." * (GAME.space.n_cells - 1),
        text_of((3, 0, 1)),  # the second player first
        text_of((3, 0, 0), (2, 0, 0)),  # two stones of one player
        text_of((3, 1, 0)),  # in the air
        text_of((3, 0, 0), (3, 1, 1), (3, 3, 0)),  # in the air, higher up
        text_of((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1)),
        text_of(
            (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 0, 0)
        ),
    ],
)
def test_a_board_no_game_reaches_is_refused(text):
    with pytest.raises(ValueError, match=r"board|stones|air|line"):
        start_of(GAME, text)


def test_a_full_board_is_refused():
    columns = [
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ]
    stones = [(c, r, p) for c, col in enumerate(columns) for r, p in enumerate(col)]
    with pytest.raises(ValueError, match="full"):
        start_of(GAME, text_of(*stones))


def test_the_person_moves_first_and_the_player_replies(table):
    state = table.new("fit", 0)
    assert state["to_move"] == 0
    assert state["moves"] == []
    state = table.move(state["game"], 3)
    assert state["moves"][0]["by"] == "person"
    assert state["moves"][0]["cell"] == 3 * ROWS
    assert state["moves"][0]["column"] == 3
    assert state["moves"][0]["why"] is None
    assert state["to_move"] == 1
    state = table.reply(state["game"])
    assert state["moves"][1]["by"] == "fit"
    assert state["moves"][1]["why"]["named"] is True
    assert state["to_move"] == 0


def test_the_player_moves_first_when_the_person_sits_second(table):
    state = table.new("fit", 1)
    assert state["to_move"] == 0
    with pytest.raises(ValueError, match="your move"):
        table.move(state["game"], 3)
    state = table.reply(state["game"])
    assert state["moves"][0]["player"] == 0
    assert state["to_move"] == 1


def test_a_person_cant_have_the_player_move_out_of_turn(table):
    state = table.new("fit", 0)
    with pytest.raises(ValueError, match="player's move"):
        table.reply(state["game"])


def test_a_scripted_player_has_no_explanation(table):
    state = table.new("random", 1)
    state = table.reply(state["game"])
    assert state["moves"][0]["why"] is None
    assert state["about"] == "Any column."


def test_a_column_off_the_board_or_full_is_refused(table):
    number = table.new("random", 0)["game"]
    with pytest.raises(ValueError, match="column"):
        table.move(number, 7)
    with pytest.raises(ValueError, match="column"):
        table.move(number, -1)
    with pytest.raises(TypeError, match="column"):
        table.move(number, True)  # a boolean is never an integer
    text = text_of((0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 3, 1), (0, 4, 0), (0, 5, 1))
    number = table.new("random", 0, text)["game"]
    with pytest.raises(ValueError, match="column"):
        table.move(number, 0)
    assert table.state(number)["moves"] == []


def test_a_game_of_another_number_or_a_rung_of_another_name_is_refused(table):
    with pytest.raises(ValueError, match="no game"):
        table.state(99)
    with pytest.raises(ValueError, match="no rung"):
        table.new("nonesuch", 0)
    with pytest.raises(ValueError, match="seat"):
        table.new("fit", 2)
    with pytest.raises(ValueError, match="seat"):
        table.new("fit", -1)


def test_two_rungs_of_one_name_are_refused(tmp_path):
    with pytest.raises(ValueError, match="share"):
        Table(GAME, (LADDER[0], LADDER[0]), tmp_path / "x.json", seed=0)


def win_for_the_person(table, rung="random"):
    """Return a game the person won in one move, from a board with a win ready."""
    text = text_of((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1))
    number = table.new(rung, 0, text)["game"]
    return table.move(number, 0)


def test_a_won_game_says_who_won_and_where(table):
    state = win_for_the_person(table)
    assert state["over"]
    assert state["winner"] == 0
    assert state["result"] == "won"
    assert state["to_move"] is None
    assert state["legal"] == []
    assert state["line"] == [0, 1, 2, 3]


def test_a_game_lost_and_a_game_drawn_are_recorded_so(tmp_path):
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=0)
    # The person holds the second seat and the rung to move has one legal win: fit takes it.
    text = text_of((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1))
    state = table.reply(table.new("fit", 1, text)["game"])
    assert state["result"] == "lost"
    # Full but for one cell, whose stone makes no line.
    columns = [
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0],
    ]
    stones = [(c, r, p) for c, col in enumerate(columns) for r, p in enumerate(col)]
    state = table.new("random", 1, text_of(*stones))
    assert state["to_move"] == 1
    state = table.move(state["game"], 6)
    assert state["result"] == "drawn"
    assert state["winner"] is None
    assert state["line"] == []
    results = [r["result"] for r in json.loads(table.path.read_text())]
    assert results == ["lost", "drawn"]


def test_a_game_is_kept_when_it_ends_and_not_before(table):
    number = table.new("random", 0)["game"]
    table.move(number, 3)
    assert not table.path.exists()
    win_for_the_person(table)
    records = json.loads(table.path.read_text())
    assert len(records) == 1
    record = records[0]
    assert record["rung"] == "random"
    assert record["seat"] == 0
    assert record["result"] == "won"
    assert record["note"] is None
    assert record["moves"][0]["cell"] == 3
    assert record["moves"][0]["column"] == 0
    assert record["moves"][0]["seconds"] >= 0.0
    assert record["start"].count(".") == GAME.space.n_cells - 6
    assert "about" not in record  # what a rung says of itself isn't part of the game


def test_a_note_is_kept_with_its_game(table):
    number = win_for_the_person(table)["game"]
    table.note(number, 4, [6], "took the column")
    other = win_for_the_person(table)["game"]
    records = {r["game"]: r for r in json.loads(table.path.read_text())}
    assert records[number]["note"] == {"rating": 4, "interesting": [6], "text": "took the column"}
    assert records[other]["note"] is None


def test_a_later_note_replaces_an_earlier(table):
    number = win_for_the_person(table)["game"]
    table.note(number, 1, [], "first thought")
    state = table.note(number, None, [], "second thought")
    assert state["note"]["rating"] is None
    records = json.loads(table.path.read_text())
    assert len(records) == 1
    assert records[0]["note"]["text"] == "second thought"


@pytest.mark.parametrize(
    ("rating", "interesting", "error"),
    [
        (0, [], ValueError),
        (6, [], ValueError),
        (True, [], TypeError),
        (2.5, [], TypeError),
        (3, [99], ValueError),
        (3, [0], ValueError),  # the game started at ply 6
        (3, [True], TypeError),
    ],
)
def test_a_bad_note_is_refused_and_keeps_nothing(table, rating, interesting, error):
    number = win_for_the_person(table)["game"]
    with pytest.raises(error):
        table.note(number, rating, interesting, "")
    assert json.loads(table.path.read_text())[0]["note"] is None


def test_a_note_on_a_game_still_going_is_refused(table):
    number = table.new("random", 0)["game"]
    with pytest.raises(ValueError, match="isn't over"):
        table.note(number, 3, [], "")


def test_a_later_run_adds_to_the_records_and_numbers_on(tmp_path):
    path = tmp_path / "games.json"
    first = Table(GAME, LADDER, path, seed=0)
    number = win_for_the_person(first)["game"]
    first.note(number, 5, [], "good")
    second = Table(GAME, LADDER, path, seed=0)
    later = win_for_the_person(second)["game"]
    assert later > number
    records = json.loads(path.read_text())
    assert [r["game"] for r in records] == [number, later]
    assert records[0]["note"]["text"] == "good"


def test_a_file_of_something_else_is_refused_and_left_alone(tmp_path):
    path = tmp_path / "games.json"
    path.write_text('{"not": "records"}')
    with pytest.raises(ValueError, match="list of records"):
        Table(GAME, LADDER, path, seed=0)
    assert path.read_text() == '{"not": "records"}'


def test_a_file_that_isnt_json_is_refused_and_left_alone(tmp_path):
    path = tmp_path / "games.json"
    path.write_text("half a reco")
    with pytest.raises(ValueError, match="Expecting value"):
        Table(GAME, LADDER, path, seed=0)
    assert path.read_text() == "half a reco"


def test_the_seed_and_the_games_number_set_a_players_draws(tmp_path):
    noisy = (Rung("noisy", "Draws noise.", replace(SEARCH, simulations=2, noise=True)),)

    def replies(name, seed, games):
        table = Table(GAME, noisy, tmp_path / f"{name}.json", seed)
        return [
            table.reply(table.new("noisy", 1, noise=True)["game"])["moves"][0]["why"]["columns"]
            for _ in range(games)
        ]

    again = replies("a", 8, 8)
    assert again == replies("b", 8, 8)
    assert again != replies("c", 9, 8)
    assert any(game != again[0] for game in again[1:])  # each game has its own draws


def test_standing_counts_each_rungs_results_and_mean_rating(tmp_path):
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=0)
    for rating in (2, 5, None):
        table.note(win_for_the_person(table)["game"], rating, [], "")
    table.reply(
        table.new(
            "fit", 1, text_of((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1))
        )["game"]
    )
    standing = {rung["name"]: rung for rung in table.setup()["rungs"]}
    assert (standing["random"]["won"], standing["random"]["rating"]) == (3, 3.5)
    assert (standing["fit"]["lost"], standing["fit"]["rating"]) == (1, None)
    assert standing["fit"]["about"] == "A small search."


def test_setup_draws_every_cell_with_row_0_at_the_bottom(table):
    setup = table.setup()
    assert (setup["columns"], setup["rows"]) == GAME.shape
    assert len(setup["cells"]) == GAME.space.n_cells
    bottom, top = setup["cells"][0], setup["cells"][ROWS - 1]
    assert bottom["y"] > top["y"]  # y runs down the picture
    assert setup["cells"][ROWS]["x"] > bottom["x"]
    assert 0 < bottom["x"] < setup["width"]
    assert 0 < top["y"] < setup["height"]


@pytest.fixture
def served(table):
    server = serve(table, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()
    server.server_close()


def call(server, path, body=None, headers=None, host=None, raw=None):
    """Return the status and body of a request, as bytes."""
    data = raw if body is None else json.dumps(body).encode()
    sent = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}", data, {**sent, **(headers or {})}
    )
    if host is not None:
        request.add_unredirected_header("Host", host)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read()


def test_the_page_is_served_with_the_reports_styles(served):
    status, body = call(served, "/")
    assert status == 200
    text = body.decode()
    assert "<title>Connect Four Table</title>" in text
    assert "/* STYLE */" not in text
    assert "--first:" in text
    assert text == page()


def test_setup_is_served_as_json(served):
    status, body = call(served, "/api/setup")
    assert status == 200
    assert [r["name"] for r in json.loads(body)["rungs"]] == ["random", "fit"]


def test_a_game_is_played_through_the_routes(served, table):
    status, body = call(served, "/api/new", {"rung": "fit", "seat": 0})
    assert status == 200
    number = json.loads(body)["game"]
    status, body = call(served, "/api/move", {"game": number, "column": 3})
    assert (status, json.loads(body)["to_move"]) == (200, 1)
    status, body = call(served, "/api/reply", {"game": number})
    state = json.loads(body)
    assert (status, len(state["moves"]), state["moves"][1]["by"]) == (200, 2, "fit")
    status, body = call(served, "/api/state", {"game": number})
    assert json.loads(body) == state


def test_a_note_is_kept_through_the_route(served, table):
    number = win_for_the_person(table)["game"]
    fields = {"game": number, "rating": 4, "interesting": [6], "text": "a </script> note"}
    assert call(served, "/api/note", fields)[0] == 200
    assert json.loads(table.path.read_text())[0]["note"]["text"] == "a </script> note"
    fields["rating"] = None
    assert call(served, "/api/note", fields)[0] == 200
    assert json.loads(table.path.read_text())[0]["note"]["rating"] is None


@pytest.mark.parametrize("noise", [False, True])
def test_noise_is_asked_for_through_the_route(served, noise):
    _, body = call(served, "/api/new", {"rung": "fit", "seat": 0, "noise": noise})
    assert json.loads(body)["noise"] is noise


def test_noise_stays_off_unless_a_request_asks(served):
    _, body = call(served, "/api/new", {"rung": "fit", "seat": 0})
    assert json.loads(body)["noise"] is False


def test_a_board_is_started_from_through_the_route(served):
    text = text_of((3, 0, 0))
    _, body = call(served, "/api/new", {"rung": "random", "seat": 1, "board": text})
    state = json.loads(body)
    assert state["start"] == text
    assert state["to_move"] == 1


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/new", {"seat": 0}),
        ("/api/new", {"rung": "fit"}),
        ("/api/new", {"rung": 3, "seat": 0}),
        ("/api/new", {"rung": "fit", "seat": "0"}),
        ("/api/new", {"rung": "fit", "seat": True}),
        ("/api/new", {"rung": "fit", "seat": 0, "board": 7}),
        ("/api/new", {"rung": "fit", "seat": 0, "noise": 1}),
        ("/api/new", {"rung": "fit", "seat": 0, "board": "nonsense"}),
        ("/api/new", ["rung", "fit"]),
        ("/api/move", {"game": 0, "column": 3}),
        ("/api/move", {"game": "0", "column": 3}),
        ("/api/note", {"game": 0, "rating": "good", "interesting": [], "text": ""}),
        ("/api/note", {"game": 0, "rating": 3, "interesting": "all", "text": ""}),
        ("/api/note", {"game": 0, "rating": 3, "interesting": [], "text": 5}),
    ],
)
def test_a_wrong_request_is_answered_400_saying_why(served, path, body):
    status, answer = call(served, path, body)
    assert status == 400
    assert json.loads(answer)["error"]


def test_a_body_that_isnt_json_is_answered_400(served):
    assert call(served, "/api/state", raw=b"{half")[0] == 400


def test_an_unknown_address_is_answered_404(served):
    assert call(served, "/nothing")[0] == 404
    assert call(served, "/api/nothing", {})[0] == 404


def test_a_request_addressed_to_another_name_is_refused(served, table):
    host = "evil.example"
    assert call(served, "/", host=host)[0] == 403
    assert call(served, "/api/new", {"rung": "fit", "seat": 0}, host=host)[0] == 403
    # Nothing started: the next game is still the first.
    assert table.new("random", 0)["game"] == 0


def test_localhost_is_this_machine(served):
    assert call(served, "/", host=f"localhost:{served.server_port}")[0] == 200


def test_a_write_that_doesnt_say_it_is_json_is_refused(served, table):
    status, _ = call(
        served, "/api/new", {"rung": "fit", "seat": 0}, headers={"Content-Type": "text/plain"}
    )
    assert status == 415
    assert table.new("random", 0)["game"] == 0


def test_the_server_listens_on_this_machine_only(served):
    assert served.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("mover", [0, 1])
def test_expects_falls_back_on_the_positions_value_for_a_column_never_visited(mover):
    stones = [(3, 0, 0)] if mover == 1 else []
    position = position_of(GAME, board(*stones), len(stones))
    result = replace(SEARCH, simulations=2).search(GAME, position, np.random.default_rng(0))
    cell = next(
        int(c) for c in np.flatnonzero(GAME.legal(position)[PLACE]) if np.isnan(result.q[PLACE, c])
    )
    why = explain(GAME, FIT, position, result._replace(action=Action(PLACE, cell)))
    assert why.expects == (result.value if mover == 0 else 1.0 - result.value)
    assert why.expects != 0.5  # else the two sides can't be told apart


def test_a_record_names_what_its_player_was_built_from(tmp_path):
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=0, players="fit.json abc")
    assert finished(table)["players"] == "fit.json abc"
    assert json.loads((tmp_path / "games.json").read_text())[0]["players"] == "fit.json abc"


def test_the_setup_counts_only_games_against_the_same_players(tmp_path):
    path = tmp_path / "games.json"
    earlier = Table(GAME, LADDER, path, seed=0, players="old")
    finished(earlier)
    assert sum(_counts(earlier.setup())) == 1
    later = Table(GAME, LADDER, path, seed=0, players="new")
    assert sum(_counts(later.setup())) == 0
    assert len(later.records) == 1  # kept, only not counted
    assert sum(_counts(Table(GAME, LADDER, path, seed=0, players="old").setup())) == 1


def _counts(setup):
    return [rung["won"] + rung["lost"] + rung["drawn"] for rung in setup["rungs"]]


def test_a_length_that_is_no_integer_is_answered_not_dropped(tmp_path):
    table = Table(GAME, LADDER, tmp_path / "games.json", seed=0)
    server = serve(table, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.putrequest("POST", "/api/new")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "many")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert "Content-Length" in json.loads(response.read())["error"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fingerprint_changes_with_a_file_s_contents_not_its_folder(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one, two = tmp_path / "a" / "fit.json", tmp_path / "b" / "fit.json"
    one.write_text("1")
    two.write_text("1")
    assert fingerprint(str(one)) == fingerprint(str(two))
    two.write_text("2")
    assert fingerprint(str(one)) != fingerprint(str(two))
    assert fingerprint(str(one)).startswith("fit.json ")
    assert fingerprint(str(one), str(two)).count(",") == 1


def finished(table):
    """Return a game the person has just won: three stones in column 0, and their move."""
    stones = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1)]
    return table.move(table.new("random", 0, text_of(*stones))["game"], 0)
