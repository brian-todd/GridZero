import numpy as np
import pytest

from examples.games.tactics import (
    RandomPlayer,
    Verb,
    event_text,
    summarize,
    summary_text,
)
from examples.games.tactics.text import where
from gridzero.game import play, replay, series
from tactics_demo import TINY


def test_every_event_of_a_game_reads_as_a_sentence():
    record = play(TINY, [RandomPlayer(), RandomPlayer()], TINY.start(2))
    state = record.final.state
    kinds = set()
    for event in (e for position in replay(TINY, record) for e in position.events):
        sentence = event_text(TINY, state, event)
        assert sentence
        kinds.add(type(event).__name__)
    assert {"Placed", "Moved", "Attacked", "Ended", "RoundStarted", "GameOver"} <= kinds


def test_event_text_refuses_what_is_not_an_event():
    with pytest.raises(TypeError, match="event"):
        event_text(TINY, TINY.start(0).state, (Verb.END, 3))  # ty: ignore[invalid-argument-type]


def test_where_writes_axial_coordinates(at):
    state = TINY.start(0).state
    assert where(state, at(3, 7)) == "(3, 7)"
    assert where(state, np.int64(at(0, 0))) == "(0, 0)"


def test_a_summary_reads_its_counts_and_has_a_row_per_kind():
    records = series(TINY, [("a", RandomPlayer()), ("b", RandomPlayer())], 1, seed=4)
    summary = summarize(TINY, records)
    first, endings, blank, header, rule, *rows = summary_text(TINY, summary).splitlines()
    assert endings.startswith("Ended by: killed ")
    counts = summary.tally
    assert first.startswith(f"2 games: wins {counts.wins}, draws {counts.draws}, by team ")
    assert first.endswith(f"{summary.rounds:.1f} rounds and {counts.plies:.0f} decisions a game.")
    assert (blank, header, rule) == (
        "",
        "| Kind | Placed | Damage dealt | Damage taken | Kills | Losses |",
        "|---|---|---|---|---|---|",
    )
    assert len(rows) == len(TINY.units)
    for k, (row, stats) in enumerate(zip(rows, TINY.units, strict=True)):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells == [
            stats.name,
            str(summary.placed[k]),
            str(summary.damage_dealt[k]),
            str(summary.damage_taken[k]),
            str(summary.kills[k]),
            str(summary.losses[k]),
        ]
