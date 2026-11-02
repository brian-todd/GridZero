"""Tests of reading and showing training runs (`examples.games.curves`)."""

import json

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from examples.games.curves import RUNS, Curves, grid, log, name, new_log, read

FIRST = {
    "generation": 3,
    "loss": 1.5,
    "steps": [1, 2, 3],
    "scores": {"uniform": [9, 1, 0], "note": "text"},
    "opening": [1, [0.5, 0.5], 0.25],
    "done": True,
    "missing": None,
}


def write(path, *entries, partial=""):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries) + partial)
    return path


def test_read_takes_numbers_by_key_and_a_list_of_numbers_as_one_metric(tmp_path):
    generations, metrics = read(write(tmp_path / "a_generations.jsonl", FIRST))
    np.testing.assert_array_equal(generations, [3.0])
    assert list(metrics) == [
        "loss",
        "steps",
        "scores.uniform",
        "opening[0]",
        "opening[1]",
        "opening[2]",
    ]
    np.testing.assert_array_equal(metrics["loss"], [[1.5]])
    np.testing.assert_array_equal(metrics["steps"], [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(metrics["scores.uniform"], [[9.0, 1.0, 0.0]])
    np.testing.assert_array_equal(metrics["opening[1]"], [[0.5, 0.5]])
    assert all(m.dtype == np.float64 for m in metrics.values())


def test_read_fills_what_a_generation_lacks_with_nan(tmp_path):
    second = {"generation": 4, "steps": [5, 6], "new": 2}
    _, metrics = read(write(tmp_path / "a_generations.jsonl", FIRST, second))
    np.testing.assert_array_equal(metrics["loss"], [[1.5], [np.nan]])
    np.testing.assert_array_equal(metrics["steps"], [[1, 2, 3], [5, 6, np.nan]])
    np.testing.assert_array_equal(metrics["new"], [[np.nan], [2]])


def test_read_numbers_lines_without_a_generation_and_skips_one_being_written(tmp_path):
    path = write(tmp_path / "a_generations.jsonl", {"loss": 1}, {"loss": 2}, partial='{"lo')
    generations, metrics = read(path)
    np.testing.assert_array_equal(generations, [1.0, 2.0])
    np.testing.assert_array_equal(metrics["loss"], [[1.0], [2.0]])


def test_a_run_is_found_by_its_file_or_its_prefix_and_named_for_it(tmp_path):
    path = write(tmp_path / "a_generations.jsonl", FIRST)
    assert log(path) == path
    assert log(tmp_path / "a") == path
    assert name(path) == "a"
    folder = tmp_path / "trial"
    folder.mkdir()
    assert name(write(folder / "generations.jsonl", FIRST)) == "trial"
    with pytest.raises(FileNotFoundError, match="no run"):
        log(tmp_path / "b")


def runs(tmp_path):
    a = write(tmp_path / "a_generations.jsonl", FIRST, FIRST | {"generation": 4})
    b = write(tmp_path / "b_generations.jsonl", {"generation": 1, "loss": 2, "other": 1})
    return a, b


def lines(curves):
    return [(line.get_color(), len(line.get_xdata())) for line in curves.ax.get_lines()]


def test_curves_show_the_first_metric_of_every_run_that_has_it(tmp_path):
    curves = Curves(Figure(), runs(tmp_path))
    assert curves.metric == "loss"
    assert curves.ax.get_title() == "loss"
    assert lines(curves) == [(RUNS[0], 2), (RUNS[1], 1)]
    assert curves.metrics[-1] == "other"
    legend = curves.ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


def test_choosing_a_metric_draws_each_of_its_values_in_a_style_of_its_own(tmp_path):
    curves = Curves(Figure(), runs(tmp_path))
    curves.buttons.set_active(curves.metrics.index("steps"))
    assert curves.ax.get_title() == "steps"
    assert [line.get_linestyle() for line in curves.ax.get_lines()] == ["-", "--", ":"]


def test_up_and_down_choose_the_previous_and_the_next_metric_round_the_list(tmp_path):
    curves = Curves(Figure(), runs(tmp_path))
    canvas = curves.figure.canvas
    for key in ("up", "down", "down", "x"):
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key))
        if key == "up":
            assert curves.metric == curves.metrics[-1]
    assert curves.metric == curves.metrics[1]


def test_refresh_redraws_only_when_a_run_has_grown(tmp_path):
    a, b = runs(tmp_path)
    curves = Curves(Figure(), [a, b])
    marker = curves.ax.axhline(0)
    assert not curves.refresh()
    assert marker in curves.ax.get_lines()  # not drawn again
    with a.open("a") as f:
        f.write(json.dumps(FIRST | {"generation": 5, "loss": 0.5}) + "\n")
    assert curves.refresh()
    assert lines(curves) == [(RUNS[0], 3), (RUNS[1], 1)]


def test_the_timer_keeps_reading_the_runs_after_a_tick_with_nothing_new(tmp_path):
    a, b = runs(tmp_path)
    curves = Curves(Figure(), [a, b])
    curves.timer._on_timer()  # ty: ignore[unresolved-attribute] # a tick (private, unstubbed)
    assert len(curves.timer.callbacks) == 1  # Matplotlib drops a callback returning False
    with a.open("a") as f:
        f.write(json.dumps(FIRST | {"generation": 5, "loss": 0.5}) + "\n")
    curves.timer._on_timer()  # ty: ignore[unresolved-attribute] # private, as above
    assert lines(curves) == [(RUNS[0], 3), (RUNS[1], 1)]


def test_a_new_log_refuses_a_prefix_a_run_already_used(tmp_path):
    path = new_log(str(tmp_path / "run"))
    assert path == tmp_path / "run_generations.jsonl"
    assert not path.exists()
    path.write_text("")
    with pytest.raises(FileExistsError, match="prefix of its own"):
        new_log(str(tmp_path / "run"))


def test_curves_need_a_run(tmp_path):
    with pytest.raises(ValueError, match="at least one run"):
        Curves(Figure(), [])


def test_a_grid_draws_every_metric_with_one_legend_of_every_run(tmp_path):
    a, b = runs(tmp_path)
    figure = Figure()
    shown = [(name(p), *read(p)) for p in (a, b)]
    grid(figure, shown)
    assert [ax.get_title() for ax in figure.axes] == [
        "loss",
        "steps",
        "scores.uniform",
        "opening[0]",
        "opening[1]",
        "opening[2]",
        "other",
    ]
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["a", "b"]
