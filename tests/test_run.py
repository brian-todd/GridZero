import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import run

ROOT = Path(__file__).parents[1]


def test_every_experiment_on_disk_is_named_and_every_name_is_on_disk():
    on_disk = {
        ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for pattern in (
            "examples/simulations/*/experiment.py",
            "examples/games/*/experiments/*.py",
            "examples/games/experiments/*.py",
        )
        for p in ROOT.glob(pattern)
        if p.name != "__init__.py"
    }
    assert set(run.EXPERIMENTS.values()) == on_disk


def test_each_simulation_runs_under_its_name_and_each_game_experiment_under_the_game():
    for name, module in run.EXPERIMENTS.items():
        parts = module.split(".")
        if parts[1] == "simulations":
            assert name == parts[2]
        elif parts[2] == "experiments":  # over any game's output
            assert name == f"games.{parts[3]}"
        else:
            assert name == f"{parts[2]}.{parts[4]}"


@pytest.mark.parametrize("name", list(run.EXPERIMENTS))
def test_every_experiment_has_a_summary_and_takes_strings(name):
    module = importlib.import_module(run.EXPERIMENTS[name])
    assert (inspect.getdoc(module) or "").strip()
    for parameter in inspect.signature(module.main).parameters.values():
        assert parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL)
        assert parameter.annotation in (str, str | None), parameter


def test_no_arguments_list_the_experiments_and_fail(capsys):
    assert run.main([]) == 2
    out = capsys.readouterr().out
    assert all(f"  {name} " in out for name in run.EXPERIMENTS)


def test_help_lists_the_experiments_and_succeeds(capsys):
    assert run.main(["--help"]) == 0
    assert "tactics.zero_check" in capsys.readouterr().out


def test_an_unknown_experiment_fails_and_says_so(capsys):
    assert run.main(["nonesuch"]) == 2
    assert "no experiment 'nonesuch'" in capsys.readouterr().err


def test_too_many_arguments_fail_before_the_experiment_runs(capsys):
    assert run.main(["tactics.tournament", "1", "a", "extra"]) == 2
    err = capsys.readouterr().err
    assert "too many positional arguments" in err
    assert "run.py tactics.tournament [starts] [prefix]" in err


def test_missing_arguments_fail_before_the_experiment_runs(capsys):
    assert run.main(["tactics.zero_check"]) == 2
    assert "run.py tactics.zero_check <weights> [starts]" in capsys.readouterr().err


def test_help_on_an_experiment_prints_its_docstring(capsys):
    assert run.main(["predators", "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("uv run python run.py predators [years] [prefix]")
    assert "A trophic cascade" in out


def test_any_number_of_trailing_arguments_is_shown_as_optional():
    assert run.usage("tactics.search").startswith(
        "uv run python run.py tactics.search [starts] [budgets]"
    )


def test_the_script_runs_an_experiment_with_its_arguments(tmp_path):
    prefix = tmp_path / "settlers"
    done = subprocess.run(
        [sys.executable, "run.py", "settlers", "1", str(prefix)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "year     1:" in done.stdout
    assert Path(f"{prefix}_map.png").exists()
    assert Path(f"{prefix}_history.png").exists()


def test_the_script_fails_with_status_2_on_bad_arguments():
    done = subprocess.run(
        [sys.executable, "run.py", "nonesuch"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 2
    assert "no experiment" in done.stderr
