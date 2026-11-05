"""Every experiment that finishes in seconds at its smallest, run once through ``run.py``.

Not run here: ``tactics.watch`` (a GIF of hundreds of frames, a minute),
``in_a_row.code_report`` (a deep search over hundreds of positions, whose parts `test_report`
runs small), ``in_a_row.play`` (a server, until stopped, whose parts `test_play` starts and plays),
and the training experiments ``tactics.zero_scratch``, ``in_a_row.zero_scratch`` and
``colonies.muzero_scratch`` (the GPU, and half a minute to minutes each).
`test_run` imports each of them and checks its arguments.
"""

import json

import pytest
import torch
from PIL import Image, ImageSequence

import run
from examples.games.colonies import EVAL_SEED, Colonies, Raider
from examples.games.colonies.experiments import muzero_ladder
from examples.games.colonies.learned import player
from examples.games.in_a_row import (
    MOVE_FEATURES,
    VALUE_FEATURES,
    FittedEvaluator,
    InARow,
    Lessons,
)
from examples.games.tactics import Tactics
from examples.games.voronoi import EVAL_SEED as VORONOI_SEED
from examples.games.voronoi import Greedy, Voronoi
from examples.simulations.biomes import Biomes
from examples.simulations.predators import Predators
from gridzero.agents import UniformPlayer
from gridzero.agents.muzero_net import WorldNet
from gridzero.agents.net import PolicyNet
from gridzero.game import play

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """Untrained weights of Tactics', in_a_row's and Colonies' networks, as experiments use."""
    folder = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    tactics, four = folder / "tactics.pt", folder / "four.pt"
    torch.save(PolicyNet(Tactics()).state_dict(), tactics)
    torch.save(PolicyNet(InARow(), channels=32, blocks=4).state_dict(), four)
    colonies = folder / "colonies.pt"
    torch.save(WorldNet(Colonies()).state_dict(), colonies)
    return {"tactics": str(tactics), "four": str(four), "colonies": str(colonies)}


def ran(capsys, name, *args):
    assert run.main([name, *(str(a) for a in args)]) == 0
    return capsys.readouterr().out


@pytest.mark.parametrize(("name", "picture"), [("competition", "map"), ("migration", "globes")])
def test_a_simulation_without_years_saves_its_picture(capsys, tmp_path, name, picture):
    prefix = tmp_path / name
    out = ran(capsys, name, prefix)
    assert f"saved {prefix}_{picture}.png" in out
    assert (tmp_path / f"{name}_{picture}.png").exists()


def test_biomes_runs_past_the_wolves(capsys):
    out = ran(capsys, "biomes", Biomes().wolves_arrive + 1)
    assert "apparent competition" in out


@pytest.mark.parametrize(
    ("name", "years"), [("predators", Predators().foxes_arrive + 1), ("settlers", 2)]
)
def test_a_simulation_with_years_saves_its_map_and_history(capsys, tmp_path, name, years):
    prefix = tmp_path / name
    out = ran(capsys, name, years, prefix)
    assert f"saved {prefix}_map.png and {prefix}_history.png" in out
    assert (tmp_path / f"{name}_map.png").exists()
    assert (tmp_path / f"{name}_history.png").exists()


def test_population_evolution_prints_its_species_and_saves_its_pictures(capsys, tmp_path):
    prefix = tmp_path / "evolution"
    out = ran(capsys, "population_evolution", 100, prefix)
    assert "step    100:" in out
    assert "1 species" in out
    assert (tmp_path / "evolution_map.png").exists()
    assert (tmp_path / "evolution_history.png").exists()


def test_population_evolution_warms_from_halfway(capsys, tmp_path):
    out = ran(capsys, "population_evolution", 100, tmp_path / "warm", 0.001)
    assert "warming 0.001 a step from step 50" in out


def test_tournament(capsys, tmp_path):
    out = ran(capsys, "tactics.tournament", 1, tmp_path / "tactics")
    assert "Check: greedy won" in out
    assert (tmp_path / "tactics_end.png").exists()


def test_voronoi_tournament(capsys, tmp_path):
    for players in (2, 3):
        out = ran(capsys, "voronoi.tournament", 1, tmp_path / f"voronoi{players}", players)
        assert "Greedy against uniform in every other seat:" in out
        assert "wins by seat [" in out
        assert (tmp_path / f"voronoi{players}_end.png").exists()


def test_colonies_tournament(capsys, tmp_path):
    for players in (2, 3):
        out = ran(capsys, "colonies.tournament", 1, tmp_path / f"colonies{players}", players)
        assert out.count("Raider against uniform in every other seat:") == 2  # fog, and not
        assert "Without fog:" in out
        assert "wins by seat [" in out
        assert (tmp_path / f"colonies{players}_end.png").exists()


def test_colonies_watch_saves_every_round_of_its_start(capsys, tmp_path, weights):
    out = ran(capsys, "colonies.watch", 3, tmp_path / "watch", weights["colonies"], "raider")
    game = Colonies()
    record = play(game, [player(weights["colonies"])[1], Raider()], game.start(EVAL_SEED + 3))
    people = record.final.state.people[:2].sum(axis=1)
    rounds = record.final.round
    assert f": {rounds} rounds, " in out
    assert f"(people {people[0]:.1f}, {people[1]:.1f})" in out
    with Image.open(tmp_path / "watch_game.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == rounds + 1


def test_voronoi_watch_saves_every_stone_of_its_start(capsys, tmp_path):
    out = ran(capsys, "voronoi.watch", 2, tmp_path / "watch", "uniform", "greedy", "greedy")
    game = Voronoi(players=3)
    record = play(game, [UniformPlayer(), Greedy(), Greedy()], game.start(VORONOI_SEED + 2))
    land = ", ".join(f"{a:.4f}" for a in game.areas(record.final.state))
    assert f"uniform against greedy against greedy: {len(record.actions)} stones" in out
    assert f"(land {land})" in out
    with Image.open(tmp_path / "watch_game.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == len(record.actions) + 1


def test_games_curves_saves_every_metric_of_every_run(capsys, tmp_path):
    for run_name, loss in (("a", 1.0), ("b", 2.0)):
        lines = [json.dumps({"generation": g, "loss": loss / g, "steps": [g, g]}) for g in (1, 2)]
        (tmp_path / f"{run_name}_generations.jsonl").write_text("\n".join(lines) + "\n")
    out = ran(capsys, "games.curves", tmp_path / "both", tmp_path / "a", tmp_path / "b")
    assert "2 metrics of 2 runs" in out
    assert (tmp_path / "both_curves.png").exists()


def test_colonies_muzero_ladder(capsys, weights):
    out = ran(capsys, "colonies.muzero_ladder", 1, 2, weights["colonies"])
    assert "Ratings (Elo scale, uniform 0):" in out
    assert "colonies" in out  # the weights file's name


def test_colonies_muzero_ladder_refuses_two_players_of_one_name(weights):
    spec = str(weights["colonies"])
    with pytest.raises(ValueError, match="two players named 'colonies'"):
        muzero_ladder.main("1", "2", spec, spec)


def test_colonies_muzero_check(capsys, weights):
    out = ran(capsys, "colonies.muzero_check", weights["colonies"], 1, 2)
    assert "network alone against uniform" in out
    assert "MuZero, 2 simulations against" in out


def test_search(capsys):
    out = ran(capsys, "tactics.search", 1, 1)
    assert "Check: search beats greedy at 1 simulations:" in out


def test_zero_check(capsys, weights):
    out = ran(capsys, "tactics.zero_check", weights["tactics"], 1, 2, "net")
    assert "Against net: " in out


def test_in_a_row_zero_check(capsys, weights):
    out = ran(capsys, "in_a_row.zero_check", weights["four"], 1, 2)
    assert "agent, 2 simulations against threat" in out


def test_in_a_row_code_fit_writes_a_fit_and_lessons_per_round(capsys, tmp_path, weights):
    prefix = tmp_path / "code"
    out = ran(capsys, "in_a_row.code_fit", weights["four"], 1, prefix, 2)
    assert "Round 1: " in out
    assert "at 64 simulations against threat" in out
    assert FittedEvaluator.load(f"{prefix}_round1.json")
    assert len(Lessons.load(f"{prefix}_lessons.npz")) > 0


def test_in_a_row_code_ladder(capsys, tmp_path, weights):
    fit = tmp_path / "fit.json"
    FittedEvaluator((0.0,) * len(VALUE_FEATURES), 0.0, (0.0,) * len(MOVE_FEATURES)).save(fit)
    out = ran(capsys, "in_a_row.code_ladder", weights["four"], fit, 1, 2)
    assert "teacher: after each first column" in out
    assert "us a simulation at 64 simulations" in out
    for name in ("search", "lines", "fit", "teacher"):
        assert f"2 simulations, {name} against uniform" in out
    assert "against teacher at 2 " in out


def test_in_a_row_code_fit_refuses_a_prefix_already_used(capsys, tmp_path, weights):
    prefix = tmp_path / "code"
    (tmp_path / "code_round2.json").write_text("{}")  # a round the new run might not reach
    with pytest.raises(FileExistsError, match="prefix of its own"):
        ran(capsys, "in_a_row.code_fit", weights["four"], 1, prefix, 1)
    assert not (tmp_path / "code_lessons.npz").exists()
