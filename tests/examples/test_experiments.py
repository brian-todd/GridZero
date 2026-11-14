"""Every experiment that finishes in seconds at its smallest, run once through ``run.py``.

Not run here: ``tactics.watch`` (a GIF of hundreds of frames, a minute),
``tactics.code_games`` (a minute a game, whose parts `test_lessons` runs small),
``in_a_row.code_report`` (a deep search over hundreds of positions, whose parts `test_report`
runs small), ``in_a_row.play`` (a server, until stopped, whose parts `test_play` starts and plays),
and the training experiments ``tactics.zero_scratch``, ``tactics.ppo``,
``in_a_row.zero_scratch`` and ``colonies.muzero_scratch`` (the GPU, and half a minute to minutes
each).
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
from examples.games.tactics import EVAL_SEED as TACTICS_SEED
from examples.games.tactics import Lessons as TacticsLessons
from examples.games.tactics import SearchPlayer, Tactics, lessons_of, small_rules
from examples.games.tactics.experiments import code_ladder
from examples.games.tactics.fit import LOOKAHEAD
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


def test_tactics_code_games_refuses_a_prefix_already_used(capsys, tmp_path):
    (tmp_path / "code_lessons.npz").write_text("")
    with pytest.raises(FileExistsError, match="has been used"):
        ran(capsys, "tactics.code_games", 1, tmp_path / "code")


def test_tactics_code_ladder(capsys):
    out = ran(capsys, "tactics.code_ladder", "hand", 1, 15, 1)
    assert "1 simulations: " in out
    assert " ms a decision, " in out
    assert " ms a unit's turn." in out
    assert "Against random " in out
    assert ", greedy " in out


def test_tactics_code_ladder_plays_a_replica_and_rivals_by_name(capsys):
    out = ran(capsys, "tactics.code_ladder", "hand", 1, 15, "1,2", 1, "greedy")
    assert "Against greedy " in out
    assert "random" not in out
    assert "2 simulations: " in out


def test_tactics_code_ladder_replicas_have_starts_of_their_own():
    assert code_ladder.first_start(0) == TACTICS_SEED
    assert code_ladder.first_start(1) - code_ladder.first_start(0) >= 1000


def test_tactics_fog_compares_the_open_game_with_fog(capsys):
    out = ran(capsys, "tactics.fog", "hand", 1, 15, 1)
    assert "In the open, greedy against itself: 2 games" in out
    assert "Under fog, greedy against itself: 2 games" in out
    assert out.count("Greedy with (1, 2, 2, 1, 1) against (3, 2, 1, 1, 1)") == 3
    assert "under fog, every kind seeing 3" in out
    assert out.count("The code player at 1 simulations against greedy") == 2


def test_tactics_code_ladder_refuses_an_unknown_rival(capsys):
    with pytest.raises(KeyError, match="nobody"):
        ran(capsys, "tactics.code_ladder", "hand", 1, 15, 1, 0, "nobody")


def test_tactics_code_fit_saves_constants_the_ladder_plays(capsys, tmp_path):
    game = small_rules()
    teacher = SearchPlayer(simulations=2, plans=2, depth=1, rollout_turns=1)
    path = tmp_path / "games_lessons.npz"
    games = [lessons_of(game, teacher, seed) for seed in range(3)]
    TacticsLessons.join(games).save(path)
    out = ran(capsys, "tactics.code_fit", tmp_path / "code", path)
    assert "Value on 1 games kept out" in out
    assert "Prior on 1 games kept out" in out
    saved = tmp_path / "code_fit.json"
    assert set(json.loads(saved.read_text())) == {"value", "intercept", "prior", "lookahead"}
    assert json.loads(saved.read_text())["lookahead"] == 0
    played = tmp_path / "code_player.json"
    assert json.loads(played.read_text())["lookahead"] == LOOKAHEAD
    assert "Against random " in ran(capsys, "tactics.code_ladder", played, 1, 15, 1)


def test_tactics_code_fit_refuses_a_prefix_already_used(capsys, tmp_path):
    (tmp_path / "code_fit.json").write_text("")
    with pytest.raises(FileExistsError, match="has been used"):
        ran(capsys, "tactics.code_fit", tmp_path / "code", "lessons.npz")


def test_tactics_code_fit_needs_lessons(capsys, tmp_path):
    with pytest.raises(ValueError, match="lessons"):
        ran(capsys, "tactics.code_fit", tmp_path / "code")


@pytest.mark.slow
def test_rules(capsys, tmp_path):
    out = ran(capsys, "tactics.rules", "greedy", 1, tmp_path / "rules", 1, 5)
    assert "### Styles, 4 rounds to hold" in out
    report = json.loads((tmp_path / "rules_rules.json").read_text())
    assert set(report["self_play"]) == {"hold 5"}
    assert "seeker v sitter" in report["styles"]
    assert report["own_picks"] is None
    assert report["rosters"]["scores"]


@pytest.mark.slow
def test_rules_with_a_code_player_reports_its_own_picks_and_no_styles(capsys, tmp_path):
    out = ran(capsys, "tactics.rules", "hand", 1, tmp_path / "rules", 2, 5)
    assert "### Its own picks, 4 rounds to hold" in out
    report = json.loads((tmp_path / "rules_rules.json").read_text())
    assert report["styles"] == {}
    assert sum(report["own_picks"]["picked"].values()) == 4
    assert report["rosters"]["scores"]


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
