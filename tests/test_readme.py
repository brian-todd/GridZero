"""The examples in the README run as written and print what they say."""

import re
from pathlib import Path

from game_demo import check_game
from gridzero.agents import UniformPlayer
from gridzero.core import LatticeSpace, icosphere
from model_demo import check_model

README = Path(__file__).parents[1] / "README.md"


def blocks() -> list[str]:
    """Return the README's Python blocks, in order."""
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


def test_the_readme_examples_run_and_print_what_they_say(capsys):
    assert blocks()
    for code in blocks():
        exec(compile(code, str(README), "exec"), {})  # each block on its own
    assert capsys.readouterr().out.splitlines() == [
        "[774 338 168  93]",
        "0 13",
        "Score(won=79, lost=21, drawn=0)",
    ]


def test_the_readme_model_keeps_the_promises_of_a_model():
    names: dict[str, object] = {}
    exec(compile(blocks()[0], str(README), "exec"), names)  # the canonical model
    life = names["Life"]()  # ty: ignore[call-non-callable]  # a class the README defines
    check_model(life, LatticeSpace((8, 8), tiling="triangle", periodic=True), steps=5)
    check_model(life, icosphere(2), steps=5)


def test_the_readme_game_keeps_the_promises_of_a_game():
    names: dict[str, object] = {}
    exec(compile(blocks()[1], str(README), "exec"), names)  # the canonical game
    for shape in [(5, 5), (2, 3)]:
        game = names["Stakes"](shape)  # ty: ignore[call-non-callable]  # a class the README defines
        check_game(game, UniformPlayer(), seeds=range(4))
