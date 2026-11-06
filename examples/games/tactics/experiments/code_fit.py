"""Fit a code player to a search player's games, and print how it does on games kept out.

Reads what ``tactics.code_games`` saved, on `small_rules`. Prints the fit's value and prior beside
simple baselines on the same games, then its constants by name, and saves them as
``{prefix}_fit.json``, for ``tactics.code_ladder``.

Run with ``uv run python run.py tactics.code_fit prefix lessons.npz [more.npz...]``. Fit a teacher
that works apart from one that doesn't: one that never works says nothing of when to.
"""

import time
from pathlib import Path

from examples.games import CORES
from examples.games.tactics import MOVE_FEATURES, VALUE_FEATURES, Lessons, small_rules
from examples.games.tactics.fit import cases_of, fit

SEED = 6_500_000  # which games are kept out


def main(prefix: str, *lessons: str) -> None:
    """Fit, printing the scores and the constants, and save the constants.

    Parameters
    ----------
    prefix : str
        Every file written starts with it.
    *lessons : str
        The files of lessons to fit to, at least one.

    Raises
    ------
    ValueError
        If no lessons are given.
    FileExistsError
        If a run already used the prefix.
    """
    if not lessons:
        raise ValueError("give at least one file of lessons")
    path = Path(f"{prefix}_fit.json")
    if path.exists():
        raise FileExistsError(f"{prefix} has been used: give the new run a prefix of its own")
    game = small_rules()
    start = time.perf_counter()
    cases = cases_of(game, Lessons.join([Lessons.load(p) for p in lessons]), workers=CORES)
    replayed = time.perf_counter() - start
    evaluator, score = fit(cases, SEED)
    print(
        f"{cases.chosen.size} decisions, {cases.moves.shape[0]} moves: "
        f"replayed in {replayed:.0f} s, fit in {time.perf_counter() - start - replayed:.0f} s."
    )
    print(f"Value on {score.games_out} games kept out, of {cases.games}:")
    for name, value in score.value.items():
        print(
            f"  {name:9} correlation {value.correlation:.3f}, off by {value.error:.3f}, "
            f"loss on who won {value.winner_loss:.3f}"
        )
    print(f"Prior on {score.games_out} games kept out, of {cases.games}:")
    for name, prior in score.prior.items():
        print(f"  {name:9} agrees {prior.agreement:.3f}, loss {prior.loss:.3f}")
    print(f"Value: {_named(VALUE_FEATURES, evaluator.value)}, at 0 {evaluator.intercept:+.3f}")
    print(f"Prior: {_named(MOVE_FEATURES, evaluator.prior)}")
    evaluator.save(path)


def _named(names: tuple[str, ...], weights: tuple[float, ...]) -> str:
    """Return weights by name, as text."""
    return ", ".join(f"{n} {w:+.3f}" for n, w in zip(names, weights, strict=True))
