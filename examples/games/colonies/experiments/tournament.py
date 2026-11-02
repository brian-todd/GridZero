"""A tournament of Colonies players, under fog and without: who beats whom, and seat by seat.

All play on the boards players are scored on (`EVAL_SEED`), each start once per rotation of
the seats, on every core, once under fog and once without. The raider and the settler each
play the uniform player in every other seat (`game.ladder`), then the raider plays the
settler. Then the raider plays every seat, with dice of its own, to show which seat the game
favors. Fog matters if the scores move between the two: without it, a player may gather toward
any land, not only what it sees. It saves a picture of the end of the first game of the raider
against itself under fog as ``{prefix}_end.png``.

Run with ``uv run python run.py colonies.tournament [starts] [prefix] [players]``: by default
10 starts, the picture in ``colonies_end.png`` in the working directory, and 2 players. A game
of 30 rounds takes about 0.1 s.
"""

import dataclasses
import time

from matplotlib.figure import Figure

from examples.games import CORES
from examples.games.colonies import EVAL_SEED, Colonies, Raider, Settler, colors
from gridzero.agents import UniformPlayer
from gridzero.game import fairness, ladder, series, tally
from gridzero.render import globe


def main(starts: str = "10", prefix: str = "colonies", players: str = "2") -> None:
    """Play the raider and the settler against uniform and each other, and save a game's end.

    Parameters
    ----------
    starts : str, default "10"
        Starts each series, each played once per rotation of the seats.
    prefix : str, default "colonies"
        Where the picture goes: ``{prefix}_end.png``.
    players : str, default "2"
        How many players a game has.
    """
    game = Colonies(players=int(players))
    start = time.perf_counter()
    ends = {}  # the end of the first game of the raider against itself, by fog
    for fog in (True, False):
        rules = dataclasses.replace(game, fog=fog)
        print("Under fog:" if fog else "Without fog:")
        matches = (
            ("Raider", Raider(), "uniform", UniformPlayer()),
            ("Settler", Settler(), "uniform", UniformPlayer()),
            ("Raider", Raider(), "settler", Settler()),
        )
        for name, player, rival, other in matches:
            scores = ladder(rules, player, {rival: other}, int(starts), EVAL_SEED, workers=CORES)
            won, lost, drawn = scores[rival]
            print(
                f"  {name} against {rival} in every other seat: "
                f"{won} wins, {lost} losses, {drawn} draws."
            )
        mirror = series(
            rules,
            [("raider", Raider())] * rules.players,
            int(starts),
            EVAL_SEED,
            same_dice=False,
            workers=CORES,
        )
        counts = tally(mirror)
        fair = fairness(counts)
        edges = ", ".join(f"{e:+.2f}" for e in fair.seat_edges)
        print(
            f"  Raider in every seat: wins by seat {counts.seat_wins.tolist()}, "
            f"{counts.draws} draws; seat edges {edges}, "
            f"{fair.excess_edge():.2f} beyond a fair game's noise."
        )
        ends[fog] = mirror[0].final
    print(f"Done in {time.perf_counter() - start:.1f} s on {CORES} processes.")

    final = ends[True]
    people = ", ".join(f"{p:.0f}" for p in game.totals(final.state))
    figure = Figure(figsize=(6, 6))
    ax = figure.add_subplot()
    ax.imshow(globe(game.space, size=512).image(colors(final.state), borders=True))
    ax.set_axis_off()
    winner = "a draw" if final.winner is None else f"player {final.winner} won"
    ax.set_title(f"Raider against itself, under fog: {winner} (people {people})")
    picture = f"{prefix}_end.png"
    figure.savefig(picture, dpi=100, bbox_inches="tight")
    print(f"The end of the first game: {picture}.")
