"""A tournament of Voronoi players: greedy against uniform, and greedy in every seat.

Both play on the boards players are scored on (`EVAL_SEED`), each start once per rotation of
the seats, on every core. Greedy plays the uniform player in every other seat (`game.ladder`),
and should win nearly every game. Then greedy plays every seat, with dice of their own, to show
which seat the game favors: the last to place often takes the most. It saves a picture of the
end of the first game of greedy against itself as ``{prefix}_end.png``.

Run with ``uv run python run.py voronoi.tournament [starts] [prefix] [players]``: by default
10 starts, the picture in ``voronoi_end.png`` in the working directory, and 2 players. A game
of greedy players takes about 0.1 s.
"""

import time

from matplotlib.figure import Figure

from examples.games import CORES
from examples.games.voronoi import EVAL_SEED, Greedy, Voronoi, colors
from gridzero.agents import UniformPlayer
from gridzero.game import fairness, ladder, series, tally
from gridzero.render import globe


def main(starts: str = "10", prefix: str = "voronoi", players: str = "2") -> None:
    """Play greedy against uniform and against itself, and save the end of a game.

    Parameters
    ----------
    starts : str, default "10"
        Starts each series, each played once per rotation of the seats.
    prefix : str, default "voronoi"
        Where the picture goes: ``{prefix}_end.png``.
    players : str, default "2"
        How many players a game has.
    """
    game = Voronoi(players=int(players))
    start = time.perf_counter()
    scores = ladder(
        game, Greedy(), {"uniform": UniformPlayer()}, int(starts), EVAL_SEED, workers=CORES
    )
    won, lost, drawn = scores["uniform"]
    print(f"Greedy against uniform in every other seat: {won} wins, {lost} losses, {drawn} draws.")

    mirror = series(
        game,
        [("greedy", Greedy())] * game.players,
        int(starts),
        EVAL_SEED,
        same_dice=False,
        workers=CORES,
    )
    counts = tally(mirror)
    fair = fairness(counts)
    edges = ", ".join(f"{e:+.2f}" for e in fair.seat_edges)
    print(
        f"Greedy in every seat: wins by seat {counts.seat_wins.tolist()}, {counts.draws} draws; "
        f"seat edges {edges}, {fair.excess_edge():.2f} beyond a fair game's noise."
    )
    games = 2 * len(mirror)
    print(f"{games} games in {time.perf_counter() - start:.1f} s on {CORES} processes.")

    final = mirror[0].final
    areas = ", ".join(f"{a:.2f}" for a in game.areas(final.state))
    figure = Figure(figsize=(6, 6))
    ax = figure.add_subplot()
    ax.imshow(globe(game.space, size=512).image(colors(final.state), borders=True))
    ax.set_axis_off()
    winner = "a draw" if final.winner is None else f"player {final.winner} won"
    ax.set_title(f"Greedy against itself: {winner} (areas {areas})")
    picture = f"{prefix}_end.png"
    figure.savefig(picture, dpi=100, bbox_inches="tight")
    print(f"The end of the first game: {picture}.")
