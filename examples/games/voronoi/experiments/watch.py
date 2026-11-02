"""Watch a game of Voronoi stone by stone: the board, and the land each player holds.

Players play one of the boards players are scored on (`EVAL_SEED`), and a window shows the
board on a world map (Mollweide: every cell, by its area) after each stone, each cell in its
holder's color and each stone darker, with each player's land area stone by stone below. The
slider, the arrow keys, home and end step through the stones, and space plays them. With a
prefix, the stones go to ``{prefix}_game.gif`` instead of a window.

Run with ``uv run python run.py voronoi.watch [start] [prefix] [players...]``: by default the
first start, a window (an empty prefix, ``""``), and greedy against uniform. Players are
``uniform`` or ``greedy``, one per seat in seat order; their number sets the game's.
"""

import numpy as np
from numpy.typing import NDArray

from examples.games.browse import figure, finish
from examples.games.voronoi import EVAL_SEED, PLAYERS, Greedy, Voronoi, VoronoiPosition, colors
from gridzero.agents import UniformPlayer
from gridzero.game import Player, play, replay
from gridzero.render import Browser, world_map


def main(start: str = "0", prefix: str = "", *players: str) -> None:
    """Play one game, then browse its stones in a window or save them.

    Parameters
    ----------
    start : str, default "0"
        Which start of the scored games: ``game.start(EVAL_SEED + start)``.
    prefix : str, default ""
        Where the stones go, ``{prefix}_game.gif``, or empty for a window.
    *players : str
        One player per seat by name; greedy and uniform by default.
    """
    names = list(players or ("greedy", "uniform"))
    game = Voronoi(players=len(names))
    seated = [chosen(name) for name in names]
    record = play(game, seated, game.start(EVAL_SEED + int(start)), names=names)
    positions = list(replay(game, record))
    areas = np.array([game.areas(p.state) for p in positions])
    winner = "a draw" if record.winner is None else f"{record.winner} wins"
    final = ", ".join(f"{a:.4f}" for a in areas[-1])
    print(f"{' against '.join(names)}: {len(record.actions)} stones, {winner} (land {final}).")

    shown = figure(prefix, (10, 8))
    board = shown.add_axes((0.02, 0.36, 0.96, 0.58))
    raster = world_map(game.space, 1024)
    image = raster.show(board, colors(positions[0].state), borders=True)
    chart = shown.add_axes((0.1, 0.11, 0.85, 0.2))
    for seat, name in enumerate(names):
        chart.plot(areas[:, seat], color=PLAYERS[seat], linewidth=2, label=name, marker="o")
    now = chart.axvline(0, color="0.4", linewidth=1)
    chart.set_xlabel("stone")
    chart.set_ylabel("land held")
    chart.set_xlim(0, len(positions) - 1)
    chart.legend(loc="upper left", frameon=False)
    for side in ("top", "right"):
        chart.spines[side].set_visible(False)

    def draw(frame: int) -> None:
        position = positions[frame]
        image.set_data(raster.image(colors(position.state), borders=True))
        now.set_xdata([frame, frame])
        board.set_title(title(position, names, areas[frame], winner))

    finish(Browser(shown, len(positions), draw, label="stone"), prefix)


def title(
    position: VoronoiPosition, names: list[str], areas: NDArray[np.float64], winner: str
) -> str:
    """Return a position's title: the stones placed, the land each holds, and who won."""
    held = ", ".join(f"{n} {a:.2f}" for n, a in zip(names, areas, strict=True))
    end = f": {winner}" if position.over else ""
    return f"Stone {position.ply}{end} (land {held})"


def chosen(name: str) -> Player[Voronoi, VoronoiPosition]:
    """Return a player by name."""
    players: dict[str, Player[Voronoi, VoronoiPosition]] = {
        "uniform": UniformPlayer(),
        "greedy": Greedy(),
    }
    if name not in players:
        raise ValueError(f"a player is one of {', '.join(players)}, got {name!r}")
    return players[name]
