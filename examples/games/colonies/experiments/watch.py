"""Watch a game of Colonies round by round: the board or a player's view, and each side's people.

Players play one of the games players are scored on (`EVAL_SEED`), and a window shows the board
on a world map (Mollweide: every cell, by its area) after each round, each cell in the color of
the player with the most people there, fuller the nearer its capacity, with each player's people
round by round below. The slider, the arrow keys, home and end step through the rounds, and
space plays them. The buttons beside the chart choose what the map shows: the whole board, or
what one player sees (`Colonies.view`), the cells it doesn't see darkened. With a prefix, the
rounds of the whole board go to ``{prefix}_game.gif`` instead of a window.

Run with ``uv run python run.py colonies.watch [start] [prefix] [players...]``: by default the
first start, a window (an empty prefix, ``""``), and the settler against the raider. Players
are ``uniform``, ``settler``, ``raider``, or specs of saved MuZero weights
(`examples.games.colonies.learned`, such as ``runs/muzero/colonies_scratch_all_gen_028.pt``),
one per seat in seat order.
"""

import numpy as np
from matplotlib.widgets import RadioButtons

from examples.games.browse import figure, finish
from examples.games.colonies import (
    EVAL_SEED,
    PLAYERS,
    Colonies,
    ColoniesView,
    Raider,
    Resolved,
    Settler,
    colors,
    view_colors,
)
from examples.games.colonies.learned import player
from gridzero.agents import UniformPlayer
from gridzero.game import Player, play, replay
from gridzero.render import Browser, world_map


def main(start: str = "0", prefix: str = "", *players: str) -> None:
    """Play one game, then browse its rounds in a window or save them.

    Parameters
    ----------
    start : str, default "0"
        Which start of the scored games: ``game.start(EVAL_SEED + start)``.
    prefix : str, default ""
        Where the rounds go, ``{prefix}_game.gif``, or empty for a window.
    *players : str
        One player per seat, by name or weights spec; the settler and the raider by default.
    """
    game = Colonies()
    specs = players or ("settler", "raider")
    if len(specs) != game.players:
        raise ValueError(f"the game has {game.players} players, got {len(specs)}")
    seated = [chosen(spec) for spec in specs]
    names = [name for name, _ in seated]
    record = play(game, [p for _, p in seated], game.start(EVAL_SEED + int(start)), names=names)
    rounds = [
        p
        for p in replay(game, record)
        if p.ply == 0 or any(isinstance(e, Resolved) for e in p.events)
    ]
    people = np.array([p.state.people[: game.players].sum(axis=1) for p in rounds])
    winner = "a draw" if record.winner is None else f"{record.winner} wins"
    final = ", ".join(f"{v:.1f}" for v in people[-1])
    print(f"{' against '.join(names)}: {len(rounds) - 1} rounds, {winner} (people {final}).")

    shown = figure(prefix, (10, 8))
    board = shown.add_axes((0.02, 0.36, 0.96, 0.58))
    raster = world_map(game.space, 1024)
    image = raster.show(board, colors(rounds[0].state), borders=True)
    chart = shown.add_axes((0.34, 0.11, 0.62, 0.2))
    for seat, name in enumerate(names):
        chart.plot(people[:, seat], color=PLAYERS[seat], linewidth=2, label=name)
    now = chart.axvline(0, color="0.4", linewidth=1)
    chart.set_xlabel("round")
    chart.set_ylabel("people")
    chart.set_xlim(0, len(rounds) - 1)
    chart.legend(loc="upper left", frameon=False)
    for side in ("top", "right"):
        chart.spines[side].set_visible(False)
    views = ["board", *names]
    buttons = RadioButtons(
        shown.add_axes((0.01, 0.11, 0.26, 0.2), frameon=False), views, label_props={"fontsize": [8]}
    )

    def draw(frame: int) -> None:
        position = rounds[frame]
        seat = views.index(buttons.value_selected) - 1
        if seat < 0:
            cells = colors(position.state)
        else:
            cells = view_colors(position.state, game.view(position, seat))
        image.set_data(raster.image(cells, borders=True))
        now.set_xdata([frame, frame])
        totals = ", ".join(f"{n} {v:.0f}" for n, v in zip(names, people[frame], strict=True))
        end = f": {winner}" if position.over else ""
        seen = "" if seat < 0 else f", as {names[seat]} sees it"
        board.set_title(f"Round {position.round} of {game.rounds}{end} ({totals}){seen}")

    browser = Browser(shown, len(rounds), draw, label="round")
    buttons.on_clicked(lambda _: browser.redraw())
    finish(browser, prefix)


def chosen(spec: str) -> tuple[str, Player[Colonies, ColoniesView]]:
    """Return a player by name, or the MuZero player of saved weights, and its name."""
    scripted: dict[str, Player[Colonies, ColoniesView]] = {
        "uniform": UniformPlayer(),
        "settler": Settler(),
        "raider": Raider(),
    }
    return (spec, scripted[spec]) if spec in scripted else player(spec)
