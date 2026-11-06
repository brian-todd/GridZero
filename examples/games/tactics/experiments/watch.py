"""Watch a game of Tactics action by action: the board, what each step did, and the material.

Players play one of the boards players are scored on (`EVAL_SEED`), and a window shows the
board (`pictures.draw`) after each decision, with what that step did below it (`event_text`)
and team 0's share of the material on the board (`material`) decision by decision. The slider,
the arrow keys, home and end step through the decisions, and space plays them. With a prefix,
the decisions go to ``{prefix}_game.gif`` instead, 10 a second.

Run with ``uv run python run.py tactics.watch [start] [prefix] [players...]``: by default the
first start, a window (an empty prefix, ``""``), and greedy against random. Players are
``random``, ``greedy``, ``sitter`` or ``search`` (minutes a game), one per team, team 0 (red) first.
"""

from examples.games.browse import figure, finish
from examples.games.tactics import (
    EVAL_SEED,
    GreedyPlayer,
    RandomPlayer,
    SearchPlayer,
    SitterPlayer,
    Tactics,
    TacticsPosition,
    event_text,
)
from examples.games.tactics.pictures import TEAM_COLORS, draw
from examples.games.tactics.search import material
from gridzero.game import Player, play, replay
from gridzero.render import Browser

LINES = 4
"""The most events shown for one step."""


def main(start: str = "0", prefix: str = "", *players: str) -> None:
    """Play one game, then browse its decisions in a window or save them.

    Parameters
    ----------
    start : str, default "0"
        Which start of the scored games: ``game.start(EVAL_SEED + start)``.
    prefix : str, default ""
        Where the decisions go, ``{prefix}_game.gif``, or empty for a window.
    *players : str
        One player per team by name; greedy and random by default.
    """
    game = Tactics()
    names = list(players or ("greedy", "random"))
    if len(names) != game.n_players:
        raise ValueError(f"the game has {game.n_players} players, got {len(names)}")
    seated = [chosen(name) for name in names]
    record = play(game, seated, game.start(EVAL_SEED + int(start)), names=names)
    positions = list(replay(game, record))
    shares = [material(game, p) for p in positions]
    winner = f"{record.winner} wins"
    print(
        f"{' against '.join(names)}: {len(record.actions)} decisions, {record.final.round} "
        f"rounds, {winner} (team 0's material {shares[-1]:.3f})."
    )

    shown = figure(prefix, (9, 9))
    board = shown.add_axes((0.02, 0.33, 0.96, 0.62))
    caption = shown.text(0.05, 0.3, "", va="top", fontsize=8, family="monospace")
    chart = shown.add_axes((0.1, 0.1, 0.85, 0.11))
    chart.plot(shares, color=TEAM_COLORS[0], linewidth=1.5)
    chart.axhline(0.5, color="0.6", linewidth=0.8, linestyle="--")
    now = chart.axvline(0, color="0.4", linewidth=1)
    chart.set_ylim(0, 1)
    chart.set_xlim(0, len(positions) - 1)
    chart.set_xlabel("decision")
    chart.set_ylabel(f"{names[0]}'s\nmaterial")
    for side in ("top", "right"):
        chart.spines[side].set_visible(False)

    def show(frame: int) -> None:
        position = positions[frame]
        board.clear()
        draw(game, position, board)
        board.set_title(title(position, names, winner))
        lines = [event_text(game, position.state, e) for e in position.events]
        if len(lines) > LINES:
            lines = [*lines[: LINES - 1], f"... and {len(lines) - LINES + 1} more"]
        caption.set_text("\n".join(lines))
        now.set_xdata([frame, frame])

    finish(Browser(shown, len(positions), show, label="decision"), prefix, fps=10)


def title(position: TacticsPosition, names: list[str], winner: str) -> str:
    """Return a position's title: the decision, the round, who holds the point, who won."""
    when = "deploying" if position.deploying else f"round {position.round}"
    if position.hold.owner >= 0:
        when += f", {names[position.hold.owner]} on the point for {position.hold.clock}"
    end = f": {winner}" if position.over else ""
    return f"{names[0]} (red) against {names[1]} (blue), decision {position.ply}, {when}{end}"


def chosen(name: str) -> Player[Tactics, TacticsPosition]:
    """Return a player by name."""
    players: dict[str, Player[Tactics, TacticsPosition]] = {
        "random": RandomPlayer(),
        "greedy": GreedyPlayer(),
        "sitter": SitterPlayer(),
        "search": SearchPlayer(),
    }
    if name not in players:
        raise ValueError(f"a player is one of {', '.join(players)}, got {name!r}")
    return players[name]
