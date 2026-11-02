"""Tactics as text: cells, events and summaries, for terminals and logs.

Cells are written ``(q, r)`` in axial coordinates.
"""

from typing import SupportsIndex

from .events import (
    Attacked,
    Deferred,
    Died,
    Ended,
    Event,
    GameOver,
    Moved,
    Placed,
    RoundStarted,
    Swapped,
    Turned,
)
from .game import Tactics
from .match import Summary
from .state import TacticsState


def where(state: TacticsState, cell: SupportsIndex) -> str:
    """Return a cell's axial coordinates as ``"(q, r)"``.

    Parameters
    ----------
    state : TacticsState
        The board.
    cell : int
        A cell id.

    Returns
    -------
    str
        Such as ``"(4, 7)"``.
    """
    q, r = state.space.unravel(int(cell))
    return f"({q}, {r})"


def summary_text(game: Tactics, summary: Summary) -> str:
    """Return what a set of games showed: who won, from which side, and what each kind did.

    Parameters
    ----------
    game : Tactics
        The rules the games were played by, for the kinds' names.
    summary : Summary
        What the games show, from `match.summarize`.

    Returns
    -------
    str
        A line of wins, draws, wins by team, and the mean rounds and decisions a game, a blank
        line, then a Markdown table of the damage dealt and taken, kills and losses by kind.
    """
    counts = summary.tally
    lines = [
        f"{counts.games} games: wins {counts.wins}, draws {counts.draws}, "
        f"by team {counts.seat_wins.tolist()}, "
        f"{summary.rounds:.1f} rounds and {counts.plies:.0f} decisions a game.",
        "",
        "| Kind | Damage dealt | Damage taken | Kills | Losses |",
        "|---|---|---|---|---|",
    ]
    for k, stats in enumerate(game.units):
        lines.append(
            f"| {stats.name} | {summary.damage_dealt[k]} | {summary.damage_taken[k]} | "
            f"{summary.kills[k]} | {summary.losses[k]} |"
        )
    return "\n".join(lines)


def event_text(game: Tactics, state: TacticsState, event: Event) -> str:
    """Return one event as a sentence.

    Parameters
    ----------
    game : Tactics
        The rules, for the kinds' names.
    state : TacticsState
        Any state on the game's board, for coordinates.
    event : Event
        The event.

    Returns
    -------
    str
        Such as ``"unit 3 (archer, team 0) attacks unit 14 (knight) at (5, 6): rolls 12,
        needs 10, hits for 2"``.
    """
    name = [stats.name for stats in game.units]
    match event:
        case Placed(uid, kind, team, cell):
            return f"unit {uid} ({name[kind]}, team {team}) deploys at {where(state, cell)}"
        case Moved(uid, route):
            return f"unit {uid} moves " + " -> ".join(where(state, c) for c in route)
        case Swapped(uid, other, cell, other_cell):
            return (
                f"unit {uid} at {where(state, cell)} swaps with unit {other} at "
                f"{where(state, other_cell)}"
            )
        case Turned(uid, facing):
            return f"unit {uid} turns to face direction {facing}"
        case Deferred(uid, tile):
            return f"unit {uid} defers its attack onto {where(state, tile)}"
        case Attacked():
            kind = "a deferred attack on" if event.deferred else "attacks"
            outcome = f"hits for {event.damage}" if event.hit else "misses"
            flank = ", flanking" if event.flank else ""
            return (
                f"unit {event.attacker} ({name[event.attacker_kind]}, team {event.attacker_team})"
                f" {kind} unit {event.target} ({name[event.target_kind]}) at "
                f"{where(state, event.cell)}{flank}: Atk {event.atk} against Def "
                f"{event.defense}, rolls {event.roll}, needs {event.needed}, {outcome}"
            )
        case Died(uid, kind, team, cell):
            return f"unit {uid} ({name[kind]}, team {team}) dies at {where(state, cell)}"
        case Ended(uid):
            return f"unit {uid} ends its turn"
        case RoundStarted(round_):
            return f"round {round_} begins"
        case GameOver(winner):
            return "the game is a draw" if winner is None else f"team {winner} wins"
        case _:
            raise TypeError(f"not an event: {event!r}")
