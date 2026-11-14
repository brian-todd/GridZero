"""Play Connect Four against any player, on a page served on this machine.

Loads constants saved by ``code_fit``, and, if given, weights saved by ``zero_scratch`` (32
channels, 4 blocks) for rungs above the code player's. Serves a page to choose an opponent, a seat
and whether search players draw noise, and to play; it says why a search played each move, and
asks for a note after each game.
Every finished game goes with its note into ``{prefix}_games.json``, which later runs add to; a
record names the files its player came from, and the page counts only games against these files.
Stop it with Ctrl-C; a game left unfinished isn't kept.

Run with ``uv run python run.py in_a_row.play fit.json [prefix] [port] [weights.pt]``: by default
prefix ``play`` and port 8000, where the links in ``code_report``'s cards point. Open the address
it prints.
"""

import hashlib
from pathlib import Path

from examples.games.in_a_row import FittedEvaluator, InARow
from examples.games.in_a_row.play import Table, rungs, serve
from examples.games.in_a_row.report import PLAY_PORT
from gridzero.agents.net import NetEvaluator

from .zero_check import saved_net

SEED = 7_000_000  # the players' draws, per game


def fingerprint(*paths: str) -> str:
    """Return what files hold, by name and a hash of their contents, for a game's record.

    Parameters
    ----------
    *paths : str
        The files.

    Returns
    -------
    str
        Each file's name and the first 12 digits of its SHA-256, joined by commas.
    """
    return ", ".join(
        f"{Path(path).name} {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]}"
        for path in paths
    )


def main(
    fit: str, prefix: str = "play", port: str = str(PLAY_PORT), weights: str | None = None
) -> None:
    """Serve the page until stopped.

    Parameters
    ----------
    fit : str
        The constants saved by ``code_fit``.
    prefix : str, default "play"
        Where the games are kept: ``{prefix}_games.json``.
    port : str, default "8000"
        The port on this machine.
    weights : str, optional
        The teacher's saved weights, which add its rungs.
    """
    game = InARow()
    teacher = None if weights is None else NetEvaluator(saved_net(game, weights))
    table = Table(
        game,
        rungs(FittedEvaluator.load(fit), teacher),
        f"{prefix}_games.json",
        SEED,
        fingerprint(fit, *([] if weights is None else [weights])),
    )
    server = serve(table, int(port))
    print(f"Open http://localhost:{server.server_port}/ and press Ctrl-C here to stop.")
    print(f"Games are kept in {table.path}.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()
