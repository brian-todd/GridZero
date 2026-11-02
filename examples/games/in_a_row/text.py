"""Boards of k in a row as text, for terminals and logs."""

import numpy as np

from gridzero.render import lattice_text

from .game import InARow
from .state import InARowPosition

STONES = ("X", "O")
"""The glyph of each player's stones; a free cell is ``.``."""


def board_text(game: InARow, position: InARowPosition) -> str:
    """Draw the board, the top row first, with the column numbers under it.

    Parameters
    ----------
    game : InARow
        The rules.
    position : InARowPosition
        A position of them.

    Returns
    -------
    str
        One line per row, then the column numbers (`render.lattice_text`).
    """
    owner = position.state.owner
    glyphs = np.where(owner == 0, STONES[0], np.where(owner == 1, STONES[1], "."))
    return lattice_text(game.space, glyphs)
