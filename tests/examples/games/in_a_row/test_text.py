"""Tests of boards of k in a row as text."""

from examples.games.in_a_row import PLACE, InARow, board_text


def test_the_board_is_drawn_bottom_row_last_with_stones_by_player():
    game = InARow((4, 3), 3)
    position = game.start(0)
    for cell in (3, 0, 4):  # X at (1, 0), O at (0, 0), X at (1, 1)
        position = game.step(position, (PLACE, cell))
    assert board_text(game, position) == ("  2 . . . .\n  1 . X . .\n  0 O X . .\ni   0 1 2 3")
