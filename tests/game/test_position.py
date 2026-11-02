import dataclasses

import numpy as np
import pytest

from game_demo import Claimed, Claims, ClaimsPosition, StripState
from gridzero.game import Action, check_action

GAME = Claims()


def test_an_action_is_a_verb_and_a_cell():
    action = Action(1, 3)
    assert (action.verb, action.cell) == (1, 3)
    assert action == (1, 3)


def test_an_action_is_checked_to_plain_ints():
    action = check_action(GAME.start(0), (np.int8(1), np.uint64(4)), 2)
    assert action == (1, 4)
    assert type(action) is Action
    assert type(action.verb) is int
    assert type(action.cell) is int


def test_an_action_check_refuses_a_game_over():
    position = GAME.start(0)
    for cell in range(GAME.length):
        position = GAME.step(position, (0, cell))
    assert position.over
    with pytest.raises(ValueError, match="over"):
        check_action(position, (0, 0), 2)


def test_an_action_check_refuses_bad_verbs_and_cells(not_an_integer):
    start = GAME.start(0)
    with pytest.raises(TypeError, match="verb"):
        check_action(start, (not_an_integer, 0), 2)
    with pytest.raises(TypeError):
        check_action(start, (0, not_an_integer), 2)
    with pytest.raises(ValueError, match="verb"):
        check_action(start, (2, 0), 2)
    with pytest.raises(ValueError, match="verb"):
        check_action(start, (-1, 0), 2)
    with pytest.raises(IndexError):
        check_action(start, (0, GAME.length), 2)
    with pytest.raises(IndexError):
        check_action(start, (0, -1), 2)


def test_positions_compare_by_identity():
    a = GAME.start(0)
    assert a == a
    assert a != GAME.start(0)


def test_a_key_ignores_how_a_position_was_reached():
    a = GAME.step(GAME.start(0), (0, 2))
    b = dataclasses.replace(a, seed=99, drew=True, events=(Claimed(1, 0, False),))
    assert b.key() == a.key()


def test_a_key_tells_positions_apart_by_their_values():
    a = GAME.step(GAME.start(0), (0, 2))
    owner = a.state.owner.copy()
    owner[3] = 1
    others = [
        dataclasses.replace(a, state=a.state.replace({StripState.owner: owner})),
        dataclasses.replace(a, ply=a.ply + 1),
        dataclasses.replace(a, over=True),
        dataclasses.replace(a, winner=0),
        dataclasses.replace(a, flips=1),  # a field of the subclass
    ]
    keys = {p.key() for p in [a, *others]}
    assert len(keys) == len(others) + 1


def test_a_key_reads_every_field_of_the_state():
    a = GAME.start(0)
    flipped = np.zeros(a.state.n_cells, dtype=np.bool_)
    flipped[0] = True
    b = dataclasses.replace(a, state=a.state.replace({StripState.flipped: flipped}))
    assert b.key() != a.key()


def test_a_winner_of_0_and_a_draw_have_different_keys():
    over = dataclasses.replace(GAME.start(0), over=True)
    assert dataclasses.replace(over, winner=0).key() != over.key()


def test_a_position_of_a_subclass_defaults_its_own_fields():
    position = ClaimsPosition(GAME.start(0).state, 0)
    assert (position.ply, position.over, position.winner, position.drew) == (0, False, None, False)
    assert position.events == ()
    assert position.flips == 0
