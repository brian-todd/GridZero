import numpy as np
import pytest

from agents_demo import check_learnable
from examples.games.tactics import Action, RandomPlayer, Verb
from game_demo import check_game
from tactics_demo import ARCHER, CAPTAIN, FOOT, KNIGHT, SMALL, TINY, mirrored, piece, play_random


def plane(game, obs, name):
    return obs[game.planes.index(name)]


def test_observations_have_one_float_plane_per_name_in_zero_to_one():
    position = TINY.start(1)
    obs = TINY.observe(position, 0)
    names = TINY.planes
    assert len(names) == len(set(names)) == 4 + 3 * 4 + 1 + 6 + 5 + 3 + 2
    assert obs.shape == (len(names), 81)
    assert obs.dtype == np.float32
    assert obs.min() >= 0
    assert obs.max() <= 1


def test_units_are_own_or_enemy_with_their_hit_points_and_facing(make_battle, at):
    pieces = [piece(1, 1, KNIGHT, 0, hp=7, facing=2), piece(4, 4, ARCHER, 1, hp=1, facing=5)]
    position = make_battle(pieces, walls=[(2, 2)])
    obs = SMALL.observe(position, 0)
    assert np.flatnonzero(plane(SMALL, obs, "own knight")).tolist() == [at(1, 1)]
    assert np.flatnonzero(plane(SMALL, obs, "enemy archer")).tolist() == [at(4, 4)]
    assert plane(SMALL, obs, "own archer").sum() == 0
    hp = plane(SMALL, obs, "hp")
    assert hp[at(1, 1)] == 1.0
    assert hp[at(4, 4)] == pytest.approx(1 / 3)
    assert plane(SMALL, obs, "facing 2")[at(1, 1)] == 1
    assert plane(SMALL, obs, "facing 5")[at(4, 4)] == 1
    assert np.flatnonzero(plane(SMALL, obs, "wall")).tolist() == [at(2, 2)]
    assert np.flatnonzero(plane(SMALL, obs, "active")).tolist() == [at(1, 1)]


def test_flags_and_watched_tiles_are_planes(make_battle, at):
    pieces = [
        piece(1, 1, ARCHER, 0, watch=at(3, 3), moved=1, attacked=True),
        piece(6, 6, ARCHER, 1, watch=at(5, 5), engaged=True, done=True),
    ]
    obs = SMALL.observe(make_battle(pieces), 0)
    assert np.flatnonzero(plane(SMALL, obs, "watched by own")).tolist() == [at(3, 3)]
    assert np.flatnonzero(plane(SMALL, obs, "watched by enemy")).tolist() == [at(5, 5)]
    assert np.flatnonzero(plane(SMALL, obs, "moved")).tolist() == [at(1, 1)]
    assert np.flatnonzero(plane(SMALL, obs, "attacked")).tolist() == [at(1, 1)]
    assert np.flatnonzero(plane(SMALL, obs, "engaged")).tolist() == [at(6, 6)]
    assert np.flatnonzero(plane(SMALL, obs, "done")).tolist() == [at(6, 6)]


def test_scalars_fill_their_planes():
    position = TINY.start(1)
    obs = TINY.observe(position, 0)
    assert (plane(TINY, obs, "deploying") == 1).all()
    assert (plane(TINY, obs, "round") == 0).all()
    assert (plane(TINY, obs, "to deploy captain") == 1).all()
    cell = int(np.flatnonzero(TINY.legal(position)[Verb.PLACE + FOOT])[0])
    after = TINY.step(position, (Verb.PLACE + FOOT, cell))
    assert (plane(TINY, TINY.observe(after, 0), "to deploy footsoldier") == 0.5).all()
    assert (plane(TINY, TINY.observe(after, 1), "to deploy footsoldier") == 1).all()
    assert (plane(TINY, TINY.observe(after, 1), "active") == 0).all()


def test_team_one_sees_the_board_turned_with_itself_as_own(make_battle, at):
    position = make_battle([piece(0, 0, CAPTAIN, 1, facing=1), piece(8, 8, FOOT, 0)])
    obs = SMALL.observe(position, 1)
    assert np.flatnonzero(plane(SMALL, obs, "own captain")).tolist() == [at(8, 8)]
    assert plane(SMALL, obs, "facing 4")[at(8, 8)] == 1
    assert np.flatnonzero(plane(SMALL, obs, "enemy footsoldier")).tolist() == [at(0, 0)]
    own_home = plane(SMALL, obs, "own home")
    np.testing.assert_array_equal(own_home, position.state.home == 0)  # always team 0's corner


def test_a_mirrored_position_looks_the_same_to_the_other_team():
    checked = 0
    rng = np.random.default_rng(3)
    for position, _ in play_random(TINY, 3, rng):
        if position.over or rng.random() > 0.2:
            continue
        flipped = mirrored(TINY, position)
        for team in (0, 1):
            np.testing.assert_array_equal(
                TINY.observe(flipped, 1 - team), TINY.observe(position, team)
            )
        checked += 1
    assert checked > 10


def test_team_must_be_zero_or_one(not_an_integer):
    position = TINY.start(1)
    with pytest.raises(ValueError, match="team"):
        TINY.observe(position, 2)
    with pytest.raises(TypeError, match="team"):
        TINY.observe(position, not_an_integer)


def test_legal_actions_in_a_teams_frame_are_those_of_the_mirrored_position():
    checked = 0
    rng = np.random.default_rng(4)
    for position, _ in play_random(TINY, 4, rng):
        # In deployment, whose turn it is follows the ply, which mirroring leaves alone.
        if position.over or position.deploying or rng.random() > 0.3:
            continue
        flipped = mirrored(TINY, position)
        legal = TINY.legal(position)
        np.testing.assert_array_equal(TINY.frame(1).into(legal), TINY.legal(flipped))
        assert TINY.frame(0).into(legal) is legal
        checked += 1
    assert checked > 10


def test_turning_actions_twice_gives_them_back(rng):
    n_verbs, n_cells = Verb.PLACE + len(TINY.units), 81
    values = rng.random((2, n_verbs, n_cells))
    turned = TINY.frame(1).into(values)
    assert not np.array_equal(turned, values)
    np.testing.assert_array_equal(TINY.frame(1).back(turned), values)
    for verb in range(n_verbs):
        for cell in (0, 7, 40, 80):
            action = Action(verb, cell)
            assert TINY.frame(1).action_back(TINY.frame(1).action_into(action)) == action
            assert TINY.frame(0).action_into(action) == action


def test_one_action_turns_as_the_values_over_actions_do():
    n_verbs, n_cells = Verb.PLACE + len(TINY.units), 81
    for verb in range(n_verbs):
        for cell in (0, 5, 40, 79):
            one = np.zeros((n_verbs, n_cells), dtype=bool)
            one[verb, cell] = True
            turned = TINY.frame(1).action_into(Action(verb, cell))
            assert np.argwhere(TINY.frame(1).into(one)).tolist() == [list(turned)]


def test_facing_turns_by_half_a_turn_in_team_ones_frame(at):
    assert TINY.frame(1).action_into(Action(Verb.FACE + 1, at(0, 0, TINY))) == (
        Verb.FACE + 4,
        at(8, 8, TINY),
    )
    assert TINY.frame(1).action_into(Action(Verb.MOVE, at(2, 3, TINY))) == (
        Verb.MOVE,
        at(6, 5, TINY),
    )


def test_frames_need_a_team_and_values_over_the_games_actions(not_an_integer):
    legal = TINY.legal(TINY.start(1))
    with pytest.raises(ValueError, match="team"):
        TINY.frame(2).into(legal)
    with pytest.raises(TypeError, match="team"):
        TINY.frame(not_an_integer).into(legal)
    with pytest.raises(ValueError, match="team"):
        TINY.frame(-1).action_into(Action(0, 0))
    with pytest.raises(ValueError, match="shape"):
        TINY.frame(1).into(legal[:, :-1])
    with pytest.raises(ValueError, match="shape"):
        TINY.frame(0).into(legal[:-1])


def test_tactics_is_learnable():
    check_learnable(TINY, check_game(TINY, RandomPlayer(), range(1)))
    assert TINY.space is TINY.start(0).state.space
    assert TINY.frame(1) is TINY.frame(1)
