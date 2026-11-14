"""Fog of war in Tactics: what a side is shown, what it may do on it, and what it remembers."""

import dataclasses

import numpy as np
import pytest

from examples.games.tactics import GreedyPlayer, RandomPlayer, SearchPlayer, Tactics, Verb
from game_demo import check_game
from gridzero.agents import AlphaZeroPlayer, self_play
from gridzero.core import hops, sight
from gridzero.game import play, replay
from tactics_demo import FOOT, KNIGHT, SMALL, TINY, WORKS, piece

FOG = dataclasses.replace(SMALL, fog=True)
"""The first rules under fog: footsoldiers and captains see 3 tiles, archers 4, knights 5."""

NEAR = dataclasses.replace(
    SMALL, units=tuple(dataclasses.replace(u, reach=1, sight=1.0) for u in SMALL.units), fog=True
)
"""Fog where every unit sees and strikes only the tiles beside it."""

GAME = dataclasses.replace(WORKS, fog=True)
"""The default kinds under fog on a 9x9 board, for whole games."""


def known(q, r, kind, team, at, game=FOG, **extra):
    """Return a unit that the other side last saw where it stands, at full health, in round 1."""
    seen = {"spotted": at(q, r), "spotted_hp": game.units[kind].hp, "spotted_round": 1}
    return piece(q, r, kind, team, **(seen | extra))


def vision(game, position, team):
    """Return the tiles a team's units see, by `core.sight`: shape (n_cells,), bool."""
    state = position.state
    u = state.units
    seen = np.zeros(state.n_cells, dtype=bool)
    for cell in np.flatnonzero(state.occupied & (u.team == team)):
        seen |= sight(state.space, cell, state.wall, radius=game.units[u.kind[cell]].sight)
    return seen


def where(position, uid):
    """Return the cell of the unit ``uid`` in a position."""
    return int(np.flatnonzero(position.state.units.uid == uid)[0])


# What a side is shown.


def test_without_fog_a_view_is_the_position_and_nothing_is_remembered():
    record = play(TINY, [RandomPlayer()] * 2, TINY.start(0))
    for position in replay(TINY, record):
        assert TINY.view(position, 0) is position
        assert (position.state.units.spotted == -1).all()


def test_deployment_and_the_end_are_in_the_open_under_fog():
    record = play(GAME, [GreedyPlayer()] * 2, GAME.start(0))
    positions = list(replay(GAME, record))
    deploying = [p for p in positions if p.deploying]
    assert deploying
    for position in deploying:
        for team in (0, 1):
            shown = GAME.view(position, team)
            assert shown.state is position.state  # all of it
            assert (shown.belief, shown.deploying) == (team, True)
            assert shown.seed != position.seed
    assert GAME.view(positions[-1], 0) is positions[-1]


def test_a_belief_of_deployment_plays_on_into_a_battle_without_fog():
    position = GAME.start(0)
    shown = GAME.view(position, 0)
    rng = np.random.default_rng(0)
    while shown.deploying:
        shown = GAME.step(
            shown, GreedyPlayer()(GAME, GAME.player(shown), shown, GAME.legal(shown), rng)
        )
    assert shown.belief == 0
    assert (shown.state.units.spotted == -1).all()  # nothing to remember where all is seen
    assert GAME.view(shown, 1) is shown


def test_the_battle_starts_with_every_unit_last_seen_where_it_was_deployed():
    record = play(GAME, [GreedyPlayer()] * 2, GAME.start(0))
    first = next(p for p in replay(GAME, record) if not p.deploying)
    u = first.state.units
    cells = np.flatnonzero(first.state.occupied)
    np.testing.assert_array_equal(u.spotted[cells], cells)
    np.testing.assert_array_equal(u.spotted_hp[cells], u.hp[cells])
    assert (u.spotted_round[cells] == 1).all()


def test_a_unit_that_moves_out_of_sight_is_shown_where_it_was_last_seen(make_battle, at):
    # The enemy footsoldier goes first and walks away from a tile nobody sees: 6 tiles off.
    pieces = [known(6, 4, FOOT, 1, at), known(0, 4, FOOT, 0, at)]
    position = FOG.step(make_battle(pieces, FOG), (Verb.MOVE, at(8, 4)))
    assert where(position, 0) == at(8, 4)
    shown = FOG.view(position, 0)
    assert shown.belief == 0
    assert where(shown, 0) == at(6, 4)  # team 0 still believes it there
    assert where(FOG.view(position, 1), 0) == at(8, 4)  # its own side knows better
    assert where(FOG.view(position, 1), 1) == at(0, 4)


def test_a_unit_in_sight_is_shown_as_it_is(make_battle, at):
    # Three tiles apart: each sees the other. The footsoldier hits the knight for 2 (a roll of 20).
    pieces = [known(3, 4, FOOT, 0, at, facing=0), known(4, 4, KNIGHT, 1, at, facing=3)]
    position = make_battle(pieces, FOG)
    seed = next(
        s
        for s in range(200)
        if (after := FOG.step(dataclasses.replace(position, seed=s), (Verb.ATTACK, at(4, 4))))
        .events[0]
        .hit  # ty: ignore[unresolved-attribute]  # an Attacked
    )
    after = FOG.step(dataclasses.replace(position, seed=seed), (Verb.ATTACK, at(4, 4)))
    hp = int(after.state.units.hp[at(4, 4)])
    assert hp < FOG.units[KNIGHT].hp
    for team in (0, 1):
        shown = FOG.view(after, team)
        assert where(shown, 1) == at(4, 4)
        assert shown.state.units.hp[at(4, 4)] == hp
        assert shown.state.units.engaged[at(4, 4)]


def test_a_ghost_keeps_the_hit_points_and_facing_it_last_showed(make_battle, at):
    pieces = [
        known(6, 4, FOOT, 1, at, hp=1, facing=2, spotted_hp=4, spotted_facing=5, engaged=True),
        known(0, 4, FOOT, 0, at),
    ]
    position = FOG.step(make_battle(pieces, FOG), (Verb.MOVE, at(8, 4)))
    ghost = FOG.view(position, 0).state.units
    assert (ghost.hp[at(6, 4)], ghost.facing[at(6, 4)]) == (4, 5)
    assert not ghost.engaged[at(6, 4)]
    assert ghost.moved[at(6, 4)] == 0  # it has done nothing the other side knows of
    own = FOG.view(position, 1).state.units
    assert (own.hp[at(8, 4)], own.moved[at(8, 4)]) == (1, 2)


def test_a_view_doesnt_depend_on_where_the_units_it_hides_are(make_battle, at):
    # The same battle with the hidden enemy on two different tiles, last seen on a third.
    views, legal = [], []
    for q, r in ((8, 4), (8, 0)):
        pieces = [known(0, 4, FOOT, 0, at), known(q, r, FOOT, 1, at, spotted=at(7, 7), hp=2)]
        position = make_battle(pieces, FOG)
        views.append(FOG.view(position, 0))
        legal.append(FOG.legal(position))
    assert views[0].key() == views[1].key()
    np.testing.assert_array_equal(legal[0], legal[1])
    assert where(views[0], 1) == at(7, 7)


def test_a_view_has_a_seed_of_its_own_and_no_events(make_battle, at):
    pieces = [known(6, 4, FOOT, 1, at), known(0, 4, FOOT, 0, at)]
    position = FOG.step(make_battle(pieces, FOG, seed=11), (Verb.MOVE, at(8, 4)))
    assert position.events
    shown = FOG.view(position, 0)
    assert shown.seed != position.seed
    assert shown.events == ()
    assert FOG.view(dataclasses.replace(position, seed=12), 0).seed != shown.seed


# What a side remembers.


def test_a_ghost_that_isnt_where_it_was_is_believed_on_the_nearest_tile_out_of_sight(
    make_battle, at
):
    # The enemy was last seen on (6, 4), a tile out of sight, and walks off. The footsoldier then
    # steps up, sees that tile empty, and can't see where the enemy went.
    pieces = [known(6, 4, FOOT, 1, at), known(2, 4, FOOT, 0, at)]
    position = make_battle(pieces, FOG)
    for action in ((Verb.MOVE, at(8, 4)), (Verb.END, at(8, 4))):
        position = FOG.step(position, action)
    assert where(FOG.view(position, 0), 0) == at(6, 4)
    position = FOG.step(position, (Verb.MOVE, at(4, 4)))
    seen = vision(FOG, position, 0)
    assert seen[at(6, 4)]
    assert not seen[at(8, 4)]
    believed = where(FOG.view(position, 0), 0)
    hidden = np.flatnonzero(~seen)
    away = hops(FOG.space, hidden, at(6, 4))
    assert believed == hidden[away.argmin()]  # the nearest, the lowest cell among equals
    assert not seen[believed]


def test_a_unit_glimpsed_on_its_way_is_last_seen_there_and_sees_from_there_too(make_battle, at):
    # The walker passes (7, 4), three tiles from the watcher, between two tiles out of its sight.
    # Its side believed the watcher far away, on (0, 0).
    pieces = [known(8, 3, FOOT, 1, at), known(4, 4, FOOT, 0, at, spotted=at(0, 0))]
    position = make_battle(pieces, FOG)
    before = vision(FOG, position, 0)
    assert not before[at(8, 3)]
    assert before[at(7, 4)]
    assert not before[at(7, 5)]
    position = FOG.step(position, (Verb.MOVE, at(7, 5)))
    assert position.events[0].route == (at(8, 3), at(7, 4), at(7, 5))  # ty: ignore[unresolved-attribute]  # a Moved
    hidden = np.flatnonzero(~vision(FOG, position, 0))
    nearest = hidden[hops(FOG.space, hidden, at(7, 4)).argmin()]
    assert where(FOG.view(position, 0), 0) == nearest  # beside where it was glimpsed
    assert where(FOG.view(position, 1), 1) == at(4, 4)  # the walker saw the watcher from (7, 4)
    assert not vision(FOG, position, 1)[at(4, 4)]


def test_two_ghosts_never_share_a_tile_and_none_stands_in_sight(make_battle, at):
    # Two enemies last seen on the same side of the board walk off; the knight rides up and
    # sees both tiles empty.
    pieces = [
        known(6, 4, FOOT, 1, at),
        known(6, 5, FOOT, 1, at),
        known(0, 4, KNIGHT, 0, at),
    ]
    position = make_battle(pieces, FOG)
    for action in (
        (Verb.MOVE, at(8, 4)),
        (Verb.END, at(8, 4)),
        (Verb.MOVE, at(8, 5)),
        (Verb.END, at(8, 5)),
        (Verb.MOVE, at(2, 4)),
    ):
        position = FOG.step(position, action)
    seen = vision(FOG, position, 0)
    assert seen[at(6, 4)]
    assert seen[at(6, 5)]
    shown = FOG.view(position, 0)
    ghosts = [where(shown, 0), where(shown, 1)]
    assert ghosts[0] != ghosts[1]
    assert not seen[ghosts].any()


# What a side may do.


def test_legal_actions_are_those_of_the_movers_belief(make_battle, at):
    # A ghost on (2, 4) blocks the way in the mover's belief, though the unit is elsewhere.
    pieces = [known(0, 4, FOOT, 0, at, game=NEAR), known(8, 4, FOOT, 1, at, spotted=at(2, 4))]
    position = make_battle(pieces, NEAR)
    legal = NEAR.legal(position)
    assert not legal[Verb.MOVE, at(2, 4)]
    assert legal[Verb.MOVE, at(1, 4)]
    np.testing.assert_array_equal(legal, NEAR.legal(NEAR.view(position, 0)))
    with pytest.raises(ValueError, match="illegal"):
        NEAR.step(position, (Verb.MOVE, at(2, 4)))


def test_a_move_stops_before_a_unit_it_didnt_know_of_and_then_sees_it(make_battle, at):
    # The enemy stands two tiles off, where the mover is going, believed far away.
    pieces = [known(2, 4, FOOT, 0, at, game=NEAR), known(4, 4, FOOT, 1, at, spotted=at(8, 8))]
    position = make_battle(pieces, NEAR)
    assert NEAR.legal(position)[Verb.MOVE, at(4, 4)]
    position = NEAR.step(position, (Verb.MOVE, at(4, 4)))
    assert where(position, 0) == at(3, 4)
    assert position.events[0].route == (at(2, 4), at(3, 4))  # ty: ignore[unresolved-attribute]  # a Moved
    units = position.state.units
    assert units.moved[at(3, 4)] == 1  # it keeps the tile it didn't walk
    assert where(NEAR.view(position, 0), 1) == at(4, 4)
    legal = NEAR.legal(position)
    assert not legal[Verb.MOVE, at(4, 4)]
    assert legal[Verb.ATTACK, at(4, 4)]


def test_a_belief_is_played_on_as_a_game_that_hides_nothing(make_battle, at):
    pieces = [known(0, 4, FOOT, 0, at), known(8, 4, FOOT, 1, at, spotted=at(6, 4))]
    position = make_battle(pieces, FOG)
    shown = FOG.view(position, 0)
    assert FOG.view(shown, 0) is shown
    assert FOG.view(shown, 1) is shown
    after = FOG.step(shown, (Verb.MOVE, at(2, 4)))
    assert after.belief == 0
    assert where(after, 1) == at(6, 4)  # the ghost stays: nothing is spotted in a belief
    np.testing.assert_array_equal(
        after.state.units.spotted[[at(2, 4), at(6, 4)]], [at(0, 4), at(6, 4)]
    )


# Whole games.


@pytest.mark.parametrize("player", [RandomPlayer(), GreedyPlayer()])
def test_every_view_of_a_game_shows_what_its_side_sees_and_ghosts_only_out_of_sight(player):
    record = play(GAME, [player] * 2, GAME.start(3))
    hidden = 0
    for position in replay(GAME, record):
        if position.deploying or position.over:
            continue
        true = position.state.units
        for team in (0, 1):
            shown = GAME.view(position, team)
            u = shown.state.units
            np.testing.assert_array_equal(
                np.sort(u.uid[u.uid >= 0]), np.sort(true.uid[true.uid >= 0])
            )
            seen = vision(GAME, position, team)
            for cell in np.flatnonzero(position.state.occupied):
                uid = int(true.uid[cell])
                there = where(shown, uid)
                if true.team[cell] == team or seen[cell]:
                    assert there == cell
                    assert u.hp[there] == true.hp[cell]
                    assert u.facing[there] == true.facing[cell]
                else:
                    assert not seen[there]
                    hidden += 1
    assert hidden > 0


def test_fog_keeps_the_promises_of_a_game():
    check_game(GAME, RandomPlayer(), range(3))


def test_search_players_finish_a_game_under_fog():
    tiny = dataclasses.replace(TINY, fog=True)
    record = play(tiny, [SearchPlayer(simulations=6, rollout_turns=2)] * 2, tiny.start(0))
    assert record.final.over


def test_self_play_searches_what_each_side_believes():
    tiny = dataclasses.replace(TINY, fog=True)
    judged = []

    def uniform(game, positions, legal):
        judged.extend(positions)
        return [np.zeros(int(mask.sum())) for mask in legal], np.full(len(positions), 0.5)

    self_play(tiny, AlphaZeroPlayer(uniform, simulations=4, considered=4, noise=True), [0])
    battle = [p for p in judged if not p.deploying]
    assert battle
    assert all(p.belief >= 0 for p in battle)  # never the position with what is hidden


# What a network is shown.


def test_the_observation_marks_what_is_seen_and_how_old_a_ghost_is(make_battle, at):
    assert FOG.planes[-2:] == ("seen", "ghost age")
    assert len(FOG.planes) == len(SMALL.planes) + 2
    pieces = [known(0, 4, FOOT, 0, at), known(8, 4, FOOT, 1, at, spotted=at(6, 4))]
    position = make_battle(pieces, FOG, round=3)
    obs = FOG.observe(FOG.view(position, 0), 0)
    seen, age = obs[-2], obs[-1]
    np.testing.assert_array_equal(seen.astype(bool), vision(FOG, position, 0))
    assert age[at(6, 4)] == pytest.approx(3 / FOG.max_rounds)  # last seen in round 1, now round 3
    assert np.count_nonzero(age) == 1
    assert obs[FOG.planes.index("enemy footsoldier"), at(6, 4)] == 1


def test_during_deployment_everything_counts_as_seen():
    obs = GAME.observe(GAME.start(0), 0)
    assert obs[-2].all()
    assert not obs[-1].any()


# Its parameters.


def test_fog_is_the_default_a_bool_and_needs_every_kind_to_see_as_far_as_the_furthest_reach():
    assert Tactics().fog
    with pytest.raises(TypeError, match="fog"):
        Tactics(fog=1)  # ty: ignore[invalid-argument-type]  # not a bool: refused
    short = tuple(dataclasses.replace(u, sight=2.0) for u in Tactics().units)
    Tactics(units=short, fog=False)  # fine without fog
    with pytest.raises(ValueError, match="see at least as far"):
        Tactics(units=short, fog=True)


def test_the_default_kinds_see_three_tiles_but_archers_four_and_knights_five():
    assert [u.sight for u in Tactics().units] == [3, 4, 5, 3, 3]
    with pytest.raises(ValueError, match="sight"):
        dataclasses.replace(Tactics().units[0], sight=0.0)
