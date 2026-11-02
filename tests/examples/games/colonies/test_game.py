"""Tests of the rules of Colonies: the start, orders, resolution, fog, players out, errors."""

import dataclasses

import numpy as np
import pytest

from agents_demo import check_revealable
from examples.games.colonies import (
    GATHER,
    HOLD,
    MAX_PLAYERS,
    MIN_PEOPLE,
    Colonies,
    ColoniesPosition,
    ColoniesState,
    Ordered,
    Out,
    Resolved,
    board,
)
from game_demo import check_game
from gridzero.agents import UniformPlayer
from gridzero.core import Context, MeshSpace, regions, steps
from gridzero.game import Action, play, replay

SMALL = Colonies(frequency=2, rounds=4, fog=False)
"""42 cells, few rounds and no fog: quick to play to the end, and any land is a target."""

FOGGY = Colonies(frequency=2, rounds=4)
"""The same under fog."""

STILL = dataclasses.replace(SMALL, fight=0.0, growth=0.0)
"""People only move: nobody fights or grows."""


def at(game, people, land=None, capacity=None):
    """Return round 0 of ``game`` with ``people`` per player, all land by default."""
    n_cells = game.space.n_cells
    rows = np.zeros((MAX_PLAYERS, n_cells))
    rows[: len(people)] = people
    land = np.ones(n_cells, dtype=np.bool_) if land is None else np.asarray(land)
    capacity = np.full(n_cells, 10.0) * land if capacity is None else np.asarray(capacity)
    state = ColoniesState(
        game.space,
        {ColoniesState.land: land, ColoniesState.capacity: capacity, ColoniesState.people: rows},
    )
    return ColoniesPosition(state, 0, out=(False,) * game.players)


def colony(n_cells, cell, people=10.0):
    """Return a row of people: ``people`` in ``cell``, none elsewhere."""
    row = np.zeros(n_cells)
    row[cell] = people
    return row


def order(game, position, *orders):
    """Return the position after each order in turn."""
    for each in orders:
        position = game.step(position, each)
    return position


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize(("frequency", "sea"), [(1, 0.0), (2, 0.3)])
def test_the_game_keeps_the_promises_of_a_game(players, frequency, sea):
    game = Colonies(frequency=frequency, players=players, sea=sea, rounds=3)
    check_game(game, UniformPlayer(), seeds=range(3))


def test_the_board_is_an_icosphere_shared_by_every_position():
    assert isinstance(SMALL.space, MeshSpace)
    assert SMALL.space.n_cells == 42
    assert SMALL.space is board(2) is SMALL.start(0).state.space is SMALL.start(5).state.space


def test_the_start_comes_from_the_seeds_stream_0():
    position = SMALL.start(9)
    rng = np.random.default_rng(np.random.SeedSequence(9, spawn_key=(0,)))  # Context's stream
    sea = np.sort(rng.choice(42, round(0.3 * 42), replace=False))
    np.testing.assert_array_equal(np.flatnonzero(~position.state.land), sea)
    area = SMALL.space.measures / SMALL.space.measures.mean()
    richness = rng.uniform(0.5, 1.5, 42) * 10.0
    np.testing.assert_array_equal(position.state.capacity, richness * area * position.state.land)
    assert SMALL.start(9).key() == position.key()
    assert len({SMALL.start(seed).key() for seed in range(6)}) == 6


@pytest.mark.parametrize(
    ("frequency", "sea"), [(2, 0.3), (2, 0.8), (1, 0.65)]
)  # landmasses too small
def test_each_player_starts_with_its_settlers_in_a_cell_of_its_own(frequency, sea):
    game = Colonies(frequency=frequency, players=4, sea=sea)
    for seed in range(12):
        people = game.start(seed).state.people
        assert (people[4:] == 0).all()
        assert ((people[:4] > 0).sum(axis=1) == 1).all()
        np.testing.assert_array_equal(people[:4].sum(axis=1), 10.0)
        assert len(set(np.argmax(people[:4], axis=1).tolist())) == 4


def test_the_colonies_start_on_the_largest_landmass_while_it_has_room():
    game = Colonies(frequency=2, players=4, sea=0.8)
    split = 0
    for seed in range(12):
        state = game.start(seed).state
        landmass = regions(state.space, state.land)
        split += landmass.max() > 0
        on = landmass[np.argmax(state.people[:4], axis=1)]
        assert (on == 0).sum() == min(4, (landmass == 0).sum())  # filled before any other
    assert split > 6  # most of these starts have several landmasses, some too small


def test_each_next_colony_is_the_land_farthest_from_those_before_it_dealt_at_random():
    game = dataclasses.replace(SMALL, players=3)
    dealt = set()
    for seed in range(8):
        state = game.start(seed).state
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))
        rng.choice(42, round(0.3 * 42), replace=False)  # the sea
        rng.uniform(0.5, 1.5, 42)  # the capacities
        largest = np.flatnonzero(regions(state.space, state.land) == 0)
        colonies = [int(largest[rng.integers(largest.size)])]
        for _ in range(2):
            moves = steps(state.space, colonies, blocked=~state.land)
            free = state.land & np.isfinite(moves)
            free[colonies] = False
            far = moves[free].max()
            colonies.append(int(np.flatnonzero(free & (moves == far))[0]))  # the lowest id
        seats = rng.permutation(colonies)
        np.testing.assert_array_equal(np.argmax(state.people[:3], axis=1), seats)
        dealt.add(seats.tolist().index(colonies[0]))
    assert dealt == {0, 1, 2}  # the first colony drawn goes to any seat


def test_the_seats_order_in_turn_and_the_last_order_resolves_the_round():
    game = dataclasses.replace(SMALL, players=3)
    start = game.start(0)
    land = int(np.flatnonzero(start.state.land)[0])
    position = order(game, start, (HOLD, 0), (GATHER, land))
    assert game.player(position) == 2
    assert position.orders == (Action(HOLD, 0), Action(GATHER, land))
    assert position.round == 0
    assert position.events == (Ordered(1),)
    resolved = game.step(position, (HOLD, 0))
    assert resolved.round == 1
    assert resolved.orders == ()
    assert game.player(resolved) == 0
    assert resolved.events[0] == Ordered(2)
    assert isinstance(resolved.events[1], Resolved)
    assert resolved.events[1].people == tuple(game.totals(resolved.state).tolist())


def test_orders_before_the_last_change_no_field_and_draw_nothing():
    game = dataclasses.replace(SMALL, players=4)
    start = game.start(0)
    land = int(np.flatnonzero(start.state.land)[0])
    position = order(game, start, (HOLD, 0), (GATHER, land), (HOLD, 0))
    assert position.state is start.state
    assert not position.drew
    assert game.step(position, (HOLD, 0)).drew


def test_a_resolution_draws_only_from_the_steps_stream():
    start = SMALL.start(4)
    a, b = np.flatnonzero(start.state.land)[[0, -1]]
    position = order(SMALL, start, (GATHER, a))
    resolved = SMALL.step(position, (GATHER, b))
    dice = Context(seed=4, spawn_key=(1, 1))
    seed = int(dice.rng.integers(2**63))
    expected = SMALL.resolve(position.state, {0: Action(GATHER, a), 1: Action(GATHER, b)}, seed)
    np.testing.assert_array_equal(resolved.state.people, expected.people)


def test_without_fog_gather_on_any_land_and_hold_on_cell_0_are_the_legal_orders():
    position = SMALL.start(1)
    legal = SMALL.legal(position)
    assert legal.shape == (2, 42)
    np.testing.assert_array_equal(legal[GATHER], position.state.land)
    np.testing.assert_array_equal(np.flatnonzero(legal[HOLD]), [0])
    sea = int(np.flatnonzero(~position.state.land)[0])
    with pytest.raises(ValueError, match="illegal"):
        SMALL.step(position, (GATHER, sea))
    with pytest.raises(ValueError, match="illegal"):
        SMALL.step(position, (HOLD, 3))


def test_marching_conserves_each_players_people():
    position = STILL.start(2)
    before = STILL.totals(position.state)
    land = int(np.flatnonzero(position.state.land)[-1])
    after = order(STILL, position, (GATHER, land), (HOLD, 0)).state
    np.testing.assert_allclose(STILL.totals(after), before, rtol=1e-12)
    assert (after.people >= 0).all()
    assert (after.people[:, ~after.land] == 0).all()


def test_gathering_people_move_toward_the_target_and_stay_there():
    space = STILL.space
    target = 30
    moves = steps(space, target)
    start = int(np.flatnonzero(moves == 3)[0])
    position = at(STILL, [colony(42, start), colony(42, target)])
    after = order(STILL, position, (GATHER, target), (GATHER, target)).state

    def mean_moves(row):
        return (row * moves).sum() / row.sum()

    assert mean_moves(after.people[0]) < 3 - 1
    np.testing.assert_array_equal(after.people[1], colony(42, target))  # already there


def test_people_beyond_reach_of_the_target_only_drift():
    space = STILL.space
    target = 30
    moves = steps(space, target)
    far = int(np.flatnonzero(moves == STILL.reach + 1)[0])
    position = at(STILL, [colony(42, far), colony(42, target)])
    after = order(STILL, position, (GATHER, target), (HOLD, 0)).state
    assert after.people[0, far] == pytest.approx(10 * (1 - STILL.drift) ** STILL.steps, rel=0.1)
    near = int(np.flatnonzero(moves == STILL.reach)[0])
    position = at(STILL, [colony(42, near), colony(42, target)])
    after = order(STILL, position, (GATHER, target), (HOLD, 0)).state
    assert after.people[0, near] < 10 * (1 - STILL.march) ** (STILL.steps - 1)


def test_holding_people_only_drift_to_neighbors():
    position = at(STILL, [colony(42, 0), colony(42, 30)])
    held = order(STILL, position, (HOLD, 0), (HOLD, 0)).state
    moves = steps(STILL.space, 0)
    assert held.people[0, 0] == pytest.approx(10 * (1 - STILL.drift) ** STILL.steps, rel=0.1)
    assert (held.people[0, moves > STILL.steps] == 0).all()


def test_growth_approaches_the_capacity_and_never_passes_it():
    game = dataclasses.replace(SMALL, fight=0.0, drift=0.0, rounds=30)
    position = at(game, [colony(42, 5, 1.0), colony(42, 20, 1.0)])
    for _ in range(game.rounds):
        position = order(game, position, (HOLD, 0), (HOLD, 0))
        people = position.state.people.sum(axis=0)
        assert (people <= position.state.capacity).all()
    np.testing.assert_allclose(position.state.people[[0, 1], [5, 20]], 10.0, rtol=1e-6)


def test_a_crowded_cell_loses_at_most_a_share_growth_of_its_people_per_step():
    game = dataclasses.replace(SMALL, fight=0.0, drift=0.0, steps=1)
    capacity = np.full(42, 1.0)
    for people, kept in ((1.5, 1.5 * (1 - 0.3 * 0.5)), (100.0, 100.0 * (1 - 0.3))):
        position = at(game, [colony(42, 5, people), colony(42, 30, 1.0)], capacity=capacity)
        after = order(game, position, (HOLD, 0), (HOLD, 0)).state
        assert after.people[0, 5] == pytest.approx(kept)


def test_players_lose_people_only_where_they_meet():
    game = dataclasses.replace(SMALL, growth=0.0, march=0.0, drift=0.0)
    alone, shared = 3, 25
    first = colony(42, alone) + colony(42, shared, 4.0)
    position = at(game, [first, colony(42, shared, 8.0)])
    after = order(game, position, (HOLD, 0), (HOLD, 0)).state
    assert after.people[0, alone] == 10.0
    assert after.people[0, shared] < 4.0
    assert after.people[1, shared] < 8.0
    assert after.people[0, shared] < after.people[1, shared]  # outnumbered, it fares worse


def test_how_much_a_fight_costs_is_drawn_per_cell():
    game = dataclasses.replace(SMALL, growth=0.0, march=0.0, drift=0.0, sea=0.0)
    position = at(game, [colony(42, 3) + colony(42, 25), colony(42, 3) + colony(42, 25)])
    holds = {0: Action(HOLD, 0), 1: Action(HOLD, 0)}
    one, other = (game.resolve(position.state, holds, seed).people for seed in (1, 2))
    assert one[0, 3] != one[0, 25]  # the same fight, another factor
    assert one[0, 3] != other[0, 3]  # another seed, another factor


def test_a_round_treats_every_seat_alike():
    game = dataclasses.replace(SMALL, sea=0.0)
    a, b = colony(42, 3) + colony(42, 25, 4.0), colony(42, 25, 8.0) + colony(42, 40)
    one = game.resolve(at(game, [a, b]).state, {0: Action(GATHER, 25), 1: Action(HOLD, 0)}, 7)
    other = game.resolve(at(game, [b, a]).state, {0: Action(HOLD, 0), 1: Action(GATHER, 25)}, 7)
    np.testing.assert_array_equal(one.people[[1, 0]], other.people[:2])


def test_the_most_people_after_the_last_round_wins():
    game = dataclasses.replace(SMALL, rounds=2)
    record = play(game, [UniformPlayer()] * 2, game.start(0))
    assert record.final.over
    assert record.final.ply == 4
    totals = game.totals(record.final.state)
    assert record.final.winner == int(np.argmax(totals))
    assert [e.round for e in record.events if isinstance(e, Resolved)] == [1, 2]


def test_the_same_seed_plays_the_same_game_and_other_seeds_others():
    game = dataclasses.replace(SMALL, players=3)
    keys = [play(game, [UniformPlayer()] * 3, game.start(s)).final.key() for s in (0, 0, 1)]
    assert keys[0] == keys[1] != keys[2]


def test_start_refuses_a_seed_that_is_negative_or_not_an_integer(not_an_integer):
    with pytest.raises(TypeError, match="seed"):
        SMALL.start(not_an_integer)
    with pytest.raises(ValueError, match="seed"):
        SMALL.start(-1)


def test_the_rules_refuse_bad_parameters(not_an_integer, not_a_real):
    for name in ("frequency", "players", "rounds", "steps", "reach", "vision"):
        with pytest.raises(TypeError, match=name):
            Colonies(**{name: not_an_integer})
    for name in ("sea", "settlers", "richness", "march", "drift", "fight", "growth"):
        with pytest.raises(TypeError, match=name):
            Colonies(**{name: not_a_real})
    bad = [
        ("frequency", lambda: Colonies(frequency=0)),
        ("players", lambda: Colonies(players=1)),
        ("rounds", lambda: Colonies(rounds=0)),
        ("steps", lambda: Colonies(steps=0)),
        ("reach", lambda: Colonies(reach=0)),
        ("vision", lambda: Colonies(vision=-1)),
        ("sea", lambda: Colonies(sea=1.0)),
        ("settlers", lambda: Colonies(settlers=0.0)),
        ("richness", lambda: Colonies(richness=-1.0)),
        ("march", lambda: Colonies(march=1.5)),
        ("drift", lambda: Colonies(drift=-0.1)),
        ("fight", lambda: Colonies(fight=-0.1)),
        ("growth", lambda: Colonies(growth=-1.0)),
    ]
    for name, make in bad:
        with pytest.raises(ValueError, match=name):
            make()
    with pytest.raises(ValueError, match="players"):
        Colonies(players=MAX_PLAYERS + 1)
    for fog in (1, None, np.True_):
        with pytest.raises(TypeError, match="fog"):
            Colonies(fog=fog)  # ty: ignore[invalid-argument-type]  # not a bool, on purpose
    with pytest.raises(ValueError, match="land cells"):
        Colonies(frequency=1, players=4, sea=0.8)  # 2 land cells of 12


def seen_by(game, state, player):
    """Return the land within ``game.vision`` moves of the player's cells with enough people."""
    own = np.flatnonzero(state.people[player] >= MIN_PEOPLE)
    return np.isfinite(steps(state.space, own, blocked=~state.land, limit=game.vision))


def test_a_player_sees_the_land_within_its_vision_of_its_own_people():
    game = dataclasses.replace(FOGGY, players=3)
    position = order(game, game.start(5), (HOLD, 0), (HOLD, 0), (HOLD, 0))
    for player in range(3):
        view = game.view(position, player)
        np.testing.assert_array_equal(view.seen, seen_by(game, position.state, player))
        assert view.seen.sum() < 42
        np.testing.assert_array_equal(view.people, position.state.people[:3] * view.seen)
        np.testing.assert_array_equal(view.land, position.state.land)
        np.testing.assert_array_equal(view.capacity, position.state.capacity)
        assert (view.round, view.out) == (1, (False,) * 3)


def test_cells_with_too_few_people_give_no_sight():
    far = int(np.argmax(steps(FOGGY.space, 5)))  # beyond sight of cell 5
    position = at(FOGGY, [colony(42, 5, MIN_PEOPLE) + colony(42, far, MIN_PEOPLE / 2)])
    seen = FOGGY.view(position, 0).seen
    assert seen[5]
    assert not seen[far]
    assert not FOGGY.view(position, 1).seen.any()  # no people, no sight
    np.testing.assert_array_equal(np.flatnonzero(FOGGY.legal(position)[HOLD]), [0])


def test_under_fog_only_seen_land_is_a_legal_target():
    position = FOGGY.start(3)
    legal = FOGGY.legal(position)
    seen = FOGGY.view(position, 0).seen
    np.testing.assert_array_equal(legal[GATHER], seen & position.state.land)
    hidden = int(np.flatnonzero(position.state.land & ~seen)[0])
    with pytest.raises(ValueError, match="illegal"):
        FOGGY.step(position, (GATHER, hidden))
    mover = FOGGY.step(position, (HOLD, 0))  # now player 1 moves, and its sight counts
    np.testing.assert_array_equal(
        FOGGY.legal(mover)[GATHER], FOGGY.view(mover, 1).seen & position.state.land
    )


def test_without_fog_every_view_is_whole():
    position = SMALL.start(3)
    for player in range(2):
        view = SMALL.view(position, player)
        assert view.seen.all()
        np.testing.assert_array_equal(view.people, position.state.people[:2])


@pytest.mark.parametrize("players", [2, 3, 4])
def test_a_view_and_the_legal_orders_tell_nothing_hidden(players):
    game = dataclasses.replace(FOGGY, players=players, rounds=6)
    record = play(game, [UniformPlayer()] * players, game.start(players))
    rng = np.random.default_rng(0)
    for position in replay(game, record):
        if position.over:
            continue
        mover = game.player(position)
        for player in range(players):
            seen = game.view(position, player).seen
            people = position.state.people.copy()
            others = np.arange(MAX_PLAYERS) != player
            hidden = others[:, None] & ~seen
            people[hidden] = rng.uniform(0, 20, hidden.sum())
            orders = tuple(Action(HOLD, 0) for _ in position.orders)  # other orders
            other = dataclasses.replace(
                position,
                state=position.state.replace({ColoniesState.people: people}),
                orders=orders,
            )
            for a, b in zip(game.view(position, player), game.view(other, player), strict=True):
                np.testing.assert_array_equal(a, b)
            if player == mover:
                np.testing.assert_array_equal(game.legal(position), game.legal(other))


def test_a_player_left_with_too_few_people_is_out_and_its_seat_is_skipped():
    game = dataclasses.replace(SMALL, players=3, growth=0.0, march=0.0, drift=0.0)
    rows = [colony(42, 5, 0.4), colony(42, 20), colony(42, 30)]  # too few, though alive
    position = order(game, at(game, rows), (HOLD, 0), (HOLD, 0), (HOLD, 0))
    assert position.out == (True, False, False)
    assert Out(0) in position.events
    assert (position.state.people[0] == 0).all()  # its last people are gone
    assert game.totals(position.state)[0] == 0
    assert not position.over
    assert game.player(position) == 1
    position = game.step(position, (HOLD, 0))
    assert game.player(position) == 2
    position = game.step(position, (HOLD, 0))
    assert position.round == 2
    assert game.player(position) == 1
    assert position.events[0] == Ordered(2)


def test_the_last_player_left_wins_at_once():
    game = dataclasses.replace(SMALL, growth=0.0, march=0.0, drift=0.0, fight=1.0, rounds=30)
    position = order(
        game, at(game, [colony(42, 5, 0.6), colony(42, 5, 10.0)]), (HOLD, 0), (HOLD, 0)
    )
    assert position.over
    assert position.winner == 1
    assert position.round == 1
    assert position.events[2:] == (Out(0),)
    assert not game.legal(position).any()


def test_players_out_together_leave_a_draw():
    game = dataclasses.replace(SMALL, growth=0.0, march=0.0, drift=0.0, fight=2.0)
    position = order(game, at(game, [colony(42, 5, 0.6), colony(42, 5, 0.6)]), (HOLD, 0), (HOLD, 0))
    assert position.over
    assert position.out == (True, True)
    assert position.winner is None


def test_check_game_holds_on_games_where_players_go_out():
    game = Colonies(frequency=1, players=4, sea=0.0, fight=1.0, rounds=20)
    records = check_game(game, UniformPlayer(), seeds=range(4))
    outs = [e for record in records for e in record.events if isinstance(e, Out)]
    assert len(outs) >= 4
    assert any(record.final.round < game.rounds for record in records)  # one left, early


@pytest.mark.parametrize("players", [2, 3])
def test_the_game_keeps_the_promises_of_a_revealable_game(players):
    game = dataclasses.replace(FOGGY, players=players)
    check_revealable(game, check_game(game, UniformPlayer(), seeds=range(2)))


def test_an_observation_shows_people_seen_by_whom_and_the_map_squashed_into_zero_to_one():
    game = dataclasses.replace(FOGGY, players=3, richness=4.0)
    rows = [colony(42, 0, 4.0), colony(42, 1, 12.0), colony(42, 2, 4.0)]
    position = dataclasses.replace(at(game, rows), round=1)
    view = game.view(position, 1)
    obs = game.observe(view, 1)
    assert game.planes == ("own", "others", "seen", "land", "capacity", "round")
    assert obs.dtype == np.float32
    assert obs.shape == (6, 42)
    np.testing.assert_allclose(obs[0], (view.people[1] / (view.people[1] + 4.0)), rtol=1e-6)
    others = view.people[0] + view.people[2]
    np.testing.assert_allclose(obs[1], others / (others + 4.0), rtol=1e-6)
    assert obs[0, 1] == 0.75  # 12 people at richness 4
    np.testing.assert_array_equal(obs[2], view.seen)
    np.testing.assert_array_equal(obs[3], view.land)
    np.testing.assert_allclose(obs[4], 10.0 / 14.0, rtol=1e-6)
    assert (obs[5] == 1 / game.rounds).all()


def test_what_is_revealed_is_the_observation_without_fog():
    record = play(FOGGY, [UniformPlayer()] * 2, FOGGY.start(5))
    clear = dataclasses.replace(FOGGY, fog=False)
    hidden = False
    for position in replay(FOGGY, record):
        for player in range(2):
            revealed = FOGGY.reveal(position, player)
            assert FOGGY.revealed == ("own", "others")
            seen = clear.observe(clear.view(position, player), player)
            np.testing.assert_array_equal(revealed, seen[:2])
            fogged = FOGGY.observe(FOGGY.view(position, player), player)
            hidden |= bool((fogged[1] < revealed[1]).any())
    assert hidden  # the fog hid some enemies


def test_every_player_sees_the_board_as_it_is():
    frame = FOGGY.frame(1)
    assert frame.identity
    assert frame.cells.size == 42
