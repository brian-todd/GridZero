"""Tests of the Colonies players: their rules on boards built by hand, and their strength."""

import dataclasses

import numpy as np
import torch

from examples.games.colonies import (
    GATHER,
    HOLD,
    MAX_PLAYERS,
    Colonies,
    ColoniesPosition,
    ColoniesState,
    Raider,
    Settler,
)
from gridzero.agents import UniformPlayer
from gridzero.agents.muzero import MuZeroPlayer
from gridzero.agents.muzero_net import NetModel, WorldNet
from gridzero.core import steps
from gridzero.game import Action, ladder, play, replay

GAME = Colonies(frequency=2)
"""42 cells, under fog."""

RNG = np.random.default_rng(0)
"""Neither player draws."""


def at(people, capacity=10.0, game=GAME):
    """Return round 0 of ``game`` on all land with ``people``, a dict of player to cell amounts."""
    rows = np.zeros((MAX_PLAYERS, 42))
    for player, cells in people.items():
        for cell, amount in cells.items():
            rows[player, cell] = amount
    capacity = np.broadcast_to(np.asarray(capacity, dtype=np.float64), (42,)).copy()
    state = ColoniesState(
        game.space, {ColoniesState.capacity: capacity, ColoniesState.people: rows}
    )
    return ColoniesPosition(state, 0, out=(False,) * game.players)


def decide(player, position, seat=0, game=GAME):
    """Return what ``player`` orders in ``seat`` at ``position``."""
    return player(game, seat, game.view(position, seat), game.legal(position), RNG)


def test_the_settler_holds_while_its_land_fills():
    assert decide(Settler(), at({0: {5: 7.0}})) == Action(HOLD, 0)


def test_the_settler_then_gathers_toward_the_seen_land_with_most_room():
    capacity = np.full(42, 10.0)
    seen = GAME.view(at({0: {5: 9.0}}), 0).seen
    richest = int(np.flatnonzero(seen & (np.arange(42) != 5))[-1])
    capacity[richest] = 30.0
    hidden = int(np.flatnonzero(~seen)[0])
    capacity[hidden] = 90.0  # richer, but unseen
    assert decide(Settler(), at({0: {5: 9.0}}, capacity)) == Action(GATHER, richest)


def test_the_settler_holds_when_no_seen_land_has_room():
    capacity = np.zeros(42)
    capacity[5] = 9.0
    assert decide(Settler(full=0.0), at({0: {5: 9.0}}, capacity)) == Action(HOLD, 0)


def test_the_raider_attacks_the_enemies_it_outnumbers_nearby():
    neighbor = int(np.flatnonzero(steps(GAME.space, 5) == 1)[0])
    position = at({0: {5: 9.0, neighbor: 9.0}, 1: {neighbor: 5.0}})
    assert decide(Raider(), position) == Action(GATHER, neighbor)


def test_the_raider_leaves_enemies_alone_when_it_lacks_the_edge():
    neighbor = int(np.flatnonzero(steps(GAME.space, 5) == 1)[0])
    position = at({0: {5: 7.0}, 1: {neighbor: 6.0}})
    assert decide(Raider(), position) == decide(Settler(), position) == Action(HOLD, 0)
    assert decide(Raider(edge=1.0), position) == Action(GATHER, neighbor)


def test_the_raider_attacks_the_prey_with_the_most_enemies():
    a, b = np.flatnonzero(steps(GAME.space, 5) == 1)[:2]
    for small, big in ((a, b), (b, a)):
        position = at({0: {5: 20.0}, 1: {int(small): 2.0, int(big): 3.0}})
        assert decide(Raider(), position) == Action(GATHER, int(big))


def test_the_raider_settles_when_it_has_no_prey():
    capacity = np.full(42, 10.0)
    capacity[int(np.flatnonzero(steps(GAME.space, 5) == 2)[0])] = 30.0
    position = at({0: {5: 9.0}}, capacity)
    assert decide(Settler(), position).verb == GATHER
    assert decide(Raider(), position) == decide(Settler(), position)


def test_the_raider_attacks_only_beside_its_own_people_even_without_fog():
    far = int(np.argmax(steps(GAME.space, 5)))
    position = at({0: {5: 7.0}, 1: {far: 0.1}})
    seeing = dataclasses.replace(GAME, fog=False)
    assert GAME.view(position, 0).people[1, far] == 0  # hidden under fog
    assert decide(Raider(edge=0.0), position, game=seeing) == Action(HOLD, 0)


def test_under_fog_the_scripted_players_beat_uniform():
    for player in (Raider(), Settler()):
        won, lost, _ = ladder(Colonies(), player, {"uniform": UniformPlayer()}, 8)["uniform"]
        assert won > 2 * lost


def test_the_raider_beats_uniform_without_fog_too():
    game = Colonies(players=3, fog=False)
    won, lost, _ = ladder(game, Raider(), {"uniform": UniformPlayer()}, 8)["uniform"]
    assert won > lost


def test_a_muzero_player_decides_alike_whatever_it_cant_see():
    """Enemies under fog and the round's earlier orders change nothing in its search."""
    torch.manual_seed(0)
    net = WorldNet(GAME, channels=8, blocks=1).eval()
    with torch.no_grad():
        for p in net.parameters():  # an untrained net that still reads every input
            p.add_(0.1 * torch.randn_like(p))
    player = MuZeroPlayer(NetModel(net), simulations=8, considered=4, noise=True)
    record = play(GAME, [UniformPlayer()] * 2, GAME.start(2))
    rng = np.random.default_rng(1)
    checked = 0
    for position in replay(GAME, record):
        if position.over:
            continue
        mover = GAME.player(position)
        seen = GAME.view(position, mover).seen
        people = position.state.people.copy()
        others = np.arange(MAX_PLAYERS) != mover
        hidden = others[:, None] & ~seen
        people[hidden] = rng.uniform(0, 20, hidden.sum())
        other = dataclasses.replace(
            position,
            state=position.state.replace({ColoniesState.people: people}),
            orders=tuple(Action(HOLD, 0) for _ in position.orders),
        )
        results = []
        for p in (position, other):
            obs = GAME.observe(GAME.view(p, mover), mover)
            legal = GAME.frame(mover).into(GAME.legal(p))
            results.append(player.search(obs, legal, np.random.default_rng(7)))
        assert results[0].action == results[1].action
        np.testing.assert_array_equal(results[0].policy, results[1].policy)
        checked += hidden.any()
    assert checked > 10  # positions where something was hidden
