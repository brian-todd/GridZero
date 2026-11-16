"""Tactics' rules as arrays against the game: the same positions, given the same dice."""

import dataclasses
import functools
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_demo import Compiled, check_array_game, differences
from examples.games.tactics import (
    Attacked,
    Broke,
    Built,
    Deferred,
    Died,
    GameOver,
    GreedyPlayer,
    Moved,
    Swapped,
    Verb,
    earlier_rules,
    small_rules,
)
from examples.games.tactics.arrays import Dice, State, TacticsArrays
from examples.games.tactics.rules import hit_chance
from tactics_demo import ARCHER, CAPTAIN, ENGINEER, FOOT, KNIGHT, WORKS, piece

RULES = {
    "small, fog": small_rules,
    "nine": lambda: WORKS,
    "earlier": lambda: dataclasses.replace(earlier_rules(), shape=(15, 15), home_depth=3),
    "flags off": lambda: dataclasses.replace(
        small_rules(),
        turn_on_move=False,
        free_facing=False,
        face_center=False,
        engaged_until_turn=False,
        shuffle_each_round=True,
    ),
}
"""Rules that between them turn every switch: fog, the earlier kinds (a double move, no charge
or brace, turns shuffled each round) and each flag."""

UNDER_FOG = {
    "nine, fog": lambda: dataclasses.replace(WORKS, fog=True),
    "near": lambda: dataclasses.replace(
        WORKS,
        units=tuple(dataclasses.replace(u, reach=1, sight=1.0) for u in WORKS.units),
        fog=True,
    ),
}
"""Rules under fog for positions built by hand: the default kinds on `WORKS`' board, and kinds
that see and strike only the tiles beside them."""

CENTER = (4, 4)  # the point of `WORKS`' board


@functools.cache
def compiled(name: str, hidden: bool = True) -> Compiled[TacticsArrays]:
    """The array rules of one of `RULES` or `UNDER_FOG`, compiled once a session."""
    return Compiled(TacticsArrays((RULES | UNDER_FOG)[name](), hidden=hidden))


def played(
    compiled_, position, forced=(), *, seed=0, plies=None, watch=None, greedy=0.0
) -> Counter[str]:
    """Check both engines from ``position``: the ``forced`` game actions first, then the greedy
    player's for a share ``greedy`` of the battle's decisions, and the rest at random, with a
    liking for the rarer verbs."""
    arrays = compiled_.arrays
    todo = list(forced)
    player = GreedyPlayer()

    def choose(game, position, legal, rng):
        if todo:
            return arrays.local(position, todo.pop(0))
        if not position.deploying and rng.random() < greedy:
            mover = game.player(position)
            shown = game.view(position, mover)
            return arrays.local(position, player(game, mover, shown, game.legal(position), rng))
        verbs = {local: arrays.action(position, local)[0] for local in legal}
        rare = [a for a, v in verbs.items() if v in (Verb.DEFER, Verb.SWAP, Verb.BUILD, Verb.BREAK)]
        attacks = [a for a, v in verbs.items() if v == Verb.ATTACK]
        acts = [a for a, v in verbs.items() if v != Verb.END]
        roll = rng.random()
        for some, share in ((rare, 0.25), (attacks, 0.6), (acts, 0.9)):
            if some and roll < share:
                return int(rng.choice(some))
        return int(rng.choice(legal))

    rng = np.random.default_rng(seed)

    def outcome(before, after) -> tuple[int, float]:
        """A step's hits as bits, and their chance from the Atk and Def of its attacks."""
        game, which, chance = arrays.game, 0, 1.0
        attacks = [event for event in after.events if isinstance(event, Attacked)]
        for nth, event in enumerate(attacks):
            hit = hit_chance(event.atk, event.defense, game.hit_base, game.hit_step)
            which += event.hit << nth
            chance *= hit if event.hit else 1.0 - hit
        return which, chance

    return check_array_game(
        arrays.game, compiled_, position, choose, rng=rng, plies=plies, watch=watch, outcome=outcome
    )


def events_of(seen: Counter[object]):
    """Return a `watch` that counts a step's events by kind into ``seen``."""

    def watch(position) -> None:
        for event in position.events:
            seen[type(event)] += 1
            if isinstance(event, Attacked):
                seen["deferred"] += event.deferred
                seen["strikes"] += event.braced
                seen["hits"] += event.hit
                seen["flanks"] += event.flank
            if isinstance(event, GameOver):
                seen[event.reason] += 1
        seen["most rolls"] = max(
            seen["most rolls"], sum(isinstance(event, Attacked) for event in position.events)
        )

    return watch


# Whole games, from the first unit placed.


def whole_games(name, seeds) -> tuple[Counter[str], Counter[object]]:
    """Check whole games of one of `RULES`: beliefs under fog, taken at the start or midway."""
    game, run = RULES[name](), compiled(name)
    greedy = GreedyPlayer()
    found: Counter[str] = Counter()
    seen: Counter[object] = Counter()
    for seed in seeds:
        position = game.start(seed)
        rng = np.random.default_rng(seed)
        if game.fog and seed % 2:  # a belief of a game under way, ghosts and all
            for _ in range(60 + 20 * (seed % 5)):
                me = game.player(position)
                action = greedy(game, me, game.view(position, me), game.legal(position), rng)
                position = game.step(position, action)
                if position.over:
                    break
        if position.over:
            continue
        position = game.view(position, seed % 2)  # the position itself without fog
        share = (0.0, 0.5, 0.8, 0.95)[seed % 4]  # random games wander, greedy ones fight
        found += played(run, position, seed=seed, watch=events_of(seen), greedy=share)
    return found, seen


@pytest.mark.parametrize("name", ["small, fog", "nine"])
def test_whole_games_are_the_games_ply_by_ply(name):
    found, seen = whole_games(name, range(4))
    assert set(found) == {"plies"}, found
    assert found["plies"] > 400
    for event in (Moved, Attacked, Died, Deferred, Swapped, Built, Broke, GameOver):
        assert seen[event] > 0, event


@pytest.mark.slow
@pytest.mark.parametrize("name", list(RULES))
def test_many_games_of_every_rule_are_the_games_ply_by_ply(name):
    found, seen = whole_games(name, range(4, 34))
    assert set(found) == {"plies"}, found
    assert found["plies"] > 3000
    assert {"killed", "held", "deadline"} <= set(seen)
    assert seen["deferred"] > 0


def real_games(name, seeds) -> tuple[Counter[str], Counter[object]]:
    """Check whole games under fog of one of `RULES`, as the game plays them: from the first
    unit placed, on the positions themselves, not on beliefs. Counts the plies that hid units
    in ``"hidden"``, and the ghosts (units out of the other side's sight) in ``"ghosts"``."""
    game, run = (RULES | UNDER_FOG)[name](), compiled(name)
    found: Counter[str] = Counter()
    seen: Counter[object] = Counter()
    count = events_of(seen)

    def watch(position) -> None:
        count(position)
        if position.round > 0 and not position.over:
            seen["hidden"] += 1
            units, cells = position.state.units, position.state.occupied.nonzero()[0]
            seen["ghosts"] += int((units.spotted[cells] != cells).sum())

    for seed in seeds:
        share = (0.0, 0.5, 0.8, 0.95)[seed % 4]
        found += played(run, game.start(seed), seed=seed, watch=watch, greedy=share)
    return found, seen


def test_whole_games_under_fog_are_the_games_ply_by_ply():
    found, seen = real_games("small, fog", range(2))
    assert set(found) == {"plies"}, found
    assert seen["hidden"] > 400
    assert seen["ghosts"] > 100
    assert seen[Died] > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", ["small, fog", "nine, fog"])
def test_many_games_under_fog_are_the_games_ply_by_ply(name):
    found, seen = real_games(name, range(2, 32))
    assert set(found) == {"plies"}, found
    assert seen["hidden"] > 3000
    assert {"killed", "held", "deadline"} & set(seen)


# Hand-made positions for what whole games seldom show.


@pytest.fixture
def works(make_battle):
    """Battles of `WORKS` built by hand, on its own board: the point at the center."""
    return functools.partial(make_battle, game=WORKS, point=CENTER)


def test_a_brace_strikes_first_and_may_kill_the_charger(works, at):
    seen: Counter[object] = Counter()
    pieces = [
        piece(5, 4, KNIGHT, 0, moved=3, hp=4),  # one strike from dying
        piece(4, 4, FOOT, 1, facing=0),
        piece(3, 5, FOOT, 1, facing=0),
    ]
    for seed in range(24):
        found = played(
            compiled("nine"),
            works(pieces, seed=seed),
            [(Verb.ATTACK, at(4, 4, WORKS))],
            plies=3,
            watch=events_of(seen),
        )
        assert set(found) == {"plies"}, (seed, found)
    assert seen["strikes"] == 24
    assert 0 < seen[Died] < 24  # the knight died of some strikes, and attacked after others
    assert seen[Attacked] > 24


@pytest.mark.parametrize(
    ("target", "friend", "strikes"),
    [
        ({}, FOOT, True),
        ({"engaged": True}, FOOT, False),  # attacked this round: it has lost its footing
        ({}, ARCHER, False),  # no friend that forms up beside it
        ({"facing": 3}, FOOT, True),  # charged from behind: it braces all the same
    ],
)
def test_a_footsoldier_braces_only_on_its_footing_beside_a_friend_in_formation(
    works, at, target, friend, strikes
):
    seen: Counter[object] = Counter()
    pieces = [
        piece(5, 4, KNIGHT, 0, moved=3),
        piece(4, 4, FOOT, 1, **{"facing": 0, **target}),
        piece(3, 5, friend, 1, facing=0),
    ]
    for seed in range(8):
        found = played(
            compiled("nine"),
            works(pieces, seed=seed),
            [(Verb.ATTACK, at(4, 4, WORKS))],
            plies=1,
            watch=events_of(seen),
        )
        assert set(found) == {"plies"}, (seed, found)
    assert seen["strikes"] == 8 * strikes


def test_archers_fire_in_order_at_a_unit_walking_through_their_tiles(works, at):
    seen: Counter[object] = Counter()
    walker = piece(6, 4, FOOT, 1)
    archers = [
        piece(4, 7, ARCHER, 0, watch=at(4, 4, WORKS), done=True),
        piece(1, 4, ARCHER, 0, watch=at(4, 4, WORKS), done=True),
        piece(5, 1, ARCHER, 0, watch=at(5, 4, WORKS), done=True),
    ]
    for seed in range(40):
        found = played(
            compiled("nine"),
            works([walker, *archers], seed=seed),
            [(Verb.MOVE, at(4, 4, WORKS))],
            plies=2,
            watch=events_of(seen),
        )
        assert set(found) == {"plies"}, (seed, found)
    assert seen["most rolls"] == 3
    assert 0 < seen[Died] < 40  # some walks end on the first tile, some reach the second
    assert seen["deferred"] > 60


def test_a_swap_springs_the_attacks_on_both_tiles(works, at):
    seen: Counter[object] = Counter()
    pieces = [
        piece(4, 3, FOOT, 1, hp=2),
        piece(5, 3, CAPTAIN, 1, hp=2),
        piece(2, 3, ARCHER, 0, watch=at(5, 3, WORKS), done=True),
        piece(7, 3, ARCHER, 0, watch=at(4, 3, WORKS), done=True),
        piece(5, 6, ARCHER, 0, watch=at(5, 3, WORKS), done=True),
    ]
    for seed in range(30):
        found = played(
            compiled("nine"),
            works(pieces, seed=seed),
            [(Verb.SWAP, at(5, 3, WORKS))],
            plies=2,
            watch=events_of(seen),
        )
        assert set(found) == {"plies"}, (seed, found)
    assert seen[Swapped] == 30
    assert seen["most rolls"] == 3
    assert seen[Died] > 0


def test_an_archer_that_walked_out_of_reach_keeps_its_tile_and_does_not_fire(works, at):
    pieces = [
        piece(6, 4, FOOT, 1),
        piece(0, 8, ARCHER, 0, watch=at(5, 4, WORKS), done=True),  # too far
        piece(5, 0, ARCHER, 0, watch=at(5, 4, WORKS), done=True),  # behind a wall
    ]
    position = works(pieces, walls=[(5, 2)])
    seen: Counter[object] = Counter()
    found = played(
        compiled("nine"), position, [(Verb.MOVE, at(5, 4, WORKS))], plies=1, watch=events_of(seen)
    )
    assert set(found) == {"plies"}, found
    assert seen[Attacked] == 0


def test_a_move_goes_around_a_watched_tile_as_the_game_does(works, at):
    for watched in ((5, 4), (5, 5), (6, 5), (4, 5)):
        pieces = [
            piece(6, 4, FOOT, 1),
            piece(5, 2, ARCHER, 0, watch=at(*watched, WORKS), done=True),
            piece(7, 6, ARCHER, 0, watch=at(5, 5, WORKS), done=True),
        ]
        for goal in ((4, 5), (4, 6), (5, 6)):
            found = played(
                compiled("nine"), works(pieces), [(Verb.MOVE, at(*goal, WORKS))], plies=1
            )
            assert set(found) == {"plies"}, (watched, goal, found)


def test_an_engineers_tiles_are_the_games_among_walls_rubble_and_the_edge(works, at):
    seen: Counter[object] = Counter()
    rng = np.random.default_rng(5)
    for trial in range(30):
        q, r = (int(v) for v in rng.integers(0, 9, size=2))  # the edge and corners too
        if (q, r) == CENTER:
            continue
        around = [(q + dq, r + dr) for dq in range(-2, 3) for dr in range(-2, 3)]
        around = [c for c in around if c != (q, r) and all(0 <= v < 9 for v in c)]
        kinds = rng.choice(3, size=len(around), p=[0.5, 0.3, 0.2])
        walls = [c for c, k in zip(around, kinds, strict=True) if k == 1]
        rubble = [c for c, k in zip(around, kinds, strict=True) if k == 2]
        open_ = [c for c, k in zip(around, kinds, strict=True) if k == 0 and c != CENTER]
        pieces = [piece(q, r, ENGINEER, 0), piece(*open_[0], FOOT, 1, done=True)]
        position = works(pieces, walls=walls, rubble=rubble, seed=trial)
        found = played(compiled("nine"), position, plies=4, seed=trial, watch=events_of(seen))
        assert set(found) == {"plies"}, (trial, found)
    assert seen[Built] > 5
    assert seen[Broke] > 5


def test_a_round_ends_on_the_point_and_the_game_at_its_deadline(works, at):
    seen: Counter[object] = Counter()
    for hold, round_, seed in [((1, 3), 5, 0), ((-1, 0), 8, 1), ((0, 1), 12, 2), ((1, 1), 3, 3)]:
        pieces = [piece(4, 4, FOOT, hold[0] % 2), piece(0, 0, FOOT, 1 - hold[0] % 2)]
        position = works(pieces, round=round_, hold=type(WORKS.start(0).hold)(*hold), seed=seed)
        ends = [(Verb.END, at(4, 4, WORKS)), (Verb.END, at(0, 0, WORKS))]
        found = played(compiled("nine"), position, ends, plies=6, watch=events_of(seen))
        assert set(found) == {"plies"}, found
    assert seen[GameOver] >= 2


# Under fog: what a side believes, and the game played on what it doesn't know.


def test_a_move_stops_before_a_unit_it_didnt_know_of(make_battle, at):
    # The enemy stands two tiles off, where the mover is going, believed in the far corner.
    game = UNDER_FOG["near"]()
    pieces = [piece(2, 4, FOOT, 0), piece(4, 4, FOOT, 1, spotted=at(8, 8))]
    position = make_battle(pieces, game, point=CENTER)
    move = (Verb.MOVE, at(4, 4))
    assert game.step(position, move).state.units.uid[at(3, 4)] == 0  # one tile short
    for seed in range(4):
        found = played(compiled("near"), dataclasses.replace(position, seed=seed), [move], plies=6)
        assert set(found) == {"plies"}, (seed, found)


@pytest.mark.parametrize(
    ("pieces", "moves"),
    [
        # Last seen on (6, 4), out of sight, the enemy walks off; then the footsoldier sees
        # that tile empty, and the ghost moves on to the nearest tile out of sight.
        (
            [piece(6, 4, FOOT, 1), piece(2, 4, FOOT, 0)],
            [((8, 4), Verb.MOVE), ((8, 4), Verb.END), ((4, 4), Verb.MOVE)],
        ),
        # Glimpsed on its way at (7, 4), the walker is last seen there, hurt since it was
        # seen before, and sees from there the watcher its side believed on (0, 0).
        (
            [
                piece(8, 3, FOOT, 1, hp=2, spotted_hp=WORKS.units[FOOT].hp),
                piece(4, 4, FOOT, 0, spotted=0),
            ],
            [((7, 5), Verb.MOVE)],
        ),
        # Two ghosts move on together, never to one tile.
        (
            [piece(6, 4, FOOT, 1), piece(6, 5, FOOT, 1), piece(0, 4, KNIGHT, 0)],
            [
                ((8, 4), Verb.MOVE),
                ((8, 4), Verb.END),
                ((8, 5), Verb.MOVE),
                ((8, 5), Verb.END),
                ((2, 4), Verb.MOVE),
            ],
        ),
    ],
    ids=["moved on", "glimpsed", "two ghosts"],
)
def test_what_each_side_remembers_is_the_games(make_battle, at, pieces, moves):
    position = make_battle(pieces, UNDER_FOG["nine, fog"](), point=CENTER)
    forced = [(verb, at(*tile)) for tile, verb in moves]
    for seed in range(4):
        start = dataclasses.replace(position, seed=seed)
        found = played(compiled("nine, fog"), start, forced, plies=len(forced) + 4)
        assert set(found) == {"plies"}, (seed, found)


def test_a_view_is_the_games_but_for_the_seed_of_a_belief():
    game = small_rules()
    run = compiled("small, fog")
    arrays = run.arrays
    view = jax.jit(arrays.view)
    greedy = GreedyPlayer()
    rng = np.random.default_rng(1)
    position, beliefs = game.start(5), 0
    while not position.over:
        if position.ply % 5 == 0:
            state = arrays.to_arrays(position)
            for team in (0, 1):
                shown = game.view(position, team)
                found = view(state, team)
                assert differences(found, arrays.to_arrays(shown)) == ["seed"], position.ply
                assert np.asarray(found.seed).tolist() != np.asarray(state.seed).tolist()
                np.testing.assert_array_equal(view(state, team).seed, found.seed)
                assert not differences(view(found, 1 - team), found)  # a belief already
                beliefs += position.round > 0
        me = game.player(position)
        action = greedy(game, me, game.view(position, me), game.legal(position), rng)
        position = game.step(position, action)
    state = arrays.to_arrays(position)
    assert not differences(view(state, 0), state)  # once over, all is shown
    assert beliefs > 20


def test_without_fog_a_view_is_the_state_itself():
    arrays = compiled("nine").arrays
    state = arrays.to_arrays(WORKS.start(3))
    assert not differences(arrays.view(state, 1), jax.tree.map(jnp.asarray, state))


def test_rules_that_leave_fog_out_play_beliefs_as_the_whole_rules_do():
    game, run = small_rules(), compiled("small, fog", hidden=False)
    assert not run.arrays.hidden
    assert compiled("small, fog").arrays.hidden
    position = game.start(7)
    rng = np.random.default_rng(7)
    greedy = GreedyPlayer()
    for _ in range(90):  # into the battle, ghosts and all
        me = game.player(position)
        position = game.step(
            position, greedy(game, me, game.view(position, me), game.legal(position), rng)
        )
    found = played(run, game.view(position, game.player(position)), seed=7, plies=200, greedy=0.5)
    assert set(found) == {"plies"}, found
    with pytest.raises(TypeError, match="hidden"):
        TacticsArrays(game, hidden=1)  # ty: ignore[invalid-argument-type]  # not a bool, on purpose


# Between the engines.


def test_a_position_goes_to_arrays_and_back_unchanged():
    game = small_rules()
    arrays = compiled("small, fog").arrays
    greedy = GreedyPlayer()
    rng = np.random.default_rng(0)
    position = game.start(3)
    for ply in range(150):
        if ply % 10 == 0:
            for shown in (game.view(position, 0), game.view(position, 1)):
                back = arrays.to_position(arrays.to_arrays(shown))
                for field in shown.state:
                    np.testing.assert_array_equal(back.state[field], shown.state[field], str(field))
                    assert back.state[field].dtype == shown.state[field].dtype
                for name in ("seed", "ply", "over", "winner", "round", "hold", "belief"):
                    assert getattr(back, name) == getattr(shown, name), name
        me = game.player(position)
        position = game.step(
            position, greedy(game, me, game.view(position, me), game.legal(position), rng)
        )
        if position.over:
            break


def test_a_state_is_small_arrays_of_the_declared_dtypes():
    game = small_rules()
    state = compiled("small, fog").arrays.to_arrays(game.start(2**40 + 7))
    assert isinstance(state, State)
    assert state.cell.dtype == np.int16
    assert state.kind.dtype == np.int8
    assert state.done.dtype == np.bool_
    assert state.ply.dtype == np.int32
    assert state.seed.tolist() == [7, 2**8]  # the seed's low and high 32 bits
    assert state.cell.shape == (16,)
    assert state.wall.shape == (225,)
    assert sum(np.asarray(value).nbytes for value in state) < 1024  # a node of a search


def test_local_actions_name_the_games_actions_and_back():
    game = WORKS
    arrays = compiled("nine").arrays
    assert arrays.n_actions == 133
    assert (arrays.move0, arrays.attack0, arrays.defer0, arrays.swap0) == (0, 36, 72, 108)
    assert (arrays.build0, arrays.break0, arrays.face0, arrays.end0) == (114, 120, 126, 132)
    position = game.start(0)
    home = np.flatnonzero(position.state.home == 0)
    assert arrays.home_size == len(home) == 16
    assert arrays.action(position, 0) == (Verb.PLACE, home[0])
    assert arrays.action(position, 16 + 3) == (Verb.PLACE + 1, home[3])
    assert arrays.local(position, (Verb.PLACE + 4, int(home[15]))) == 4 * 16 + 15
    with pytest.raises(ValueError, match="names nothing"):
        arrays.action(position, 5 * 16)
    with pytest.raises(ValueError, match="no local action"):
        arrays.local(position, (Verb.MOVE, 0))
    with pytest.raises(ValueError, match="names nothing"):
        arrays.action(position, 133)


def test_local_actions_of_a_battle_are_the_tiles_around_the_acting_unit(works, at):
    arrays = compiled("nine").arrays
    position = works([piece(0, 0, ARCHER, 0), piece(8, 8, FOOT, 1)])
    corner = at(0, 0, WORKS)
    assert arrays.action(position, arrays.end0) == (Verb.END, corner)
    assert arrays.action(position, arrays.face0 + 2) == (Verb.FACE + 2, corner)
    assert arrays.local(position, (Verb.DEFER, at(3, 0, WORKS))) in range(72, 108)
    assert arrays.local(position, (Verb.BREAK, at(1, 0, WORKS))) in range(120, 126)
    named = set()
    for local in range(arrays.n_actions):
        try:
            named.add(arrays.action(position, local))
        except ValueError:  # a tile off the board
            continue
        assert arrays.local(position, arrays.action(position, local)) == local
    assert len(named) < arrays.n_actions  # the corner has tiles off the board around it
    with pytest.raises(ValueError, match="no local action"):
        arrays.local(position, (Verb.MOVE, at(4, 0, WORKS)))  # four tiles away
    with pytest.raises(ValueError, match="over"):
        arrays.action(dataclasses.replace(position, over=True), 0)


def test_only_boards_and_teams_of_the_game_convert(make_battle, works):
    arrays = compiled("nine").arrays
    with pytest.raises(ValueError, match="point at the center"):
        arrays.to_arrays(make_battle([piece(0, 0, FOOT, 0)], game=WORKS))  # no point
    with pytest.raises(ValueError, match="hex lattice of shape"):
        arrays.to_arrays(small_rules().start(0))
    four = [piece(q, 0, ARCHER, 0) for q in range(4)]
    with pytest.raises(ValueError, match="units that defer"):
        arrays.to_arrays(works(four))
    eleven = [piece(q, r, FOOT, 0) for q in range(4) for r in range(3)][:11]
    with pytest.raises(ValueError, match="uids"):
        arrays.to_arrays(works(eleven))
    with pytest.raises(TypeError, match="Tactics"):
        TacticsArrays(WORKS.start(0))  # ty: ignore[invalid-argument-type]


# Its own dice.


def test_a_roll_of_20_hits_and_a_roll_of_1_misses_whatever_the_odds(works, at):
    run = compiled("nine")
    arrays = run.arrays
    position = works([piece(4, 3, ARCHER, 0), piece(4, 4, CAPTAIN, 1, facing=3)])
    state = arrays.to_arrays(position)
    attack = arrays.local(position, (Verb.ATTACK, at(4, 4, WORKS)))
    first, order = jnp.int32(0), jnp.arange(10)
    hit = run.apply(state, attack, Dice(jnp.array([20, 1, 1]), first, order))
    miss = run.apply(state, attack, Dice(jnp.array([1, 20, 20]), first, order))
    assert int(hit.hp[1]) < int(state.hp[1])
    assert int(miss.hp[1]) == int(state.hp[1])
    assert bool(hit.engaged[1])
    assert bool(miss.engaged[1])
    assert differences(hit, miss) == ["hp"]


def played_alone(run, state, plies: int, seed: int) -> State:
    """Play a state on with its own dice, taking a legal action at random each ply."""
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        legal = np.flatnonzero(np.asarray(run.legal(state)))
        if not legal.size:
            break
        state = run.step(state, int(rng.choice(legal)))
    return state


def test_a_step_rolls_the_states_own_dice_the_same_each_time():
    run = compiled("nine")
    arrays = run.arrays
    start = arrays.to_arrays(WORKS.start(1))
    state = played_alone(run, start, 60, seed=0)
    assert int(state.round) > 0
    legal = np.flatnonzero(np.asarray(run.legal(state)))
    for action in legal[:6].tolist():
        once, again = run.step(state, action), run.step(state, action)
        assert differences(once, again) == []
        given = run.apply(state, action, arrays.draws(state))
        assert differences(once, given) == []
    dice = arrays.draws(state)
    assert dice.rolls.shape == (arrays.n_rolls,)
    assert 1 <= int(dice.rolls.min()) <= int(dice.rolls.max()) <= 20
    assert int(dice.first) in (0, 1)
    assert sorted(dice.order.tolist()) == list(range(10))
    later = arrays.draws(state._replace(ply=state.ply + 1))
    other = arrays.draws(arrays.to_arrays(WORKS.start(2))._replace(ply=state.ply))
    assert (
        later.order.tolist() != dice.order.tolist() or later.rolls.tolist() != dice.rolls.tolist()
    )
    assert (
        other.order.tolist() != dice.order.tolist() or other.rolls.tolist() != dice.rolls.tolist()
    )


def test_states_played_by_the_arrays_alone_are_positions_of_the_game():
    run = compiled("nine")
    arrays = run.arrays
    ended = 0
    for seed in range(6):
        state = played_alone(run, arrays.to_arrays(WORKS.start(seed)), 40 + 25 * seed, seed=seed)
        position = arrays.to_position(state)
        if position.over:
            ended += 1
            assert position.winner in (0, 1)
            assert not WORKS.legal(position).any()
            continue
        found = played(run, position, plies=25, seed=seed)  # the game takes it from here
        assert set(found) == {"plies"}, (seed, found)
    state = played_alone(run, arrays.to_arrays(WORKS.start(9)), 5000, seed=9)
    assert bool(state.over)
    assert int(state.winner) in (0, 1)
    assert 1 <= int(state.round) <= WORKS.max_rounds


def test_a_batch_of_games_is_each_game_alone():
    run = compiled("nine")
    arrays = run.arrays
    states = [
        played_alone(run, arrays.to_arrays(WORKS.start(seed)), 30 + seed, seed) for seed in range(8)
    ]
    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *states)
    legal = np.asarray(jax.jit(jax.vmap(arrays.legal))(batch))
    actions = np.array([int(np.flatnonzero(row)[0]) if row.any() else 0 for row in legal])
    together = jax.jit(jax.vmap(arrays.step))(batch, actions)
    planes = np.asarray(jax.jit(jax.vmap(arrays.observe))(batch, np.arange(8) % 2))
    for row, state in enumerate(states):
        np.testing.assert_array_equal(legal[row], run.legal(state))
        alone = run.step(state, int(actions[row]))
        one = State(*(leaf[row] for leaf in together))
        assert differences(one, alone) == []
        assert planes[row].tobytes() == np.asarray(run.observe(state, row % 2)).tobytes()


def test_the_battles_first_turn_goes_to_either_team_and_any_unit():
    run = compiled("nine")
    arrays = run.arrays
    rng = np.random.default_rng(0)
    position = last = WORKS.start(0)
    while position.deploying:
        last = position
        legal = np.argwhere(WORKS.legal(position))
        position = WORKS.step(position, tuple(legal[rng.integers(len(legal))]))
    state = arrays.to_arrays(last)  # one unit to place
    action = int(np.flatnonzero(np.asarray(run.legal(state)))[0])
    seeds = np.arange(400, dtype=np.uint32)
    batch = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (400, *np.shape(leaf))), state)
    batch = batch._replace(seed=np.stack([seeds, np.zeros(400, np.uint32)], axis=1))
    after = jax.jit(jax.vmap(arrays.step, in_axes=(0, None)))(batch, action)
    initiative, team = np.asarray(after.initiative), np.asarray(after.team)
    assert (np.asarray(after.round) == 1).all()
    assert (np.sort(initiative, axis=1) == np.arange(10)).all()  # each place once
    first = team[np.arange(400), initiative.argmin(axis=1)]
    assert 150 < first.sum() < 250  # a coin
    order = np.argsort(initiative, axis=1)
    assert (team[np.arange(400)[:, None], order][:, ::2] == first[:, None]).all()  # they alternate
    leads = np.bincount(initiative.argmin(axis=1), minlength=10)
    assert (leads > 15).all()  # any unit may lead: 40 each on average
