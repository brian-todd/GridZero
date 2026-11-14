"""`batched`'s queries against `core`'s: moves on any space, and sight and paths around a cell."""

import jax
import numpy as np
import pytest

from gridzero import batched
from gridzero.batched import FAR
from gridzero.core import (
    Adjacency,
    LatticeSpace,
    NeighborOperator,
    icosphere,
    in_front,
    path,
    reach,
    sight,
    snap,
    steps,
    within,
)

SPACES = {
    "squares": lambda: LatticeSpace((7, 6), periodic=False),
    "hexes on a torus": lambda: LatticeSpace((8, 7), tiling="hex"),
    "triangles": lambda: LatticeSpace((8, 6), tiling="triangle", periodic=False),
    "planet": lambda: icosphere(3),
}

BOARDS = {
    "hexes": lambda: LatticeSpace((9, 9), tiling="hex", periodic=False),
    "squares": lambda: LatticeSpace((8, 7), periodic=False),
    "hexes on a torus": lambda: LatticeSpace((9, 8), tiling="hex"),
    "squares, wrapped one way": lambda: LatticeSpace((9, 6), periodic=(True, False)),
    "a small board": lambda: LatticeSpace((3, 4), tiling="hex", periodic=False),
}


@pytest.fixture(params=list(SPACES))
def space(request):
    return SPACES[request.param]()


@pytest.fixture(params=list(BOARDS))
def board(request):
    return BOARDS[request.param]()


def counted(found):
    """`core.steps`' costs as `batched` counts them: int32, `FAR` for inf."""
    return np.where(np.isfinite(found), found, FAR).astype(np.int32)


# Moves on any space.


@pytest.mark.parametrize("limit", [None, 0, 2, 5])
def test_steps_are_cores_around_blocked_cells(space, limit, rng):
    op = batched.operator(Adjacency().compile(space))
    for share in (0.0, 0.3, 0.6):
        blocked = rng.random(space.n_cells) < share
        sources = np.zeros(space.n_cells, bool)
        sources[rng.choice(space.n_cells, size=2, replace=False)] = True  # blocked or not
        want = steps(space, sources, blocked=blocked, limit=np.inf if limit is None else limit)
        found = batched.steps(op, sources, blocked=blocked, limit=limit)
        assert found.dtype == np.int32
        np.testing.assert_array_equal(found, counted(want))


def test_steps_without_blocked_cells_or_sources(space):
    op = batched.operator(Adjacency().compile(space))
    one = np.arange(space.n_cells) == 3
    np.testing.assert_array_equal(batched.steps(op, one), counted(steps(space, one)))
    none = np.zeros(space.n_cells, bool)
    assert (np.asarray(batched.steps(op, none)) == FAR).all()


@pytest.mark.parametrize("budget", [None, 0, 3])
def test_reach_is_cores_and_a_flood_without_a_budget(space, budget, rng):
    op = batched.operator(Adjacency().compile(space))
    blocked = rng.random(space.n_cells) < 0.45
    sources = np.arange(space.n_cells) == int(np.flatnonzero(~blocked)[0])
    want = reach(space, sources, np.inf if budget is None else budget, blocked=blocked)
    found = batched.reach(op, sources, budget, blocked=blocked)
    assert found.dtype == np.bool_
    np.testing.assert_array_equal(found, want)


def test_moves_go_from_a_cell_to_its_neighbors_not_back():
    chain = NeighborOperator.from_table(np.array([[1], [2], [3], [-1]]))  # 0 -> 1 -> 2 -> 3
    op = batched.operator(chain)
    assert batched.steps(op, np.arange(4) == 0).tolist() == [0, 1, 2, 3]
    assert batched.steps(op, np.arange(4) == 2).tolist() == [FAR, FAR, 0, 1]


def test_a_batch_of_searches_is_each_alone(space, rng):
    op = batched.operator(Adjacency().compile(space))
    blocked = rng.random((5, space.n_cells)) < 0.3
    sources = np.zeros((5, space.n_cells), bool)
    sources[np.arange(5), rng.integers(0, space.n_cells, 5)] = True
    search = jax.jit(jax.vmap(lambda s, b: batched.steps(op, s, blocked=b)))
    for row, s, b in zip(search(sources, blocked), sources, blocked, strict=True):
        np.testing.assert_array_equal(row, counted(steps(space, s, blocked=b)))


def test_bad_limits_raise(not_an_integer):
    op = batched.operator(Adjacency().compile(LatticeSpace((3, 3))))
    sources = np.arange(9) == 0
    if not_an_integer is not None:  # None is the default: no limit
        with pytest.raises(TypeError):
            batched.steps(op, sources, limit=not_an_integer)
        with pytest.raises(TypeError):
            batched.reach(op, sources, not_an_integer)
    with pytest.raises(ValueError, match="limit"):
        batched.steps(op, sources, limit=-1)
    with pytest.raises(ValueError, match="budget"):
        batched.reach(op, sources, -1)


# Patches.


def cells_at(space, cell, offsets):
    """The cell at each offset from ``cell``, ``n_cells`` beyond the edge: an oracle for `at`."""
    coords = space.unravel(cell) + offsets
    shape, wraps = np.array(space.shape), np.array(space.periodic)
    coords = np.where(wraps, coords % shape, coords)
    inside = ((coords >= 0) & (coords < shape)).all(axis=-1)
    return np.where(inside, coords @ [space.shape[1], 1], space.n_cells)


def test_a_patch_holds_the_cells_within_its_radius_nearest_first(board):
    radius = 2.0
    tables = batched.patch(board, radius)
    assert tables.offsets[0].tolist() == [0, 0]
    assert tables.n_cells == board.n_cells
    assert tables.at.shape == (board.n_cells + 1, tables.size)
    assert tables.at.dtype == np.int32
    assert (np.diff(tables.distance) > -1e-12).all()
    at = np.asarray(tables.at)
    assert (at[-1] == board.n_cells).all()  # no cell: nowhere
    for cell in range(board.n_cells):
        np.testing.assert_array_equal(at[cell], cells_at(board, cell, tables.offsets))
        here = at[cell][at[cell] < board.n_cells]
        assert sorted(here.tolist()) == sorted({cell, *within(board, cell, radius).tolist()})


def test_within_counts_the_slots_of_a_smaller_radius():
    board = LatticeSpace((9, 9), tiling="hex", periodic=False)
    tables = batched.patch(board, 4.0)
    center = board.ravel([4, 4])
    for radius in (0.0, 1.0, np.sqrt(3), 2.0, 3.0, 4.0):
        count = tables.within(radius)
        assert count == len(within(board, center, radius))
        np.testing.assert_array_equal(tables.offsets[:count], batched.patch(board, radius).offsets)
    assert [tables.within(r) for r in (1.0, 2.0, 3.0)] == [7, 19, 37]


def test_beside_is_the_adjacent_slot_in_each_direction(board):
    tables = batched.patch(board, 2.5)
    directions = board.directions()
    rows = {tuple(o): i for i, o in enumerate(tables.offsets.tolist())}
    beside = np.asarray(tables.beside)
    assert beside.shape == (tables.size, len(directions))
    for i, offset in enumerate(tables.offsets):
        for d, step in enumerate(directions):
            assert beside[i, d] == rows.get(tuple((offset + step).tolist()), tables.size)


def test_slot_finds_offsets_and_marks_those_beyond(board):
    tables = batched.patch(board, 2.0)
    found = batched.slot(tables, tables.offsets)
    assert found.dtype == np.int32
    assert found.tolist() == list(range(tables.size))
    far = np.array([[tables.span + 1, 0], [0, -tables.span - 1], [tables.span, tables.span]])
    assert batched.slot(tables, far).tolist() == [tables.size] * 3
    assert batched.slot(tables, tables.offsets.reshape(1, -1, 2)).shape == (1, tables.size)


def test_turned_and_front_are_cores_directions(board):
    tables = batched.patch(board, 2.5)
    n = len(board.directions())
    for i, offset in enumerate(tables.offsets):
        assert np.asarray(tables.turned)[i].tolist() == [snap(board, offset, d) for d in range(n)]
    for d in range(n):
        np.testing.assert_array_equal(tables.front[d], in_front(board, d, tables.offsets))
    np.testing.assert_array_equal(tables.step, tables.offsets)


@pytest.mark.parametrize(
    ("shape", "tiling", "radius", "moves"),
    [
        ((15, 15), "hex", 3.0, 3),  # on hexes, three steps are within a distance of three
        ((15, 15), "hex", 1.0, 1),
        ((15, 15), "hex", 5.0, 5),
        ((15, 15), "hypercubic", 3.0, 3),
        ((15, 15), "hypercubic", 1.5, 1),  # the diagonal cells are two moves away
        ((3, 3), "hex", 5.0, 9),  # the whole board: as many as there are cells
    ],
)
def test_moves_is_how_far_a_piece_surely_stays_inside(shape, tiling, radius, moves):
    board = LatticeSpace(shape, tiling, periodic=False)
    tables = batched.patch(board, radius)
    assert tables.moves == moves
    center = board.n_cells // 2
    if moves < board.n_cells:
        inside = set(np.asarray(tables.at)[center].tolist())
        assert set(np.flatnonzero(steps(board, center) <= moves).tolist()) <= inside
        assert not set(np.flatnonzero(steps(board, center) <= moves + 1).tolist()) <= inside


def test_patches_need_squares_or_hexes_long_enough_to_wrap():
    with pytest.raises(ValueError, match="squares or hexes"):
        batched.patch(LatticeSpace((8, 6), tiling="triangle"), 1.0)
    with pytest.raises(ValueError, match="periodic axis"):
        batched.patch(LatticeSpace((6, 5), tiling="hex"), 3.0)
    with pytest.raises(TypeError, match="LatticeSpace"):
        batched.patch(icosphere(1), 1.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="radius"):
        batched.patch(LatticeSpace((5, 5)), -1.0)


# Sight.


def test_clear_is_one_viewers_sight_by_slot(board, rng):
    radius = 3.0 if min(board.shape) > 6 or not any(board.periodic) else 2.0
    tables = batched.patch(board, radius)
    at = np.asarray(tables.at)
    look = jax.jit(lambda walls, cell: batched.clear(tables, walls, cell))
    for share in (0.1, 0.4):
        walls = rng.random(board.n_cells) < share
        for cell in range(board.n_cells):
            found = np.asarray(look(walls, cell))
            assert found.dtype == np.bool_
            on = at[cell] < board.n_cells
            assert not found[~on].any()
            np.testing.assert_array_equal(
                found[on], sight(board, cell, walls, radius)[at[cell][on]]
            )
    assert not np.asarray(look(walls, board.n_cells)).any()  # no viewer


def test_sight_is_cores_for_viewers_that_see_differently_far(board, rng):
    radii = (1.0, 2.0)
    tables = batched.patch(board, max(radii))
    walls = rng.random(board.n_cells) < 0.3
    viewers = np.array([board.n_cells - 1, -1, 4, 0, 4])  # one slot empty, one cell twice
    far = rng.integers(0, 2, size=viewers.size)
    count = np.array([tables.within(radii[k]) for k in far])
    want = np.zeros(board.n_cells, bool)
    for viewer, k in zip(viewers.tolist(), far.tolist(), strict=True):
        if viewer >= 0:
            want |= sight(board, viewer, walls, radii[k])
    found = batched.sight(tables, walls, viewers, count)
    assert found.dtype == np.bool_
    np.testing.assert_array_equal(found, want)
    live = viewers[viewers >= 0]
    np.testing.assert_array_equal(
        batched.sight(tables, walls, viewers), sight(board, live, walls, max(radii))
    )
    np.testing.assert_array_equal(
        batched.sight(tables, walls, viewers, tables.within(1.0)), sight(board, live, walls, 1.0)
    )
    assert not np.asarray(batched.sight(tables, walls, np.array([-1, -1]))).any()


# Moves around a cell.


@pytest.mark.parametrize("moves", [0, 1, 2])
def test_steps_around_are_cores_within_the_moves(board, moves, rng):
    tables = batched.patch(board, 2.0)
    moves = min(moves, tables.moves)
    at = np.asarray(tables.at)
    search = jax.jit(lambda cell, blocked: batched.steps_around(tables, cell, blocked, moves))
    for share in (0.0, 0.35):
        blocked = rng.random(board.n_cells) < share
        for cell in range(board.n_cells):
            found = np.asarray(search(cell, blocked))
            assert found.dtype == np.int32
            want = counted(steps(board, cell, blocked=blocked, limit=moves))
            on = at[cell] < board.n_cells
            assert (found[~on] == FAR).all()
            np.testing.assert_array_equal(found[on], want[at[cell][on]])
            reached = at[cell][found < FAR]  # no cell of the board is missed
            assert sorted(reached.tolist()) == np.flatnonzero(want < FAR).tolist()


def test_steps_around_with_costs_are_cores(rng):
    board = LatticeSpace((9, 9), tiling="hex", periodic=False)
    tables = batched.patch(board, 3.0)
    at = np.asarray(tables.at)
    for trial in range(40):
        blocked = rng.random(board.n_cells) < 0.25
        cost = rng.choice([20, 21, 23], size=board.n_cells)  # no cheaper way is longer
        cell = int(rng.integers(board.n_cells))
        found = np.asarray(batched.steps_around(tables, cell, blocked, 3, cost))
        want = steps(board, cell, cost=cost, blocked=blocked, limit=3 * 23)
        want = np.where(steps(board, cell, blocked=blocked) <= 3, want, np.inf)
        on = at[cell] < board.n_cells
        np.testing.assert_array_equal(found[on], counted(want)[at[cell][on]], str(trial))


@pytest.mark.parametrize("costs", [None, (20, 21), (5, 5)])
def test_paths_around_are_cores_with_its_ties(board, costs, rng):
    tables = batched.patch(board, 2.0)
    moves = min(2, tables.moves)
    at = np.asarray(tables.at)
    walk = jax.jit(
        lambda cell, goal, blocked, cost: batched.path_around(
            tables, cell, goal, blocked, moves, None if costs is None else cost
        )
    )
    checked = 0
    for share in (0.0, 0.3):
        blocked = rng.random(board.n_cells) < share
        cost = np.ones(board.n_cells, int) if costs is None else rng.choice(costs, board.n_cells)
        within_moves = None
        for cell in range(board.n_cells):
            within_moves = steps(board, cell, blocked=blocked) <= moves
            for goal in range(tables.size):
                route, length = walk(cell, goal, blocked, cost)
                route, length = np.asarray(route), int(length)
                assert route.dtype == np.int32
                target = at[cell, goal]
                if target == board.n_cells or not within_moves[target]:
                    assert length == -1
                    assert route.tolist() == [0] * (moves + 1)
                    continue
                want = path(
                    board, cell, int(target), cost=None if costs is None else cost, blocked=blocked
                )
                assert length == len(want) - 1
                assert at[cell][route[: length + 1]].tolist() == want.tolist(), (cell, goal)
                assert (route[length:] == goal).all()  # it stays at the goal
                checked += 1
    assert checked > board.n_cells


def test_a_goal_too_far_or_walled_in_is_not_reached():
    board = LatticeSpace((7, 7), tiling="hex", periodic=False)
    tables = batched.patch(board, 3.0)
    center = board.ravel([3, 3])
    open_ = np.zeros(49, bool)
    far = int(batched.slot(tables, np.array([3, 0])))
    route, length = batched.path_around(tables, center, far, open_, 2)
    assert (int(length), route.tolist()) == (-1, [0, 0, 0])
    route, length = batched.path_around(tables, center, far, open_, 3)
    assert int(length) == 3
    walls = np.zeros(49, bool)
    walls[np.asarray(tables.at)[center, 1:7]] = True  # shut in
    assert int(batched.path_around(tables, center, far, walls, 3)[1]) == -1
    assert batched.steps_around(tables, center, walls, 3).tolist() == [0] + [FAR] * 36
    route, length = batched.path_around(tables, center, 0, walls, 3)
    assert (int(length), route.tolist()) == (0, [0, 0, 0, 0])  # to itself: no move


def test_moves_beyond_the_patch_raise(not_an_integer):
    tables = batched.patch(LatticeSpace((9, 9), tiling="hex", periodic=False), 2.0)
    blocked = np.zeros(81, bool)
    with pytest.raises(ValueError, match="at most 2"):
        batched.steps_around(tables, 40, blocked, 3)
    with pytest.raises(ValueError, match="moves"):
        batched.path_around(tables, 40, 1, blocked, -1)
    with pytest.raises(TypeError):
        batched.steps_around(tables, 40, blocked, not_an_integer)
    with pytest.raises(TypeError):
        batched.path_around(tables, 40, 1, blocked, not_an_integer)
