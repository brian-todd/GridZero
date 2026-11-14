"""Moving through cells: steps, reach, path and hops, on every space and both implementations."""

import itertools
import math

import numpy as np
import pytest

from gridzero.core import (
    AdjacencyKind,
    Hops,
    LatticeSpace,
    hops,
    icosphere,
    path,
    reach,
    steps,
)

pytestmark = pytest.mark.usefixtures("each_impl")

KINDS = ["facet", "vertex"]


def random_costs(space, seed, walls=0.2):
    """Costs between 0.5 and 4, with some cells impassable: rounding makes sums differ."""
    rng = np.random.default_rng(seed)
    cost = rng.uniform(0.5, 4.0, space.n_cells)
    cost[rng.random(space.n_cells) < walls] = np.inf
    return cost


def relaxed(space, sources, kind, cost):
    """Cheapest costs by relaxing every edge until nothing changes (Bellman-Ford)."""
    graph = space.adjacency(kind)
    step = np.ones(space.n_cells) if cost is None else cost
    dist = np.full(space.n_cells, np.inf)
    dist[sources] = 0.0
    while True:
        through = np.full(space.n_cells, np.inf)
        np.minimum.at(through, graph.indices, dist[graph.sources] + step[graph.indices])
        new = np.minimum(dist, through)
        if np.array_equal(new, dist):
            return dist
        dist = new


@pytest.mark.parametrize("kind", KINDS)
def test_steps_count_the_hops_to_each_cell(any_space, kind):
    hops = Hops(3, kind).compile(any_space).graph
    for source in (0, any_space.n_cells // 2):
        found = steps(any_space, source, kind=kind)
        assert found.dtype == np.float64
        expected = np.full(any_space.n_cells, np.inf)
        expected[hops.neighbors(source)] = hops.edge_data["hops"][
            hops.indptr[source] : hops.indptr[source + 1]
        ]
        expected[source] = 0.0
        near = expected <= 3
        np.testing.assert_array_equal(found[near], expected[near])
        assert (found[~near] > 3).all()


@pytest.mark.parametrize(
    ("kind", "distance"),
    [
        ("facet", lambda d: np.abs(d).sum(axis=-1)),  # Manhattan
        ("vertex", lambda d: np.abs(d).max(axis=-1)),  # Chebyshev
    ],
)
def test_steps_on_squares_are_manhattan_or_chebyshev_distances(kind, distance):
    space = LatticeSpace((7, 5), periodic=False)
    start = space.ravel([2, 3])
    expected = distance(space.coords - [2, 3]).astype(np.float64)
    np.testing.assert_array_equal(steps(space, start, kind=kind), expected)


def test_steps_on_hexes_are_hex_distances():
    space = LatticeSpace((6, 7), tiling="hex", periodic=False)
    dq, dr = (space.coords - [2, 3]).T
    expected = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2
    np.testing.assert_array_equal(steps(space, space.ravel([2, 3])), expected)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_steps_with_costs_match_relaxing_every_edge_bit_for_bit(any_space, kind, seed):
    cost = random_costs(any_space, seed)
    sources = np.random.default_rng(seed).choice(any_space.n_cells, 2, replace=False)
    expected = relaxed(any_space, sources, kind, cost)
    found = steps(any_space, sources, kind=kind, cost=cost)
    assert np.array_equal(found.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("kind", KINDS)
def test_every_reached_cell_is_entered_from_a_cell_as_cheap(any_space, kind):
    cost = random_costs(any_space, 7)
    dist = steps(any_space, 0, kind=kind, cost=cost)
    graph = any_space.adjacency(kind)
    through = dist[graph.sources] + cost[graph.indices]
    assert (dist[graph.indices] <= through).all()  # no edge offers a cheaper way
    tight = np.zeros(any_space.n_cells, bool)
    tight[graph.indices[through == dist[graph.indices]]] = True
    reached = np.isfinite(dist)
    reached[0] = False
    assert tight[reached].all()


def test_impassable_cells_are_unreachable_but_a_source_on_one_costs_nothing():
    space = LatticeSpace((5, 1), periodic=False)
    cost = np.array([np.inf, 1.0, np.inf, 1.0, 1.0])
    np.testing.assert_array_equal(steps(space, 0, cost=cost), [0, 1, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(steps(space, 2, cost=cost), [np.inf, 1, 0, 1, 2])


@pytest.mark.parametrize("weighted", [False, True])
def test_blocked_cells_are_the_same_as_infinite_costs(any_space, weighted):
    rng = np.random.default_rng(8)
    walls = rng.random(any_space.n_cells) < 0.25
    walls[[0, 1]] = False
    cost = random_costs(any_space, 9, walls=0.0) if weighted else np.ones(any_space.n_cells)
    inf = np.where(walls, np.inf, cost)
    given = cost if weighted else None
    np.testing.assert_array_equal(
        steps(any_space, 0, cost=given, blocked=walls), steps(any_space, 0, cost=inf)
    )
    np.testing.assert_array_equal(
        reach(any_space, 0, 3.0, cost=given, blocked=walls), reach(any_space, 0, 3.0, cost=inf)
    )
    goal = any_space.n_cells - 1
    np.testing.assert_array_equal(
        path(any_space, 0, goal, cost=given, blocked=walls), path(any_space, 0, goal, cost=inf)
    )


def test_a_source_on_a_blocked_cell_costs_nothing_and_blocked_wins_over_cost():
    space = LatticeSpace((5, 1), periodic=False)
    walls = np.array([False, False, True, False, False])
    cost = np.array([1.0, 1.0, 0.5, 2.0, 2.0])
    np.testing.assert_array_equal(steps(space, 0, blocked=walls), [0, 1, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(steps(space, 2, cost=cost, blocked=walls), [2, 1, 0, 2, 4])
    np.testing.assert_array_equal(path(space, 0, 4, blocked=walls), np.empty(0, np.int64))


@pytest.mark.parametrize(
    ("blocked", "error"),
    [(np.zeros(25, np.int64), TypeError), (np.zeros(24, bool), ValueError), (True, ValueError)],
)
def test_bad_blocked_masks_raise(blocked, error):
    space = LatticeSpace((5, 5))
    for call in (
        lambda: steps(space, 0, blocked=blocked),
        lambda: reach(space, 0, 1, blocked=blocked),
        lambda: path(space, 0, 3, blocked=blocked),
    ):
        with pytest.raises(error, match="blocked"):
            call()


@pytest.mark.parametrize("limit", [0.0, 2.0, 2.5, 7.0])
def test_a_limit_keeps_the_cells_within_it_and_drops_the_rest(any_space, limit):
    cost = random_costs(any_space, 3, walls=0.0)
    full = steps(any_space, 1, cost=cost)
    expected = np.where(full <= limit, full, np.inf)
    np.testing.assert_array_equal(steps(any_space, 1, cost=cost, limit=limit), expected)


def test_a_limit_includes_cells_exactly_at_it():
    space = LatticeSpace((6, 1), periodic=False)
    np.testing.assert_array_equal(steps(space, 0, limit=3), [0, 1, 2, 3, np.inf, np.inf])


def test_several_sources_give_the_nearest_ones_cost(any_space):
    cost = random_costs(any_space, 4)
    sources = [0, any_space.n_cells - 1, any_space.n_cells // 3]
    rows = steps(any_space, sources, cost=cost, separate=True)
    assert rows.shape == (3, any_space.n_cells)
    for row, source in zip(rows, sources, strict=True):
        np.testing.assert_array_equal(row, steps(any_space, source, cost=cost))
    np.testing.assert_array_equal(steps(any_space, sources, cost=cost), rows.min(axis=0))


def test_sources_may_be_a_mask_or_repeat():
    space = LatticeSpace((5, 5))
    mask = np.zeros(space.n_cells, bool)
    mask[[3, 17]] = True
    expected = steps(space, [3, 17])
    np.testing.assert_array_equal(steps(space, mask), expected)
    np.testing.assert_array_equal(steps(space, [17, 3, 17]), expected)
    assert steps(space, np.int8(3))[3] == 0


def test_no_sources_reach_nothing():
    space = LatticeSpace((4, 4))
    np.testing.assert_array_equal(steps(space, []), np.full(16, np.inf))
    np.testing.assert_array_equal(steps(space, np.zeros(16, bool)), np.full(16, np.inf))
    assert steps(space, [], separate=True).shape == (0, 16)


@pytest.mark.parametrize("budget", [0, 1, 2.5, math.inf])
def test_reach_is_the_cells_within_the_budget(any_space, budget):
    cost = random_costs(any_space, 5)
    sources = [0, 2]
    dist = steps(any_space, sources, cost=cost)
    expected = np.isfinite(dist) & (dist <= budget)
    found = reach(any_space, sources, budget, cost=cost)
    assert found.dtype == np.bool_
    np.testing.assert_array_equal(found, expected)
    assert found[sources].all()


def test_reach_on_squares_is_a_diamond():
    space = LatticeSpace((7, 7), periodic=False)
    found = space.grid_view(reach(space, space.ravel([3, 3]), 2))
    expected = np.abs(np.indices((7, 7)) - 3).sum(axis=0) <= 2
    np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_path_is_adjacent_cells_whose_costs_add_up_to_the_steps(any_space, kind, seed):
    cost = random_costs(any_space, seed, walls=0.1)
    rng = np.random.default_rng(seed)
    graph = any_space.adjacency(kind)
    for start, goal in rng.integers(0, any_space.n_cells, (5, 2)):
        dist = steps(any_space, start, kind=kind, cost=cost)
        cells = path(any_space, start, goal, kind=kind, cost=cost)
        assert cells.dtype == np.int64
        if np.isinf(dist[goal]):
            assert cells.size == 0
            continue
        assert (cells[0], cells[-1]) == (start, goal)
        for a, b in itertools.pairwise(cells):
            assert b in graph.neighbors(a)
        total = np.cumsum(cost[cells[1:]])[-1] if cells.size > 1 else 0.0
        assert total == dist[goal]  # the costs in the order they are paid


def test_a_path_steps_back_to_the_lowest_cell_each_time():
    # From (2, 2) back to (0, 0) on squares: (1, 2) and (2, 1) are as near, and (1, 2) is the
    # lower id; then (0, 2) before (1, 1). The path runs along axis 1 first, then axis 0.
    space = LatticeSpace((5, 5), periodic=False)
    cells = path(space, 0, space.ravel([2, 2]))
    np.testing.assert_array_equal(space.unravel(cells), [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])


def test_a_path_goes_around_walls():
    space = LatticeSpace((5, 3), periodic=False)
    cost = np.ones(space.n_cells)
    cost[space.ravel([[1, 0], [1, 1]])] = np.inf
    cells = path(space, space.ravel([0, 0]), space.ravel([2, 0]), cost=cost)
    np.testing.assert_array_equal(
        space.unravel(cells), [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0]]
    )


def test_a_path_to_itself_is_one_cell_and_to_an_unreachable_cell_none():
    space = LatticeSpace((4, 1), periodic=False)
    np.testing.assert_array_equal(path(space, 2, 2), [2])
    cost = np.array([1.0, np.inf, 1.0, 1.0])
    assert path(space, 0, 3, cost=cost).size == 0
    assert path(space, 0, 1, cost=cost).size == 0


def test_a_path_wraps_around_a_periodic_axis():
    space = LatticeSpace((8, 1), periodic=(True, False))
    np.testing.assert_array_equal(path(space, 1, 6), [1, 0, 7, 6])


def test_a_path_on_the_sphere_reaches_the_far_side():
    space = icosphere(4)
    goal = int(np.argmin(space.centers @ space.centers[0]))
    cells = path(space, 0, goal)
    assert len(cells) - 1 == steps(space, 0)[goal]


def test_a_path_refuses_costs_that_rounding_hides():
    space = LatticeSpace((3,), periodic=False)
    with pytest.raises(ValueError, match="too small"):
        path(space, 0, 2, cost=[1.0, 1e20, 1e-10])


@pytest.mark.parametrize(
    ("cost", "error"),
    [
        (np.ones(3, bool), TypeError),
        (np.array(["1"] * 25), TypeError),
        (np.ones(24), ValueError),
        (np.ones((25, 1)), ValueError),
        (np.r_[np.ones(24), 0.0], ValueError),
        (np.r_[np.ones(24), -1.0], ValueError),
        (np.r_[np.ones(24), np.nan], ValueError),
    ],
)
def test_bad_costs_raise(cost, error):
    space = LatticeSpace((5, 5))
    for call in (
        lambda: steps(space, 0, cost=cost),
        lambda: reach(space, 0, 1, cost=cost),
        lambda: path(space, 0, 3, cost=cost),
    ):
        with pytest.raises(error):
            call()


def test_integer_costs_are_accepted():
    space = LatticeSpace((4,), periodic=False)
    np.testing.assert_array_equal(steps(space, 0, cost=np.array([1, 2, 3, 4])), [0, 2, 5, 9])


def test_bad_limits_and_budgets_raise(not_a_real):
    space = LatticeSpace((5, 5))
    with pytest.raises(TypeError, match="limit"):
        steps(space, 0, limit=not_a_real)
    with pytest.raises(TypeError, match="budget"):
        reach(space, 0, not_a_real)
    for bad in (-1.0, np.nan, -np.inf):
        with pytest.raises(ValueError, match="limit"):
            steps(space, 0, limit=bad)
        with pytest.raises(ValueError, match="budget"):
            reach(space, 0, bad)


def test_bad_sources_raise(not_cell_ids):
    space = LatticeSpace((5, 5))
    values, error = not_cell_ids
    if values.dtype == np.bool_:
        error = ValueError  # a mask of the wrong shape
    with pytest.raises(error):
        steps(space, values)
    with pytest.raises(IndexError):
        steps(space, [0, 25])
    with pytest.raises(IndexError):
        steps(space, -1)
    with pytest.raises(ValueError, match="1-D"):
        steps(space, [[0, 1]])


def test_bad_ends_of_a_path_raise(not_an_integer):
    space = LatticeSpace((5, 5))
    with pytest.raises(TypeError):
        path(space, not_an_integer, 0)
    with pytest.raises(TypeError):
        path(space, 0, not_an_integer)
    with pytest.raises(IndexError):
        path(space, 0, 25)


def test_bad_kinds_raise():
    space = LatticeSpace((5, 5))
    with pytest.raises(ValueError, match="adjacency kind"):
        steps(space, 0, kind="edge")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="adjacency kind"):
        path(space, 0, 1, kind=2)


HOP_LATTICES = [
    ((6, 5), "hypercubic", False),
    ((6, 5), "hypercubic", (True, False)),
    ((4, 3, 5), "hypercubic", (True, False, True)),
    ((3,), "hypercubic", True),
    ((7, 5), "hex", False),
    ((4, 9), "hex", True),
    ((9, 3), "hex", True),  # a thin torus with slanted periods
    ((3, 8), "hex", (True, False)),
    ((7, 5), "triangle", False),
    ((6, 4), "triangle", True),
    ((7, 4), "triangle", (False, True)),
    ((6, 5), "triangle", (True, False)),
]


@pytest.mark.parametrize(("shape", "tiling", "periodic"), HOP_LATTICES)
def test_hops_between_every_pair_are_the_steps_of_a_search(shape, tiling, periodic):
    space = LatticeSpace(shape, tiling=tiling, periodic=periodic)
    ids = np.arange(space.n_cells)
    kinds: list[AdjacencyKind] = ["facet", "vertex", *range(space.dim)]
    for kind in kinds:
        found = hops(space, ids[:, None], ids, kind=kind)
        assert found.dtype == np.int64
        np.testing.assert_array_equal(found, steps(space, ids, kind=kind, separate=True))


def test_hops_on_a_thin_hex_torus_take_the_shortest_way_round():
    space = LatticeSpace((3, 56), tiling="hex")
    ids = np.arange(space.n_cells)
    np.testing.assert_array_equal(hops(space, 0, ids), steps(space, 0))
    np.testing.assert_array_equal(hops(space, 100, ids), steps(space, 100))


def test_hops_on_squares_are_manhattan_or_chebyshev_distances_across_the_edges():
    space = LatticeSpace((7, 9), periodic=(True, False))
    a, b = space.ravel([[1, 2], [0, 0]]), space.ravel([[6, 7], [4, 8]])
    np.testing.assert_array_equal(hops(space, a, b), [2 + 5, 3 + 8])
    np.testing.assert_array_equal(hops(space, a, b, kind="vertex"), [5, 8])


def test_hops_broadcast_ids_and_are_symmetric():
    space = LatticeSpace((6, 4), tiling="triangle")
    a = np.array([[0], [5], [23]])
    b = np.array([1, 7, 12, 23])
    found = hops(space, a, b)
    assert found.shape == (3, 4)
    np.testing.assert_array_equal(found, hops(space, b[:, None], a[:, 0]).T)
    assert hops(space, 5, 5).shape == ()
    assert hops(space, 5, 5) == 0


def test_hops_count_the_steps_hops_neighborhoods_use():
    space = LatticeSpace((7, 6), tiling="hex", periodic=False)
    graph = Hops(3).compile(space).graph
    found = hops(space, graph.sources, graph.indices)
    np.testing.assert_array_equal(found, graph.edge_data["hops"])


def test_hops_on_empty_ids_are_empty():
    space = LatticeSpace((5, 5), tiling="hex")
    assert hops(space, np.empty(0, np.int64), 3).shape == (0,)


def test_bad_hops_raise(not_cell_ids):
    space = LatticeSpace((5, 5))
    values, error = not_cell_ids
    if values.dtype == np.bool_:
        error = TypeError
    with pytest.raises(error):
        hops(space, values, 0)
    with pytest.raises(error):
        hops(space, 0, values)
    with pytest.raises(IndexError):
        hops(space, 0, 25)
    with pytest.raises(TypeError, match="LatticeSpace"):
        hops(icosphere(1), 0, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="adjacency kind"):
        hops(space, 0, 1, kind=2)


@pytest.mark.parametrize("shape", [(9, 9), (15, 15)])
def test_a_path_with_unit_costs_is_the_path_with_no_costs(shape):
    # Moves leave out the costs when nothing is watched, so the two must never pick different ways.
    space = LatticeSpace(shape, tiling="hex", periodic=False)
    rng = np.random.default_rng(0)
    for _ in range(200):
        blocked = rng.random(space.n_cells) < rng.choice([0.05, 0.2, 0.35])
        a, b = rng.choice(space.n_cells, 2, replace=False)
        blocked[[a, b]] = False
        plain = path(space, int(a), int(b), blocked=blocked)
        costed = path(space, int(a), int(b), cost=np.ones(space.n_cells), blocked=blocked)
        np.testing.assert_array_equal(plain, costed)
