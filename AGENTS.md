# AGENTS.md

Guidance for anyone working in this repository, human or AI agent. `CLAUDE.md` imports this
file, so keep shared conventions here.

## Project

**GridZero** (package `gridzero`): agents that learn games by self-play, and the simulations
that share their engine, on grids of arbitrary polygonal or polytope cells: flat n-D lattices,
hexagonal and triangular grids, and spherical (planet) meshes. Each cell has rich state,
neighborhoods can be arbitrary, and updates are driven by actions. The flagship is the games,
their self-play agents and the simulations, as decided on 2026-10-29.

This is a research playground. Correctness, clarity and reproducibility come first. Performance
comes second but is taken seriously.

**Current focus:** the `core` package (`src/gridzero/core/`), which covers
grids, neighborhoods, state, actions and the update loop. Beside it, `worlds` and `render`
build planets, terrain and pictures from `core`. `game` holds the game layer beside `core`,
`agents` what plays any game: the AlphaZero search, network and training, and LLM players,
and `examples/games/` four games on them: `tactics`, a two-player skirmish on hexes,
`in_a_row`, from tic-tac-toe to Connect Four, `voronoi`, stones placed on a planet by any
number of players, and `colonies`, orders on a planet under fog of war, resolved by `core`'s
loop. `game` plays any number of players on any space since
2026-10-30, the last change planned before calling the API stable; the learners in `agents`
are for two players on 2-D lattices. There are no notebooks for now: the expansion model and
both notebooks were set aside on 2026-10-18, to come back to.

## Layout

```
src/gridzero/core/    spaces, regions, random points, queries (paths, ranges, sight, directions),
                      neighborhoods, operators, fields, intents, the loop
src/gridzero/game/    games beside simulations: the Game and Player protocols, Position, play,
                      series, records, replay, tally, ladder, fairness, board helpers (boards.py)
src/gridzero/agents/  players of any game: a uniform one, Observable, Revealable, Learnable and
                      Frame, the AlphaZero search (zero.py), its network (net.py) and training
                      (zero_train.py), MuZero's world model (muzero.py, muzero_net.py,
                      muzero_train.py), imitation (imitate.py), PPO (ppo.py), LLM players
                      (llm.py), exact values of small games (exact.py); Torch only in the
                      networks and the training modules, which the package doesn't import
src/gridzero/worlds/  Planet (planet.py), optional terrain (terrain.py), noise and relief
                      (noise.py)
src/gridzero/         render.py (globes, maps, flat maps, cell polygons, text)
examples/             whole models on the package, outside it: games/ (tactics/, in_a_row/,
                      voronoi/, colonies/) and simulations/ (competition, migration, predators,
                      biomes, settlers, population_evolution), each simulation a core.Model: a
                      frozen dataclass of parameters with start(space, seed) and phases(). Each
                      has a README.md and its experiments beside it: a simulation's
                      experiment.py, a game's experiments/; games/experiments/ holds those over
                      any game's output (games.curves), games/browse.py and games/curves.py
                      what they share; README.md lists them all
run.py                runs any experiment: EXPERIMENTS names them all
tests/                mirrors src/ and examples/. Shared fixtures and factories go in
                      conftest.py, shared State classes in demo.py, the fake game and the checks
                      every game runs in game_demo.py, those every simulation's model runs in
                      model_demo.py, the test game of agents and check_learnable in
                      agents_demo.py, and helpers the Tactics tests import in tactics_demo.py
benchmarks/           timing scripts, run as modules (python -m benchmarks.bench): bench.py times
                      both paths and the step target
compose.yaml          the local LLM server for LLM players (llama.cpp, pinned image)
```

## Commands

```bash
uv sync                          # install the project plus dev tools (ruff, ty, pytest)
uv run pytest                    # full suite (exit code 5 means no tests collected)
uv run pytest -m "not slow"      # fast loop
uv run ruff check --fix .        # lint, with autofix
uv run ruff format .             # format
uv run ty check                  # type-check src/, examples/, tests/ and benchmarks/
uv run python run.py                             # list the experiments; run.py <name> [args] runs one
taskset -c 4 uv run python -m benchmarks.bench   # timings, pinned to one performance core
docker compose up -d             # the local LLM server (llama.cpp) for LLM players; `down` stops it
```

**Before calling work done:**
`uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest` must all pass.

- Python 3.15. Run Python only through `uv run`, never `pip install`.
- Add dependencies only with `uv add` or `uv add --dev`, and **ask before adding one**.
- Language models are reached over HTTP with the standard library (`agents/llm.py`),
  not a client package. The suite never needs the server: tests pass a fake `Chat`. No
  example has an LLM player since Tactics' was retired on 2026-10-31.
- We build on what is already locked: NumPy, SciPy, Numba, Matplotlib, pandas and
  scikit-learn. Torch and CUDA are available. GPU work in `core` is deferred; the networks of
  `agents` train on the GPU (RTX 3080, 10 GB: mind the memory) and play on the CPU.
  JupyterLab is a dev tool, kept for when notebooks return.

## Code style

- **Ruff** handles linting and formatting. The config is in `pyproject.toml`. Don't disable a rule inline without a comment
  saying why.
- **NumPy-style docstrings** on every public module, class and function, with Parameters,
  Returns and Raises sections. Ruff's `D417` checks that a Parameters section names every
  argument. Document parameters in the class docstring, not in `__init__`. Private helpers get
  at least a summary. **Always state the shape and dtype of arrays.**
- **Types everywhere in `src/` and `examples/`:** every function is annotated (ruff's `ANN`
  rules), arrays as `numpy.typing.NDArray[np.float32]`, and ty reports nothing, not even a
  warning. In `tests/` and `benchmarks/`, ty checks the code against the package's types, but
  functions need no annotations.
  - A parameter's type is what it accepts, and an attribute's what it holds. Integers checked by
    `check_int` or `check_cell` are `SupportsIndex`, so NumPy integers pass. A constructor that
    converts its inputs gets an explicit `__init__` (`@dataclass(init=False)`), not
    `__post_init__`, which would declare the converted type as the input's.
  - No bare `Any` where a type can be written, and no bare generic: `tuple[int, ...]`, not
    `tuple` (ty's `missing-type-argument`, turned on in `pyproject.toml`). No
    `# ty: ignore[rule]` without a comment saying why, and none without a rule. SciPy ships no
    type stubs, so calls into it are mostly unchecked.
  - Use `typing.Protocol` for extension points. Use frozen dataclasses for specs and
    configuration.
  - Type parameters use the `class C[T]` syntax, except on a generic frozen dataclass that must
    vary with its parameter (`Position`, `Record`, `Phase`): from Python 3.13 it gets a
    `__replace__` taking each field, so type checkers infer it invariant. Such a class declares
    `TypeVar(..., covariant=True)` (or `contravariant=True`) and subclasses `Generic`, with a
    comment saying why.
- Prefer small pure functions and composition. Avoid deep class hierarchies.
- Use ASCII in code and docstrings: write `x`, not `×`. Ruff flags characters that look alike.
- Naming:
  - `cells`/`ids` are flat `int64` cell ids, and `n_cells` is their count.
  - `idx`/`weight` are neighbor arrays. `indptr`/`indices`/`data` are CSR arrays.
  - `rng` is a `np.random.Generator` that the caller passes in.

```python
def pull(x: NDArray[np.float32], op: NeighborOperator) -> NDArray[np.float32]:
    """Aggregate neighbor values into each cell.

    Parameters
    ----------
    x : ndarray of float32, shape (..., n_cells)
        Per-cell values. Leading axes (components, batches) are broadcast.
    op : NeighborOperator
        Compiled neighborhood: CSR structure with per-edge weights.

    Returns
    -------
    ndarray of float32, shape (..., n_cells)
        ``y[..., i] = sum_j w_ij * x[..., j]``.
    """
```

## Core design rules (decided)

1. **Separate the layers:**
   - a **Space** holds the cells, their ids, geometry, adjacency and areas,
   - a **Neighborhood** is any method that maps cells to `(idx, weight)`,
   - the **State** is a set of typed fields,
   - **phases** emit actions or intents,
   - **resolvers** apply intents and enforce invariants.
2. **Flat cell ids** (`int64`, `0..n_cells-1`) identify cells everywhere. Fields are stored as
   `(*leading, n_cells)`, cells last. Leading axes are components such as groups or attributes,
   and later batches. Lattices expose a structured grid view by reshaping, without copying.
3. **Neighborhoods** compile to a `NeighborOperator`: CSR neighbors with weights, plus a cached
   transpose. Missing neighbors are `-1` in padded tables, and simply absent in CSR.
   - Every operation on operators, intents and action spaces has a NumPy/SciPy **reference
     path**. Fast paths (Numba kernels in `core/_kernels.py`) do the same arithmetic in the same
     order and **must give bit-identical results** (NaN payloads aside), which `test_kernels.py`
     checks bit for bit. `implementation("reference" | "numba" | "auto")` chooses between them.
   - To add a fast path: write the kernel in `core/_kernels.py`, branch on `fast()` in the
     operation and leave the reference code as it is, and add cases to `test_kernels.py`.
4. **Fields declare** dtype, component shape, static or dynamic, and **extensive** (amounts:
   conserved and summed) or **intensive** (densities: averaged).
   - Cell areas differ: about 2× between the smallest and largest icosphere cells. **Store
     amounts and compute densities** by dividing by area.
5. **Rules propose, the framework enforces.** Phases return intents: flows, sources, claims and
   assignments. Resolvers apply them and guarantee conservation, bounds and masks.
   - **Conserved quantities always move as fluxes** (a push through the transpose). Never move
     them by having each cell pull independently: two cells could both claim the same amount.
6. **Phases are synchronous** over the snapshot they read, and run in schedule order. This is
   operator splitting, so the order of phases is part of the model.
   - What a step means in time is up to the scenario. Rates are per step, and resolvers clip
     outflow fractions to at most 1.
7. **Determinism:** all randomness comes from one root seed, split with `np.random.SeedSequence`
   per step and per phase. The same seed must give bit-identical results.
   - Numba kernels reseed per fixed chunk of work, never per thread or per cell. Results then
     don't depend on the thread count.

## Performance: what we have measured

`benchmarks/bench.py` reproduces the current numbers.
**Measure before optimizing.**

- **Never loop over cells in Python in hot paths.** That costs about 1.3 µs per cell, roughly
  500× slower than the same loop in Numba.
- At our working sizes (up to about 40k cells), **Python call overhead dominates, not
  arithmetic.** Keep the number of operation calls per step low, preallocate buffers, and avoid
  temporaries inside loops.
- **The loop itself is cheap.** A phase costs about 0.4 µs plus its work. Each `Assign` copies
  its values into a new array: 0.6 µs for 10k booleans, 4–6 µs for 3×10k float64. `ctx.rng` costs
  7 µs on first use in a phase call, so touch it only when drawing.
- **NumPy:**
  - On large lattices of narrow dtypes (uint8 states at 10⁶ cells), shifted views of a padded
    array summed a neighborhood about 6× faster than a CSR product. For float data they lose
    (see "Lattice stencils" below).
  - Write rules as boolean or arithmetic expressions. 2-D fancy indexing such as
    `lut[s, n]` was 12× slower.
  - Scatter-add with `np.bincount`, **never `np.add.at`**, which was 10× slower.
  - For a sum over CSR rows, a SciPy sparse matvec was 6–8× faster than `np.add.reduceat` or
    `np.bincount` (28 µs against 187 µs at 10k cells).
  - Gather padded neighbor tables with `np.take`, then overwrite the `-1` slots with
    `np.copyto(..., where=...)`. Assigning through a boolean mask was about 8× slower.
  - Gather along the last axis with `np.take(x, idx, axis=-1)`. `x[..., idx]` was about 2×
    slower (80 µs against 38 µs for 60k edges).
  - Don't use `scipy.ndimage.convolve` or `uniform_filter`. They were 2–8× slower than views.
  - Watch accumulator dtypes: a `uint8` sum overflows once there are more than 255 neighbors.
  - Avoid broadcasting over a short last axis, such as the slots of a padded table: NumPy then
    loops over rows. On a `(3, 10242, 7)` table, `np.where(valid, x, 0)` took 650 µs against
    135 µs for `x * float_mask`, and `x.sum(axis=-1)` took 340 µs against 150 µs for adding the
    slots one at a time (which also keeps the sum order fixed, unlike a BLAS `x @ ones`).
- **Numba** is the CPU engine for logic per cell, per edge and per sample. It was 10–90× faster
  than NumPy for that kind of work.
  - `parallel=True` *loses* on small problems because of thread overhead. Keep serial variants
    and choose by size.
  - Use `cache=True`. Jitted functions can be passed as arguments; each distinct function
    compiles its own specialization. A call costs about 0.2 µs. A kernel version compiles in
    about 80 ms the first time and loads from the disk cache in about 15 ms after that.
  - **Index with unsigned integers.** Numba checks every signed index for wraparound: a CSR sum
    over int64 indices was 1.5× slower than SciPy, and over uint64 views of them it was faster.
    Never mix uint64 and int64 in index arithmetic: Numba promotes the pair to float64.
  - **Index arrays directly in inner loops.** Taking a row view (`w[c, i]`) per row counts
    references: the allocation fractions ran 3× slower that way.
  - **Call a kernel from Python, not from another jitted function,** when it matters: the same
    CSR kernel ran 35% slower inlined into a jitted driver.
  - **Avoid branches on data.** On random scores, `NeighborChoice.best` with a branch per
    comparison mispredicted and ran 1.8× slower than with selects. Write `a & (b == c)`, not
    `a and b == c`, which short-circuits into a branch. A branch that guards a write is
    another matter: in `choice_rates`, `and` and `&` ran equally fast.
  - **`max` and `min` break ties differently from NumPy.** Numba's `max(a, b)` keeps `a` when
    they are equal, so `max(-0.0, 0.0)` is -0.0. NumPy's float32 and float64 `maximum` keep the
    later operand (0.0), and its float16 `maximum` the earlier. Clip with a select,
    `w if w > 0.0 else 0.0`, to match `np.maximum(w, 0.0)`.
  - **Divide per cell, not per edge,** and use `error_model="numpy"` where nothing divides by 0
    (the default checks each divisor). A fused `Spread` kernel that divided on every edge took
    152 µs, slower than the reference's own order: divide per cell, then push.
  - **Kernel times vary by up to 30% between processes** for the same code and data, from
    memory addresses. On the i9-12900K, also pin benchmarks to a performance core
    (`taskset -c 4`): the efficiency cores run at 3.9 GHz.
- **Sparse:**
  - A Numba CSR sum with components in blocks of 4/3/2/1 (each edge read once per block) is
    about even with SciPy at one component and 2.3–3× faster at 2–8. SciPy copies a transposed
    input first. It is bit-identical to SciPy, which adds each row in CSR order from 0.
  - A serial Numba scatter in source order is fine at working sizes. `Flow` delivery took 83 µs
    that way, against 277 µs for gathering, multiplying, and a product with the target
    incidence matrix (f=32, 3 components).
  - `np.divide(..., where=mask)` is slow on scattered masks: rationing with half the cells short
    took 242 µs in NumPy and 25 µs in a kernel.
  - Do a push (transpose) by pulling through a **cached CSR transpose**. That was 4–8× faster
    than scattering.
  - Multiply by one contiguous vector at a time. `S @ r.T` with a transposed `(61440, 3)` view
    took 242 µs, because SciPy copies it first, against 3 × 29 µs for one row at a time. On
    `(k, n_cells)` blocks the two are close (100 against 91 µs at 3 rows, 196 against 241 µs at
    8), so `pull` keeps `S @ x.T`.
  - **SciPy sparse products overflow to inf silently,** even under `np.errstate(over="raise")`.
    Check their results when values can be huge.
- **Searches** (`steps`, `reach`, `path`; 200x200 hexes, 40k cells): SciPy's `dijkstra` and a
  Numba binary heap tie on a whole weighted search (2.8 against 2.6 ms). Numba wins where
  searches stop early or need no heap: a budget of 5 moves in 12-18 µs against 250-290 µs, a
  path to a nearby cell in 20 µs against 3 ms (SciPy can't stop at a goal), and a
  breadth-first search for unit costs (walls too) in 176 µs against 2.8 ms. `sight` within 10 cells takes
  70-120 µs, about 0.3 µs a line, against 25-43 ms on the reference path. Set-up repeated per
  call had cost more than the lines: cache what depends only on the space and a radius, and
  give many viewers one call (`sight(space, viewers, ...)`).
- **Lattice stencils don't pay for float data.** Shifted views of a padded array, and a Numba
  stencil over flat offsets, both lost or tied against the Numba CSR kernel at every size tried
  (101² to 1024², 32³), and they add neighbors in another order than the reference.
- **FFT and separable tricks** only apply to planar lattices with kernels that depend only on
  displacement. **They don't work on the sphere.**
- **GPU (Torch, RTX 3080) is deferred for `core`.** It is slower than the CPU on small grids.
  Networks are another matter: AlphaZero and MuZero train on it.

## Grids: facts worth knowing

- **Icosphere planet:**
  - Subdivide an icosahedron with frequency `f`, then use `SphericalVoronoi` for polygons and
    areas. Adjacency comes from shared polygon edges, as for every `MeshSpace`. The Delaunay
    triangulation (`scipy.spatial.ConvexHull` of the sites) is the independent test oracle.
  - `n_cells = 10*f**2 + 2`: exactly 12 pentagons, and the rest hexagons. Naive subdivision gives
    a largest-to-smallest area ratio of about 2.
  - Working sizes: `f=16` gives 2,562 cells (fast iteration), and `f=32` gives 10,242 (default).
    Both build in under 0.5 s.
  - `f=1` (12 cells) and `f=2` (42 cells) are good test fixtures.
  - The cell containing a point is the one with the nearest center, because the cells are the
    Voronoi cells of their centers. `MeshSpace.locate` finds it with a cached k-d tree, and
    `render` draws whole images this way: about 0.1 s for 512x512 pixels at `f=32`.
- **Adjacency from geometry:** cells are j-adjacent if they share a face of dimension ≥ j, which
  is the affine rank of their shared vertices.
  - On hypercubes, shared facets give von Neumann neighborhoods and shared vertices give Moore.
  - This assumes a face-to-face tiling.
- **Hex lattice:** a 2-D array in axial coordinates, with the six offsets `(±1, 0)`, `(0, ±1)`,
  `(1, -1)` and `(-1, 1)`. Each row starts half a cell further right, so the grid is a rhombus.
  A hex grid that wraps along x still has a rectangle as one period, and `flat_map` draws that.
- **Triangle lattice:** equilateral triangles in straight rows, so the grid is a rectangle.
  Cell `(i, j)` is in column `i` of row `j` and points up when `i + j` is even. Centers of
  neighbors are 1 apart, so edges are √3 long and each cell's area is 3√3/4 (the basis is
  `diag(√3/2, 3/2)`, one column and one row). A center lies 1/4 below `basis @ coords` (the
  middle of its row) when it points up, and 1/4 above when it points down.
  - The two orientations are the tiling's two **types** of cell. A triangle pointing down has
    the opposite offsets of one pointing up. Facet neighbors: `(±1, 0)` and `(0, -1)` from one
    pointing up. Corner neighbors span 5 columns and 3 rows.
  - Wrapped axes need an even length, so that the pattern continues across the edge, and at
    least 6 along axis 0 and 4 along axis 1.
  - Triangles are the Voronoi cells of their centers, so "locate is the nearest center" still
    holds.
- **Locating points on a lattice is exact:** `LatticeSpace.locate` rounds lattice coordinates
  (cube rounding on hexes: round `q`, `r` and `-q - r`, then recompute the one that moved most;
  on triangles, round to the row, then compare the height in the row with the distance from the
  column pointing up), wraps periodic axes, and gives -1 beyond a fixed edge.
- **Periodic axes** need `L >= 2*reach + 1` for anything built from offsets: lattice adjacency,
  `radius_graph`, stencils. Otherwise one neighbor is reached through two offsets and counted
  twice. `Hops` searches the adjacency graph instead, so it counts each cell once on any torus.
- **The nearest copy across wrapped axes:** rounding a step's coordinates finds it along
  orthogonal periods (squares, cubes, triangles). Hexes that wrap both ways have slanted
  periods: reduce them first (Lagrange and Gauss), then try the 9 copies around the rounded
  one. Without the reduction, on thin tori (hexes 3x56 or 56x3), the 9 copies missed the
  nearest by up to 1.6, for points within 12 of each other.
- **Lines on 2-D lattices are exact in integers.** Every center and corner is a lattice point
  of a frame (`_Frame` on the `_Tiling`): twice the coordinates on squares, three times the
  axial ones on hexes, and columns by quarters of a unit on triangles. The side of a line a
  corner lies on is then an integer cross product, so passing through a corner is settled
  exactly. On hexes, a line between centers toward a corner of a hexagon (offset `(2, -1)`,
  and the like) runs along the edges between two cells for its whole way: a common case, not
  a rare one.
- **Regular polygons tile the plane only as triangles, squares and hexagons.** Octagons and
  12-gons appear only beside smaller polygons, as in octagons with squares, and no convex polygon
  with 7 or more sides tiles the plane. Regular n-gons for any n tile only the hyperbolic plane.
  On mixed regular tilings, the cell holding a point is still the nearest center if each center
  is weighted by its cell's size: squared distance minus inradius squared. That was checked on
  octagons with squares, where plain nearest center was wrong for a third of the points.
- **Neighbor counts to use as test oracles:**

  | Grid | Counts |
  |---|---|
  | square | 4 (facet) / 8 (vertex) |
  | hex, by hop distance | 6, 12, 18, … |
  | cube | 6 / 18 / 26 (facet / edge / vertex) |
  | BCC | 14 |
  | FCC | 12 (facet) / 18 (vertex) |
  | triangles | 3 / 12 |
  | icosphere | degree 5 for exactly 12 cells, 6 for the rest |
  | cells at exactly k hops | square 4k (facet) / 8k (vertex), hex 6k, triangle 3k (facet) / 12k (vertex), cube 4k²+2 (facet), icosphere pentagon 5k for k ≤ f |
  | cells within a radius | square 4, 8, 12, 20, 28 at r = 1, √2, 2, √5, 3; hex 6, 12, 18, 30, 36 at r = 1, √3, 2, √7, 3; triangle 3, 9, 12, 18, 24 at r = 1, √3, 2, √7, 3 |

## Testing

- **pytest.** Tests mirror the source layout: `tests/core/test_<module>.py`. Warnings are
  errors. NumPy `RuntimeWarning`s (division by zero, invalid values) are bugs in a simulation,
  so handle them explicitly with `np.divide(..., where=...)` or `np.errstate`.
- **Build inputs with fixtures and factories, not monkeypatching.**
  - `tests/conftest.py` holds factory fixtures such as `make_lattice`, `make_icosphere`,
    `make_state` and `rng`. They return small, deterministic objects.
  - Prefer dependency injection: pass `rng`, spaces and operators explicitly, so tests never
    need `monkeypatch` or `unittest.mock`.
  - For a protocol, write a small fake implementation, for example a hand-written
    `Neighborhood`.
- **Mock data is tiny and checkable by hand:** 3×3 or 5×5 lattices, `f=1`/`f=2` icospheres, and
  fields with known totals.
- **Test behavior and invariants, not implementation details:**
  - **conservation:** totals before and after flows are equal within a stated tolerance,
  - **adjointness:** `dot(pull(x), y) == dot(x, push(y))`,
  - **grid structure:** adjacency is symmetric, degree counts are right, areas sum to the
    surface area,
  - **fast path equals reference:** test modules of operations with fast paths run every test
    on both paths, with `pytestmark = pytest.mark.usefixtures("each_impl")`. `test_kernels.py`
    compares the two paths' outcomes (results bit for bit, or the same error) on hard inputs,
  - **determinism:** the same seed gives identical arrays, and different seeds give different
    arrays,
  - **bounds:** no negative amounts after resolution, and no flow into masked cells.
- **Test at the edges, not only at typical values.** The milestone 2–3 review found bugs only
  there:
  - **dtype limits:** run every conversion, reduction and normalization over each numeric dtype
    at its smallest and largest values. The result must be exact or an error, never wrapped or
    infinite. The one exception is float64 arithmetic that overflows, which gives inf as
    NumPy does (`pull` of `1e308` values); a result that doesn't fit a narrower dtype still
    raises. Floats need subnormal values too (`1 / 1e-310` overflows), and integer scores
    need values past `2**53`, where float64 can't tell them apart.
  - **scale invariance:** where only ratios matter, as for allocation weights and attractions,
    multiply the input by 1e-310 and by values near the float64 maximum, and compare.
  - **bad inputs:** every parameter that takes an integer or cell ids gets the `not_an_integer`
    or `not_cell_ids` fixture from `conftest.py`. A line that passes a wrong type on purpose
    carries `# ty: ignore[rule]`, and the `pytest.raises` around it says why.
  - **error context:** errors raised while building or running a simulation name the phase,
    neighborhood or observer, and the step.
- **Test with inputs that can tell implementations apart.** The milestone 4 and 5 spot checks
  and reviews found bugs that the obvious inputs hide:
  - **directed operators:** on a symmetric operator `pull` equals `push`, so a missing transpose
    passes. Build one with `NeighborOperator.from_table` from random targets.
  - **intents together:** with one taker, rationing quietly repairs a missing clip. Combine a
    flow with a removal to see each rule on its own.
  - **adjustments together:** when two rules adjust one input, use inputs where their order
    changes the result. The domain test's rates added up to 0.75, so it never saw the domain
    applied before rates past 1 were scaled down.
  - **special values in random places:** NaN only in cell 0 is always a row's first neighbor,
    where even a max that drops NaN keeps it. The milestone 5 spot check found that.
  - **special values in every input:** `-0.0` went into `pull`'s values but not into
    allocation weights, so an allocation kernel that kept `-0.0` passed.
  - **an assertion as strict as the promise:** `assert_array_equal` treats `-0.0` and `0.0` as
    equal, and every NaN as equal. The milestone 5 tests planted `-0.0` and still passed two
    kernels that returned it where the reference returned `+0.0`. Compare bits
    (`float_bits` in `test_kernels.py`).
  - **work done in blocks:** test that the result does not depend on the block size, with
    blocks small enough to cross boundaries. Default lattice blocks hold tens of thousands of
    cells, so a bug that restarted each block at cell 0 passed every test, including the
    512x512 one. `LatticeSpace._offset_graph` takes `block=` for that.
  - **empty axes:** no cells, no edges and no components. `reshape(-1, 0)` cannot infer its
    size, and both paths of `pull` and allocation flows crashed on it. Two paths that fail the
    same way still agree, so also test that such calls succeed.
  - **every documented input form, on a path that succeeds:** a scalar, a `(1,)` and a `(k, 1)`
    array, compared with the full array. A test that raises before the code under test covers
    nothing: every scalar `Flow` rate in the milestone 4 tests sat in an error test, and the
    plain scalar crashed.
  - **moments, not only counts:** counts per cell (chi-square) and the share of points in disks
    around centers both passed boxes that cut 3% off each triangle's far corner. The mean of
    the points around their cell's center, and their mean squared distance from it, caught it.
- **Test the state a model claims to reach, not only that it moves.** The first expansion model
  let a fixed fraction leave every cell toward any room, however little. It spread nicely and
  conserved people, but its land never settled: cells stayed between 0.7 and 1.7 times their
  capacity, and 39% of the people moved every step. A test that the land fills to capacity
  found it.
- **Spot-check tests by planting bugs** (mutants), one at a time, after a milestone. Run each
  with `PYTHONDONTWRITEBYTECODE=1`. Python accepts cached bytecode when the source's size and
  its modification time in whole seconds match. A mutant of the same length, restored within
  the second, therefore leaves its bytecode behind for the next run. That made a passing test
  fail once in milestone 6.
- One behavior per test, named for that behavior, e.g.
  `test_flow_resolver_conserves_total_when_outflow_exceeds_stock`. Use
  `pytest.mark.parametrize` to run the same test across spaces and implementations.
- Keep the default suite fast. Mark expensive tests with `@pytest.mark.slow`. Pay the Numba
  compile cost once, in session-scoped fixtures.

## Design decisions

The rules below are the ones to keep in mind while writing code.

- **"There should be one-- and preferably only one --obvious way to do it."** This is the
  standard for the package's primitives and for how models, experiments and tests are written.
  Before adding something, check how the other examples already do it and do it the same way.
  A second way needs a reason, and a rule for when to use which.
  This is an experimental project, so the standard won't always hold, but we stay as close to
  it as we can. The README's "A first simulation" is the canonical model, and its "A first
  game" (`Stakes`) the canonical game.

- **Scope:** the planet (a spherical mesh) is the primary space. Flat lattices are test beds,
  general building blocks and game boards. Agents and the `Env` wrapper are shelved. There is
  no layer or stack axis, population attributes are generic field axes or fields, and what a
  time step means is up to the scenario.
- **Where code goes:** `core` holds what means only cells, spaces and fields. `worlds` holds
  content with a meaning: land, sea, climate, terrain colors, longitude and latitude.
  - Questions about cells that need only a space are **functions in `core`** that take and
    return cell ids or points, such as `regions`, `random_points` and the queries.
  - `worlds`, `render` and `examples` use `core`'s public API. `core` never imports them.
  - Model-specific notions, such as habitability, belong to the model, not to `worlds`.
  - **A game board is a `core.State` on a 2-D `LatticeSpace`.** No board class until one earns
    its place beyond its `State` subclass. Costs and obstacles are array arguments, not field
    names.
- **A model's fields are its state's type:** they are `Field` attributes of a subclass of
  `State`, **named `*State`** (`BoardState`, `GameState(TerrainState, PiecesState)`). On a
  state the attribute is the field's array (`board.owner`, typed), and on the class it names
  the field everywhere (`BoardState.owner`). There are no string field names, and values are
  keyed by field. Rules annotate their state (`def claim(s: BoardState, ctx: Context)`), so
  phases and simulations are typed by it. The arrays underneath are plain NumPy arrays: the
  class only organizes them.
  - **A state whose rules need a kind of space annotates it** (`space: LatticeSpace`, or
    `LatticeSpace | MeshSpace`): `state.space` is then typed by it, and states are made only on
    such a space. Don't narrow `state.space` with `isinstance` in rules. **That annotation is
    where a model says what space it needs:** its `start` takes exactly that kind (`Space`
    when the state has none), and `check_model` checks it.
  - **Arrays computed from fields are derived fields** on the same class
    (`@derived(height) def land(self) -> NDArray[np.bool_]`), not loose arrays in scripts or
    recomputed in each rule. They are cached while their fields keep their versions.
  - **Fields that belong together are a `NestedState`,** nested under a name
    (`foxes = Nested(Species)`, read as `state.foxes.population`), for different kinds of
    thing or one kind under different rules. A `NestedState` subclass is named for what it
    holds (`Species`, `Settlers`), not `*State`: it can't be simulated alone. Amounts carried
    by other amounts go in one nested state and move as one (`Spread(WorldState.settlers,
    ...)`). **Groups of one kind under the same rules stay a component axis** of one field
    (`people = Field(np.float64, (2,))`): one intent and one kernel call instead of one per
    group. When a parameter sets how many groups there are (a game's players), size the axis
    to the most the model allows (`MAX_PLAYERS`) and keep the rows past a run's count at 0:
    component shapes are fixed by the class.
- **Values are immutable:** spaces, graphs, operators and states. `copy` and `deepcopy` return
  the same object, and pickling rebuilds through the constructor. A write makes a new `State`
  that shares the arrays it left unchanged.
- **`CellGraph` rows are canonical:** sorted, with no edge twice. A duplicate edge raises, and
  callers combine duplicates first, because no single merge rule fits every attribute.
- **Neighborhoods are space-independent descriptions** with `compile(space)`, and **check
  their parameters when made**, because equal descriptions share one compiled operator.
- **Order by cell id, never by direction:** padded slots, action-space slots, and ties
  (regions by area, then lowest id; a path steps back to the lowest neighbor). A planet has no
  global directions.
- **Clip proposals, don't reject them:** a negative rate counts as 0, so a bad rule makes a
  dull world, not a broken one. Non-finite values and overflows raise.
- **Values keep their meaning when stored:** floats are never truncated to integers, and a
  value that doesn't fit its field raises instead of wrapping or becoming infinite.
- **Errors:** a value of the wrong type raises TypeError, and a value of the right type but out
  of range raises ValueError (IndexError for a cell id).
  - Booleans are never integers: `every=True` raises.
  - Arrays of ids or offsets need an integer dtype. Floats are never truncated, and a uint64 too
    large for int64 raises instead of wrapping to a negative id.
  - Shared checks live in `core/_arrays.py`: `check_int`, `check_real`, `int_array`,
    `check_cell`, `check_cells`, `check_ids` and `cell_mask`. `check_int` and `check_real` are
    public (`core.check_int`): models and games check their parameters and seeds with them,
    as `Stakes` does.
- **Implementations are bit-identical:** equal dtypes, shapes and bits (a zero's sign too), or
  the same error. NaN payloads and NumPy's warnings are exempt. Random draws come only from
  NumPy, kernels are serial, and `implementation(...)` is the one switch, with no keyword per
  call.
- **Tilings are data:** adding a tiling means a new `_Tiling` record in `core/lattice.py`, not a
  new class.
- **Simulations and games are data:** a frozen dataclass of parameters builds the state, the
  phases and the simulation. A simulation's is a `core.Model` (`start(space, seed)`,
  `phases()`): its variants are parameters, one seed sets the whole run, and what happens once
  during a run (an arrival) is a phase, not a second `Simulation`.
  - A variant that is off leaves its phase out of the schedule (`wildfire`, `seasons`,
    `warming`); a phase that acts only from a step, or at one, returns None on the other
    steps (`seasons_from`, an arrival).
  - Values computed from the parameters live by what they depend on: on static fields too, a
    static field set by `start`; on changing fields, a method of the parameters; on the
    parameters alone (not per cell, such as a matrix over components), computed once in
    `phases()` and closed over.
- **Every experiment runs one way:** `uv run python run.py <name> [args]`, from the
  repository's root. Examples live outside the library (`examples/`, which it never imports).
  An experiment is a module beside its example (a simulation's `experiment.py`, a game's
  `experiments/<name>.py`) with a docstring and `main(*args: str)`, and a line in
  `run.py`'s `EXPERIMENTS`: a simulation's is named after it (`predators`), a game's
  `<game>.<experiment>` (`tactics.zero_check`), and one over any game's output, not one
  game's, `games.<experiment>` (`games.curves`, in `examples/games/experiments/`). Notebooks,
  when they come back, go beside the example they explore.
  - **Every example has a `README.md`,** in one shape: what it is (and what it tests), its
    model or rules, its players (games), its experiments (a game's as a table of commands),
    what they found (dated), and
    its files (games). The reference stays in the docstrings: the README orients and links.
    Update it when an experiment is added, retired or rerun with new results.
    `examples/README.md` lists every example, what it tests and its main findings, and holds
    the shapes of experiments; the top `README.md` says what the project aims at and can do,
    and links there rather than listing examples.
  - **Watching a game is one experiment per game, `<game>.watch [start] [prefix]
    [players...]`:** it plays `game.start(EVAL_SEED + start)`, then steps a figure through
    the positions with `render.Browser`, built and finished by `examples/games/browse.py`
    (`figure`, `finish`): a window for an empty prefix, else `{prefix}_game.gif`.
  - **A simulation's experiment has one shape.** A module `SEED`, with a comment on what it
    sets. `main(length, prefix)`: the length in the model's own unit (`years`, `steps`), left
    out when a run's length is fixed, then any of the model's parameters it varies, and every
    picture saved as `{prefix}_<what>.png`. The model held under its own name (`settlement =
    Settlement()`), and each run one `Simulation(model.start(space, SEED), model.phases(),
    seed=SEED)`; an experiment comparing variants runs each through a `run(model, space,
    length)` helper. What it shows is read from its states, never rebuilt from the seed, and it
    doesn't repeat the model's tests (`check_model` compares the paths).
  - **A game is written one way** (`Stakes` in the README): a `*State` with its `space:`
    annotation, a `*Position` subclass, events as named tuples, and the rules a frozen
    dataclass with a `space` property (the board is a rule, cached per shape), `start(seed)`
    (the only way to start: the board drawn from `Context(seed=seed, spawn_key=(0,)).rng`),
    `player` (raising once the game is over), `view`, `legal` and `step` (opening with
    `check_action`, drawing from `(1, ply)`, and deciding by scores with `game.winner`).
    Players are frozen dataclasses called with `(game, player, view, legal, rng)`: they read
    which player they are from `player`, never from `game.player`. Its tests run `check_game`.
    - **Moves made at once** are orders given in turn and kept hidden (Colonies): each player
      still in gives one per ply, in seat order, kept on the position and out of every view
      and event (`Ordered(player)` names no order); a step before the round's last changes
      no field and draws nothing, and the last one resolves them all together, the same
      whatever the order they came in. `ply` counts orders, and the game's own `round`
      counts resolutions.
    - **A step that runs a `Simulation`** seeds it from its own stream, a seed drawn as
      `int(dice.rng.integers(2**63))` with `dice = Context(seed=seed, spawn_key=(1, ply))`.
      The loop keys its streams `(t, phase)` from its seed, so seeded with the game's own
      seed they would meet the game's `(0,)`, `(1, ply)` and `(2, ply)`.
  - **A game's experiment has one shape.** Players are scored on the game's `EVAL_SEED` (in
    its `match.py`) in every experiment, so scores compare across experiments and stay clear
    of training; a module `SEED`, with a comment, sets anything else it draws (self-play
    starts, first weights), and a `seed` argument only sets replicas or continues a run.
    `main(*inputs, amount, prefix, *varied)`: inputs such as `weights`, then how much to do
    in its own unit (`starts` of a series, `minutes` of a time box, `games` or `iterations` of
    training), the prefix when it writes files (every file `{prefix}_<what>.<ext>`), then what
    it varies; rivals chosen by name are trailing arguments. The rules built in `main` as
    `game` and passed to helpers, never a module constant. Play counted in starts (`series`,
    `ladder`, `SelfPlay`), each played once per rotation of the seats, on every core
    (`workers=CORES`, from `examples.games`). Players are compared by `series` and `tally` (or
    the game's own `summarize`, which reads its events), one player against rivals by `ladder`,
    the rival in every other seat. An experiment that does what another
    game's does takes its name and argument order (`zero_scratch`, `zero_check`).
- **Notebooks are saved without outputs.** Their pictures and animations run to megabytes. A
  test runs the README's code.

## Workflow

- Add a line to "Design decisions" above only for a rule that should shape new code.
- Keep experiments and notebooks thin: logic lives in the package or the example's own
  modules, and has tests. Each experiment is run at its smallest in
  `tests/examples/test_experiments.py` when that takes seconds, and imported and its arguments
  checked (`tests/test_run.py`) when it doesn't.
- Benchmarks go in `benchmarks/`. Record the machine and library versions with any numbers.
- Run long experiments (series of games, sweeps) on every core, in processes
  (`game.series(..., workers=CORES)`). Pin to one performance core only for timings.
- Don't commit generated data or notebook outputs that contain large arrays.
