# GridZero

Agents that learn games by self-play, and the simulations that share their engine, on worlds
made of cells, from flat boards to whole planets.

## Goals

GridZero is a research playground with three aims.

- **Agents that learn games they were not written for.** A game is a protocol: rules that
  start, step and say what is legal, and what each player is shown. Agents read a game only
  through it and learn by self-play: searching the rules themselves where a game shows
  everything (AlphaZero), and a model of the rules learned from what a player is shown where it
  hides things (MuZero). The aim is one way of learning for many kinds of game: chance, hidden
  information, moves made at once, and more than two players.
- **Worlds of any shape.** Boards and worlds are spaces of cells: squares, hexagons and
  triangles, cubes in any number of dimensions, or the cells of a planet. Rules are written
  over cells and their neighbors, not coordinates, so one rule runs on any of them.
- **One engine for games and simulations.** Living worlds, where populations grow, move,
  compete and hunt, and games share one state, one way of writing rules and one loop, so a game
  can be played on a living world.

Correctness, clarity and reproducibility come first, and performance is a close second. One
seed gives the same bits every time, every fast path matches a plain NumPy reference bit for
bit, and the framework, not each rule, keeps amounts conserved and non-negative. There is one
obvious way to write a model, a game, a player and an experiment.

## What it can do

- **Spaces:** lattices of squares, cubes and hypercubes, hexagons and triangles, each axis
  wrapped or with edges; planets as icospheres of up to tens of thousands of cells, with
  continents, seas and climates. Paths, ranges and exact lines of sight between cells.
- **Simulations:** typed state per cell, rules that propose changes as intents (flows,
  sources, claims), and a loop that applies them, conserving amounts, with Numba fast paths:
  thousands of steps a second at ten thousand cells.
- **Games:** any number of players on any space, with dice, hidden information, orders given
  at once and players knocked out. Games are played, replayed and run in series on every core,
  then scored by tallies, Elo ladders and tests of whether a seat has the edge. One set of
  checks says whether a new game keeps the promises of a game.
- **Agents:** a uniform player; exact values of small games; Gumbel AlphaZero with chance
  nodes; MuZero, planning on a learned model from a player's view alone; imitation and PPO.
  Self-play runs in actor processes around one GPU inference server.
- **Pictures:** globes and maps of planets, flat maps of boards, a viewer that steps through a
  game in a window or saves a GIF, and training curves that refresh while a run trains.

## Where it stands

AlphaZero and MuZero both learn from scratch, in 25 minutes on one GPU, to beat the scripted
players of the games they were tried on, and MuZero does so under fog of war on a planet. The
four games and six simulations in [`examples/`](examples/README.md) are the proving grounds:
its README lists them and what they have shown.

Not done yet: the learners play two players only, AlphaZero needs a game with nothing hidden
on a 2-D lattice, and self-play in the board games drifts into draws. `core` runs on the CPU
only. The `game` protocols have had their last planned change; the rest may still move.

## Getting started

You need Python 3.15 (`uv python install 3.15`) and [uv](https://docs.astral.sh/uv/). Torch
is a dependency; training needs an NVIDIA GPU (developed on an RTX 3080 with 10 GB), and the
few tests that need one skip without it.

```bash
uv sync                      # install the package and its development tools
uv run pytest -m "not slow"  # the fast tests, in about 2 minutes; without -m, about 5
uv run python run.py         # list the experiments of the examples
```

## A first simulation

Every simulation is written one way: a **state** says what the cells hold, and a **model**, a
frozen dataclass of parameters, says how a run starts and what happens each step. Life on a
board of hexagons:

```python
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np

from gridzero.core import (
    Adjacency,
    Assign,
    Context,
    Field,
    LatticeSpace,
    Phase,
    Simulation,
    Space,
    State,
    check_int,
    phase,
    pull,
)


class LifeState(State):  # what each cell holds
    alive = Field(bool)


@dataclass(frozen=True)
class Life:  # the rules: a core.Model
    density: float = 0.3  # the share of cells alive at the start

    def start(self, space: Space, seed: SupportsIndex) -> LifeState:
        rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
        return LifeState(space, {LifeState.alive: rng.random(space.n_cells) < self.density})

    def phases(self) -> list[Phase[LifeState]]:
        @phase(writes=LifeState.alive, hoods={"near": Adjacency()})
        def life(s: LifeState, ctx: Context) -> Assign:
            # Hexagonal Life (B2/S34): born with 2 live neighbors, survives with 3 or 4.
            n = pull(s.alive, ctx.hoods["near"])  # live neighbors of each cell
            born = ~s.alive & (n == 2)
            survives = s.alive & ((n == 3) | (n == 4))
            return Assign(LifeState.alive, born | survives)

        return [life]


life = Life()
board = LatticeSpace((64, 64), tiling="hex", periodic=True)  # 4,096 hexagons on a torus
sim = Simulation(life.start(board, 0), life.phases(), seed=0)
history = sim.run(100, observers={"alive": lambda s: s.alive.sum()}, every=25)
print(history["alive"])  # [774 338 168  93]: the random start thins out
```

**The state** is a subclass of `State`, named `*State`, whose `Field` attributes declare what
cells hold. On a state, an attribute is the field's array: `sim.state.alive` is typed
`NDArray[np.bool_]`. On the class, it names the field everywhere, as in `LifeState.alive`.
Every state carries its space (`state.space`). When the rules need a kind of space, such as a
2-D lattice for a wind, the state's class says so (`space: LatticeSpace`) and `start` takes
that kind. Life's rule never sees what kind of space it runs on, so `tiling="triangle"`, a cube
of cells, or a planet's `icosphere(16)` runs it unchanged. Arrays computed from fields are
`derived` fields on the same class (see the board below).

**The model** holds parameters and no state. `start(space, seed)` builds step 0, and one seed
sets the whole run: the start, and through `Simulation`, every random draw after it.
`phases()` returns the rules of a step, run in order. Values that depend on the parameters
alone are computed once in `phases()`, beside the rules that use them. Variants are parameters:
a variant that is off leaves its phase out of the list, and a phase that acts only from some
step returns `None` before it. Something that happens once during a run, such as an arrival,
is a phase too, never a second `Simulation`.

**A rule never writes arrays itself.** It returns **intents**, and the framework applies them.
`Assign` overwrites a field. `Flow`, `Spread`, `Source` and `Claim` move, add and remove
amounts, and that is where conservation, non-negative amounts and fair rationing are
enforced, so a rule can't break them.

**A model's tests** start with `check_model(model, space, steps)` (in `tests/model_demo.py`):
it checks that `start` takes its state's kind of space, that a seed gives the same bits on
both implementations, and that amounts stay finite and non-negative. The model's own tests
then check what it claims to do.

## A first game

Every game is written one way too: a **state** says what the cells hold, a **position** adds
what isn't per cell, and the **rules**, a frozen dataclass of parameters, say how a game
starts, whose turn it is, what is legal and what an action does. A game of stakes on a board
of hexagons, where a player claims a free cell outright or gambles on it: heads, it takes the
cell and every free neighbor; tails, the cell goes to the other player.

```python
import functools
from dataclasses import dataclass
from typing import NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.agents import UniformPlayer
from gridzero.core import (
    Adjacency,
    Context,
    Field,
    LatticeSpace,
    State,
    check_int,
    pull,
)
from gridzero.game import (
    Action,
    Position,
    check_action,
    play,
    series,
    tally,
    winner,
)

CLAIM, GAMBLE = 0, 1  # the verbs: take a free cell, or flip a coin for it and its neighbors


class StakesState(State):  # what each cell holds
    space: LatticeSpace  # the rules need a board
    owner = Field(np.int8, default=-1)  # the player holding the cell, -1 for nobody


class Staked(NamedTuple):  # an event: what a step did
    player: int
    cells: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class StakesPosition(Position[StakesState, Staked]):
    """The state, the seed, the ply, the result and the step's events."""


@dataclass(frozen=True)
class Stakes:  # the rules: a game.Game
    shape: tuple[int, int] = (5, 5)  # the board is a rule: 25 hexagons, with edges

    @property
    def n_players(self) -> int:
        return 2

    @property
    def n_verbs(self) -> int:
        return 2

    @property
    def space(self) -> LatticeSpace:
        return board(self.shape)

    def start(self, seed: SupportsIndex) -> StakesPosition:
        return StakesPosition(StakesState(self.space), check_int(seed, "seed", minimum=0))

    def player(self, position: StakesPosition) -> int:
        if position.over:
            raise ValueError("the game is over")  # nobody moves
        return position.ply % 2  # the players take turns, 0 first

    def view(self, position: StakesPosition, player: int) -> StakesPosition:
        return position  # nothing is hidden

    def legal(self, position: StakesPosition) -> NDArray[np.bool_]:
        free = (position.state.owner < 0) & (not position.over)
        return np.stack([free, free])  # (n_verbs, n_cells): any free cell, either way

    def step(
        self, position: StakesPosition, action: tuple[SupportsIndex, SupportsIndex]
    ) -> StakesPosition:
        verb, cell = check_action(position, action, self.n_verbs)
        owner = position.state.owner.copy()
        if owner[cell] >= 0:
            raise ValueError(f"cell {cell} is taken")
        player = position.ply % 2
        dice = Context(seed=position.seed, spawn_key=(1, position.ply))  # this step's stream
        taken = np.array([cell])
        if verb == GAMBLE and dice.rng.random() < 0.5:  # heads: the free neighbors too
            near = self.space.adjacency().neighbors(cell)
            taken = np.append(taken, near[owner[near] < 0])
        elif verb == GAMBLE:  # tails: the cell goes to the other player
            player = 1 - player
        owner[taken] = player
        over = bool((owner >= 0).all())
        cells = np.bincount(owner[owner >= 0], minlength=2)  # each player's cells
        return StakesPosition(
            position.state.replace({StakesState.owner: owner}),
            position.seed,
            ply=position.ply + 1,
            over=over,
            winner=winner(cells) if over else None,  # more cells wins; equal is a draw
            drew=dice.drew,
            events=(Staked(player, tuple(int(c) for c in taken)),),
        )


@functools.cache
def board(shape: tuple[int, int]) -> LatticeSpace:  # one space per shape, shared by positions
    return LatticeSpace(shape, tiling="hex", periodic=False)


@dataclass(frozen=True)
class Gambler:  # a player: a game.Player
    nerve: int = 3  # gambles when a cell has at least this many free neighbors

    def __call__(
        self,
        game: Stakes,
        player: int,  # which player decides: this one plays both sides alike
        position: StakesPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        free = pull(legal[CLAIM], Adjacency().compile(game.space))  # free neighbors of each cell
        best = np.flatnonzero(legal[CLAIM] & (free == free[legal[CLAIM]].max()))
        verb = GAMBLE if free[best[0]] >= self.nerve else CLAIM
        return Action(verb, int(rng.choice(best)))


game = Stakes()
record = play(game, [Gambler(), UniformPlayer()], game.start(0))
print(record.final.winner, record.final.ply)  # 0 13: the gambler won, in 13 decisions

records = series(game, [("gambler", Gambler()), ("never", Gambler(nerve=7))], starts=50)
print(tally(records).score("gambler"))  # Score(won=79, lost=21, drawn=0): gambling pays
```

**The state** is a `State` like a simulation's, so the rules use the same spaces, fields and
queries. **The position** is a `game.Position`, named `*Position`, which holds the state, the
seed, the ply and the result; a game adds what isn't per cell by subclassing it, such as
Tactics' round. **The events** say what each step did, in the game's own types.

**The rules** (`game.Game`) hold parameters and no position, and every step is pure: it makes
a new position and leaves the old one as it was, so a search tree or a replay is just
positions. The board is a rule (Connect Four is 7x6), so `start` takes only a seed, and its
space is cached per shape. One seed sets the whole game: stream `(0,)` of it is the game's own,
such as a board drawn at random, a step draws only from stream `(1, ply)`, through
`core.Context`, and players get `(2, ply)`. `step` opens with `check_action`.

**A player** (`game.Player`) is a frozen dataclass of its parameters, called with the rules,
which player it is, what the game shows it, the legal actions and a generator of its own, and
returns an `Action`: a verb and a cell. `play` plays one game and keeps its `Record`; `series`
plays players against each other from several starts, each start once per rotation of the
seats, in processes; `tally`, `ladder` and `fairness` say who won and whether a seat had the
edge. Players in `agents` read a game only through these protocols, so `UniformPlayer` and
the AlphaZero agent play games they were not written for.

**A game's tests** start with `check_game(game, player, seeds)` (in `tests/game_demo.py`): it
plays games to their end and checks that steps are pure, that replays give the same
positions, that only legal actions are accepted, and that a step that drew nothing gives the
same position under any seed. The game's own tests then check its rules.

## Building on it

A new simulation or game goes in its own package under `examples/`, beside the others:

1. Write the state and the model, or the state, position and rules, as above. Start from the
   example closest to yours and do it the same way.
2. Test it with `check_model` or `check_game` first, then test what it claims to do: a model
   against its theory, a game against its rules.
3. Write each experiment as a module with `main(*args: str)`, and name it in `run.py`'s
   `EXPERIMENTS`.
4. Give it a `README.md` in the shape the others share.

`agents.UniformPlayer` plays a new game at once. The learners need a little more from it:
AlphaZero an `agents.Learnable` (observations as planes per cell, in each player's frame), and
MuZero an `agents.Observable` and `agents.Revealable` (also the whole position, which it learns
to guess). [`examples/README.md`](examples/README.md) gives the
shape of an experiment, and [`AGENTS.md`](AGENTS.md) every convention.

## The engine

| Layer | What it is | Main names |
|---|---|---|
| Space | Cells with flat integer ids, centers, areas and adjacency, the connected regions and borders of masks of cells, and random points spread evenly over cells | `icosphere`, `MeshSpace`, `LatticeSpace`, `regions`, `border`, `random_points` |
| Queries | Moving between cells with costs, straight-line ranges, and lines of sight | `steps`, `reach`, `path`, `distance`, `within`, `line`, `visible`, `sight` |
| Neighborhood | Which cells each cell sees, compiled to a sparse operator | `Adjacency`, `Hops`, `Ball`, `Explicit`, `NeighborOperator` |
| Operations | Aggregating values over neighbors | `pull` (sum, mean, max, min), `push`, `gather` |
| State | Typed per-cell fields, as immutable values: whatever the cells hold, from populations to the pieces of a game, nested states of fields that belong together, and arrays derived from them | `State`, `Field`, `NestedState`, `Nested`, `derived` |
| Phase | A rule that reads the state and returns intents | `phase`, `Phase` |
| Intents | Overwrite a field, move amounts along edges, add or remove amounts, claim a scarce resource | `Assign`, `Flow`, `Spread`, `Source`, `Claim` |
| Actions | Per-cell choices (stay, or move to a neighbor) turned into flows | `NeighborChoice`, `NeighborAllocation`, `Box` |
| Simulation | Runs the phases in order, with seeded random streams per step and phase | `Simulation` |

A few principles hold throughout:

- **Rules propose, the framework enforces.** Takers of one amount share it in proportion when
  they ask for more than there is. Out-of-range proposals are clipped rather than rejected, so a
  bad rule makes a dull world, not a broken one.
- **Conserved quantities move as flows** between cells. A cell never takes from its neighbors
  on its own.
- **One seed, one result.** Each phase draws from its own random stream, derived from the root
  seed, the step and the phase name. Snapshots are free, because states are immutable.
- **Two implementations, identical bits.** Operations, intents, action spaces, searches and
  sight each have a NumPy/SciPy reference path and a Numba fast path, and the two give
  bit-identical results. `implementation("reference")` runs a whole simulation on the reference
  path.

## Games, agents and worlds

| Package | What it does |
|---|---|
| `game` | The `Game` and `Player` protocols, positions and actions, `play`, `series` in processes, records and `replay`, `tally`, `ladder` and `fairness`, and helpers that make a board fair (`symmetric`, `zones`, `joined`) |
| `agents` | Players of any game: `UniformPlayer`, Gumbel AlphaZero search with chance nodes (`AlphaZeroPlayer`), its residual network and training loop on the GPU (`net`, `zero_train`), MuZero's search over a learned world model (`muzero`, `muzero_net`, `muzero_train`), imitation and PPO, and exact values of small games (`exact`). Only the networks and their training import Torch, and the package doesn't import them |
| `worlds` | A `Planet` (`generate_planet`) has continents, seas and a climate, with land, coasts and landmasses, on an icosphere or projected onto a lattice as its map. `generate_terrain` makes continents and seas for any 2-D lattice, such as a board for a game, and one seed gives the same continents at every resolution |
| `render` | `globe` and `world_map` (Mollweide or equirectangular) picture a spherical mesh as images, and `flat_map` a 2-D lattice, such as a board or a planet's map, with `colorize`, `mix` and `over` for per-cell colors; `Browser` steps a figure through its frames, such as the positions of a game |

A board for a game, from `core` and `worlds` alone, whose cells hold a height and an owner:

```python
import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Context,
    Field,
    LatticeSpace,
    State,
    derived,
    path,
    random_points,
    reach,
    regions,
    sight,
)
from gridzero.worlds import generate_terrain


class TerrainState(State):  # what a board's cells hold
    space: LatticeSpace  # sight traces lines on a 2-D lattice
    height = Field(np.float64, static=True)
    owner = Field(np.int8, default=-1)  # -1 for nobody

    @derived(height)  # computed from height when first read, then cached
    def land(self) -> NDArray[np.bool_]:
        return self.height > 0

    @derived(height)
    def walk(self) -> NDArray[np.float64]:  # what entering each cell costs: land only
        return np.where(self.land, 1.0, np.inf)


seed = 3  # a game's seed: as in a game's start, its stream (0,) draws the board
rng = Context(seed=seed, spawn_key=(0,)).rng
space = LatticeSpace((48, 48), tiling="hex", periodic=False)  # hexagons, with edges
height = generate_terrain(space, seed=int(rng.integers(2**63)))
state = TerrainState(space, {TerrainState.height: height})
starts = space.locate(random_points(space, rng, 4, where=regions(space, state.land) == 0))
owner = state.owner.copy()
owner[starts] = [0, 1, 2, 3]  # four players, on the largest landmass
state = state.replace({TerrainState.owner: owner})  # state.values_at(cell): height and owner

moves = reach(space, starts[0], 5, cost=state.walk)  # where player 0 may go in 5 moves
route = path(space, starts[0], starts[1], cost=state.walk)  # a cheapest way to player 1
peaks = state.height > 0.6
seen = sight(space, starts[0], blocked=peaks, radius=8.0)  # high ground hides what's behind
```

What a board's cells hold is up to the game: a subclass of `TerrainState` declares more
fields, and `state.extend(SubclassState, values)` gives a state those fields too. Arrays
computed from fields, such as `land` and `walk`, are declared with `derived` and read like
fields. They are cached while the fields they depend on are unchanged, so a board's `land` is
computed once however many steps read it. `tiling="triangle"` gives a board of triangles, in
rows that point up and down in turn.

Pieces and players ask questions of the cells: `steps`, `reach` and `path` move between
adjacent cells on any space, with a cost for entering each cell; `distance` and `within`
measure straight lines between centers; and on a 2-D lattice `line`, `visible` and `sight`
trace lines exactly, through corners and along edges too. A line is blocked by a blocked
cell it passes through, or where blocked cells touch it from both sides.

## Performance

One step on one core of an i9-12900K, on the Numba path:

| Model | Steps per second |
|---|---|
| Logistic growth, spread toward room and random shocks, 10,242 cells | 2,900 |
| Conway's Life, 256x256 | 5,600 |

`taskset -c 4 uv run python -m benchmarks.bench` prints these and the times of the individual
operations, on both paths. At these sizes, Python call overhead counts for as much as
arithmetic, so the fast paths gain 1.1-1.6x on whole models and up to 4x on single operations.
Queries gain more where the fast path can stop early or needs no heap: 12x for `reach` within
5 moves, and over 100x for `sight` and for a `path` to a nearby cell.

## Layout

```
src/gridzero/
    core/          spaces, queries (paths, ranges, sight), neighborhoods, operators,
                   fields, intents, actions, the loop
    game/          games beside simulations: protocols, play, series, records, ladders
    agents/        players of any game: AlphaZero and MuZero, networks, training
    worlds/        planets, and terrain for lattices
    render.py      globes and maps of planets, and flat maps of lattices
examples/          whole models on the package, each with its experiments (README.md):
                   simulations/ (competition, migration, predators, biomes, settlers,
                   population_evolution) and games/ (tactics, in_a_row, voronoi,
                   colonies)
run.py             runs any experiment of the examples
tests/             mirrors src/ and examples/
benchmarks/        timing scripts, run as modules (python -m benchmarks.bench)
```

## Development

Before calling work done, run:

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [`AGENTS.md`](AGENTS.md) holds the conventions and the design rules to follow while writing
  code, for people and coding agents alike.
