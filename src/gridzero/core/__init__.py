"""Core simulation loop: spaces of polygonal cells, neighborhoods, state, and updates.

Spaces
------
Space
    Protocol shared by every space: cell ids, centers, measures, adjacency, radius graphs.
LatticeSpace
    n-dimensional hypercubic grids, and hexagonal and triangular grids in the plane.
MeshSpace
    Polygonal cells with explicit geometry, on a plane or a sphere.
icosphere
    Planet grid: 12 pentagons, with all other cells hexagons.
CellGraph
    CSR adjacency with per-edge attributes.
regions, border
    The connected regions of a mask of cells, numbered by area, and the cells of a mask that
    touch cells outside it.
random_points
    Points drawn at random, evenly by area over the cells of a space or of a mask.
distance, within
    How far apart cells' centers are, and the cells within a distance of one: a range.
snap, turns, in_front
    Directions on squares and hexes (`LatticeSpace.directions`): the one nearest a vector, the
    steps between two, and whether vectors point into the half a piece faces.
line, visible, sight
    Lines between cells' centers on a 2-D lattice: the cells one crosses, whether it is clear
    of blocked cells, and every cell a viewer sees.
steps, reach, path, hops
    Moving between adjacent cells, with costs: what reaching each cell costs from the nearest
    source, the cells a budget reaches, a cheapest path, and the fewest moves between any cells
    of a lattice.

Neighborhoods
-------------
Neighborhood
    Protocol: a space-independent rule, compiled on a space to a `NeighborOperator`.
Adjacency, Hops, Ball, Explicit
    Built-in neighborhoods: touching cells, cells within k hops, cells within a distance, and a
    given graph.

Operators
---------
NeighborOperator
    A compiled neighborhood: CSR neighbors with weights, a cached transpose, padded tables.
Neighbors
    Padded neighbor tables returned by `NeighborOperator.query`.
pull, push, gather
    Aggregate neighbor values into cells, send values out to neighbors (the transpose), and
    collect raw neighbor values.

State
-----
State, Field
    Immutable values of every field at one step. A model's states are a subclass of `State`
    whose `Field` attributes declare its fields: dtype, component shape, static or dynamic,
    extensive or intensive, and a default. ``BoardState.owner`` names a field everywhere, and
    ``board.owner`` is a state's array of it, typed. A reference to a state is a snapshot. It
    also holds what the cells of a game board hold: `values_at` reads one cell, and `replace`
    and `extend` make new states.
Edit
    Writable copies of some fields (`State.edit`), changed in place by a game's step, moving,
    swapping and clearing cells across all of them, then frozen into a new state once.
derived, Derived
    Arrays computed from fields, declared as methods of a state's class and read like fields,
    such as ``board.land``. Cached while the fields they depend on are unchanged.
NestedState, Nested
    Fields that belong together, such as a species, declared once on a `NestedState` and nested in
    states under a name: ``foxes = Nested(Species)``. ``state.foxes.population`` is the array,
    and ``EcosystemState.foxes.population`` the field, stored as ``"foxes.population"``.
    `Flow` and `Spread` move a nested state's amounts together.

Updates
-------
Phase, phase
    One process of a step: a rule that reads the state and returns intents.
Assign
    The intent to overwrite a field.
Flow, Spread
    Intents that move amounts between neighboring cells, conserving their total: edge by edge,
    or as a leave rate split toward attractive neighbors.
Source, Claim
    Intents that add or remove amounts, and that move amounts between fields within a cell,
    rationing a scarce supply.
resolve
    Applies a phase's intents to a state, enforcing conservation, bounds and domains.
Simulation
    Runs a schedule of phases over a state, with seeded random streams per step and phase.
Model
    Protocol of a simulation's rules: ``start(space, seed)`` and ``phases()``.
Context
    What a phase gets besides the state: compiled neighborhoods and its random generator.
StateNeighborhood
    Protocol for neighborhoods that depend on fields, recompiled when those fields change.

Actions
-------
ActionSpace
    Protocol: the shape of an action, sampling and a validity check.
NeighborAllocation, NeighborChoice
    Split each cell's amount over staying and its neighbors, or pick one of them, with the
    same row of slots for cells of every degree. Both turn actions into a `Flow`.
Box
    A bounded real value per cell, such as a rate a rule controls.

Implementations
---------------
implementation, Impl
    Choose between the NumPy/SciPy reference operations and their Numba fast paths, which
    give bit-identical results.

Argument checks
---------------
check_int, check_real
    Check that an argument is an integer (never a boolean) or a finite real number in range,
    raising TypeError or ValueError as `core` itself does: what models and games use to check
    their parameters and seeds.
"""

from gridzero.core._arrays import check_int, check_real
from gridzero.core.actions import (
    ActionSpace,
    Box,
    NeighborAllocation,
    NeighborChoice,
)
from gridzero.core.directions import in_front, snap, turns
from gridzero.core.fields import (
    Derived,
    Edit,
    Field,
    Nested,
    NestedState,
    State,
    derived,
)
from gridzero.core.graph import CellGraph
from gridzero.core.implementations import Impl, implementation
from gridzero.core.intents import (
    Assign,
    Claim,
    Flow,
    Intent,
    Intents,
    Source,
    Spread,
    resolve,
)
from gridzero.core.lattice import LatticeSpace, Tiling
from gridzero.core.loop import Context, Model, Phase, Simulation, phase
from gridzero.core.mesh import MeshSpace, Metric, icosphere
from gridzero.core.neighborhoods import (
    Adjacency,
    Ball,
    Explicit,
    Hops,
    Neighborhood,
    StateNeighborhood,
)
from gridzero.core.operators import (
    NeighborOperator,
    Neighbors,
    Reduction,
    gather,
    pull,
    push,
)
from gridzero.core.paths import hops, path, reach, steps
from gridzero.core.points import random_points
from gridzero.core.ranges import distance, within
from gridzero.core.regions import border, regions
from gridzero.core.sight import line, sight, visible
from gridzero.core.space import AdjacencyKind, Space

__all__ = [
    "ActionSpace",
    "Adjacency",
    "AdjacencyKind",
    "Assign",
    "Ball",
    "Box",
    "CellGraph",
    "Claim",
    "Context",
    "Derived",
    "Edit",
    "Explicit",
    "Field",
    "Flow",
    "Hops",
    "Impl",
    "Intent",
    "Intents",
    "LatticeSpace",
    "MeshSpace",
    "Metric",
    "Model",
    "NeighborAllocation",
    "NeighborChoice",
    "NeighborOperator",
    "Neighborhood",
    "Neighbors",
    "Nested",
    "NestedState",
    "Phase",
    "Reduction",
    "Simulation",
    "Source",
    "Space",
    "Spread",
    "State",
    "StateNeighborhood",
    "Tiling",
    "border",
    "check_int",
    "check_real",
    "derived",
    "distance",
    "gather",
    "hops",
    "icosphere",
    "implementation",
    "in_front",
    "line",
    "path",
    "phase",
    "pull",
    "push",
    "random_points",
    "reach",
    "regions",
    "resolve",
    "sight",
    "snap",
    "steps",
    "turns",
    "visible",
    "within",
]
