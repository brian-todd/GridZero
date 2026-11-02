"""Neighborhoods: rules that pick, for every cell, the cells it interacts with.

A neighborhood is a small, immutable description such as "cells within 2 hops" that does not
depend on any particular space. Compiling it on a space gives a `NeighborOperator`, which the
operations in `gridzero.core.operators` use. The same description therefore
works on a lattice and on the planet.

Write a custom neighborhood as any object with a ``compile(space)`` method that returns a
`NeighborOperator`, for example built with `NeighborOperator.from_table`. A neighborhood that
depends on the state, such as "the nearest settlements", follows `StateNeighborhood` instead:
it names the fields it reads, and the simulation recompiles it when one of them changes.
"""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal, Protocol, SupportsIndex, runtime_checkable

import numpy as np
import scipy.sparse as sp

from gridzero.core._arrays import check_int
from gridzero.core.fields import Field, State
from gridzero.core.graph import CellGraph
from gridzero.core.operators import NeighborOperator
from gridzero.core.space import AdjacencyKind, Space, check_kind, check_radius


@runtime_checkable
class Neighborhood(Protocol):
    """A rule mapping each cell to its neighbors and their weights.

    Implementations are descriptions, independent of any space. `compile` evaluates the rule
    on a space.
    """

    def compile(self, space: Space) -> NeighborOperator:
        """Evaluate the neighborhood on a space.

        Parameters
        ----------
        space : Space
            The cells to connect.

        Returns
        -------
        NeighborOperator
            Neighbors and weights of every cell of ``space``.
        """
        ...


@runtime_checkable
class StateNeighborhood(Protocol):
    """A neighborhood that depends on the state, such as the nearest settlements.

    The simulation compiles it with the current state before a phase that uses it, and reuses
    the operator until one of the fields in `depends_on` is written. `compile` must read no
    other field, because changes to other fields do not trigger a rebuild.

    Many neighborhoods that seem state-dependent are a fixed set of candidates with weights
    that change, such as adjacent cells weighted by how attractive they are now. Keep those
    static, and apply the changing weights in the phase: rebuilding an operator costs far more
    than reweighting.
    """

    @property
    def depends_on(self) -> tuple[Field[Any], ...]:
        """The fields `compile` reads."""
        ...

    def compile(self, space: Space, state: State) -> NeighborOperator:
        """Evaluate the neighborhood on a space in a given state.

        Parameters
        ----------
        space : Space
            The cells to connect.
        state : State
            The current state. Only the fields in `depends_on` may be read.

        Returns
        -------
        NeighborOperator
            Neighbors and weights of every cell of ``space``.
        """
        ...


@dataclass(frozen=True)
class Adjacency:
    """Cells that touch: the space's adjacency graph.

    Parameters
    ----------
    kind : {"facet", "vertex"} or int, default "facet"
        How cells must touch (see `AdjacencyKind`).
    weight : str, optional
        Edge attribute to use as weights: ``"distance"`` (between centers) or ``"boundary"``
        (measure of the shared face). If omitted, every weight is 1.

    Raises
    ------
    TypeError
        If ``kind`` is neither a string nor an integer, or ``weight`` is not a string.
    ValueError
        If ``kind`` is an unknown string or a negative integer, or ``weight`` names another
        attribute. Whether an integer kind fits a space is checked when compiling.
    """

    kind: AdjacencyKind = "facet"
    weight: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", check_kind(self.kind))
        _check_weight(self.weight, ("distance", "boundary"))

    def compile(self, space: Space) -> NeighborOperator:
        """Return the adjacency graph of ``space`` as an operator.

        Parameters
        ----------
        space : Space
            The cells to connect.

        Returns
        -------
        NeighborOperator
            Operator over the space's own (cached) adjacency graph, with its ``"distance"``
            and ``"boundary"`` edge attributes.
        """
        return NeighborOperator(space.adjacency(self.kind), self.weight)


@dataclass(frozen=True, init=False)
class Hops:
    """Cells reachable in at most ``reach`` steps between adjacent cells, excluding the cell.

    Hop distance counts steps through the adjacency graph, so the neighborhood follows the
    shape of the grid: a diamond of facet hops on squares, a hexagon on hexes and on the
    planet. Each cell appears once, even when it can be reached in several ways.

    Parameters
    ----------
    reach : int
        Largest number of steps, at least 1. ``Hops(1, kind)`` has the same neighbors as
        ``Adjacency(kind)``.
    kind : {"facet", "vertex"} or int, default "facet"
        Adjacency used for each step (see `AdjacencyKind`).
    weight : str, optional
        Edge attribute to use as weights. The only one is ``"hops"``. If omitted, every
        weight is 1.

    Raises
    ------
    TypeError
        If ``reach`` is not an integer, ``kind`` is neither a string nor an integer, or
        ``weight`` is not a string.
    ValueError
        If ``reach`` is less than 1, ``kind`` is an unknown string or a negative integer, or
        ``weight`` names another attribute.
    """

    reach: int
    kind: int | Literal["facet", "vertex"]
    weight: str | None

    def __init__(
        self, reach: SupportsIndex, kind: AdjacencyKind = "facet", weight: str | None = None
    ) -> None:
        object.__setattr__(self, "reach", check_int(reach, "reach", minimum=1))
        object.__setattr__(self, "kind", check_kind(kind))
        object.__setattr__(self, "weight", _check_weight(weight, ("hops",)))

    def compile(self, space: Space) -> NeighborOperator:
        """Find every cell's neighbors within ``reach`` hops.

        Parameters
        ----------
        space : Space
            The cells to connect.

        Returns
        -------
        NeighborOperator
            Operator with edge attribute ``"hops"``, the smallest number of steps from a cell
            to its neighbor (1 to ``reach``).
        """
        return NeighborOperator(_hop_graph(space.adjacency(self.kind), self.reach), self.weight)


@dataclass(frozen=True)
class Ball:
    """Cells whose centers lie within a distance of the cell's center, excluding the cell.

    Parameters
    ----------
    radius : float
        Largest center-to-center distance, finite and non-negative. It is measured as in
        `Space.radius_graph`: across periodic boundaries, and along great circles on a sphere.
    weight : str, optional
        Edge attribute to use as weights. The only one is ``"distance"``. If omitted, every
        weight is 1.

    Raises
    ------
    TypeError
        If ``radius`` is not a real number, or ``weight`` is not a string.
    ValueError
        If ``radius`` is negative, infinite, or NaN, or ``weight`` names another attribute.
    """

    radius: float
    weight: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", check_radius(self.radius))
        _check_weight(self.weight, ("distance",))

    def compile(self, space: Space) -> NeighborOperator:
        """Find every cell's neighbors within ``radius``.

        Parameters
        ----------
        space : Space
            The cells to connect.

        Returns
        -------
        NeighborOperator
            Operator with edge attribute ``"distance"``.
        """
        return NeighborOperator(space.radius_graph(self.radius), self.weight)


@dataclass(frozen=True, eq=False)
class Explicit:
    """A fixed, given graph, such as rivers, roads, or trade routes.

    Parameters
    ----------
    graph : CellGraph
        Row ``i`` lists the neighbors of cell ``i``. It need not be symmetric.
    weight : str, optional
        Edge attribute of ``graph`` to use as weights. If omitted, every weight is 1.

    Raises
    ------
    TypeError
        If ``graph`` is not a `CellGraph`, or ``weight`` is not a string.
    ValueError
        If ``weight`` is not an edge attribute of ``graph``.
    """

    graph: CellGraph
    weight: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, CellGraph):
            raise TypeError(f"graph must be a CellGraph, got {type(self.graph).__name__}")
        _check_weight(self.weight, self.graph.edge_data)

    def compile(self, space: Space) -> NeighborOperator:
        """Return the graph as an operator, after checking that it fits ``space``.

        Parameters
        ----------
        space : Space
            The cells the graph connects.

        Returns
        -------
        NeighborOperator
            Operator over ``graph`` itself, with all its edge attributes.

        Raises
        ------
        ValueError
            If the graph and the space have different numbers of cells.
        """
        if self.graph.n_cells != space.n_cells:
            raise ValueError(
                f"the graph has {self.graph.n_cells} cells but the space has {space.n_cells}"
            )
        return NeighborOperator(self.graph, self.weight)


def _check_weight(weight: object, offered: Collection[str]) -> str | None:
    """Return ``weight`` after checking that it is None or names one of the edge attributes.

    Raises TypeError if it is neither None nor a string, and ValueError if it is not offered.
    """
    if weight is None:
        return None
    if not isinstance(weight, str):
        raise TypeError(f"weight must be a str or None, got {type(weight).__name__}")
    if weight not in offered:
        raise ValueError(f"no edge attribute {weight!r} here; there are {sorted(offered)}")
    return weight


def _hop_graph(graph: CellGraph, reach: int) -> CellGraph:
    """Link each cell to every cell at most ``reach`` steps away, recording the hop count.

    A breadth-first search from every cell at once: ``frontier`` holds the cells first reached
    at the current distance, one row per starting cell, as a sparse matrix.
    """
    step = graph.to_scipy()
    reached = sp.eye_array(graph.n_cells, format="csr")
    frontier = reached
    sources, targets, hops = [], [], []
    for distance in range(1, reach + 1):
        candidates = frontier @ step  # entries count the paths, so they are positive
        # Drop the cells reached earlier.
        frontier = candidates - candidates.multiply(reached)
        frontier.eliminate_zeros()
        if frontier.nnz == 0:
            break
        frontier.data[:] = 1.0
        found = frontier.tocoo()
        sources.append(found.row)
        targets.append(found.col)
        hops.append(np.full(found.nnz, float(distance)))
        reached = reached + frontier
    if not sources:
        return CellGraph.from_edges([], [], graph.n_cells, edge_data={"hops": []})
    return CellGraph.from_edges(
        np.concatenate(sources),
        np.concatenate(targets),
        graph.n_cells,
        edge_data={"hops": np.concatenate(hops)},
    )
