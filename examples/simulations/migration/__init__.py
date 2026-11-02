"""Herds following the rains around a planet: seven ways to move, and the ideal free distribution.

Rain falls in a band around the planet, and grass grows where it falls. Seven herds share the
planet, one for each way of moving amounts tried here, and none sees the others: each is
a world of its own on the same rain. A herd's animals split the grass of their cell equally,
so each eats ``rain / density``.

Theory (the ideal free distribution, Fretwell and Lucas 1970): animals that move to wherever
they would eat more stop moving when every occupied cell gives the same intake, so a herd's
density is proportional to the rain. `distance` measures how far each herd is from that, as
the share of its animals that would have to move to reach it (the total variation distance,
0 at the ideal free distribution and 1 at the opposite).

1. **still rain:** the band stays at the equator. A method reaches the ideal free distribution
   only if it is a resting state of the method: if animals stop leaving where no neighbor
   offers more. A `Spread` or an allocation that always sends a share ``roam`` away is
   diffusion once every cell offers the same, so it settles short of the ideal. Greedy choice
   keeps sending ``roam`` to a neighbor that offers a little more, and so overshoots and never
   settles. The random walk ignores the grass altogether.
2. **seasons:** the band swings north and south once a year. The herds chase it, and the
   distance shows how far each lags behind.

``python run.py migration`` runs both on an icosphere of 2,562 cells.
"""

from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Assign,
    Ball,
    Context,
    Field,
    Flow,
    NeighborAllocation,
    NeighborChoice,
    NeighborOperator,
    Phase,
    Space,
    Spread,
    State,
    phase,
    pull,
)


class HerdsState(State):
    """The rain, and the seven herds, one per way of moving."""

    rain = Field(np.float64, default=1.0)  # grass grown per unit area per step: intensive
    spreading = Field(np.float64, default=1.0, extensive=True)
    settling = Field(np.float64, default=1.0, extensive=True)
    flowing = Field(np.float64, default=1.0, extensive=True)
    allocating = Field(np.float64, default=1.0, extensive=True)
    choosing = Field(np.float64, default=1.0, extensive=True)
    ranging = Field(np.float64, default=1.0, extensive=True)
    wandering = Field(np.float64, default=1.0, extensive=True)


HERDS = {
    "Spread toward food": HerdsState.spreading,
    "Spread, leave if better": HerdsState.settling,
    "Flow down the gradient": HerdsState.flowing,
    "NeighborAllocation": HerdsState.allocating,
    "NeighborChoice, best": HerdsState.choosing,
    "same, Ball reach": HerdsState.ranging,
    "random walk": HerdsState.wandering,
}
"""Each herd's field, by the way it moves, in the order its phase runs."""


@dataclass(frozen=True)
class Migration:
    """The parameters of the rain and the herds: a `core.Model` of their states and phases.

    Parameters
    ----------
    seasons : bool, default False
        Whether the rain band swings north and south once a year. Without, it stays still at
        the equator.
    seasons_from : int, default 0
        The first step the band swings, with ``seasons``: until then it stays still.
    year : int, default 1000
        Steps in a year of seasons.
    swing : float, default 0.6
        How far north and south the rain band moves: the sine of its latitude.
    width : float, default 0.3
        Width of the band, in units of the sphere's radius.
    dry : float, default 0.1
        Rain outside the band, as a share of the rain in its middle.
    roam : float, default 0.1
        Share of each cell's animals that moves each step, in the methods that set one.
    floor : float, default 0.01
        Density added to a cell's own, so that an empty cell is not infinitely attractive.
    reach : float, default 0.2
        Radius of the long-range herd's neighborhood: about three cells at frequency 16.
    """

    seasons: bool = False
    seasons_from: int = 0
    year: int = 1000
    swing: float = 0.6
    width: float = 0.3
    dry: float = 0.1
    roam: float = 0.1
    floor: float = 0.01
    reach: float = 0.2

    def band(self, z: NDArray[np.float64], middle: float) -> NDArray[np.float64]:
        """Return the rain at each height, for a band centered on ``middle``.

        Parameters
        ----------
        z : ndarray of float64, shape (n_cells,)
            Heights of the cells' centers on the unit sphere: the sines of their latitudes.
        middle : float
            The sine of the latitude of the band's middle.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            Grass grown per unit area per step.
        """
        return self.dry + np.exp(-(((z - middle) / self.width) ** 2))

    def start(self, space: Space, seed: SupportsIndex) -> HerdsState:
        """Return every herd spread evenly, under the rain of step 0, at the equator.

        Parameters
        ----------
        space : Space
            A sphere: the rain follows the third coordinate of the cells' centers.
        seed : int
            Unused: the start draws nothing.

        Returns
        -------
        HerdsState
            One animal per cell in each herd.
        """
        return HerdsState(space, {HerdsState.rain: self.band(space.centers[:, 2], 0.0)})

    def _intake(self, s: HerdsState, herd: Field[np.float64]) -> NDArray[np.float64]:
        """Return the grass each animal of a herd eats in each cell, if one more came."""
        return s.rain / (s.density(herd) + self.floor)

    def _restless(self, food: NDArray[np.float64], op: NeighborOperator) -> NDArray[np.float64]:
        """Return the share of each cell's animals that leaves: how much more the best offers.

        That is ``roam`` times the best neighbor's lead. Nothing leaves a cell whose neighbors
        offer no more, so the ideal free distribution, where every cell offers the same, is a
        resting state. Nothing leaves a cell with no neighbors either (a ``reach`` shorter than
        the distance to the nearest center), whose best is 0.
        """
        best = pull(food, op, "max")
        ratio = np.divide(food, best, out=np.ones_like(food), where=best > 0)
        return self.roam * np.maximum(1 - ratio, 0)

    def phases(self) -> list[Phase[HerdsState]]:
        """Return the phases of a step: the seasons, if any, then the herds' moves.

        Returns
        -------
        list of Phase
            ``seasons`` (with `seasons`; nothing before ``seasons_from``), then one phase per
            herd, in the order of `HERDS`: ``spread``, ``settle``, ``flow``, ``allocate``,
            ``choose``, ``range_far`` and ``wander``.
        """
        return [*([self._seasons()] if self.seasons else []), *self._moves()]

    def _seasons(self) -> Phase[HerdsState]:
        """Return the phase in which the rain band swings north and south once a year."""

        @phase(writes=HerdsState.rain)
        def seasons(s: HerdsState, ctx: Context) -> Assign | None:
            """Move the rain band to where it is at this time of the year."""
            if s.step < self.seasons_from:
                return None
            middle = self.swing * np.sin(2 * np.pi * s.step / self.year)
            return Assign(HerdsState.rain, self.band(s.space.centers[:, 2], middle))

        return seasons

    def _moves(self) -> list[Phase[HerdsState]]:
        """Return the phases that move the herds, one per herd, in the order of `HERDS`."""
        roam = self.roam
        intake = self._intake
        restless = self._restless

        @phase(writes=HerdsState.spreading, hoods={"near": Adjacency()})
        def spread(s: HerdsState, ctx: Context) -> Spread:
            """Send a share of the animals away, over the neighbors by the food each offers."""
            toward = intake(s, HerdsState.spreading)
            return Spread(HerdsState.spreading, ctx.hoods["near"], leave=roam, toward=toward)

        @phase(writes=HerdsState.settling, hoods={"near": Adjacency()})
        def settle(s: HerdsState, ctx: Context) -> Spread:
            """Spread as `spread`, but only as far as a neighbor offers more than the cell."""
            food = intake(s, HerdsState.settling)
            near = ctx.hoods["near"]
            return Spread(HerdsState.settling, near, leave=restless(food, near), toward=food)

        @phase(writes=HerdsState.flowing, hoods={"near": Adjacency()})
        def flow(s: HerdsState, ctx: Context) -> Flow:
            """Move animals along each edge toward more food, in proportion to the gap."""
            near = ctx.hoods["near"]
            food = intake(s, HerdsState.flowing)
            here, there = food[near.graph.sources], food[near.graph.indices]
            degree = np.diff(near.graph.indptr)[near.graph.sources]
            rate = roam * np.maximum(1 - here / there, 0) / degree
            return Flow(HerdsState.flowing, near, rate)

        @phase(writes=HerdsState.allocating, hoods={"near": Adjacency()})
        def allocate(s: HerdsState, ctx: Context) -> Flow:
            """Split each cell's animals over staying and its neighbors, weighted by food.

            Staying gets the weight that keeps a share ``roam`` moving when every slot offers
            the same.
            """
            near = ctx.hoods["near"]
            slots = NeighborAllocation(near)
            food = intake(s, HerdsState.allocating)
            weights = slots.values(food)
            degree = slots.valid[:, 1:].sum(axis=1)
            weights[:, 0] *= degree * (1 - roam) / roam
            return slots.flow(HerdsState.allocating, weights)

        @phase(writes=HerdsState.choosing, hoods={"near": Adjacency()})
        def choose(s: HerdsState, ctx: Context) -> Flow:
            """Send a share of each cell's animals to the slot with the most food: maybe home."""
            slots = NeighborChoice(ctx.hoods["near"])
            food = intake(s, HerdsState.choosing)
            best = slots.best(slots.values(food))
            return slots.flow(HerdsState.choosing, best, rate=roam)

        @phase(writes=HerdsState.ranging, hoods={"far": Ball(self.reach)})
        def range_far(s: HerdsState, ctx: Context) -> Spread:
            """Spread as `settle`, but to every cell within ``reach``."""
            food = intake(s, HerdsState.ranging)
            far = ctx.hoods["far"]
            return Spread(HerdsState.ranging, far, leave=restless(food, far), toward=food)

        @phase(writes=HerdsState.wandering, hoods={"near": Adjacency()})
        def wander(s: HerdsState, ctx: Context) -> Flow:
            """Send a share of each cell's animals to a random slot, whatever the food."""
            slots = NeighborChoice(ctx.hoods["near"])
            return slots.flow(HerdsState.wandering, slots.sample(ctx.rng), rate=roam)

        return [spread, settle, flow, allocate, choose, range_far, wander]


def distance(s: HerdsState) -> NDArray[np.float64]:
    """Return each herd's total variation distance from the ideal free distribution.

    Parameters
    ----------
    s : HerdsState
        A state.

    Returns
    -------
    ndarray of float64, shape (7,)
        In the order of `HERDS`: 0 where a herd's density is proportional to the rain, 1 at
        the opposite.
    """
    grass = s.rain * s.space.measures
    ideal = grass / grass.sum()
    return np.array(
        [0.5 * np.abs(s[herd] / s.total(herd) - ideal).sum() for herd in HERDS.values()]
    )
