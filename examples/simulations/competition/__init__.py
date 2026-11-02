"""Two groups competing for one food supply, checked against the R* rule of ecology.

Each land cell grows food toward a capacity. Two groups eat it, in proportion to how much food
there is and how many of them there are, and die at a steady rate. When food runs short, the
cell's food is shared out in proportion to what each group asked for (scramble competition).
A few people of each group drift to neighboring land cells every step.

Theory (Tilman's R* rule): a group alone settles where the food it finds just replaces its
deaths, at the food density ``R* = mortality / appetite``. With two groups on one food, the
group with the lower R* eats the food down to a level where the other can't replace its deaths,
and drives it out. So:

- **one niche:** every cell favors the same group, which takes over every landmass,
  however rare it starts;
- **two niches:** mortality depends on height, so each group has the lower R* on half of the
  land: lowlanders below the median height, highlanders above it. Each should hold its own
  half, with a mixed border where people drift across.

``python run.py competition`` runs both on a 64x64 hex torus and prints the checks.
"""

from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Claim,
    Context,
    Field,
    LatticeSpace,
    Phase,
    Source,
    Spread,
    State,
    derived,
    phase,
)
from gridzero.worlds import generate_terrain

GROUPS = ("lowlanders", "highlanders")


class CompetitionState(State):
    """Land and sea, the food on the land, and the two groups that eat it."""

    space: LatticeSpace  # its terrain is made on a 2-D lattice (`worlds.generate_terrain`)
    height = Field(np.float64, static=True)  # relative to sea level, in [-1, 1]
    capacity = Field(np.float64, static=True)  # food each cell grows toward, none at sea
    mortality = Field(np.float64, (2,), static=True)  # share of each group dying per step
    food = Field(np.float64, extensive=True)
    people = Field(np.float64, (2,), extensive=True)

    @derived(height)
    def land(self) -> NDArray[np.bool_]:
        """Cells above sea level: the only ones with food, and the only ones people enter."""
        return self.height > 0


@dataclass(frozen=True)
class Competition:
    """The parameters of the competition, per step: a `core.Model` of its states and phases.

    Parameters
    ----------
    split : bool, default False
        Whether there are two niches (see `niches`).
    land : float, default 0.5
        Share of the cells that are land.
    appetite : tuple of float, default (0.5, 0.25)
        Share of a cell's food density each person of each group asks for: lowlanders,
        highlanders.
    capacity : float, default 1.0
        Food per unit area that land grows toward.
    regrowth : float, default 0.1
        Share of the missing food that grows back each step.
    drift : float, default 0.02
        Share of each group that moves to neighboring land each step.
    """

    split: bool = False
    land: float = 0.5
    appetite: tuple[float, float] = (0.5, 0.25)
    capacity: float = 1.0
    regrowth: float = 0.1
    drift: float = 0.02

    @property
    def _appetite(self) -> NDArray[np.float64]:
        """`appetite` as a column, shape (2, 1)."""
        return np.array(self.appetite)[:, None]

    def rstar(self, mortality: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the food density at which each group's births replace its deaths.

        Parameters
        ----------
        mortality : ndarray of float64, shape (2, n_cells)
            Each group's share dying per step.

        Returns
        -------
        ndarray of float64, shape (2, n_cells)
            ``mortality / appetite``.
        """
        return mortality / self._appetite

    def niches(self, height: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the mortality of each group in each cell.

        With one niche, every cell favors the lowlanders (R* 0.2 against 0.4 at the default
        appetites). With two, R* rises with height for lowlanders, from 0.1 to 0.7, and falls
        for highlanders, from 0.5 to 0.3, and they cross at the median height of the land.

        Parameters
        ----------
        height : ndarray of float64, shape (n_cells,)
            Height relative to sea level: land above 0.

        Returns
        -------
        ndarray of float64, shape (2, n_cells)
            The share of each group dying per step.
        """
        land = height > 0
        if not self.split:
            return np.broadcast_to(np.array([[0.1], [0.1]]), (2, height.size)).copy()
        rank = np.zeros_like(height)
        rank[land] = (np.argsort(np.argsort(height[land])) + 0.5) / land.sum()  # 0 to 1 over land
        rstar = np.stack([0.1 + 0.6 * rank, 0.5 - 0.2 * rank])  # equal at rank 0.5
        return rstar * self._appetite

    def start(self, space: LatticeSpace, seed: SupportsIndex) -> CompetitionState:
        """Return terrain, and the highlanders settled alone beside lowlanders at 1% of them.

        Parameters
        ----------
        space : LatticeSpace
            The cells.
        seed : int
            The seed of the terrain (`worlds.generate_terrain`), with ``land`` of it land.

        Returns
        -------
        CompetitionState
            Food on land at the highlanders' R*, and each group in every land cell.
        """
        height = generate_terrain(space, seed=seed, land=self.land)
        mortality = self.niches(height)
        land = height > 0
        rstar = mortality[1] / self.appetite[1]
        food = np.where(land, rstar, 0.0) * space.measures  # highlanders alone, settled
        alone = self.regrowth * (self.capacity - rstar) / mortality[1]  # their density there
        people = np.stack([0.01 * alone, alone]) * space.measures * land
        return CompetitionState(
            space,
            {
                CompetitionState.height: height,
                CompetitionState.capacity: self.capacity * space.measures * land,
                CompetitionState.mortality: mortality,
                CompetitionState.food: food,
                CompetitionState.people: people,
            },
        )

    def phases(self) -> list[Phase[CompetitionState]]:
        """Return the phases of a step: food regrows, people eat and die, and a few drift.

        Returns
        -------
        list of Phase
            ``regrow``, ``feed`` and ``drift``, in that order.
        """
        appetite = self._appetite

        @phase(writes=CompetitionState.food)
        def regrow(s: CompetitionState, ctx: Context) -> Source:
            """Grow food back toward the capacity of each land cell."""
            return Source(CompetitionState.food, self.growth(s))

        @phase(writes=(CompetitionState.food, CompetitionState.people))
        def feed(s: CompetitionState, ctx: Context) -> list[Claim | Source]:
            """Feed each group by food density and its numbers, and let some die."""
            asked = appetite * s.density(CompetitionState.food) * s.people
            return [
                Claim(CompetitionState.food, asked, into=CompetitionState.people),
                Source(CompetitionState.people, -s.mortality * s.people),
            ]

        @phase(writes=CompetitionState.people, hoods={"near": Adjacency()})
        def drift(s: CompetitionState, ctx: Context) -> Spread:
            """Move a few people of each group to neighboring land cells."""
            near = ctx.hoods["near"]
            return Spread(CompetitionState.people, near, leave=self.drift, domain=s.land)

        return [regrow, feed, drift]

    def growth(self, state: CompetitionState) -> NDArray[np.float64]:
        """Return the food that grows back in each cell in a step: ``regrow``'s rule.

        Parameters
        ----------
        state : CompetitionState
            The state ``regrow`` reads.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            ``regrowth`` times what each cell lacks of its capacity.
        """
        return self.regrowth * (state.capacity - state.food)

    def meal(self, state: CompetitionState) -> NDArray[np.float64]:
        """Return the food density people find when they eat: after it regrows.

        Observers see the food left after eating, so this regrows it by `growth`, as the next
        step will, so that the two never disagree.

        Parameters
        ----------
        state : CompetitionState
            A state between steps.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            Food per unit area.
        """
        return (state.food + self.growth(state)) / state.space.measures
