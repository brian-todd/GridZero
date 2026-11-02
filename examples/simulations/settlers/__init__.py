"""Farmers and herders settling a world of seasons, droughts, raids and worn-out soil.

A month is one step. Each land cell grows food, faster in summer and on fertile soil. Two
peoples harvest it into stores, eat from their stores, have children when fed and die when
starving, and move toward food when their stores run low, carrying their stores with them.
Herders raid farmers they share a cell with. Droughts strike in summer, and land grazed bare
loses fertility, which comes back slowly once left alone. A people too few in a cell to
live on dies out there.

The peoples differ in what they are good at:

- **farmers** harvest more on low land and keep their stores longer, but rarely move;
- **herders** harvest as much in the hills as on the coasts, move readily and win their
  raids, but their stores spoil.

The point is less the history than the run: eight phases over thousands of steps.
``python run.py settlers`` runs it on 128x128 hexes, and prints the totals
along the run and how fast each stretch ran.
"""

from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Assign,
    Claim,
    Context,
    Field,
    LatticeSpace,
    Nested,
    NestedState,
    Phase,
    Source,
    Spread,
    State,
    check_int,
    derived,
    phase,
    random_points,
    within,
)
from gridzero.worlds import generate_terrain

PEOPLES = ("farmers", "herders")


class Settlers(NestedState):
    """The two peoples, farmers and herders, and the food each holds: it moves with them."""

    people = Field(np.float64, (2,), extensive=True)
    stores = Field(np.float64, (2,), extensive=True)


class SettlersState(State):
    """The land, its food and fertility, and the settlers on it."""

    space: LatticeSpace  # its terrain is made on a 2-D lattice (`worlds.generate_terrain`)
    height = Field(np.float64, static=True)  # relative to sea level, in [-1, 1]
    harvest = Field(np.float64, (2,), static=True)  # food a person gathers, in months of need
    fertility = Field(np.float64, default=1.0)  # in [0, 1]: worn out by grazing land bare
    food = Field(np.float64, extensive=True)
    settlers = Nested(Settlers)

    @derived(height)
    def land(self) -> NDArray[np.bool_]:
        """Return the cells above sea level: the only ones with food, and that people enter."""
        return self.height > 0


@dataclass(frozen=True)
class Settlement:
    """The parameters of the settlement, per unit area and per month: a `core.Model` of it.

    Each pair holds a value per people: farmers, then herders.

    Parameters
    ----------
    year : int, default 12
        Steps in a year.
    settled : float, default 1.0
        People per unit area for a cell to count as settled.
    minimum : float, default 0.01
        People per unit area below which a people dies out in a cell.
    capacity : float, default 400.0
        Food that fertile land grows toward.
    regrowth : float, default 0.08
        Share of the missing food that grows back in an average month.
    seasons : float, default 0.9
        How far growth swings between summer and winter: 0 for none, 1 for none in winter.
    wear : float, default 0.01
        Fertility lost each month by settled land grazed below ``grazed`` of its capacity.
    grazed : float, default 0.4
        See ``wear``.
    recovery : float, default 0.002
        Share of lost fertility that comes back each month on land left alone.
    droughts : int, default 3
        Places that lose most of their food each summer.
    drought_radius : float, default 6.0
        How far a drought reaches from its place.
    drought_loss : float, default 0.8
        Share of the food a drought takes.
    farming : float, default 3.0
        Food a farmer gathers on fertile lowland, in months of ``need``.
    hills : float, default 0.4
        Height at which farming yields a tenth of that.
    herding : float, default 1.5
        Food a herder gathers on fertile land of any height.
    need : float, default 1.0
        Food a person eats in a month.
    spoilage : pair of float, default (0.01, 0.05)
        Share of the stores lost each month.
    births : float, default 0.015
        Children per person per month, when fully fed.
    deaths : float, default 0.005
        Deaths per person per month, whatever they eat.
    starving : float, default 0.08
        Extra deaths per person per month, when nothing is eaten.
    reserve : float, default 6.0
        Months of food in store at which people stop looking for more.
    mobility : pair of float, default (0.01, 0.15)
        Share who leave a cell each month, when hungry.
    raids : float, default 0.002
        Meetings with herders per farmer per month, per herder per unit area.
    kills : pair of float, default (0.1, 0.02)
        Farmers and herders killed per meeting.
    loot : float, default 1.0
        Share of a farmer's stores taken per meeting.
    """

    year: int = 12
    settled: float = 1.0
    minimum: float = 0.01
    capacity: float = 400.0
    regrowth: float = 0.08
    seasons: float = 0.9
    wear: float = 0.01
    grazed: float = 0.4
    recovery: float = 0.002
    droughts: int = 3
    drought_radius: float = 6.0
    drought_loss: float = 0.8
    farming: float = 3.0
    hills: float = 0.4
    herding: float = 1.5
    need: float = 1.0
    spoilage: tuple[float, float] = (0.01, 0.05)
    births: float = 0.015
    deaths: float = 0.005
    starving: float = 0.08
    reserve: float = 6.0
    mobility: tuple[float, float] = (0.01, 0.15)
    raids: float = 0.002
    kills: tuple[float, float] = (0.1, 0.02)
    loot: float = 1.0

    def start(self, space: LatticeSpace, seed: SupportsIndex) -> SettlersState:
        """Return terrain, food at capacity, and each people in a few places with a year of food.

        Parameters
        ----------
        space : LatticeSpace
            The world.
        seed : int
            The seed of the terrain and of the homes.

        Returns
        -------
        SettlersState
            Half the world land, and 100 people of each people in each of 3 places on it.
        """
        height = generate_terrain(space, seed=seed, land=0.5)
        land = height > 0
        rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
        people = np.zeros((2, space.n_cells))
        for group in (0, 1):
            homes = space.locate(random_points(space, rng, 3, where=land))
            people[group, homes] = 100.0
        farming = self.farming * np.clip(1 - height / self.hills, 0.1, 1)
        return SettlersState(
            space,
            {
                SettlersState.height: height,
                SettlersState.harvest: np.stack([farming, np.full(space.n_cells, self.herding)]),
                SettlersState.food: self.capacity * space.measures * land,
                SettlersState.settlers.people: people,
                SettlersState.settlers.stores: self.year * self.need * people,
            },
        )

    def capacity_of(self, s: SettlersState) -> NDArray[np.float64]:
        """Return the food each cell grows toward: ``capacity`` per unit area on fertile land.

        Parameters
        ----------
        s : SettlersState
            A state.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            Food, none at sea.
        """
        return self.capacity * s.fertility * s.space.measures * s.land

    def hunger(self, s: SettlersState) -> NDArray[np.float64]:
        """Return how short each people is of ``reserve`` months of food.

        Parameters
        ----------
        s : SettlersState
            A state.

        Returns
        -------
        ndarray of float64, shape (2, n_cells)
            From 0 with ``reserve`` months in store (or no people) to 1 with none.
        """
        stores, people = s.settlers.stores, s.settlers.people
        months = np.divide(
            stores, self.need * people, out=np.full_like(stores, self.reserve), where=people > 0
        )
        return 1 - np.minimum(months / self.reserve, 1)

    def phases(self) -> list[Phase[SettlersState]]:
        """Return the phases of a month.

        Returns
        -------
        list of Phase
            ``regrow``, ``harvest``, ``live``, ``raid``, ``migrate``, ``drought`` (once a year,
            in summer), ``soil`` and ``die_out``.
        """
        spoilage = np.array(self.spoilage)[:, None]
        mobility = np.array(self.mobility)[:, None]
        kills = np.array(self.kills)[:, None]

        @phase(writes=SettlersState.food)
        def regrow(s: SettlersState, ctx: Context) -> Source:
            """Grow food back toward each cell's capacity, fast in summer and slowly in winter."""
            season = 1 + self.seasons * np.sin(2 * np.pi * s.step / self.year)
            capacity = self.capacity_of(s)
            return Source(SettlersState.food, self.regrowth * season * (capacity - s.food))

        @phase(writes=(SettlersState.food, SettlersState.settlers.stores))
        def harvest(s: SettlersState, ctx: Context) -> Claim:
            """Gather food into each people's stores, by the food density and their numbers."""
            density = s.density(SettlersState.food)
            asked = s.harvest * self.need * density / self.capacity * s.settlers.people
            return Claim(SettlersState.food, asked, into=SettlersState.settlers.stores)

        @phase(writes=SettlersState.settlers)
        def live(s: SettlersState, ctx: Context) -> list[Source]:
            """Feed people from their stores; they have children when fed, die when starving."""
            need = self.need * s.settlers.people
            eaten = np.minimum(s.settlers.stores, need)
            fed = np.divide(eaten, need, out=np.ones_like(need), where=need > 0)
            born = self.births * fed * s.settlers.people
            died = (self.deaths + self.starving * (1 - fed)) * s.settlers.people
            spoiled = spoilage * (s.settlers.stores - eaten)
            return [
                Source(SettlersState.settlers.stores, -(eaten + spoiled)),
                Source(SettlersState.settlers.people, born - died),
            ]

        @phase(writes=SettlersState.settlers)
        def raid(s: SettlersState, ctx: Context) -> list[Source]:
            """Where herders meet farmers, kill some of both, and give herders farmers' stores."""
            herders = s.density(SettlersState.settlers.people)[1]
            meetings = self.raids * s.settlers.people[0] * herders  # (n_cells,)
            killed = kills * meetings  # farmers, herders
            looted = np.minimum(self.loot * self.raids * herders, 1) * s.settlers.stores[0]
            # No intent moves amounts between two peoples of one field, so the loot leaves the
            # farmers with one source and reaches the herders with another. It is never more
            # than the farmers hold, so nothing is clipped and the stores are conserved.
            return [
                Source(SettlersState.settlers.people, -killed),
                Source(SettlersState.settlers.stores, np.stack([-looted, looted])),
            ]

        @phase(writes=SettlersState.settlers, hoods={"near": Adjacency()})
        def migrate(s: SettlersState, ctx: Context) -> Spread:
            """Move hungry people toward food per person, carrying their stores with them."""
            leave = mobility * self.hunger(s)
            food_per_person = s.food / (s.settlers.people.sum(axis=0) + s.space.measures)
            # The settlers move as one: each cell's stores split as its people do.
            near = ctx.hoods["near"]
            return Spread(SettlersState.settlers, near, leave, food_per_person, domain=s.land)

        @phase(writes=SettlersState.food, every=self.year, offset=self.year // 4)
        def drought(s: SettlersState, ctx: Context) -> Source:
            """Take most of the food of a few places, in summer."""
            space = s.space
            hit = np.zeros(s.n_cells)
            places = space.locate(random_points(space, ctx.rng, self.droughts, where=s.land))
            for place in places:
                hit[within(space, place, self.drought_radius)] = self.drought_loss
            return Source(SettlersState.food, -hit * s.food)

        @phase(writes=SettlersState.fertility)
        def soil(s: SettlersState, ctx: Context) -> Assign:
            """Wear out land grazed bare, and let land left alone slowly get its fertility back."""
            settled = s.density(SettlersState.settlers.people).sum(axis=0) > self.settled
            grazed = settled & (s.food < self.grazed * self.capacity_of(s))
            change = np.where(grazed, -self.wear, self.recovery * (1 - s.fertility))
            fertility = np.clip(s.fertility + change, 0, 1)
            return Assign(SettlersState.fertility, fertility, where=s.land)

        @phase(writes=SettlersState.settlers)
        def die_out(s: SettlersState, ctx: Context) -> list[Source]:
            """Remove a people too few in a cell to live on, and its stores with it.

            Amounts are continuous, so without this rule a spread leaves a trace of every
            people in every cell it can reach, which grows back wherever it would do well: a
            cell is never empty.
            """
            gone = s.density(SettlersState.settlers.people) < self.minimum
            return [
                Source(SettlersState.settlers.people, -s.settlers.people * gone),
                Source(SettlersState.settlers.stores, -s.settlers.stores * gone),
            ]

        return [regrow, harvest, live, raid, migrate, drought, soil, die_out]
