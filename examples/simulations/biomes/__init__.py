"""Grassland and forest on a planet, each with its own grazers, and wolves that hunt in both.

A step is a week. The planet is grassland in the south and forest in the north, mixed in a band
around the equator: each biome covers a share of each cell. A biome's plants grow back toward
its share of the cell, and its grazers (rabbits on the grassland, deer in the forest) eat them,
turn what they eat into young, die at a steady rate and wander toward plants within their
biome. Each biome is a nested state, with its grazers nested in it: two levels deep, as in
``state.forest.grazers.population``. The biomes' phases are written once, for any biome. Wolves
arrive later (they are in the state from the start, none until the ``wolves arrive`` phase),
hunt the grazers of both biomes and roam toward prey.

Theory (consumer-resource chains, and apparent competition, Holt 1977):

1. **apart:** without wolves, each biome's grazers hold its plants at their R*, ``deaths /
   grazing``, per unit area of a cell all one biome. In weekly steps, where deaths follow births,
   it is ``deaths / (grazing * (1 - deaths))`` (`Rates.rstar`): 0.208 on the grassland and 0.309
   in the forest. Grazers find that much when they eat, after the plants have grown, so `means`
   measures it then.
2. **wolves, no deer:** wolves hold the rabbits at their own R*, `Biomes.held`, and the grass
   grows back toward a level the theory gives (`Biomes.greener`), as in `predators`. These
   levels are for continuous time: weekly steps shift them by a few percent, as in 1.
3. **wolves and deer:** where the deer feed the wolves, the wolves press harder on the rabbits,
   though rabbits and deer share no food: apparent competition. In the band where the biomes
   mix, the rabbits fall most where the forest covers most of a cell, and its deer are most
   (`by_cover`).

``python run.py biomes`` runs it on an icosphere of 2,562 cells, with deer
and without.
"""

import itertools
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Claim,
    Context,
    Field,
    Nested,
    NestedState,
    Phase,
    Source,
    Space,
    Spread,
    State,
    derived,
    phase,
)


@dataclass(frozen=True)
class Rates:
    """What sets a biome apart: how its plants grow and how its grazers live, per week.

    Parameters
    ----------
    regrowth : float
        Share of the missing plants that grows back.
    grazing : float
        Share of the plant density each grazer asks for.
    deaths : float
        Share of the grazers that die.
    wandering : float
        Share of the grazers that wander to neighboring cells.
    """

    regrowth: float
    grazing: float
    deaths: float
    wandering: float

    @property
    def rstar(self) -> float:
        """The plant density at which the grazers' births replace their deaths, weekly."""
        return self.deaths / (self.grazing * (1 - self.deaths))

    def growth(self, state: State, biome: type[Biome]) -> NDArray[np.float64]:
        """Return the plants that grow back in a week: the rule of the biome's ``grows`` phase.

        Parameters
        ----------
        state : State
            A state with the biome.
        biome : Biome subclass
            The biome, as nested in the state's class.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            ``regrowth`` times what each cell lacks of the biome's capacity.
        """
        return self.regrowth * (state[biome.capacity] - state[biome.plants])


class Population(NestedState):
    """A species: how many there are in each cell."""

    population = Field(np.float64, extensive=True)


class Biome(NestedState):
    """Plants and the grazers that eat them, over the share of each cell the biome covers."""

    cover = Field(np.float64, static=True)  # share of the cell's area, from 0 to 1
    plants = Field(np.float64, extensive=True)
    grazers = Nested(Population)

    @derived(cover)
    def capacity(self) -> NDArray[np.float64]:
        """Return the plants the biome holds at most in each cell, shape (n_cells,)."""
        return self.cover * self.space.measures


class LandState(State):
    """The two biomes."""

    grassland = Nested(Biome)
    forest = Nested(Biome)


class WolvesState(LandState):
    """The two biomes, and the wolves that hunt in both."""

    wolves = Nested(Population)


BIOMES: dict[str, type[Biome]] = {"grassland": LandState.grassland, "forest": LandState.forest}
"""Each biome by name: its grazers are rabbits on the grassland and deer in the forest."""


@dataclass(frozen=True)
class Biomes:
    """The parameters of the biomes and the wolves, per week: a `core.Model` of them.

    Parameters
    ----------
    deer : bool, default True
        Whether the forest has deer. The grassland always has rabbits.
    year : int, default 52
        Steps in a year.
    wolves_arrive : int, default 20
        Years before the wolves arrive.
    places : int, default 20
        Cells of each biome the wolves arrive in, at random among those it covers alone.
    founders : float, default 0.05
        Wolves arriving in each of those cells.
    mixed : float, default 0.2
        Half the width of the band where the biomes mix, in sines of the latitude.
    grassland : Rates, default Rates(0.05, 0.2, 0.04, 0.05)
        The grassland's plants and rabbits.
    forest : Rates, default Rates(0.02, 0.1, 0.03, 0.03)
        The forest's plants and deer.
    hunting : float, default 0.2
        Share of each grazer density each wolf asks for.
    efficiency : float, default 0.3
        Share of the grazers eaten that become wolves.
    wolf_deaths : float, default 0.012
        Share of the wolves that die.
    roaming : float, default 0.2
        Share of the wolves that roam.
    """

    deer: bool = True
    year: int = 52
    wolves_arrive: int = 20
    places: int = 20
    founders: float = 0.05
    mixed: float = 0.2
    grassland: Rates = Rates(regrowth=0.05, grazing=0.2, deaths=0.04, wandering=0.05)
    forest: Rates = Rates(regrowth=0.02, grazing=0.1, deaths=0.03, wandering=0.03)
    hunting: float = 0.2
    efficiency: float = 0.3
    wolf_deaths: float = 0.012
    roaming: float = 0.2

    def rates(self) -> dict[str, tuple[type[Biome], Rates]]:
        """Return each biome by name, with its rates, in the order of `BIOMES`.

        Returns
        -------
        dict of str to (Biome, Rates)
            ``"grassland"`` and ``"forest"``.
        """
        return {
            "grassland": (BIOMES["grassland"], self.grassland),
            "forest": (BIOMES["forest"], self.forest),
        }

    @property
    def held(self) -> float:
        """The grazer density at which wolves hold one biome's grazers, in continuous time."""
        return self.wolf_deaths / (self.efficiency * self.hunting)

    def greener(self, rates: Rates) -> float:
        """Return a biome's plant density while wolves hold its grazers, in continuous time.

        Parameters
        ----------
        rates : Rates
            The biome's rates.

        Returns
        -------
        float
            Plants per unit area of a cell all that biome.
        """
        return rates.regrowth / (rates.regrowth + rates.grazing * self.held)

    def start(self, space: Space, seed: SupportsIndex) -> WolvesState:
        """Return a planet at capacity, grassland in the south and forest in the north.

        Parameters
        ----------
        space : Space
            A sphere: the biomes follow the third coordinate of the cells' centers.
        seed : int
            Unused: the start draws nothing.

        Returns
        -------
        WolvesState
            Plants at capacity, grazers at a tenth of it, and no wolves yet.
        """
        z = space.centers[:, 2]
        forest = np.clip(0.5 + z / (2 * self.mixed), 0.0, 1.0)
        state = WolvesState(
            space, {WolvesState.grassland.cover: 1 - forest, WolvesState.forest.cover: forest}
        )
        values = {}
        for name, biome in BIOMES.items():
            capacity = state[biome.capacity]
            values[biome.plants] = capacity
            values[biome.grazers.population] = 0.1 * capacity * (self.deer or name == "grassland")
        return state.replace(values)

    def phases(self) -> list[Phase[WolvesState]]:
        """Return the phases of a week.

        Before the wolves arrive, their phases move none.

        Returns
        -------
        list of Phase
            ``wolves arrive``; per biome, grassland first, its plants grow and its grazers
            graze, die and wander; then the wolves ``hunt``, die and ``roam``.
        """
        return [self._arrive(), *self._land(), *self._wolves()]

    def _arrive(self) -> Phase[WolvesState]:
        """Return the phase in which the wolves arrive, once, in `places` cells of each biome."""
        week = self.wolves_arrive * self.year
        wolves = WolvesState.wolves.population

        @phase(writes=wolves, name="wolves arrive")
        def arrive(s: WolvesState, ctx: Context) -> Source | None:
            """Bring ``founders`` wolves to ``places`` cells of each biome, where it is alone."""
            if s.step != week:
                return None
            founders = np.zeros(s.n_cells)
            for biome in BIOMES.values():
                alone = np.flatnonzero(s[biome.cover] == 1)
                founders[ctx.rng.choice(alone, self.places, replace=False)] = self.founders
            return Source(wolves, founders)

        return arrive

    def _land(self) -> list[Phase[LandState]]:
        """Return the phases of both biomes, grassland first."""
        return [
            each
            for name, (biome, rates) in self.rates().items()
            for each in _biome_phases(name, biome, rates)
        ]

    def _wolves(self) -> list[Phase[WolvesState]]:
        """Return the phases of the wolves: they hunt (and breed on their catch), die and roam."""
        wolves = WolvesState.wolves.population

        @phase(writes=(*(b.grazers.population for b in BIOMES.values()), wolves))
        def hunt(s: WolvesState, ctx: Context) -> list[Claim]:
            """Let wolves catch the grazers of each biome by their density, and breed on them."""
            return [
                Claim(
                    biome.grazers.population,
                    self.hunting * s.density(biome.grazers.population) * s.wolves.population,
                    into=wolves,
                    efficiency=self.efficiency,
                )
                for biome in BIOMES.values()
            ]

        @phase(writes=WolvesState.wolves, hoods={"near": Adjacency()})
        def roam(s: WolvesState, ctx: Context) -> Spread:
            """Move wolves toward the cells with the most prey, of either biome."""
            prey = s.density(WolvesState.grassland.grazers.population) + s.density(
                WolvesState.forest.grazers.population
            )
            return Spread(WolvesState.wolves, ctx.hoods["near"], self.roaming, toward=prey + 1e-3)

        return [
            hunt,
            _dying("wolves", WolvesState.wolves, self.wolf_deaths),
            roam,
        ]

    def means(self, s: WolvesState) -> NDArray[np.float64]:
        """Return each biome's plants and grazers where it covers a cell alone, and the wolves.

        Plants are counted as the grazers find them, once they have grown in the next step.

        Parameters
        ----------
        s : WolvesState
            A state.

        Returns
        -------
        ndarray of float64, shape (5,)
            Per unit area: grass, rabbits, forest plants and deer where their biome covers a
            cell alone, then wolves over the planet.
        """
        out = []
        for biome, rates in self.rates().values():
            alone = s[biome.cover] == 1
            area = s.space.measures[alone].sum()
            grown = s[biome.plants] + rates.growth(s, biome)  # as `grows` will, next step
            out += [grown[alone].sum() / area, s[biome.grazers.population][alone].sum() / area]
        wolves = s.total(WolvesState.wolves.population) / s.space.measures.sum()
        return np.array([*out, wolves])


def _biome_phases(name: str, biome: type[Biome], rates: Rates) -> list[Phase[State]]:
    """Return the phases of a biome: its plants grow, and its grazers eat, die and wander."""

    @phase(writes=biome.plants, name=f"{name} grows")
    def grow(s: State, ctx: Context) -> Source:
        """Grow the plants back toward the biome's capacity."""
        return Source(biome.plants, rates.growth(s, biome))

    @phase(writes=biome, name=f"{name} grazed")
    def graze(s: State, ctx: Context) -> Claim:
        """Let the grazers eat plants by their density and the grazers' numbers, into young."""
        asked = rates.grazing * s.density(biome.plants) * s[biome.grazers.population]
        return Claim(biome.plants, asked, into=biome.grazers.population)

    @phase(writes=biome.grazers, name=f"{name} grazers wander", hoods={"near": Adjacency()})
    def wander(s: State, ctx: Context) -> Spread:
        """Move a share of the grazers toward plants, within the biome."""
        toward = s.density(biome.plants)
        return Spread(
            biome.grazers, ctx.hoods["near"], rates.wandering, toward, domain=s[biome.cover] > 0
        )

    return [grow, graze, _dying(f"{name} grazers", biome.grazers, rates.deaths), wander]


def _dying(name: str, species: type[Population], rate: float) -> Phase[State]:
    """Return the phase in which a share ``rate`` of a species dies each week."""

    @phase(writes=species, name=f"{name} die")
    def die(s: State, ctx: Context) -> Source:
        """Kill a share of the species."""
        return Source(species.population, -rate * s[species.population])

    return die


COVER = (0.0, 1 / 3, 2 / 3, 1.0)
"""Bands of the forest's share of a cell, in the mixed band: the columns of `by_cover`."""


def by_cover(s: LandState) -> NDArray[np.float64]:
    """Return rabbits and deer per unit area in each band of forest cover.

    Parameters
    ----------
    s : LandState
        A state.

    Returns
    -------
    ndarray of float64, shape (2, 3)
        Rabbits, then deer, in the cells whose forest cover lies strictly inside each band of
        `COVER`.
    """
    forest = s[LandState.forest.cover]
    out = np.zeros((2, len(COVER) - 1))
    for b, (low, high) in enumerate(itertools.pairwise(COVER)):
        band = (forest > low) & (forest < high)
        area = s.space.measures[band].sum()
        for i, biome in enumerate(BIOMES.values()):
            out[i, b] = s[biome.grazers.population][band].sum() / area
    return out
