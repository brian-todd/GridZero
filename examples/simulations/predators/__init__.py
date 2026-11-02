"""Grass, rabbits, foxes and wildfire on a board of triangles: a trophic cascade.

A step is a week. Grass grows back toward a capacity. Rabbits eat it, have young in proportion
to what they eat and die at a steady rate, and hop toward grass and away from foxes. After ten
years, foxes arrive: they hunt rabbits, turn half of what they eat into young, and prowl
toward rabbits where their own cell has too few to live on. Lightning starts fires in dry
summers, the wind carries them downwind, and animals caught in a fire die.

Theory (consumer-resource chains, as in Oksanen et al. 1981): a consumer holds its food at the
level where its births just replace its deaths. Without foxes, rabbits hold grass at their R*,
`Predators.rabbits_hold`. With foxes, the foxes hold rabbits at theirs, `Predators.foxes_hold`,
and grass grows back to where those few rabbits eat what grows (`Predators.greener`): a trophic
cascade, in which adding a predator makes the world greener. Fires and movement keep the board
from settling, so the means only approach these levels.

The foxes are in the state from the start, none of them until the ``foxes arrive`` phase
brings a few, once. ``python run.py predators`` runs it on 192x128 triangles.
"""

import operator
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Assign,
    Box,
    CellGraph,
    Claim,
    Context,
    Field,
    Flow,
    Hops,
    LatticeSpace,
    NeighborChoice,
    NeighborOperator,
    Nested,
    NestedState,
    Phase,
    Source,
    Space,
    Spread,
    State,
    phase,
    pull,
    push,
)

BURNING = 1
"""The value of the fire field in a burning cell: 0 can burn, 2 and up are recovering."""


class MeadowState(State):
    """The grass, and where it burns."""

    space: LatticeSpace  # the wind blows across a 2-D lattice
    grass = Field(np.float64, extensive=True)
    fire = Field(np.int8)  # 0 can burn, 1 burning, 2 to ash + 1 weeks since it burnt, plus 1


class Population(NestedState):
    """A species: how many there are in each cell."""

    population = Field(np.float64, extensive=True)


class RabbitsState(MeadowState):
    """The meadow with rabbits."""

    rabbits = Nested(Population)


class FoxesState(RabbitsState):
    """The meadow with rabbits and foxes."""

    foxes = Nested(Population)


@dataclass(frozen=True)
class Downwind:
    """A neighborhood: each cell's corner neighbors that lie downwind of it, a directed graph.

    Parameters
    ----------
    wind : tuple of float
        The direction the wind blows, in the coordinates of the lattice's centers.
    """

    wind: tuple[float, float]

    def compile(self, space: Space) -> NeighborOperator:
        """Find the neighbors downwind of every cell of a 2-D lattice.

        Parameters
        ----------
        space : Space
            A 2-D `LatticeSpace`.

        Returns
        -------
        NeighborOperator
            Each cell's corner neighbors whose centers lie ahead of it along the wind, the
            short way across wrapped axes.

        Raises
        ------
        TypeError
            If ``space`` is not a lattice.
        """
        if not isinstance(space, LatticeSpace):
            raise TypeError(f"the wind needs a lattice, not a {type(space).__name__}")
        graph = space.adjacency("vertex")
        sources, targets = graph.sources, graph.indices
        step = space.displacement(space.centers[sources], space.centers[targets])  # short way
        ahead = step @ np.array(self.wind) > 0
        downwind = CellGraph.from_edges(sources[ahead], targets[ahead], space.n_cells)
        return NeighborOperator(downwind)


@dataclass(frozen=True)
class Predators:
    """The parameters of the meadow, per unit area and per week: a `core.Model` of it.

    Parameters
    ----------
    wildfire : bool, default True
        Whether fires burn.
    year : int, default 52
        Steps in a year. Summer is its first half.
    foxes_arrive : int, default 10
        Years before the foxes arrive.
    places : int, default 20
        Cells the foxes arrive in, at random.
    founders : float, default 0.5
        Foxes arriving in each of those cells.
    capacity : float, default 1.0
        Grass per unit area that the meadow grows toward.
    regrowth : float, default 0.02
        Share of the missing grass that grows back each week, a fifth of that on ash.
    ash : int, default 26
        Weeks a burnt cell takes to recover.
    fuel : float, default 0.4
        Grass density below which a cell can't burn.
    lightning : float, default 2e-5
        Chance a cell with fuel is struck in a summer week.
    catch : float, default 0.3
        Chance a fire spreads from a burning cell to each cell downwind of it.
    wind : tuple of float, default (1.0, 0.3)
        The direction the wind blows.
    scorched : float, default 0.5
        Share of the animals in a burning cell that die.
    grazing : float, default 0.1
        Share of the grass density each rabbit asks for per week.
    rabbit_deaths : float, default 0.02
        Share of the rabbits that die each week.
    hopping : float, default 0.05
        Share of the rabbits that hop each week.
    fear : float, default 10.0
        How much the foxes around a cell put rabbits off it, per fox per unit area.
    hunting : float, default 0.1
        Share of the rabbit density each fox asks for per week.
    efficiency : float, default 0.5
        Share of the rabbits eaten that become foxes.
    fox_deaths : float, default 0.01
        Share of the foxes that die each week.
    prowling : float, default 0.3
        Largest share of a cell's foxes that prowls each week.
    dispersal : float, default 0.02
        Share of the foxes that leaves even where there are rabbits enough.
    """

    wildfire: bool = True
    year: int = 52
    foxes_arrive: int = 10
    places: int = 20
    founders: float = 0.5
    capacity: float = 1.0
    regrowth: float = 0.02
    ash: int = 26
    fuel: float = 0.4
    lightning: float = 2e-5
    catch: float = 0.3
    wind: tuple[float, float] = (1.0, 0.3)
    scorched: float = 0.5
    grazing: float = 0.1
    rabbit_deaths: float = 0.02
    hopping: float = 0.05
    fear: float = 10.0
    hunting: float = 0.1
    efficiency: float = 0.5
    fox_deaths: float = 0.01
    prowling: float = 0.3
    dispersal: float = 0.02

    @property
    def rabbits_hold(self) -> float:
        """The grass density at which rabbits alone hold the meadow: their R*."""
        return self.rabbit_deaths / self.grazing

    @property
    def foxes_hold(self) -> float:
        """The rabbit density at which foxes hold them: the foxes' R*."""
        return self.fox_deaths / (self.efficiency * self.hunting)

    @property
    def greener(self) -> float:
        """The grass density the meadow grows back to while foxes hold the rabbits."""
        return self.regrowth * self.capacity / (self.regrowth + self.grazing * self.foxes_hold)

    def start(self, space: LatticeSpace, seed: SupportsIndex) -> FoxesState:
        """Return a meadow at capacity, with a few rabbits everywhere and no foxes yet.

        Parameters
        ----------
        space : LatticeSpace
            The meadow: a 2-D lattice, for the wind.
        seed : int
            The seed of the rabbits' numbers.

        Returns
        -------
        FoxesState
            Up to 0.02 rabbits per unit area in each cell, uniformly at random.
        """
        rng = np.random.default_rng(operator.index(seed))
        rabbits = rng.uniform(0, 0.02, space.n_cells) * space.measures
        grass = self.capacity * space.measures
        return FoxesState(space, {FoxesState.grass: grass, FoxesState.rabbits.population: rabbits})

    def phases(self) -> list[Phase[FoxesState]]:
        """Return the phases of a week.

        Before the foxes arrive, the phases of foxes move none, and rabbits flee none.

        Returns
        -------
        list of Phase
            ``foxes arrive``, ``grow``, ``fire`` (with `wildfire`), ``graze``, ``hunt``,
            ``foxes die``, ``rabbits die``, ``flee`` and ``prowl``.
        """
        burns = [self._fire()] if self.wildfire else []
        foxes_die = self._dying("foxes", FoxesState.foxes, self.fox_deaths)
        return [
            self._arrive(),
            self._grow(),
            *burns,
            self._graze(),
            self._hunt(),
            foxes_die,
            self._rabbits_die(),
            self._flee(),
            self._prowl(),
        ]

    def _arrive(self) -> Phase[FoxesState]:
        """Return the phase in which the foxes arrive, once, in `places` cells at random."""
        week = self.foxes_arrive * self.year
        foxes = FoxesState.foxes.population

        @phase(writes=foxes, name="foxes arrive")
        def arrive(s: FoxesState, ctx: Context) -> Source | None:
            """Bring ``founders`` foxes to each of ``places`` cells, in the week they arrive."""
            if s.step != week:
                return None
            founders = np.zeros(s.n_cells)
            founders[ctx.rng.choice(s.n_cells, self.places, replace=False)] = self.founders
            return Source(foxes, founders)

        return arrive

    def _grow(self) -> Phase[MeadowState]:
        """Return the phase in which grass grows back toward its capacity, slowly on ash."""

        @phase(writes=MeadowState.grass)
        def grow(s: MeadowState, ctx: Context) -> Source:
            """Grow grass back toward its capacity, slowly on ash."""
            rate = np.where(s.fire > BURNING, self.regrowth / 5, self.regrowth)
            return Source(MeadowState.grass, rate * (self.capacity * s.space.measures - s.grass))

        return grow

    def _fire(self) -> Phase[MeadowState]:
        """Return the phase in which fires burn out, spread downwind and start from lightning."""

        @phase(writes=(MeadowState.fire, MeadowState.grass), hoods={"wind": Downwind(self.wind)})
        def fire(s: MeadowState, ctx: Context) -> list[Assign | Source]:
            """Burn fires out into ash, spread them downwind and start them in summer."""
            burning = s.fire == BURNING
            upwind = push(burning.astype(np.float64), ctx.hoods["wind"])  # burning cells upwind
            fuel = (s.fire == 0) & (s.density(MeadowState.grass) > self.fuel)
            summer = s.step % self.year < self.year // 2
            catch = 1 - (1 - self.catch) ** upwind + (self.lightning if summer else 0.0)
            ignite = fuel & (ctx.rng.random(s.n_cells) < catch)
            recovering = s.fire > 0
            fire = np.where(recovering, s.fire + 1, 0)
            fire[fire > self.ash + 1] = 0
            fire[ignite] = BURNING
            return [
                Assign(MeadowState.fire, fire),
                Source(MeadowState.grass, -s.grass * burning),
            ]

        return fire

    def _graze(self) -> Phase[RabbitsState]:
        """Return the phase in which rabbits eat grass, and it becomes rabbits."""

        @phase(writes=(RabbitsState.grass, RabbitsState.rabbits.population))
        def graze(s: RabbitsState, ctx: Context) -> Claim:
            """Let rabbits eat grass by its density and their numbers, into rabbits."""
            asked = self.grazing * s.density(RabbitsState.grass) * s.rabbits.population
            return Claim(RabbitsState.grass, asked, into=RabbitsState.rabbits.population)

        return graze

    def _rabbits_die(self) -> Phase[MeadowState]:
        """Return the phase in which rabbits die."""
        return self._dying("rabbits", RabbitsState.rabbits, self.rabbit_deaths)

    def _dying(self, name: str, species: type[Population], rate: float) -> Phase[MeadowState]:
        """Return the phase in which a species dies: a share ``rate`` each week, more in fires.

        One rule for any species, made into a phase for each.
        """

        @phase(writes=species, name=f"{name} die")
        def die(s: MeadowState, ctx: Context) -> Source:
            """Kill a share of the species, more in fires."""
            scorched = self.scorched * (s.fire == BURNING)
            return Source(species.population, -(rate + scorched) * s[species.population])

        return die

    def _flee(self) -> Phase[FoxesState]:
        """Return the phase in which rabbits hop toward grass and away from foxes, not into fire."""

        @phase(writes=FoxesState.rabbits.population, hoods={"near": Adjacency(), "around": Hops(2)})
        def flee(s: FoxesState, ctx: Context) -> Spread:
            """Hop rabbits toward grass and away from the foxes within two hops, not into fire."""
            foxes = pull(s.density(FoxesState.foxes.population), ctx.hoods["around"], "mean")
            toward = s.density(FoxesState.grass) / (1 + self.fear * foxes)
            near = ctx.hoods["near"]
            rabbits = FoxesState.rabbits.population
            return Spread(rabbits, near, self.hopping, toward, domain=s.fire != BURNING)

        return flee

    def _hunt(self) -> Phase[FoxesState]:
        """Return the phase in which foxes catch rabbits, and a share becomes young foxes."""
        rabbits, foxes = FoxesState.rabbits.population, FoxesState.foxes.population

        @phase(writes=(rabbits, foxes))
        def hunt(s: FoxesState, ctx: Context) -> Claim:
            """Let foxes catch rabbits by the rabbit density and their numbers, into young."""
            asked = self.hunting * s.density(rabbits) * s.foxes.population
            return Claim(rabbits, asked, into=foxes, efficiency=self.efficiency)

        return hunt

    def _prowl(self) -> Phase[FoxesState]:
        """Return the phase in which foxes short of rabbits move toward more."""

        @phase(writes=FoxesState.foxes.population, hoods={"near": Adjacency("vertex")})
        def prowl(s: FoxesState, ctx: Context) -> Flow:
            """Move foxes short of rabbits to the cell around them with the most, not into fire.

            The share that moves grows with how far the cell's rabbits are below what foxes
            need, from ``dispersal`` (young foxes leaving) where there are enough to
            ``prowling`` where there are none.
            """
            rabbits = s.density(FoxesState.rabbits.population)
            slots = NeighborChoice(ctx.hoods["near"])
            unburnt = s.fire != BURNING  # changes every step, so it is given to each call
            best = slots.best(slots.values(rabbits), domain=unburnt)
            enough = self.foxes_hold  # the rabbit density at which foxes break even
            rate = Box(s.n_cells, self.dispersal, self.prowling).clip(
                self.prowling * (1 - rabbits / enough)
            )
            return slots.flow(FoxesState.foxes.population, best, rate, domain=unburnt)

        return prowl


def means(s: FoxesState) -> NDArray[np.float64]:
    """Return the mean densities of grass, rabbits and foxes, and the share burning.

    Parameters
    ----------
    s : FoxesState
        A state.

    Returns
    -------
    ndarray of float64, shape (4,)
        Grass, rabbits and foxes per unit area, and the share of cells burning.
    """
    area = s.space.measures.sum()
    fields = (FoxesState.grass, FoxesState.rabbits.population, FoxesState.foxes.population)
    return np.array([*(s.total(field) / area for field in fields), (s.fire == BURNING).mean()])
