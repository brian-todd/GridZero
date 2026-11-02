"""One founder species on an empty planet, adapting to its climates and splitting into species.

Each individual carries a heritable trait: the temperature it does best at, one of `BINS`
values. Births fall off with the distance between that temperature and the cell's own (a
Gaussian of width ``tolerance``). Offspring inherit their parent's trait, except a share
``mutation`` born into each neighboring bin. Crowding raises deaths, felt most from
individuals of similar traits (a quartic kernel of width ``competition``). A few individuals
spread to neighboring land every step, and a trait too rare in a cell to live on dies out
there.

Theory:

- **mutation against selection:** in one place, the traits settle around the best one in the
  shape of the dominant eigenvector of a step's linear part. With selection near the best
  trait as strong as ``s`` and mutation spreading traits as fast as ``D``, that shape is
  close to a Gaussian of variance ``sqrt(D / s)`` (Kimura);
- **species:** with competition narrower than the tolerance, the traits split into clusters
  with gaps between them, each the main one in its own band of climate (Pigolotti et al.
  2007; Doebeli and Dieckmann 2003). With all traits competing alike (``competition=inf``),
  they stay one smooth gradient at the default dispersal;
- **gene flow against local adaptation:** each cell's mean trait follows its temperature, but
  less steeply as more individuals move between cells (Kirkpatrick and Barton 1997);
- **a warming climate:** in one place, traits follow an optimum moving ``v`` a step at a lag
  of about ``v / (2 sqrt(D * s))``, and the population declines above a critical speed,
  where ``births - deaths - sqrt(D * s) - v**2 / (4 D)`` falls below 0 (Lynch and Lande).
  The threshold of ``minimum`` slows adaptation: traits ahead of the population are cut
  while still rare, so it lags more and dies out sooner than theory says.

``uv run python run.py population_evolution`` runs it on a planet of 2,562 cells, prints the
species and the gradient along the run, and draws the planet and the traits over time.
"""

import operator
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
from numpy.typing import NDArray

from gridzero.core import (
    Adjacency,
    Assign,
    Context,
    Field,
    LatticeSpace,
    MeshSpace,
    Phase,
    Source,
    Spread,
    State,
    derived,
    phase,
)
from gridzero.worlds import generate_planet

BINS = 32
"""The number of traits: bins of the temperature an individual does best at."""


class EvolutionState(State):
    """The planet's land and climate, and the individuals on it by trait."""

    space: MeshSpace | LatticeSpace  # a planet: a globe, or its map
    height = Field(np.float64, static=True)  # the planet's elevation, in [-1, 1]
    temperature = Field(np.float64)  # the planet's, in [0, 1]
    life = Field(np.float64, (BINS,), extensive=True)  # individuals by trait

    @derived(height)
    def land(self) -> NDArray[np.bool_]:
        """Return the cells above sea level: the only ones individuals live in and enter."""
        return self.height > 0


def spectrum(state: EvolutionState) -> NDArray[np.float64]:
    """Return the planet's individuals by trait: what a run records to show evolution.

    Parameters
    ----------
    state : EvolutionState
        A state.

    Returns
    -------
    ndarray of float64, shape (BINS,)
        Individuals of each trait, summed over the cells.
    """
    return state.total(EvolutionState.life)


def clusters(spectrum: NDArray[np.float64], cut: float = 0.01) -> list[tuple[int, int]]:
    """Return the clusters of traits: runs of bins separated by bins nearly empty.

    A bin is nearly empty when it holds less than ``cut`` of the fullest. Counting the local
    maxima instead would count ripples along a smooth gradient of traits too.

    Parameters
    ----------
    spectrum : ndarray of float64, shape (BINS,)
        Individuals by trait, as `spectrum` gives them.
    cut : float, default 0.01
        The share of the fullest bin below which a bin counts as empty.

    Returns
    -------
    list of (int, int)
        Each cluster's first and last bin, in order. None when no one lives.
    """
    full = spectrum > cut * spectrum.max() if spectrum.max() > 0 else np.zeros(BINS, bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], full.astype(np.int8), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def species(state: EvolutionState) -> NDArray[np.int64]:
    """Return each cell's species: the cluster of traits holding most of its individuals.

    Parameters
    ----------
    state : EvolutionState
        A state.

    Returns
    -------
    ndarray of int64, shape (n_cells,)
        An index into ``clusters(spectrum(state))``, so species are numbered from cold to
        warm; -1 where no one lives. Ties go to the colder species.
    """
    found = clusters(spectrum(state))
    labels = np.full(state.n_cells, -1, dtype=np.int64)
    if not found:
        return labels
    held = np.stack([state.life[a : b + 1].sum(axis=0) for a, b in found])
    lived = state.life.sum(axis=0) > 0
    labels[lived] = held[:, lived].argmax(axis=0)
    return labels


@dataclass(frozen=True)
class Evolution:
    """The parameters of the planet's evolution, per unit area and per step: a `core.Model`.

    Parameters
    ----------
    low : float, default -0.25
        The best temperature of the first bin.
    high : float, default 1.25
        The best temperature of the last bin. The bins run evenly from ``low`` to ``high``,
        beyond the planet's [0, 1], so that the ends of the range don't crowd the traits.
    tolerance : float, default 0.3
        How fast births fall off with the distance from the best temperature: the width of a
        Gaussian.
    births : float, default 0.2
        Births per individual and step, at its best temperature.
    deaths : float, default 0.05
        Deaths per individual and step, before crowding.
    capacity : float, default 100.0
        Individuals per unit area that crowding allows: at it, crowding adds ``births`` to
        the deaths.
    competition : float, default 0.15
        Width in trait of the crowding each individual feels from others in its cell, a
        quartic kernel ``exp(-|d / competition|**4)``. ``inf``: all traits crowd alike.
    mutation : float, default 0.005
        Share of births into each neighboring bin. Births at an end of the range that would
        leave it stay in their parent's bin.
    dispersal : float, default 0.05
        Share of each cell's individuals that moves to neighboring land each step.
    minimum : float, default 0.01
        Individuals per unit area below which a trait dies out in a cell.
    founders : float, default 10.0
        Individuals per unit area in the landing cell.
    land : float, default 0.3
        Share of the planet that is land.
    warming : float, default 0.0
        Rise in temperature per step, from ``warming_from`` on, everywhere: temperatures may
        rise past the planet's 1.
    warming_from : int, default 0
        The first step that warms.
    """

    low: float = -0.25
    high: float = 1.25
    tolerance: float = 0.3
    births: float = 0.2
    deaths: float = 0.05
    capacity: float = 100.0
    competition: float = 0.15
    mutation: float = 0.005
    dispersal: float = 0.05
    minimum: float = 0.01
    founders: float = 10.0
    land: float = 0.3
    warming: float = 0.0
    warming_from: int = 0

    @property
    def traits(self) -> NDArray[np.float64]:
        """The best temperature of each bin, shape (BINS,), from ``low`` to ``high``."""
        return np.linspace(self.low, self.high, BINS)

    def heredity(self) -> NDArray[np.float64]:
        """Return where births go: the mutation matrix.

        Returns
        -------
        ndarray of float64, shape (BINS, BINS)
            ``m[i, j]``, the share of bin ``j``'s births born into bin ``i``. Each column
            sums to 1.
        """
        m = np.diag(np.full(BINS, 1 - 2 * self.mutation))
        i = np.arange(BINS - 1)
        m[i + 1, i] = m[i, i + 1] = self.mutation
        m[0, 0] += self.mutation
        m[-1, -1] += self.mutation
        return m

    def crowding(self) -> NDArray[np.float64]:
        """Return how much each trait crowds each other.

        Returns
        -------
        ndarray of float64, shape (BINS, BINS)
            ``a[i, j] = exp(-|(x_i - x_j) / competition|**4)``, 1 on the diagonal, and 1
            everywhere when ``competition`` is infinite. Values below float64's smallest
            normal number are 0: the kernel's tail reaches subnormal values, which made
            ``a @ density`` 23 times slower (9 ms against 0.4 at 10,242 cells).
        """
        apart = self.traits[:, None] - self.traits[None, :]
        a = np.exp(-(np.abs(apart / self.competition) ** 4))
        return np.where(a < np.finfo(np.float64).tiny, 0.0, a)

    def fitness(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each trait's share of its best births, at each temperature.

        Parameters
        ----------
        temperature : ndarray of float64, shape (n_cells,)
            Each cell's temperature.

        Returns
        -------
        ndarray of float64, shape (BINS, n_cells)
            ``exp(-((x - t) / tolerance)**2 / 2)``, in (0, 1].
        """
        apart = (self.traits[:, None] - temperature[None, :]) / self.tolerance
        return np.exp(-0.5 * apart**2)

    def mean_trait(self, state: EvolutionState) -> NDArray[np.float64]:
        """Return the mean trait in each cell: its individuals' mean best temperature.

        Parameters
        ----------
        state : EvolutionState
            A state.

        Returns
        -------
        ndarray of float64, shape (n_cells,)
            NaN where no one lives.
        """
        total = state.life.sum(axis=0)
        mean = np.full(state.n_cells, np.nan)
        np.divide(self.traits @ state.life, total, out=mean, where=total > 0)
        return mean

    def gradient(self, state: EvolutionState) -> float:
        """Return how steeply the mean trait follows the temperature, over the occupied cells.

        Parameters
        ----------
        state : EvolutionState
            A state with individuals in at least two cells of different temperatures.

        Returns
        -------
        float
            The least-squares slope of `mean_trait` against the temperature: 1 when each
            cell's traits center on its temperature, less when migrants pull them toward
            their neighbors'.
        """
        mean = self.mean_trait(state)
        lived = ~np.isnan(mean)
        temperature = state.temperature[lived]
        apart = temperature - temperature.mean()
        return float(apart @ (mean[lived] - mean[lived].mean()) / (apart @ apart))

    def start(self, space: MeshSpace | LatticeSpace, seed: SupportsIndex) -> EvolutionState:
        """Return a planet, and the founders in one warm land cell, of the trait nearest it.

        Parameters
        ----------
        space : MeshSpace or LatticeSpace
            A spherical mesh, or a planet's map (`worlds.generate_planet`).
        seed : int
            The seed of the planet and of the landing cell.

        Returns
        -------
        EvolutionState
            ``founders`` per unit area in a land cell drawn from the warmest tenth of the land,
            in the bin whose best temperature is nearest the cell's; no one anywhere else.
        """
        planet = generate_planet(space, seed=seed, land=self.land)
        land, temperature = planet.land, planet.temperature
        warm = land & (temperature >= np.quantile(temperature[land], 0.9))
        rng = np.random.default_rng(operator.index(seed))
        cell = rng.choice(np.flatnonzero(warm))
        life = np.zeros((BINS, space.n_cells))
        life[np.argmin(np.abs(self.traits - temperature[cell])), cell] = (
            self.founders * space.measures[cell]
        )
        return EvolutionState(
            space,
            {
                EvolutionState.height: planet.elevation,
                EvolutionState.temperature: temperature,
                EvolutionState.life: life,
            },
        )

    def phases(self) -> list[Phase[EvolutionState]]:
        """Return the phases of a step: the climate warms, individuals live, spread and die out.

        Returns
        -------
        list of Phase
            ``warm`` (with ``warming``; nothing before ``warming_from``), ``live``,
            ``disperse`` and ``die_out``, in that order.
        """
        heredity, crowding = self.heredity(), self.crowding()

        @phase(writes=EvolutionState.temperature)
        def warm(s: EvolutionState, ctx: Context) -> Assign | None:
            """Raise the temperature everywhere by ``warming``, from ``warming_from`` on."""
            if s.step < self.warming_from:
                return None
            return Assign(EvolutionState.temperature, s.temperature + self.warming)

        @phase(writes=EvolutionState.life)
        def live(s: EvolutionState, ctx: Context) -> Source:
            """Give births by fitness, some into neighboring traits, and deaths by crowding."""
            born = self.births * self.fitness(s.temperature) * s.life
            crowded = crowding @ s.density(EvolutionState.life)
            died = s.life * (self.deaths + self.births * crowded / self.capacity)
            return Source(EvolutionState.life, heredity @ born - died)

        @phase(writes=EvolutionState.life, hoods={"near": Adjacency()})
        def disperse(s: EvolutionState, ctx: Context) -> Spread:
            """Move a share of each cell's individuals to neighboring land."""
            near = ctx.hoods["near"]
            return Spread(EvolutionState.life, near, leave=self.dispersal, domain=s.land)

        @phase(writes=EvolutionState.life)
        def die_out(s: EvolutionState, ctx: Context) -> Source:
            """Remove each trait too rare in a cell to live on.

            Amounts are continuous, so without this rule every trait would reach every cell
            in traces, by mutation and spreading, and grow wherever it would do well.
            """
            gone = s.density(EvolutionState.life) < self.minimum
            return Source(EvolutionState.life, -s.life * gone)

        return [*([warm] if self.warming else []), live, disperse, die_out]
