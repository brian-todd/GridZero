"""States shared by the tests, importable from any test module (``pythonpath`` in pytest's config).

`conftest.py` holds fixtures. Classes live here because a conftest module cannot be imported.
"""

from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gridzero.core import Field, Nested, NestedState, State, derived


class DemoState(State):
    """One field of each sort: a flag, amounts for two groups, and a static code."""

    alive = Field(bool)
    pop = Field(np.float64, (2,), extensive=True)
    terrain = Field(np.uint8, static=True)


def fields(**declared: Field[Any] | Nested[Any]) -> Any:
    """Return a `State` subclass with the given fields or nested states, for a test of its own.

    Typed Any: a class made at run time has attributes no type checker can know. Its states
    cannot be pickled, because pickle finds a class by its import path.
    """
    return type("Some", (State,), declared)


NESTED_CALLS: Counter[str] = Counter()
"""How many times `Species.present` was computed, by the path it is nested under."""


class Species(NestedState):
    """A population and what it caught this step."""

    population = Field(np.float64, extensive=True)
    eaten = Field(np.float64, extensive=True)

    @derived(population)
    def present(self) -> NDArray[np.bool_]:
        NESTED_CALLS[type(self).__qualname__.split(".", 1)[1]] += 1  # the path
        return self.population > 0


class EcosystemState(State):
    """Grass, and one nested state nested twice."""

    grass = Field(np.float64, extensive=True)
    rabbits = Nested(Species)
    foxes = Nested(Species)

    @derived(foxes)
    def hunting(self) -> NDArray[np.bool_]:
        return self.foxes.population > self.foxes.eaten


class Settlers(NestedState):
    """People and the stores they carry, beside one field of each kind that does not move."""

    people = Field(np.float64, (2,), extensive=True)
    stores = Field(np.float64, (2,), extensive=True)
    home = Field(np.float64, extensive=True, static=True, default=1.0)  # static
    mood = Field(np.float64)  # intensive
    count = Field(np.int64, extensive=True)  # not floating


class WorldState(State):
    """Settlers, and a field outside them."""

    settlers = Nested(Settlers)
    water = Field(np.float64, extensive=True)


class Biome(NestedState):
    """Rain, and one nested state nested twice in it: a nested state two levels deep."""

    rain = Field(np.float64, default=1.0)  # intensive: it stays when the biome moves
    rabbits = Nested(Species)
    foxes = Nested(Species)

    @derived(foxes, rain)
    def hunted(self) -> NDArray[np.bool_]:
        return (self.foxes.population > 0) & (self.rain > 0)


class Continent(NestedState):
    """A biome nested in it: species three levels deep."""

    north = Nested(Biome)


class PlanetState(State):
    """Grass, two biomes, and a continent with a third."""

    grass = Field(np.float64, extensive=True)
    north = Nested(Biome)
    south = Nested(Biome)
    europe = Nested(Continent)

    @derived(europe)
    def european_foxes(self) -> NDArray[np.bool_]:
        return self.europe.north.foxes.present
