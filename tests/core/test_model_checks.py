"""The checks every simulation's tests run (`model_demo.check_model`): they pass a sound model and
catch broken ones."""

from dataclasses import dataclass

import numpy as np
import pytest

from gridzero.core import (
    Adjacency,
    Context,
    Field,
    LatticeSpace,
    Phase,
    Source,
    Space,
    Spread,
    State,
    phase,
)
from gridzero.core.implementations import fast
from model_demo import check_model

SPACE = LatticeSpace((5, 5), periodic=True)


class HeatState(State):
    heat = Field(np.float64, extensive=True)


@dataclass(frozen=True)
class Heat:
    """Heat spreading to neighbors, with random sparks: a sound model."""

    leave: float = 0.1

    def start(self, space: Space, seed) -> HeatState:
        heat = np.random.default_rng(seed).random(space.n_cells)
        return HeatState(space, {HeatState.heat: heat})

    def phases(self) -> list[Phase[HeatState]]:
        @phase(writes=HeatState.heat, hoods={"near": Adjacency()})
        def spread(s: HeatState, ctx: Context) -> Spread:
            return Spread(HeatState.heat, ctx.hoods["near"], leave=self.leave)

        @phase(writes=HeatState.heat)
        def spark(s: HeatState, ctx: Context) -> Source:
            return Source(HeatState.heat, self.sparks(s, ctx))

        return [spread, spark]

    def sparks(self, s: HeatState, ctx: Context):
        return 0.01 * ctx.rng.random(s.n_cells)


class FreshStart(Heat):
    """Starts from fresh entropy, whatever the seed."""

    def start(self, space, seed):
        return super().start(space, None)


class FreshSparks(Heat):
    """Sparks from fresh entropy, not the phase's stream."""

    def sparks(self, s, ctx):
        return 0.01 * np.random.default_rng().random(s.n_cells)


class TwoPaths(Heat):
    """Sparks differently on the Numba path."""

    def sparks(self, s, ctx):
        return super().sparks(s, ctx) * (2.0 if fast() else 1.0)


class Debt(Heat):
    """Starts with a negative amount of heat in one cell."""

    def start(self, space, seed):
        state = super().start(space, seed)
        heat = state.heat.copy()
        heat[3] = -1.0
        return state.replace({HeatState.heat: heat})


@dataclass(frozen=True)
class Narrower:
    """Takes only a lattice, though its state lives on any space: the need said twice."""

    def start(self, space: LatticeSpace, seed) -> HeatState:
        return Heat().start(space, seed)

    def phases(self) -> list[Phase[HeatState]]:
        return Heat().phases()


class LatticeHeatState(HeatState):
    space: LatticeSpace


class Wider(Heat):
    """Takes any space, though its state lives on a lattice."""

    def start(self, space: Space, seed) -> HeatState:
        state = super().start(space, seed)
        return LatticeHeatState(space, {HeatState.heat: state.heat})


def test_a_sound_model_passes_and_gives_its_last_state():
    end = check_model(Heat(), SPACE, steps=5)
    assert isinstance(end, HeatState)
    assert end.step == 5


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (FreshStart(), "heat differs at step 0"),
        (FreshSparks(), "heat differs at step 3"),
        (TwoPaths(), "heat differs at step 3"),
        (Debt(), "heat is negative at step 0"),
        (Narrower(), "Narrower.start takes .*LatticeSpace.*, but HeatState lives on .*Space"),
        (Wider(), "Wider.start takes .*Space.*, but LatticeHeatState lives on .*LatticeSpace"),
    ],
)
def test_a_broken_model_fails(model, match):
    with pytest.raises(AssertionError, match=match):
        check_model(model, SPACE, steps=3)
