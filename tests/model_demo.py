"""The checks that every simulation's tests run: `check_model` holds to `core.Model`'s promises."""

import inspect
from typing import Any

import numpy as np

from gridzero.core import Model, Simulation, Space, State, implementation


def check_model[Sp: Space, S: State](
    model: Model[Sp, S], space: Sp, steps: int, seed: int = 0
) -> S:
    """Hold a model to the promises of `core.Model`, over a short run on each path.

    Checks that `start` takes exactly the kind of space its state's class annotates (the space a
    model needs is declared on its state, and `start` repeats it); that `start` gives step 0 and
    the same bits for the same seed; that runs of the same seed on the reference and Numba paths
    give the same bits (a draw from fresh entropy fails this too); and that every floating field
    starts and ends finite, its amounts (extensive fields) not negative.

    Parameters
    ----------
    model : Model
        The rules.
    space : Space
        The cells, of the kind the model needs.
    steps : int
        Steps to run: long enough to reach every phase that fires late, such as an arrival.
    seed : int, default 0
        The run's seed.

    Returns
    -------
    State
        The state after ``steps`` on the reference path, for further checks.
    """
    start = model.start(space, seed)
    assert_start_takes_its_states_space(model, type(start))
    assert start.step == 0
    assert_same(start, model.start(space, seed))
    assert_amounts(start)
    ends = []
    for impl in ("reference", "numba"):
        with implementation(impl):
            sim = Simulation(model.start(space, seed), model.phases(), seed=seed)
            sim.run(steps)
            ends.append(sim.state)
    first, other = ends
    assert_same(first, other)
    assert first.step == steps
    assert_amounts(first)
    return first


def assert_amounts(state: State) -> None:
    """Assert that every floating field is finite, and every amount (extensive) not negative."""
    for field in state:
        values = state[field]
        if values.dtype.kind != "f":
            continue
        assert np.isfinite(values).all(), f"{field.name} is not finite at step {state.step}"
        if field.extensive:
            assert (values >= 0).all(), f"{field.name} is negative at step {state.step}"


def assert_same(a: State, b: State) -> None:
    """Assert that two states are of one class and hold the same bits in every field."""
    assert type(a) is type(b)
    assert a.step == b.step
    for field in a:
        x, y = a[field], b[field]
        assert (x.dtype, x.shape) == (y.dtype, y.shape), field.name
        assert x.tobytes() == y.tobytes(), f"{field.name} differs at step {a.step}"


def assert_start_takes_its_states_space(model: Model[Any, Any], cls: type[State]) -> None:
    """Assert that a model's `start` takes the kind of space its state's class annotates.

    A class's own annotation narrows those of its bases, so it is the last of them; without
    one, a state lives on any `Space`. The kind `start` takes is read from the nearest class
    that annotates it, so an override without annotations (as tests write) keeps its base's.
    """
    kinds = cls._space_kinds  # the annotations State.__init_subclass__ read
    needed = kinds[-1] if kinds else Space
    for owner in type(model).__mro__:
        start = vars(owner).get("start")
        if callable(start):
            hints = inspect.get_annotations(start, eval_str=True)
            if "space" in hints:
                taken = hints["space"]
                break
    else:
        raise AssertionError(f"{type(model).__name__}.start doesn't annotate its space")
    assert taken == needed, (
        f"{type(model).__name__}.start takes {taken}, but {cls.__name__} lives on {needed}"
    )
