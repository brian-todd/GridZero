"""The update loop as arrays: phases in schedule order, for many runs at once.

A model in arrays has the phases of its `core.Model`, written once more with ``jax.numpy``:
each a rule that reads `Fields` and returns intents (`gridzero.batched.intents`), declared with
`phase` as `core.phase` declares one. `Simulation` compiles their neighborhoods on a space,
through `core`, and steps the fields of one run: a pure function, so ``jax.vmap`` runs a batch
of seeds or of parameters, and ``jax.jit`` compiles it.

As in `core`'s loop, the phases of a step run in order, each on the fields the one before it
left, and a phase with ``every`` runs on the steps ``t`` where ``t % every == offset``. Its
random draws come from its own stream of the run's key, named by the step and the phase.
"""

import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple, SupportsIndex

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from gridzero.batched.dice import stream
from gridzero.batched.intents import Intents, resolve
from gridzero.batched.operators import Operator, operator
from gridzero.batched.states import Fields
from gridzero.core import Field, Neighborhood, Space, State, StateNeighborhood, check_int


class Context(NamedTuple):
    """What a rule in arrays gets besides the fields, as `core.Context`."""

    hoods: Mapping[str, Operator]
    """The phase's compiled neighborhoods, by name."""
    key: jax.Array
    """The key of the phase's random stream at this step: uint32, shape (2,). Draw with
    `gridzero.batched.d`, or wrap it for ``jax.random``."""
    measures: jax.Array
    """The measure (area) of each cell, for densities: float32, shape (n_cells,)."""


type Rule[S: State] = Callable[[Fields[S], Context], Intents]
"""The function of a phase in arrays: reads the fields and returns intents."""


@dataclass(frozen=True, eq=False)
class Phase[S: State]:
    """One process of a step, as arrays: a rule, what it writes and the neighborhoods it reads.

    Made by `phase`. Its parts mean what `core.Phase`'s mean.
    """

    fn: Rule[S]
    writes: tuple[Field[Any], ...]
    hoods: Mapping[str, Neighborhood]
    name: str
    every: int
    offset: int


def phase[S: State](
    writes: Field[Any] | Iterable[Field[Any]] = (),
    hoods: Mapping[str, Neighborhood] | None = None,
    name: str | None = None,
    every: SupportsIndex = 1,
    offset: SupportsIndex = 0,
) -> Callable[[Rule[S]], Phase[S]]:
    """Declare a rule in arrays as a phase: a decorator, as `core.phase`.

    Parameters
    ----------
    writes : Field or iterable of Field, default ()
        The dynamic fields the rule may write.
    hoods : mapping of str to Neighborhood, optional
        The neighborhoods the rule reads, by name: descriptions such as ``Adjacency()``,
        which the simulation compiles on its space. They cannot depend on the state.
    name : str, optional
        The phase's name, which names its random stream. The rule's name by default.
    every : int, default 1
        Run on steps ``t`` with ``t % every == offset``.
    offset : int, default 0
        See ``every``.

    Returns
    -------
    callable
        Takes the rule ``fn(fields, ctx)`` and returns its `Phase`.

    Raises
    ------
    TypeError
        If ``writes`` holds something other than fields, ``every`` or ``offset`` is not an
        integer, or the name is not a string.
    ValueError
        If the name is empty, ``every`` is not positive, or ``offset`` is not in
        ``[0, every)``.
    """
    written = (writes,) if isinstance(writes, Field) else tuple(writes)
    for field in written:
        if not isinstance(field, Field):
            raise TypeError(f"a phase in arrays writes fields, got {type(field).__name__}")
    every = check_int(every, "every")
    offset = check_int(offset, "offset")
    if every < 1 or not 0 <= offset < every:
        raise ValueError(
            f"every must be positive and offset in [0, every), got {every} and {offset}"
        )

    def decorate(fn: Rule[S]) -> Phase[S]:
        label = getattr(fn, "__name__", type(fn).__name__) if name is None else name
        if not isinstance(label, str):
            raise TypeError(f"a phase name must be a string, got {label!r}")
        if not label:
            raise ValueError("a phase name must not be empty")
        return Phase(fn, written, MappingProxyType(dict(hoods or {})), label, every, offset)

    return decorate


class Simulation[S: State]:
    """The phases of a model in arrays on a space: a step, and a run, of one state's fields.

    Parameters
    ----------
    space : Space
        The cells. Each phase's neighborhoods are compiled on it once, by `core`.
    phases : sequence of Phase
        The schedule of a step, in order, with distinct names.

    Attributes
    ----------
    space : Space
        The cells.
    phases : tuple of Phase
        The schedule.

    Raises
    ------
    TypeError
        If ``space`` is not a `Space`, something is not a `Phase`, or a neighborhood has no
        ``compile`` method (one that depends on the state has none that takes a space alone).
    ValueError
        If two phases share a name.
    """

    def __init__(self, space: Space, phases: Sequence[Phase[S]]) -> None:
        if not isinstance(space, Space):
            raise TypeError(f"space must be a Space, got {type(space).__name__}")
        self.space = space
        self.phases = tuple(phases)
        names = [each.name for each in self.phases if isinstance(each, Phase)]
        if len(names) != len(self.phases):
            raise TypeError("phases must be made by gridzero.batched.phase")
        if len(set(names)) != len(names):
            raise ValueError(f"phases need distinct names, got {names}")
        compiled: dict[Any, Operator] = {}
        self._hoods: list[Mapping[str, Operator]] = []
        for each in self.phases:
            hoods = {}
            for name, hood in each.hoods.items():
                if not isinstance(hood, Neighborhood) or isinstance(hood, StateNeighborhood):
                    raise TypeError(
                        f"phase {each.name!r}: neighborhood {name!r} must compile on a space "
                        f"alone, got {type(hood).__name__}"
                    )
                key = _hood_key(hood)
                if key not in compiled:
                    compiled[key] = operator(hood.compile(space))
                hoods[name] = compiled[key]
            self._hoods.append(MappingProxyType(hoods))
        self._streams = [zlib.crc32(each.name.encode()) for each in self.phases]
        self._measures = jnp.asarray(space.measures, jnp.float32)

    def step(self, fields: Fields[S], key: ArrayLike) -> Fields[S]:
        """Advance one state's fields by one step.

        Parameters
        ----------
        fields : Fields
            The fields of one state.
        key : Array of uint32, shape (2,)
            The run's key (`gridzero.batched.root`). A phase draws from its stream named by
            the step and the phase.

        Returns
        -------
        Fields
            The fields one step later.
        """
        t = jnp.asarray(fields.step, jnp.int32)
        for rule, hoods, name in zip(self.phases, self._hoods, self._streams, strict=True):
            ctx = Context(hoods, stream(key, t, name), self._measures)
            after = resolve(fields, rule.fn(fields, ctx), writes=rule.writes)
            if rule.every > 1 and after is not fields:
                runs = t % rule.every == rule.offset
                kept = {f: jnp.where(runs, after[f], fields[f]) for f in rule.writes}
                after = fields.replace(kept)
            fields = after
        return fields.at(t + 1)

    def run(
        self,
        fields: Fields[S],
        key: ArrayLike,
        steps: SupportsIndex,
        observe: Callable[[Fields[S]], Any] | None = None,
    ) -> tuple[Fields[S], Any]:
        """Advance one state's fields by a number of steps.

        Parameters
        ----------
        fields : Fields
            The fields of one state.
        key : Array of uint32, shape (2,)
            The run's key.
        steps : int
            How many steps, at least 0.
        observe : callable, optional
            A function of the fields to record after every step, such as a total: it returns
            an array, or a tree of arrays.

        Returns
        -------
        fields : Fields
            The fields after the last step.
        history : tree of Array, or None
            What ``observe`` returned after each step, stacked along a first axis of length
            ``steps``. None without ``observe``.

        Raises
        ------
        TypeError, ValueError
            If ``steps`` is not an integer of at least 0.
        """
        steps = check_int(steps, "steps", minimum=0)
        key = jnp.asarray(key, jnp.uint32)

        def one(now: Fields[S], _: None) -> tuple[Fields[S], Any]:
            after = self.step(now, key)
            return after, None if observe is None else observe(after)

        return jax.lax.scan(one, fields, None, length=steps)


def _hood_key(hood: object) -> object:
    """Return the cache key of a neighborhood: itself if hashable, else its identity."""
    try:
        hash(hood)
    except TypeError:
        return ("unhashable", id(hood))
    return hood
