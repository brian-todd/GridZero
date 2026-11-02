"""The simulation loop: a schedule of phases applied to a state, step after step.

A `Phase` wraps a rule ``fn(state, ctx)`` that reads the current `State` and returns intents.
Phases and simulations are generic in the state's class: a rule ``def claim(s: BoardState,
ctx)`` makes a ``Phase[BoardState]``, which runs in a simulation of any state whose class is
``BoardState`` or a subclass, and ``sim.state`` is typed by the simulation's state.
Phases run in schedule order within a step, and each one sees what the earlier ones wrote. This
is operator splitting, so the order of phases is part of the model. Within a phase every cell
reads the same state, so the update is synchronous.

All randomness comes from one root seed. Each phase call gets its own stream, derived from the
seed, the step and the phase's name with `numpy.random.SeedSequence`. Equal seeds therefore give
bit-identical runs, restoring a saved state replays the same draws, and adding a phase leaves the
streams of the other phases unchanged.

A `Model` is the rules of a simulation as a value: how it starts on a space from a seed, and
its phases.
"""

import hashlib
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, NamedTuple, Protocol, Self, SupportsIndex, TypeVar

import numpy as np
from numpy.typing import NDArray

from gridzero.core._arrays import check_int
from gridzero.core.fields import Field, NestedState, State, _written
from gridzero.core.intents import Intents, _resolve
from gridzero.core.neighborhoods import Neighborhood, StateNeighborhood
from gridzero.core.operators import NeighborOperator
from gridzero.core.space import Space

type Hood = Neighborhood | StateNeighborhood
"""A neighborhood a phase can use: one compiled once, or one that depends on the state."""

type Rule[S: State] = Callable[[S, Context], Intents]
"""The function of a phase: reads the state and returns intents."""

type Observer[S: State] = Callable[[S], Any]
"""A function of the state recorded by `Simulation.run`, such as a total or a whole field."""


class Context:
    """What a phase gets besides the state: its neighborhoods and its random stream.

    The simulation makes one for each phase call. Make one by hand to call a phase directly,
    for example in a test.

    Parameters
    ----------
    hoods : mapping of str to NeighborOperator, optional
        The phase's compiled neighborhoods, by name.
    seed : int, optional
        Root entropy of the random stream. If omitted, fresh entropy is drawn on first use.
    spawn_key : tuple of int, default ()
        Selects one stream of the root seed. The simulation passes ``(step, phase key)``.
    phase : str, default ""
        Name of the phase being run.

    Attributes
    ----------
    hoods : mapping of str to NeighborOperator
        The phase's compiled neighborhoods, read-only.
    phase : str
        Name of the phase being run.
    """

    __slots__ = ("_entropy", "_rng", "_seed", "_spawn_key", "hoods", "phase")

    def __init__(
        self,
        hoods: Mapping[str, NeighborOperator] | None = None,
        seed: int | None = None,
        spawn_key: tuple[int, ...] = (),
        phase: str = "",
    ) -> None:
        if isinstance(hoods, MappingProxyType):
            self.hoods = hoods
        else:
            self.hoods = MappingProxyType({} if hoods is None else dict(hoods))
        self.phase = phase
        self._entropy = seed
        self._spawn_key = spawn_key
        self._seed: np.random.SeedSequence | None = None
        self._rng: np.random.Generator | None = None

    @property
    def seed(self) -> np.random.SeedSequence:
        """The phase's seed sequence, for code that needs its own generators or child seeds.

        Created on first use. Spawn children from it to seed fixed chunks of work, so results
        do not depend on how the work is split across threads.
        """
        if self._seed is None:
            self._seed = np.random.SeedSequence(self._entropy, spawn_key=self._spawn_key)
        return self._seed

    @property
    def rng(self) -> np.random.Generator:
        """The phase's random generator, created on first use: phases without draws pay nothing."""
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    def drew(self) -> bool:
        """Whether the random stream was touched: `seed` or `rng` was read.

        A rule that never touched it gives the same result under any root seed. A game's step
        reports this, so a search can tell steps of chance from the others.
        """
        return self._seed is not None


# Declared contravariant, so that a phase of a state runs on its subclasses' states. The syntax
# of `class Phase[S: State]` can't declare variance, and from Python 3.13 a frozen dataclass
# gets `__replace__(*, fn: Rule[S])`, from which type checkers infer invariance.
S_contra = TypeVar("S_contra", bound=State, contravariant=True)


@dataclass(frozen=True, eq=False, init=False)
class Phase(Generic[S_contra]):  # noqa: UP046 (variance can't be declared otherwise, above)
    """One process of a step, such as growth, movement or a cellular-automaton rule.

    Calling a phase calls its rule.

    Parameters
    ----------
    fn : callable
        The rule, called as ``fn(state, ctx)`` with the read-only `State` left by the earlier
        phases of the step and a `Context`. It returns intents (`Assign`, `Flow`, `Spread`,
        `Source`, `Claim`): one, an iterable of them, or None.
    writes : Field, nested state, or iterable of them, default ()
        The dynamic fields the rule may write, a `NestedState` as nested in a state class
        standing for its dynamic fields. Writing any other field is an error. Stored as a tuple
        of fields.
    hoods : mapping of str to neighborhood, optional
        Neighborhoods the rule uses, by name. The simulation compiles each one once, shares the
        result between phases that use equal neighborhoods, and passes the operators as
        ``ctx.hoods``. A `StateNeighborhood` is recompiled when a field it depends on changes.
        The phase keeps a read-only copy.
    name : str, optional
        Name of the phase, unique within a schedule. It selects the phase's random stream.
        Defaults to the function's ``__name__``, or its class name for a callable object.
    every : int, default 1
        Run on every ``every``-th step only, for slow processes.
    offset : int, default 0
        Which steps those are: the steps ``t`` with ``t % every == offset``, counting from 0.

    Raises
    ------
    TypeError
        If ``fn`` is not callable, an entry of ``writes`` is neither a declared `Field` nor a
        nested state, ``name`` is not a string, or ``every`` or ``offset`` is not an integer.
    ValueError
        If ``name`` is empty, ``every`` is not positive, or ``offset`` is not in
        ``[0, every)``.
    """

    fn: Rule[S_contra]
    writes: tuple[Field[Any], ...]
    hoods: Mapping[str, Hood]
    name: str
    every: int
    offset: int

    def __init__(
        self,
        fn: Rule[S_contra],
        writes: Field[Any] | type[NestedState] | Iterable[Field[Any] | type[NestedState]] = (),
        hoods: Mapping[str, Hood] | None = None,
        name: str | None = None,
        every: SupportsIndex = 1,
        offset: SupportsIndex = 0,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"a phase needs a callable rule, got {type(fn).__name__}")
        fields = _written(writes)
        if name is None:
            name = getattr(fn, "__name__", type(fn).__name__)
        if not isinstance(name, str):
            raise TypeError(f"a phase name must be a string, got {name!r}")
        if not name:
            raise ValueError("a phase name must not be empty")
        every = check_int(every, "every")
        offset = check_int(offset, "offset")
        if every < 1 or not 0 <= offset < every:
            raise ValueError(
                f"every must be positive and offset in [0, every), got {every} and {offset}"
            )
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "writes", fields)
        object.__setattr__(self, "hoods", MappingProxyType(dict({} if hoods is None else hoods)))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "offset", offset)

    def __call__(self, state: S_contra, ctx: Context) -> Intents:
        """Run the rule: ``fn(state, ctx)``."""
        return self.fn(state, ctx)


def phase[S: State](
    *,
    writes: Field[Any] | type[NestedState] | Iterable[Field[Any] | type[NestedState]] = (),
    hoods: Mapping[str, Hood] | None = None,
    name: str | None = None,
    every: SupportsIndex = 1,
    offset: SupportsIndex = 0,
) -> Callable[[Rule[S]], Phase[S]]:
    """Make a rule function into a `Phase`, as a decorator.

    Parameters
    ----------
    writes, hoods, name, every, offset
        As for `Phase`.

    Returns
    -------
    callable
        Decorator that takes the rule and returns the `Phase`, generic in the class of state
        that the rule's first parameter is annotated with.

    Examples
    --------
    >>> class LifeState(State):
    ...     alive = Field(bool)
    >>> @phase(writes=LifeState.alive, hoods={"near": Adjacency("vertex")})
    ... def life(state: LifeState, ctx: Context) -> Assign:
    ...     n = pull(state.alive, ctx.hoods["near"])
    ...     return Assign(LifeState.alive, (n == 3) | (state.alive & (n == 2)))
    """

    def decorate(fn: Rule[S]) -> Phase[S]:
        return Phase(fn, writes, hoods, name, every, offset)

    return decorate


class Model[Sp: Space, S: State](Protocol):
    """The rules of a simulation: how it starts and the phases of a step.

    A model is an immutable value, such as a frozen dataclass of its parameters, and holds no
    state. A `Simulation` runs it: ``Simulation(model.start(space, seed), model.phases(),
    seed=seed)``. Its rules are written per unit area, so one model runs on spaces of any size;
    ``Sp`` is the kind of space it needs, such as `LatticeSpace` for a wind across the grid.
    """

    def start(self, space: Sp, seed: SupportsIndex, /) -> S:
        """Return the first state on a space.

        Parameters
        ----------
        space : Space
            The cells.
        seed : int
            The run's root seed, non-negative: the same seed gives the same state. The
            simulation takes it too, so one seed sets the whole run.

        Returns
        -------
        State
            Step 0.
        """
        ...

    def phases(self) -> Sequence[Phase[S]]:
        """Return the schedule of a step, in order.

        Returns
        -------
        sequence of Phase
            Phases with distinct names.
        """
        ...


class _StateHood(NamedTuple):
    """A neighborhood of a phase that compiles from the state, recompiled when its fields change.

    ``key`` identifies equal neighborhoods, which share one compiled operator.
    """

    name: str
    hood: StateNeighborhood
    key: object
    depends_on: tuple[str, ...]  # field names, checked against the schema


class _Plan(NamedTuple):
    """A phase prepared to run: checked writes, its stream key and its compiled neighborhoods."""

    phase: Phase[Any]
    writes: frozenset[str]
    stream: int
    hoods: MappingProxyType[str, NeighborOperator]
    state_hoods: tuple[_StateHood, ...]


class Simulation[S: State]:
    """Runs a schedule of phases over a state, one step after another.

    Each step runs the phases that are due, in schedule order. Each phase reads the state left
    by the phases before it and returns intents, which make the next state. A step is atomic:
    if a phase raises, the simulation stays at the state before the step.

    States are immutable, so ``saved = sim.state`` is a snapshot, and ``sim.state = saved``
    restores it. Random streams depend only on the seed, the step and the phase, so a restored
    simulation replays exactly the same steps.

    A state saved to disk comes back on a rebuilt copy of its space, which this simulation
    does not accept. Resume it with a new simulation of the same phases and seed:
    ``Simulation(loaded, phases, seed=seed)`` continues exactly as the original would have.

    Parameters
    ----------
    state : State
        Initial state. Its space and class are the simulation's, and every state the
        simulation makes is of that class.
    phases : iterable of Phase
        The schedule. Names must be unique. A phase whose rule reads a base class of the
        state's class fits.
    seed : int, optional
        Root seed of all randomness, a non-negative integer. If omitted, fresh entropy is drawn
        and kept as `seed`, so the run can be repeated.

    Raises
    ------
    TypeError
        If ``state`` is not a `State`, a phase is not a `Phase`, a neighborhood has no
        ``compile`` method, or the seed is not an integer.
    KeyError
        If a phase writes, or a neighborhood depends on, a field the state does not have, or
        declares differently.
    ValueError
        If two phases share a name, a phase writes a static field, a compiled neighborhood
        does not fit the space, or the seed is negative.

    An error from compiling a neighborhood gets a note naming the phase and the neighborhood.
    """

    def __init__(
        self, state: S, phases: Iterable[Phase[S]], *, seed: SupportsIndex | None = None
    ) -> None:
        if not isinstance(state, State):
            raise TypeError(f"state must be a State, got {type(state).__name__}")
        self._state = state
        self._phases = tuple(phases)
        names: set[str] = set()
        for each in self._phases:
            if not isinstance(each, Phase):
                raise TypeError(
                    f"phases must be Phase objects, got {type(each).__name__}; "
                    "wrap rules with Phase(fn, writes=...) or @phase(writes=...)"
                )
            if each.name in names:
                raise ValueError(f"two phases are named {each.name!r}; give them distinct names")
            names.add(each.name)
        self._seed = (
            secrets.randbits(128)  # as SeedSequence() draws its entropy
            if seed is None
            else check_int(seed, "seed", minimum=0)
        )
        self._operators: dict[object, NeighborOperator] = {}
        self._state_operators: dict[object, tuple[tuple[int, ...], NeighborOperator]] = {}
        self._plans = tuple(self._plan(each) for each in self._phases)

    @property
    def state(self) -> S:
        """The current state. Assign a state of the same space and class to restore it."""
        return self._state

    @state.setter
    def state(self, state: S) -> None:
        # Spaces compare by identity, so a state unpickled onto a rebuilt space is refused.
        if not isinstance(state, State):
            raise TypeError(f"state must be a State, got {type(state).__name__}")
        if state.space is not self.space:
            raise ValueError("the state is on a different space; make a new Simulation for it")
        if type(state) is not type(self._state):
            raise ValueError(
                f"the state is a {type(state).__name__}, not a {type(self._state).__name__}; "
                "make a new Simulation for it"
            )
        self._state = state

    @property
    def space(self) -> Space:
        """The cells."""
        return self._state.space

    @property
    def phases(self) -> tuple[Phase[S], ...]:
        """The schedule."""
        return self._phases

    @property
    def seed(self) -> int:
        """Root seed of all randomness."""
        return self._seed

    def __repr__(self) -> str:
        names = [each.name for each in self._phases]
        return f"{type(self).__name__}(step={self._state.step}, phases={names}, seed={self._seed})"

    def step(self) -> S:
        """Advance one step.

        Returns
        -------
        State
            The new state, also available as `state`.

        Raises
        ------
        Exception
            Whatever a phase or the resolver raised, with a note naming the phase and step. The
            simulation then stays at the state before the step.
        """
        state = self._state
        t = state.step
        for plan in self._plans:
            rule = plan.phase
            if t % rule.every != rule.offset:
                continue
            try:
                hoods = self._hoods(plan, state)
                ctx = Context(hoods, self._seed, (t, plan.stream), rule.name)
                state = _resolve(state, rule(state, ctx), plan.writes)
            except Exception as error:
                error.add_note(f"in phase {rule.name!r} at step {t}")
                raise
        self._state = state._with({}, t + 1)
        return self._state

    def run(
        self,
        n_steps: SupportsIndex = 1,
        *,
        observers: Mapping[str, Observer[S]] | None = None,
        every: SupportsIndex = 1,
    ) -> dict[str, NDArray[Any]]:
        """Advance several steps, recording observations along the way.

        Parameters
        ----------
        n_steps : int, default 1
            Number of steps to take.
        observers : mapping of str to callable, optional
            Functions of the state, by name, such as ``lambda s: s.total(Heat.heat)``. They are
            called on the state after each recorded step. Field arrays are immutable, so an
            observer may return one without copying it.
        every : int, default 1
            Record after the steps whose new ``state.step`` is a multiple of ``every``.

        Returns
        -------
        dict of str to ndarray
            ``"step"``: int64, shape (n_records,), the step count of each recorded state. Then,
            for each observer, its results stacked along a new first axis, shape
            ``(n_records, *result_shape)``. With no records, an observer's entry is an empty
            float64 array of shape (0,), because its result shape is unknown.

        Raises
        ------
        TypeError
            If ``n_steps`` or ``every`` is not an integer.
        ValueError
            If ``n_steps`` is negative, ``every`` is not positive, an observer is named
            ``"step"``, or an observer's results change shape.

        Notes
        -----
        If a step or an observer raises, the simulation stays after the last step it completed,
        and the records of this call are lost. The error has a note naming the phase or the
        observer, and the step.
        """
        observers = {} if observers is None else dict(observers)
        n_steps = check_int(n_steps, "n_steps")
        every = check_int(every, "every")
        if n_steps < 0 or every < 1:
            raise ValueError(f"need n_steps >= 0 and every >= 1, got {n_steps} and {every}")
        if "step" in observers:
            raise ValueError("the observer name 'step' is reserved for the step counts")
        steps: list[int] = []
        records: dict[str, list[Any]] = {name: [] for name in observers}
        shapes: dict[str, tuple[int, ...]] = {}
        for _ in range(n_steps):
            state = self.step()
            if state.step % every:
                continue
            steps.append(state.step)
            for name, observe in observers.items():
                try:
                    result = observe(state)
                    shape = np.shape(result)
                except Exception as error:
                    error.add_note(f"in observer {name!r} after step {state.step}")
                    raise
                expected = shapes.setdefault(name, shape)
                if shape != expected:
                    raise ValueError(
                        f"observer {name!r} returned shape {shape} after step {state.step}, "
                        f"but shape {expected} before; an observer must keep one shape"
                    )
                records[name].append(result)
        history: dict[str, NDArray[Any]] = {"step": np.array(steps, dtype=np.int64)}
        for name, values in records.items():
            history[name] = np.stack(values) if values else np.empty(0)
        return history

    def fork(self) -> Self:
        """Return an independent simulation at the current state.

        The fork shares the schedule, the seed and the compiled neighborhoods, so it is cheap.
        Run unchanged, it repeats exactly what this simulation does. Change its state to explore
        an alternative.

        Returns
        -------
        Simulation
            A new simulation of the same type, at this one's state. Running either
            leaves the other as it was.
        """
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other._state_operators = dict(self._state_operators)
        return other

    def _plan(self, rule: Phase[S]) -> _Plan:
        """Check a phase against the state's fields and compile its fixed neighborhoods.

        Raises
        ------
        KeyError
            If the phase writes a field, or a neighborhood depends on one, that the state does
            not declare.
        ValueError
            If the phase writes a static field, or a neighborhood compiles to an operator on
            another number of cells.
        TypeError
            If a neighborhood has no compile method or compiles to something other than a
            `NeighborOperator`.
        """
        schema = self._state._schema
        for field in rule.writes:
            try:
                schema._declared(field)
            except KeyError as error:
                raise KeyError(
                    f"phase {rule.name!r} writes {field.name!r}: {error.args[0]}"
                ) from None
            if field.static:
                raise ValueError(f"phase {rule.name!r} writes static field {field.name!r}")
        hoods: dict[str, NeighborOperator] = {}
        state_hoods = []
        for name, hood in rule.hoods.items():
            key = _hood_key(hood)
            if isinstance(hood, StateNeighborhood):
                depends_on = hood.depends_on
                if isinstance(depends_on, Field | str):
                    depends_on = (depends_on,)
                depends_on = tuple(depends_on)
                try:
                    names = tuple(schema._declared(field).name for field in depends_on)
                except (KeyError, TypeError) as error:
                    error.add_note(
                        f"phase {rule.name!r}: the fields neighborhood {name!r} depends on"
                    )
                    raise
                state_hoods.append(_StateHood(name, hood, key, names))
            elif isinstance(hood, Neighborhood):
                if key not in self._operators:
                    try:
                        compiled = hood.compile(self.space)
                    except Exception as error:
                        error.add_note(f"compiling neighborhood {name!r} of phase {rule.name!r}")
                        raise
                    self._operators[key] = self._check(compiled, rule, name)
                hoods[name] = self._operators[key]
            else:
                raise TypeError(
                    f"phase {rule.name!r}: neighborhood {name!r} has no compile method, "
                    f"got {type(hood).__name__}"
                )
        stream = _stream_key(rule.name)
        return _Plan(
            rule,
            frozenset(field.name for field in rule.writes),
            stream,
            MappingProxyType(hoods),
            tuple(state_hoods),
        )

    def _hoods(self, plan: _Plan, state: S) -> MappingProxyType[str, NeighborOperator]:
        """Return a phase's operators on ``state``, compiling those whose fields changed."""
        if not plan.state_hoods:
            return plan.hoods
        hoods = dict(plan.hoods)
        for entry in plan.state_hoods:
            versions = tuple(state._versions[name] for name in entry.depends_on)
            cached = self._state_operators.get(entry.key)
            if cached is None or cached[0] != versions:
                try:
                    compiled = entry.hood.compile(self.space, state)
                except Exception as error:
                    error.add_note(f"compiling neighborhood {entry.name!r}")
                    raise
                cached = versions, self._check(compiled, plan.phase, entry.name)
                self._state_operators[entry.key] = cached
            hoods[entry.name] = cached[1]
        return MappingProxyType(hoods)

    def _check(self, compiled: object, rule: Phase[Any], name: str) -> NeighborOperator:
        """Return what a neighborhood compiled to after checking it is an operator on the space.

        Raises
        ------
        TypeError
            If it is not a `NeighborOperator`.
        ValueError
            If it has another number of cells than the space.
        """
        if not isinstance(compiled, NeighborOperator):
            raise TypeError(
                f"phase {rule.name!r}: neighborhood {name!r} compiled to "
                f"{type(compiled).__name__}, not a NeighborOperator"
            )
        if compiled.n_cells != self.space.n_cells:
            raise ValueError(
                f"phase {rule.name!r}: neighborhood {name!r} compiled to an operator on "
                f"{compiled.n_cells} cells, but the space has {self.space.n_cells}"
            )
        return compiled


def _stream_key(name: str) -> int:
    """Return a 64-bit key for a phase name, the same in every process.

    Python's built-in ``hash`` of a string changes between processes, so it would break
    reproducibility.
    """
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _hood_key(hood: object) -> object:
    """Return the cache key of a neighborhood: itself if hashable, else its identity."""
    try:
        hash(hood)
    except TypeError:
        return ("unhashable", id(hood))
    return hood
