"""Fields and states: typed per-cell quantities, declared on a class of immutable states.

A model declares its fields on a subclass of `State`, one `Field` attribute each::

    class BoardState(State):
        height = Field(np.float64, static=True)
        owner = Field(np.int8, default=-1)

A `Field` declares one quantity: its dtype, its component shape, whether phases may write it
(dynamic) or not (static), whether it holds amounts (extensive) or densities (intensive), and
the value it starts with. Its name is its attribute's. ``BoardState.owner`` is the field itself,
and names it everywhere: ``Assign(BoardState.owner, ...)``, ``writes=BoardState.owner``,
``state[BoardState.owner]``.

A state of the class holds one array per field, stored cells last as ``(*shape, n_cells)``,
together with the number of steps taken: ``BoardState(space, values)``. On a state, the
attribute is the field's array: ``board.owner``, typed ``NDArray[np.int8]``.

Fields that belong together, such as a species' population and catch, are declared on a
`NestedState` and nested in a state under a name: ``foxes = Nested(Species)``. On a state,
``state.foxes.population`` is the array, and on the class, ``EcosystemState.foxes.population``
is the field, stored under its qualified name ``"foxes.population"``.

States are immutable values. A write makes a new state of the same class that shares the arrays
of every field it leaves unchanged. Keeping a reference to a state is therefore a snapshot, and
copying one costs nothing.
"""

import annotationlib
import functools
import inspect
import itertools
import sys
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Never, Self, SupportsIndex, cast, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from gridzero.core._arrays import (
    check_cell,
    check_int,
    component_shape,
    read_only,
)
from gridzero.core.space import Space

_DTYPE_KINDS = "biuf"

_next_version = itertools.count(1).__next__
"""Issue version numbers that are unique within the process, so equal versions mean equal arrays."""


class Field[S: np.generic]:
    """One per-cell field, declared as an attribute of a `State` or `NestedState` subclass.

    The field takes its attribute's name when the class is made. On the class, the attribute is
    the field itself, which names it wherever a field is wanted. On a state, it is the state's
    read-only array of the field, typed ``NDArray[S]``. A field of a nested state is named for
    the name it is nested under too, as ``"foxes.population"`` (see `Nested`).

    The type parameter ``S`` is the scalar type of the field's values. Type checkers infer it
    from a NumPy scalar type such as ``np.float64``, from a `numpy.dtype`, or from ``bool``,
    ``int`` and ``float``. A dtype given otherwise, such as ``"f8"``, gives ``Field[Any]``.

    Fields are values: two with the same name and declaration are equal, and pickling one
    rebuilds it from them.

    Parameters
    ----------
    dtype : dtype_like
        Element type: boolean, integer or floating. Stored as a `numpy.dtype`.
    shape : int or tuple of int, default ()
        Component shape per cell: ``()`` for a scalar, ``(3,)`` for three groups. The field is
        stored as an array of shape ``(*shape, n_cells)``.
    static : bool, default False
        If true, phases may not write the field. Static fields hold geography and other inputs,
        set when the state is created.
    extensive : bool, default False
        If true, the field holds amounts, such as people or water, which add up when cells are
        combined. Otherwise it is intensive: a density, a rate, a category or a flag. Store
        amounts and derive densities with `State.density`, because cells differ in size.
    default : scalar, optional
        The value every cell starts with when a state is made without one for the field,
        converted to ``dtype`` as values are (see `as_field_array`). Without a default, a
        dynamic field starts at zero and a static field must be given.

    Raises
    ------
    TypeError
        If the dtype is missing or not boolean, integer or floating, a shape entry is not an
        integer, or the default would change kind in the field. When the class is made, also
        if the class is neither a `State` nor a `NestedState` subclass.
    ValueError
        If a shape entry is negative, a boolean field is declared extensive, or the default is
        not one value or does not fit the dtype. When the class is made, also if `State` itself
        (or `NestedState`, for a nested state's field) has an attribute of that name (such as
        ``step``, ``total`` or ``space``), or the field is already declared under another name.
        A field of one class may be declared on another under its name.
    """

    __slots__ = ("_hash", "default", "dtype", "extensive", "name", "shape", "static")

    _hash: int | None
    name: str
    """The field's name, its attribute's on the `State` class: empty until it is declared."""
    dtype: np.dtype[S]
    """Element type."""
    shape: tuple[int, ...]
    """Component shape per cell."""
    static: bool
    """Whether phases may not write the field."""
    extensive: bool
    """Whether the field holds amounts."""
    default: S | None
    """The value cells start with, a NumPy scalar of ``dtype``, or None."""

    @overload
    def __init__(
        self: Field[np.bool_],
        dtype: type[bool],
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Field[np.int_],
        dtype: type[int],
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Field[np.float64],
        dtype: type[float],
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: type[S] | np.dtype[S],
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Field[Any],
        dtype: DTypeLike,
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None: ...
    def __init__(
        self,
        dtype: DTypeLike,
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
        static: bool = False,
        extensive: bool = False,
        default: ArrayLike | None = None,
    ) -> None:
        if dtype is None:  # np.dtype(None) would quietly mean float64
            raise TypeError("field: dtype is required")
        dtype = np.dtype(dtype)
        if dtype.kind not in _DTYPE_KINDS:
            raise TypeError(f"field: dtype must be boolean, integer or floating, got {dtype}")
        if extensive and dtype.kind == "b":
            raise ValueError("field: an extensive field holds amounts, not flags")
        set_ = object.__setattr__
        set_(self, "name", "")
        set_(self, "dtype", dtype)
        set_(self, "shape", component_shape(shape, "field: "))
        set_(self, "static", static)
        set_(self, "extensive", extensive)
        if default is not None:
            if np.ndim(default) != 0:
                raise ValueError(f"field: a default is one value, got shape {np.shape(default)}")
            default = _convert("field default", dtype, (), default)[()]
        set_(self, "default", default)
        set_(self, "_hash", None)

    def __set_name__(self, owner: type, name: str) -> None:
        base = _declaring_base(owner, f"field {name!r}: a Field")
        if not name.isidentifier():
            raise ValueError(f"field name must be a Python identifier, got {name!r}")
        if hasattr(base, name):  # fields share the namespace of the class's own members
            raise ValueError(f"field {name!r}: the name is taken by {base.__name__}.{name}")
        if self.name and self.name != name:
            raise ValueError(f"field {name!r}: this Field is already declared as {self.name!r}")
        object.__setattr__(self, "name", name)  # again under its own name: classes share it

    @overload
    def __get__(self, state: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(self, state: State | NestedState, owner: type | None = None) -> NDArray[S]: ...
    @overload  # a field held by anything else, such as an intent: never a descriptor there
    def __get__(self, state: object, owner: type | None = None) -> Self: ...
    def __get__(self, state: object, owner: type | None = None) -> Self | NDArray[S]:
        if state is None:  # the class's attribute: the usual way to name a field
            return self
        try:  # not isinstance(state, State): State is a Mapping, and ABC checks cost 60 ns
            return state._arrays[self.name]  # ty: ignore[unresolved-attribute] - caught below
        except (
            AttributeError
        ):  # neither a state nor a nested state's view, the only classes with one
            return self

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("fields are immutable")

    def _key(self) -> tuple[str, np.dtype[S], tuple[int, ...], bool, bool, bytes | None]:
        """Return what equal fields share: the name and declaration, the default as bytes."""
        default = None if self.default is None else self.default.tobytes()
        return (self.name, self.dtype, self.shape, self.static, self.extensive, default)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Field):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        value = hash(self._key())
        if self.name:  # cached once named: fields are dictionary keys
            object.__setattr__(self, "_hash", value)
        return value

    def __reduce__(self) -> tuple[Callable[..., Field[Any]], tuple[object, ...]]:
        declaration = (self.dtype, self.shape, self.static, self.extensive, self.default)
        return _named_field, (self.name, *declaration)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __repr__(self) -> str:
        parts = [f"{self.name!r}" if self.name else "<undeclared>", str(self.dtype)]
        if self.shape:
            parts.append(f"shape={self.shape}")
        parts += [flag for flag in ("static", "extensive") if getattr(self, flag)]
        if self.default is not None:
            parts.append(f"default={self.default.item()!r}")
        return f"Field({', '.join(parts)})"

    def array_shape(self, n_cells: int) -> tuple[int, ...]:
        """Return the storage shape ``(*shape, n_cells)`` of the field on ``n_cells`` cells.

        Parameters
        ----------
        n_cells : int
            Number of cells of the space.

        Returns
        -------
        tuple of int
            The component shape, then ``n_cells``.
        """
        return (*self.shape, n_cells)


def _named_field(
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    static: bool,
    extensive: bool,
    default: np.generic | None,
) -> Field[Any]:
    """Rebuild a declared field from its name and declaration, as pickling does."""
    field = Field(dtype, shape, static, extensive, default)
    object.__setattr__(field, "name", name)
    return field


class Derived[S: np.generic]:
    """An array per cell computed from fields, declared with `derived` on a `State` class.

    On the class, the attribute is the derived field itself, which names it:
    ``state[BoardState.land]``. On a state, it is the computed array, typed ``NDArray[S]``
    from the method's return annotation: ``board.land``.

    The array is computed on first read and cached. The cache passes on to the states made
    from one another, by intents, `State.replace` and `State.extend`, and a cached array is
    used while the fields it depends on keep their versions. A derived field of static fields
    is therefore computed once per run.

    The method runs on a copy of the state that holds only the fields it depends on, so
    reading another field raises `KeyError`, naming it. Derived fields computed from those
    fields may be read too. The copy shares the arrays, and the method should not read the
    state's ``step``.

    A `NestedState` declares derived fields the same way.

    A derived field is not stored: iterating over a state, its length, `State.values_at` and
    pickling cover the stored fields only, and ``field in state`` is false. Writing one raises
    `TypeError`.

    Parameters
    ----------
    fn : callable
        The method, called with the state, or with the nested state's view for a derived
        field of a nested state. It returns an array whose last axis is the cells: shape
        ``(..., n_cells)``.
    depends_on : tuple of Field, Derived or nested state
        The fields the method reads. A derived field stands for the fields it depends on, and a
        nested state (`Nested`, or a `NestedState` as nested in a state class) for all of its
        fields.

    Raises
    ------
    TypeError
        If ``fn`` is not callable or a dependency is none of those. When the class is made,
        also if the class is neither a `State` nor a `NestedState` subclass.
    ValueError
        When the class is made, if `State` itself has an attribute of that name, or the
        derived field is already declared under another name.
    KeyError
        When the class is made, if the class lacks a field it depends on, or declares it
        differently.
    """

    __slots__ = ("_fields", "_nested", "depends_on", "fn", "name", "owner")

    _fields: tuple[Field[Any], ...] | None
    _nested: type[NestedState] | None
    """The nested state's class as nested, for a derived field of one: the method reads its
    view. None for a derived field of a state."""
    fn: Callable[[Any], NDArray[S]]
    """The method that computes the array."""
    depends_on: tuple[Dependency, ...]
    """The fields the method reads."""
    name: str
    """The attribute's name on its class, qualified by the name a nested state is nested under
    for a derived field of one (``"foxes.crowded"``): empty until it is declared."""
    owner: type | None
    """The class it was declared on first, or the state class a nested state is nested in: None
    until it is declared."""

    def __init__(
        self,
        fn: Callable[[Any], NDArray[S]],
        depends_on: tuple[Dependency, ...],
    ) -> None:
        if not callable(fn):
            raise TypeError(f"a derived field is computed by a method, got {fn!r}")
        _check_dependencies(depends_on)
        set_ = object.__setattr__
        set_(self, "fn", fn)
        set_(self, "depends_on", tuple(depends_on))
        set_(self, "name", "")
        set_(self, "owner", None)
        set_(self, "_fields", None)
        set_(self, "_nested", None)

    def __set_name__(self, owner: type, name: str) -> None:
        base = _declaring_base(owner, f"derived field {name!r}: it")
        if hasattr(base, name):
            raise ValueError(f"derived field {name!r}: the name is taken by {base.__name__}.{name}")
        if self.name and self.name != name:
            raise ValueError(f"derived field {name!r}: it is already declared as {self.name!r}")
        object.__setattr__(self, "name", name)
        if self.owner is None:
            object.__setattr__(self, "owner", owner)

    @overload
    def __get__(self, state: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(self, state: State | NestedState, owner: type | None = None) -> NDArray[S]: ...
    def __get__(
        self, state: State | NestedState | None, owner: type | None = None
    ) -> Self | NDArray[S]:
        if state is None:  # the class's attribute: names the derived field
            return self
        if isinstance(state, NestedState):  # a derived field of a nested state, read on its view
            return self._value(state._state)
        return self._value(state)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("derived fields are immutable")

    def __reduce__(self) -> tuple[Callable[..., object], tuple[object, ...]]:
        # The method is code, so a derived field pickles as the attribute of its class.
        return _lookup, (self.owner, self.name)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __repr__(self) -> str:
        names = ", ".join(repr(_dependency_name(each)) for each in self.depends_on)
        name = repr(self.name) if self.name else "<undeclared>"
        return f"Derived({name}, depends_on=({names}{',' if len(self.depends_on) == 1 else ''}))"

    def fields(self) -> tuple[Field[Any], ...]:
        """Return the stored fields the array depends on, those of derived fields included.

        Returns
        -------
        tuple of Field
            Each field once, in the order the dependencies name them.
        """
        fields = self._fields
        if fields is None:  # once: names are given when the class is made, after decorating
            by_name: dict[str, Field[Any]] = {}
            for each in self.depends_on:
                if isinstance(each, Field):
                    stored: Iterable[Field[Any]] = (each,)
                elif isinstance(each, Derived):
                    stored = each.fields()
                else:
                    stored = _as_nested(each, rooted=False)._schema.values()
                for field in stored:
                    by_name.setdefault(field.name, field)
            fields = tuple(by_name.values())
            object.__setattr__(self, "_fields", fields)
        return fields

    def _value(self, state: State) -> NDArray[S]:
        """Return the array on ``state``, from the cache while its fields keep their versions.

        A derived field of a nested state gets the nested state's view of the restricted state,
        however it is read: on the view, or by `State.__getitem__`.

        Raises
        ------
        KeyError
            If the method reads a field it does not depend on.
        ValueError
            If the method returns an array whose last axis is not the cells.
        """
        fields = self.fields()
        versions = state._versions
        key = tuple([versions[field.name] for field in fields])
        cached = state._derived.get(self)
        if cached is not None and cached[0] == key:
            return cached[1]
        arrays = state._arrays
        names = [field.name for field in fields]
        restricted = object.__new__(type(state))  # the fields it depends on, sharing the cache
        _fill(
            restricted,
            state._space,
            {name: arrays[name] for name in names},
            {name: versions[name] for name in names},
            state._step,
            state._derived,
        )
        nested = self._nested
        try:
            result = np.asarray(self.fn(restricted if nested is None else nested._view(restricted)))
        except KeyError as error:
            read = error.args[0] if error.args else None
            if isinstance(read, str) and read in state._schema._fields and read not in names:
                raise KeyError(
                    f"derived field {self.name!r} reads field {read!r}, which it does not "
                    f"depend on; add it to @derived(...)"
                ) from None
            error.add_note(f"computing derived field {self.name!r}")
            raise
        except Exception as error:
            error.add_note(f"computing derived field {self.name!r}")
            raise
        if result.ndim == 0 or result.shape[-1] != state.n_cells:
            raise ValueError(
                f"derived field {self.name!r} must have the cells last, shape (..., "
                f"{state.n_cells}); got shape {result.shape}"
            )
        if result.flags.writeable:  # a fresh array, or a view of one: freeze a view of it
            result = read_only(result.view())
        state._derived[self] = (key, result)
        return result


def derived[S: np.generic](
    *depends_on: Dependency,
) -> Callable[[Callable[[Any], NDArray[S]]], Derived[S]]:
    """Declare an array per cell computed from fields, as a decorator on a `State` method.

    It decorates the methods of a `NestedState` the same way, with its fields.

    >>> class BoardState(State):
    ...     height = Field(np.float64, static=True)
    ...
    ...     @derived(height)
    ...     def land(self) -> NDArray[np.bool_]:
    ...         return self.height > 0

    ``board.land`` is then the computed array, typed ``NDArray[np.bool_]``, and
    ``BoardState.land`` names it, as in ``board[BoardState.land]``. See `Derived`.

    Parameters
    ----------
    *depends_on : Field, Derived or nested state
        The fields the method reads: in the class body, the names of fields declared above it,
        or of fields and derived fields of base classes, such as ``TerrainState.height``. A
        nested state, such as ``foxes`` declared above as ``foxes = Nested(Species)``, stands
        for all of its fields.

    Returns
    -------
    callable
        Decorator that takes the method and returns the `Derived` field.

    Raises
    ------
    TypeError
        If a dependency is none of those, as when the decorator is written without
        parentheses.
    """
    _check_dependencies(depends_on)  # here, or a bare @derived would leave the method as it is

    def decorate(fn: Callable[[Any], NDArray[S]]) -> Derived[S]:
        return Derived(fn, depends_on)

    return decorate


def _check_dependencies(depends_on: tuple[object, ...]) -> None:
    """Raise `TypeError` unless each dependency of a derived field is a field or nested state."""
    for each in depends_on:
        if not (isinstance(each, Field | Derived | Nested) or _is_nested_state(each)):
            raise TypeError(
                f"derived(...) takes the fields a method depends on, got {each!r}; "
                "decorate with @derived(field, ...), with parentheses"
            )


class _Schema(Mapping[str, Field[Any]]):
    """The fields of a state: a read-only mapping from name to `Field`, in declaration order.

    Internal. Each `State` class makes one from its `Field` attributes.
    """

    __slots__ = ("_fields",)

    def __init__(self, *fields: Field[Any]) -> None:
        by_name: dict[str, Field[Any]] = {}
        for field in fields:
            if by_name.setdefault(field.name, field) != field:
                raise ValueError(f"two different fields are named {field.name!r}")
        self._fields = by_name

    def __getitem__(self, name: str) -> Field[Any]:
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(
                f"no field named {name!r}; the fields are {list(self._fields)}"
            ) from None

    def __contains__(self, field: object) -> bool:
        return isinstance(field, Field) and self._fields.get(field.name) == field

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Schema):
            return NotImplemented
        return list(self._fields.values()) == list(other._fields.values())

    def __hash__(self) -> int:
        return hash(tuple(self._fields.values()))

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        return type(self), tuple(self._fields.values())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self._fields.values()))})"

    @overload
    def _declared[T: np.generic](self, field: Field[T]) -> Field[T]: ...
    @overload
    def _declared(self, field: object) -> Field[Any]: ...
    def _declared(self, field: object) -> Field[Any]:
        """Return a field after checking that the schema declares it the same way."""
        declared = self._fields.get(getattr(field, "name", ""))  # "" names no field
        if declared is field and declared is not None:
            return declared  # the usual case: the field object of the class the state came from
        if isinstance(field, Derived):
            raise TypeError(
                f"{field!r} is computed from other fields, so it is not stored: it cannot be "
                "written, and has no version"
            )
        if not isinstance(field, Field):
            raise TypeError(f"fields are given as Field attributes of a State class, got {field!r}")
        if not field.name:
            raise TypeError(f"{field!r} is not declared on a State class")
        if declared is None:
            nested = [name for name in self._fields if name.endswith("." + field.name)]
            if nested:
                raise KeyError(
                    f"no field named {field.name!r}: {field!r} is declared on a nested state, so "
                    f"name it as nested in the state class, such as the field {nested[0]!r}"
                )
            declared = self[field.name]  # raises, listing the fields
        if declared != field:
            raise KeyError(f"field {field.name!r} is declared as {declared}, not as {field}")
        return field


class NestedState:
    """Fields that belong together, such as a species, declared once and nested in states.

    A nested state declares fields and derived fields as a state does, but it is not a `State`:
    it has no space or step of its own and can't be simulated alone. A state nests it under a
    name with `Nested`, and holds its fields as its own, each under its qualified name:

    >>> class Species(NestedState):
    ...     population = Field(np.float64, extensive=True)
    ...     eaten = Field(np.float64, extensive=True)
    >>> class EcosystemState(State):
    ...     rabbits = Nested(Species)
    ...     foxes = Nested(Species)

    Subclasses are named for what they hold (``Species``, ``Settlers``), not ``*State``: they
    are parts of states, not states.

    On a state, ``state.foxes`` is a view of the state, whose attributes are the arrays:
    ``state.foxes.population`` is the array of the field ``"foxes.population"``. On the class,
    ``EcosystemState.foxes`` is ``Species`` as nested there, a subclass of it whose attributes
    are its fields: ``EcosystemState.foxes.population`` names the field in intents, ``writes``
    and ``values``. Code written once for any species takes it as nested
    (``sp: type[Species]``) and reads through the state, as ``state[sp.population]``.

    Nested states nest in one another too, to any depth: with ``foxes = Nested(Species)`` on a
    ``Biome``, and ``north = Nested(Biome)`` on a state class, ``state.north.foxes.population``
    is the array of ``"north.foxes.population"``. On a nested state as declared, the fields are
    named relative to it (``Biome.foxes.population`` is ``"foxes.population"``), so they name
    no state's fields: only the path through a state class does.

    Nested states combine by inheritance, as states do. A view is immutable, and besides its
    fields and derived fields it has one member, ``space``. A nested state's derived fields
    depend on its own fields, including those of the states nested in it, and read them on the
    view.

    Raises
    ------
    TypeError
        If a nested state is made directly, rather than read from a state; or, when a subclass
        is made, if an attribute hides a field of a base class.
    ValueError
        When a subclass is made, if two different fields have one name.
    KeyError
        When a subclass is made, if a derived field depends on a field the class lacks.
    """

    __slots__ = ("_arrays", "_state", "_views")

    _schema: ClassVar[_Schema] = _Schema()
    """The fields: unqualified on the class as declared, qualified once nested."""
    _prefix: ClassVar[str] = ""
    """The path the class is nested under, such as ``"north.foxes"``, or "" as declared."""
    _root: ClassVar[type[State] | None] = None
    """The state class the path starts from, or None as declared or nested in a class that is
    itself as declared (``Biome.foxes``)."""
    _arrays: dict[str, NDArray[Any]]
    _state: State
    _views: dict[str, NestedState]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "_prefix" in vars(cls):  # a class as nested, which `Nested` fills in
            return
        inherited = [
            field
            for base in cls.__bases__
            if issubclass(base, NestedState)
            for field in base._schema.values()
        ]
        own: list[Field[Any]] = []
        for value in vars(cls).values():
            if isinstance(value, Field):
                own.append(value)
            elif isinstance(value, Nested) and value.bound is not None:
                own.extend(value.bound._schema.values())  # named relative to this class
        cls._schema = _class_schema(cls, inherited, own)

    def __init__(self) -> None:
        raise TypeError(
            f"{type(self).__name__} is a nested state: nest it in a state class, as "
            f"Nested({type(self).__name__}), and read it from a state"
        )

    @classmethod
    def _view(cls, state: State) -> Self:
        """Return the view of a state through the class as nested in the state's class."""
        view = object.__new__(cls)
        object.__setattr__(view, "_state", state)
        object.__setattr__(view, "_arrays", state._arrays)
        object.__setattr__(view, "_views", state._views)  # views of views are the state's too
        return view

    @property
    def space(self) -> Space:
        """The cells of the state the view reads."""
        return self._state._space

    def __setattr__(self, name: str, value: object) -> Never:
        raise AttributeError(
            f"{type(self).__qualname__} is a view of an immutable state: phases return intents, "
            "and replace makes a new state"
        )

    def __reduce__(self) -> tuple[Callable[..., object], tuple[object, ...]]:
        return _lookup, (self._state, self._prefix)

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{field.name}: {field.dtype}{list(self._arrays[field.name].shape)}"
            for field in self._schema.values()
        )
        return f"{type(self).__qualname__}({fields})"


class Nested[G: NestedState]:
    """A `NestedState` nested in a state class under the attribute's name.

    ``foxes = Nested(Species)`` on a `State` subclass gives its states the fields of
    ``Species``, named ``"foxes.population"`` and so on. On the class, the attribute is
    ``Species`` as nested there: a subclass of it, typed ``type[Species]``, whose attributes are
    the nested fields and derived fields. On a state, it is a view of the state, typed
    ``Species``, whose attributes are the arrays. Each state makes its view once, on first read.

    On a `NestedState` subclass, ``foxes = Nested(Species)`` nests ``Species`` in it, and the
    fields are named by the whole path once that class is nested in a state class:
    ``"north.foxes.population"``.

    Parameters
    ----------
    declared : type of NestedState
        The nested state, as declared.

    Raises
    ------
    TypeError
        If ``declared`` is not a `NestedState` subclass as declared. When the class is made, also
        if the class is neither a `State` nor a `NestedState` subclass.
    ValueError
        When the class is made, if `State` itself (or `NestedState`, on a nested state) has an
        attribute of that name, or the nested state is already nested under another name.
    """

    __slots__ = ("bound", "declared", "name", "path")

    declared: type[G]
    """The nested state, as declared."""
    name: str
    """The attribute's name: empty until the class is made."""
    bound: type[G] | None
    """The nested state as nested here: None until the class is made."""
    path: str
    """The bound class's path, its views' key in a state: empty until the class is made."""

    def __init__(self, declared: type[G]) -> None:
        if not (isinstance(declared, type) and issubclass(declared, NestedState)) or (
            declared._prefix
        ):
            raise TypeError(f"Nested takes a NestedState subclass as declared, got {declared!r}")
        self.declared = declared
        self.name = self.path = ""
        self.bound = None

    def __set_name__(self, owner: type, name: str) -> None:
        base = _declaring_base(owner, f"nested state {name!r}: Nested")
        if hasattr(base, name):
            raise ValueError(f"nested state {name!r}: the name is taken by {base.__name__}.{name}")
        if self.name and self.name != name:
            raise ValueError(
                f"nested state {name!r}: this Nested is already declared as {self.name!r}"
            )
        self.name = name
        if self.bound is None:
            self.bound = _bind(self.declared, owner, name)
            self.path = name

    @overload
    def __get__(self, state: None, owner: type | None = None) -> type[G]: ...
    @overload
    def __get__(self, state: State | NestedState, owner: type | None = None) -> G: ...
    def __get__(self, state: State | NestedState | None, owner: type | None = None) -> type[G] | G:
        bound = self.bound
        if bound is None:
            raise AttributeError(
                f"Nested({self.declared.__name__}) is read before its class is made"
            )
        if state is None:
            return bound
        views = state._views  # the state's, also when read on a view
        view = views.get(self.path)
        if view is None:
            root = state._state if isinstance(state, NestedState) else state
            view = views[self.path] = bound._view(root)
        return view  # ty: ignore[invalid-return-type] - the views of this path are bound's

    def __repr__(self) -> str:
        return f"Nested({self.declared.__name__}{f', as {self.name!r}' if self.name else ''})"


type Dependency = Field[Any] | Derived[Any] | Nested[Any] | type[NestedState]
"""What a derived field may depend on: a field, a derived field, or a whole nested state."""


def _bind[G: NestedState](declared: type[G], owner: type, name: str) -> type[G]:
    """Return a nested state as nested in ``owner`` under the path ``name``, with qualified fields.

    The states nested in ``declared`` are bound again under the longer path, as are their own,
    so every field is named by its whole path from ``owner``: ``"north.foxes.population"``.
    The class's qualified name is the path too, ``WorldState.north.foxes``, so pickle finds it.
    """
    namespace = {
        "__slots__": (),
        "_prefix": name,
        "_root": owner if issubclass(owner, State) else None,
        "__module__": owner.__module__,
        "__qualname__": f"{owner.__qualname__}.{name}",
    }
    bound = cast("type[G]", type(declared.__name__, (declared,), namespace))  # a subclass
    by_name: dict[str, Field[Any]] = {}
    for attr, value in _attributes(declared, Field):
        field = _named_field(
            f"{name}.{attr}", value.dtype, value.shape, value.static, value.extensive, value.default
        )
        type.__setattr__(bound, attr, field)  # not __set_name__, which would name it plainly
        by_name[field.name] = field
    for attr, value in _attributes(declared, Nested):
        inner = _bind(value.declared, owner, f"{name}.{attr}")
        nested = Nested(value.declared)
        nested.name, nested.bound, nested.path = attr, inner, inner._prefix
        type.__setattr__(bound, attr, nested)
        by_name.update((field.name, field) for field in inner._schema.values())
    for attr, value in _attributes(declared, Derived):  # base classes' first, so each finds its
        # dependencies already bound, by their names relative to ``declared``
        depends_on = tuple(_lookup(bound, _dependency_name(each)) for each in value.depends_on)
        copy = Derived(value.fn, cast("tuple[Dependency, ...]", depends_on))
        object.__setattr__(copy, "name", f"{name}.{attr}")
        object.__setattr__(copy, "owner", owner)
        object.__setattr__(copy, "_nested", bound)
        type.__setattr__(bound, attr, copy)
    bound._schema = _Schema(*(by_name[f"{name}.{each}"] for each in declared._schema))
    return bound


def _attributes[T](declared: type[NestedState], kind: type[T]) -> list[tuple[str, T]]:
    """Return a nested state's attributes of a kind with their names, base classes' first."""
    found = {
        attr: value
        for cls in reversed(declared.__mro__)
        for attr, value in vars(cls).items()
        if isinstance(value, kind)
    }
    return list(found.items())


def _class_schema(cls: type, inherited: list[Field[Any]], own: list[Field[Any]]) -> _Schema:
    """Return the schema of a `State` or `NestedState` subclass, checking its fields and derived.

    Raises
    ------
    ValueError
        If two different fields have one name.
    TypeError
        If an attribute hides a field of a base class.
    KeyError
        If a derived field depends on a field the class lacks, or declares differently.
    """
    try:
        schema = _Schema(*inherited, *own)
    except ValueError as error:
        raise ValueError(f"{cls.__name__}: {error}") from None
    for name, field in schema.items():
        seen = _lookup(cls, name, None)
        if not (isinstance(seen, Field) and seen == field):
            raise TypeError(f"{cls.__name__}.{name} hides the field of that name")
    for name, value in vars(cls).items():
        if isinstance(value, Derived):
            for field in value.fields():
                try:
                    schema._declared(field)
                except KeyError as error:
                    raise KeyError(
                        f"{cls.__name__}.{name} depends on {field.name!r}: {error.args[0]}"
                    ) from None
    return schema


def _dependency_name(each: Dependency) -> str:
    """Return the name of what a derived field depends on, for its repr."""
    if isinstance(each, Field | Derived | Nested):
        return each.name
    return each._prefix or each.__name__


def _declaring_base(owner: type, what: str) -> type:
    """Return `State` or `NestedState`, whichever ``owner`` subclasses, for a field on it."""
    if issubclass(owner, State):
        return State
    if issubclass(owner, NestedState):
        return NestedState
    raise TypeError(f"{what} is declared on a subclass of State or NestedState")


def _is_nested_state(value: object) -> bool:
    """Return whether ``value`` is a `NestedState` subclass, as declared or as nested."""
    return isinstance(value, type) and issubclass(value, NestedState)


def _as_nested(value: object, *, rooted: bool = True) -> type[NestedState]:
    """Return a nested state's class as nested, from a `Nested` or the class itself.

    With ``rooted``, the class must be nested in a state class, so that its fields are a
    state's. A derived field's dependencies need not be: they are checked against its class.

    Raises
    ------
    TypeError
        If ``value`` is a nested state as declared, whose fields no state holds under those
        names, or with ``rooted``, one nested only in a nested state as declared.
    """
    if isinstance(value, Nested):
        if value.bound is None:
            raise TypeError(f"{value!r} is not yet nested in a state class")
        value = value.bound
    if not (isinstance(value, type) and issubclass(value, NestedState) and value._prefix):
        raise TypeError(
            f"{value!r} is a nested state as declared; name it as nested in a state class, as "
            "EcosystemState.foxes"
        )
    if rooted and value._root is None:
        raise TypeError(
            f"{value.__qualname__} is nested in a nested state as declared, so its fields are "
            "no state's; name it through a state class, as WorldState.north.foxes"
        )
    return value


def _lookup(base: object, path: str, *default: object) -> object:
    """Return the attribute at a dotted path, such as ``"foxes.population"`` of a state class."""
    try:
        return functools.reduce(getattr, path.split("."), base)
    except AttributeError:
        if default:
            return default[0]
        raise


def _written(writes: object) -> tuple[Field[Any], ...]:
    """Return the fields a phase writes: fields as given, and a nested state's dynamic fields.

    Raises
    ------
    TypeError
        If an entry is neither a declared field nor a nested state.
    """
    one = isinstance(writes, Field | str) or _is_nested_state(writes)
    entries = (writes,) if one else tuple(writes)  # ty: ignore[invalid-argument-type] - else raises
    fields: list[Field[Any]] = []
    for each in entries:
        if _is_nested_state(each):
            fields.extend(field for field in _as_nested(each)._schema.values() if not field.static)
        elif isinstance(each, Field) and each.name:
            fields.append(each)
        else:
            raise TypeError(
                f"writes must be fields declared on a State class, or states nested in one, got "
                f"{writes!r}"
            )
    return tuple(fields)


class State(Mapping[Field[Any], NDArray[Any]]):
    """The values of every field at one step: an immutable snapshot.

    A model's states are a subclass that declares its fields, one `Field` attribute each:

    >>> class BoardState(State):
    ...     height = Field(np.float64, static=True)
    ...     owner = Field(np.int8, default=-1)
    >>> board = BoardState(space, {BoardState.height: height})

    On a state, each attribute is the field's read-only array, of the field's dtype and shape
    ``(*field.shape, n_cells)``: ``board.owner`` is typed ``NDArray[np.int8]``. On the class, it
    is the field itself, which names it wherever a field is wanted: ``BoardState.owner``.

    Classes combine by inheritance, as in ``class GameState(TerrainState, PiecesState)``, and
    hold their fields in declaration order, those of the base classes first. A field inherited
    twice counts once. A state's fields are exactly its class's, and `State` itself declares
    none.

    A state is also a mapping from its fields to their arrays, in declaration order:
    ``board[BoardState.owner]`` is ``board.owner``, for code that gets a field as a value.
    Looking up a field the state declares differently raises `KeyError`, so a field's static
    type is always its array's.

    Arrays computed from fields are declared on the class too, as methods decorated with
    `derived`, and read like fields: ``board.land`` and ``board[BoardState.land]``. They are
    cached, not stored (see `Derived`).

    A class whose rules need a kind of space says so with an annotation, ``space:
    LatticeSpace`` or ``space: LatticeSpace | MeshSpace``: ty then types ``board.space`` by it,
    and a state of the class (or of a subclass) is made only on such a space. Subclasses keep
    their bases' annotations and may narrow them.

    Phases read a state and return intents. The simulation applies the intents by making a new
    state of the same class, so a state never changes after it is made, and setting an
    attribute raises `AttributeError`. Two states compare equal only if they are the same
    object. Compare field arrays to compare contents. A state pickles by its class's import
    path, so states of a class made inside a function cannot be pickled.

    Parameters
    ----------
    space : Space
        The cells the fields live on.
    values : mapping of Field to array_like, optional
        Initial values by field. Each is broadcast to its field's shape and converted to its
        dtype without changing its meaning (see `as_field_array`): floats are never truncated
        to integers, and values must fit the field's range. The arrays are copied. Fields left
        out start at their default, dynamic fields without one at zero. Static fields without
        a default must be given.
    step : int, default 0
        Number of steps taken.

    Raises
    ------
    KeyError
        If ``values`` has a field that the class does not declare, or declares differently.
    ValueError
        If a static field has no value, a value does not broadcast to its field's shape or fit
        its dtype, or ``step`` is negative. When a subclass is made, also if two different
        fields have one name.
    KeyError
        When a subclass is made, if a derived field depends on a field the class lacks.
    TypeError
        If ``space`` is not a `Space` (such as a state, as if making a view of it) or not of a
        class's annotated kind, a key of ``values`` is not a declared `Field`, a value would
        change kind, such as floats into an integer field, or ``step`` is not an integer. When
        a subclass is made, also if an attribute hides a field of a base class, or ``space`` is
        annotated with something other than a class or a union of classes.
    """

    __slots__ = ("_arrays", "_derived", "_space", "_step", "_versions", "_views")

    _schema: ClassVar[_Schema] = _Schema()
    _space_kinds: ClassVar[tuple[type | types.UnionType, ...]] = ()
    """The ``space`` annotations of the class and its bases: a state's space must be each."""
    _arrays: dict[str, NDArray[Any]]
    _derived: dict[Derived[Any], tuple[tuple[int, ...], NDArray[Any]]]
    """Computed arrays, by derived field, with the versions of its fields they were made from.

    Shared with the states made from this one, so a result is reused while those versions
    hold."""
    _space: Space
    _step: int
    _versions: dict[str, int]
    _views: dict[str, NestedState]
    """The views of nested states, by name, made on first read."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        inherited = [
            field
            for base in cls.__bases__
            if issubclass(base, State)
            for field in base._schema.values()
        ]
        own: list[Field[Any]] = []
        for value in vars(cls).values():
            if isinstance(value, Field):
                own.append(value)
            elif isinstance(value, Nested):
                own.extend(_as_nested(value)._schema.values())
        cls._schema = _class_schema(cls, inherited, own)
        kinds = [
            kind for base in cls.__bases__ if issubclass(base, State) for kind in base._space_kinds
        ]
        # FORWARDREF: only this annotation is evaluated, since another may name what is
        # imported for ty alone; a name not defined is a ForwardRef, evaluated to raise.
        kind = inspect.get_annotations(cls, format=annotationlib.Format.FORWARDREF).get("space")
        if isinstance(kind, annotationlib.ForwardRef):
            kind = kind.evaluate(owner=cls)
        if isinstance(kind, str):  # as under `from __future__ import annotations`
            # Only this one: another annotation may name what is imported for ty alone.
            module = sys.modules.get(cls.__module__)
            kind = eval(kind, vars(module) if module else {}, dict(vars(cls)))
        if kind is not None:
            members = kind.__args__ if isinstance(kind, types.UnionType) else (kind,)
            if not all(isinstance(member, type) for member in members):
                raise TypeError(
                    f"{cls.__name__}.space must be annotated with a class or a union of "
                    f"classes, got {kind!r}"
                )
            kinds.append(kind)
        cls._space_kinds = tuple(dict.fromkeys(kinds))  # each once, in order

    @classmethod
    def _check_space(cls, space: Space) -> None:
        """Raise TypeError unless ``space`` is of every kind the class's annotations ask for."""
        for kind in cls._space_kinds:
            if not isinstance(space, kind):
                name = getattr(kind, "__name__", None) or str(kind)
                raise TypeError(
                    f"{cls.__name__} is made on a space of kind {name}, got {type(space).__name__}"
                )

    def __init__(
        self,
        space: Space,
        values: Mapping[Field[Any], ArrayLike] | None = None,
        *,
        step: SupportsIndex = 0,
    ) -> None:
        if not isinstance(space, Space):
            hint = (
                ": a state needs no view, because its fields are its attributes"
                if isinstance(space, State)
                else ""
            )
            raise TypeError(
                f"{type(self).__name__} is made on a space, got {type(space).__name__}{hint}"
            )
        self._build(space, {} if values is None else values, step)

    def _build(self, space: Space, values: Mapping[Any, ArrayLike], step: SupportsIndex) -> None:
        """Fill in a new state, checking ``values`` and freezing copies, as `State` documents.

        The constructor and `_restore` both build through it, so a pickled state is checked the
        same way as a new one.
        """
        self._check_space(space)
        schema = self._schema
        by_name = {schema._declared(field).name: value for field, value in values.items()}
        step = check_int(step, "step", minimum=0)
        arrays = _initial_arrays(schema, space.n_cells, by_name)
        _fill(self, space, arrays, {name: _next_version() for name in arrays}, step, {})

    @classmethod
    def _restore(cls, space: Space, arrays: dict[str, NDArray[Any]], step: int) -> Self:
        """Rebuild a pickled state, checking and freezing its arrays as the constructor does."""
        state = object.__new__(cls)
        state._build(space, {cls._schema[name]: array for name, array in arrays.items()}, step)
        return state

    def _with(self, arrays: dict[str, NDArray[Any]], step: int | None = None) -> Self:
        """Return a state of the same class with some fields replaced by checked arrays.

        Internal: the arrays must come from `as_field_array`. Replaced fields get new versions.
        """
        new = object.__new__(type(self))
        versions = self._versions | {name: _next_version() for name in arrays}
        step = self._step if step is None else step
        _fill(new, self._space, self._arrays | arrays, versions, step, self._derived)
        return new

    def __setattr__(self, name: str, value: object) -> Never:
        raise AttributeError(
            f"{type(self).__name__} is immutable: phases return intents, and replace makes a "
            "new state. To give cells more to hold, declare a Field on the class"
        )

    @property
    def space(self) -> Space:
        """The cells the fields live on."""
        return self._space

    @property
    def step(self) -> int:
        """Number of steps taken."""
        return self._step

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self._space.n_cells

    def __getitem__[T: np.generic](self, field: Field[T] | Derived[T]) -> NDArray[T]:
        if isinstance(field, Derived):
            if _lookup(type(self), field.name or "_", None) is not field:
                raise KeyError(f"{type(self).__name__} has no derived field {field!r}")
            return field._value(self)
        return self._arrays[self._schema._declared(field).name]

    def __contains__(self, field: object) -> bool:
        return field in self._schema

    def __iter__(self) -> Iterator[Field[Any]]:
        return iter(self._schema.values())

    def __len__(self) -> int:
        return len(self._arrays)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[object, ...]]:
        # Rebuild through the constructor's checks, which copy and freeze the arrays again.
        # Versions are process-local, so the copy gets new ones.
        return type(self)._restore, (self._space, self._arrays, self._step)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}: {array.dtype}{list(array.shape)}" for name, array in self._arrays.items()
        )
        return f"{type(self).__name__}(step={self._step}, {fields})"

    def version(self, field: Field[Any]) -> int:
        """Return the version of a field's values.

        Every write gives a field a new version, unique within the process. Two states whose
        field has the same version hold the same array, so cached results derived from a field
        stay valid while its version is unchanged, also across snapshots and forks.

        Parameters
        ----------
        field : Field
            The field.

        Returns
        -------
        int
            The version number.
        """
        return self._versions[self._schema._declared(field).name]

    def replace(self, values: Mapping[Field[Any], ArrayLike], /) -> Self:
        """Return a copy of the state with some fields set to new values.

        This is how to change a state between steps, for example to set up a scenario. It can
        also set static fields. Phases cannot: they return intents.

        Parameters
        ----------
        values : mapping of Field to array_like
            New values by field, broadcast and cast as in the constructor.

        Returns
        -------
        State
            A state of the same class, at the same step. Replaced fields get new versions, and
            the others keep their arrays and versions.

        Raises
        ------
        KeyError, ValueError, TypeError
            As in the constructor.
        """
        arrays = {}
        for field, value in values.items():
            declared = self._schema._declared(field)
            arrays[declared.name] = as_field_array(declared, self.n_cells, value)
        return self._with(arrays)

    def edit(self, *fields: Field[Any] | type[NestedState]) -> Edit[Self]:
        """Return writable copies of some fields, to change in place and freeze into a state.

        This is how a game's step makes several writes, such as moving a piece with all its
        fields: ``work = state.edit(GameState.pieces)``, then ``work[GameState.pieces.hp][c] -=
        1`` and ``work.move(a, b)``, then ``work.freeze()``. The fields are copied once, when
        the edit is made.

        Parameters
        ----------
        *fields : Field or nested state class
            The fields to edit, at least one: fields named on this state's class, or nested
            states as nested in it (``GameState.pieces``), which stand for all their fields.

        Returns
        -------
        Edit
            The copies, which `Edit.freeze` turns into a state of this class.

        Raises
        ------
        ValueError
            If no field is given, or a field is given twice.
        KeyError, TypeError
            If a field is not declared on this state's class, as for `replace`.
        """
        return Edit(self, fields)

    def extend[T: State](
        self, cls: type[T], values: Mapping[Field[Any], ArrayLike] | None = None, /
    ) -> T:
        """Return a copy of the state as a state of a subclass, which has more fields.

        This is how to give cells more to hold between steps, for example to add the pieces of
        a game to a board's terrain: ``terrain.extend(GameState, {GameState.owner: owner})``,
        with ``class GameState(TerrainState, PiecesState)``.

        Parameters
        ----------
        cls : State subclass
            The class of the new state: the state's own class or a subclass of it, so that the
            new state can be used wherever this one was.
        values : mapping of Field to array_like, optional
            Values by field, for the new fields and any of the state's own, broadcast and cast
            as in the constructor. New fields left out start as in the constructor.

        Returns
        -------
        State
            A state of class ``cls`` on the same space, at the same step. New and replaced
            fields get new versions, and the others keep their arrays and versions.

        Raises
        ------
        TypeError
            If ``cls`` is not a subclass of the state's class, or asks for another kind of
            space, and as in the constructor.
        KeyError, ValueError
            As in the constructor.
        """
        if not (isinstance(cls, type) and issubclass(cls, type(self))):
            name = type(self).__name__
            raise TypeError(f"a {name} extends to a subclass of {name}, got {cls!r}")
        cls._check_space(self._space)
        schema = cls._schema
        own = self._schema._fields
        old, new = {}, {}
        for field, value in ({} if values is None else values).items():
            declared = schema._declared(field)
            (old if declared.name in own else new)[declared.name] = value
        added = _Schema(*(field for name, field in schema.items() if name not in own))
        arrays = _initial_arrays(added, self.n_cells, new)
        for name, value in old.items():
            arrays[name] = as_field_array(schema[name], self.n_cells, value)
        state = object.__new__(cls)
        versions = self._versions | {name: _next_version() for name in arrays}
        _fill(state, self._space, self._arrays | arrays, versions, self._step, self._derived)
        return state

    def values_at(self, cell: SupportsIndex) -> dict[Field[Any], Any]:
        """Return what one cell holds: the value of each field at the cell.

        Parameters
        ----------
        cell : int
            Cell id.

        Returns
        -------
        dict of Field to scalar or ndarray
            By field, in declaration order: a NumPy scalar of the field's dtype for a field
            with one value per cell, and otherwise a read-only array of shape ``field.shape``.

        Raises
        ------
        TypeError
            If ``cell`` is not an integer.
        IndexError
            If ``cell`` is not a cell of the space.
        """
        cell = check_cell(cell, self.n_cells)
        return {field: self._arrays[name][..., cell][()] for name, field in self._schema.items()}

    @overload
    def density(self, field: Field[np.float64]) -> NDArray[np.float64]: ...
    @overload
    def density[T: np.generic](self, field: Field[T]) -> NDArray[T | np.float64]: ...
    def density[T: np.generic](self, field: Field[T]) -> NDArray[T | np.float64]:
        """Return a field per unit cell measure.

        Typed by the field: float64 for a float64 field, extensive or not. Whether a field is
        extensive is known only when it runs, so another dtype gives either its own or float64.

        Parameters
        ----------
        field : Field
            The field.

        Returns
        -------
        ndarray, shape (*field.shape, n_cells)
            For an extensive field, its values divided by the cell measures (float64). An
            intensive field is already per unit measure, so its stored array is returned.
        """
        field = self._schema._declared(field)
        values = self._arrays[field.name]
        if not field.extensive:
            return values
        return values / self._space.measures

    @overload
    def total[F: np.floating[Any]](self, field: Field[F]) -> NDArray[np.float64]: ...
    @overload
    def total[T: np.generic](self, field: Field[T]) -> NDArray[np.float64 | np.int_]: ...
    def total[T: np.generic](self, field: Field[T]) -> NDArray[np.float64 | np.int_]:
        """Return the amount of a field over all cells.

        Typed by the field: float64 for a floating field. An integer or boolean field gives an
        integer sum if it is extensive and float64 if not, which is known only when it runs.

        Parameters
        ----------
        field : Field
            The field.

        Returns
        -------
        ndarray, shape field.shape
            For an extensive field, the sum of its values over cells: float64 for a floating
            field, accumulated in float64 as `pull` and `push` do, and an exact integer sum
            for an integer field. For an intensive field, its integral: the sum of values times
            cell measures (float64).
        """
        field = self._schema._declared(field)
        values = self._arrays[field.name]
        if field.extensive:
            return values.sum(axis=-1, dtype=np.float64 if values.dtype.kind == "f" else None)
        return values @ self._space.measures


class Edit[S: State]:
    """Writable copies of some fields of a state, changed in place, then frozen into a state.

    Made by `State.edit`. Reading a field, ``work[field]``, gives its writable array, of the
    field's dtype and shape ``(*field.shape, n_cells)``; writes to it go into the frozen state.
    `move`, `swap` and `clear` act on every field of the edit at once, as on the pieces of a
    game, and cleared cells take each field's default (0 for a field without one). `freeze`
    makes the state once, sharing the copies with it: after that the edit refuses to be used,
    and its arrays are read-only.

    Parameters
    ----------
    state : State
        The state to edit.
    fields : sequence of Field or nested state class
        As for `State.edit`.

    Raises
    ------
    ValueError, KeyError, TypeError
        As for `State.edit`.
    """

    __slots__ = ("_arrays", "_empty", "_n_cells", "_state")

    _state: S | None
    _arrays: dict[str, NDArray[Any]]
    _empty: dict[str, np.generic]
    _n_cells: int

    def __init__(self, state: S, fields: Sequence[Field[Any] | type[NestedState]]) -> None:
        schema = state._schema
        declared: list[Field[Any]] = []
        for each in fields:
            if isinstance(each, type) and issubclass(each, NestedState):
                declared.extend(schema._declared(f) for f in each._schema.values())
            else:
                declared.append(schema._declared(each))
        if not declared:
            raise ValueError("an edit needs at least one field")
        names = [f.name for f in declared]
        if len(set(names)) != len(names):
            twice = next(n for n in names if names.count(n) > 1)
            raise ValueError(f"field {twice!r} is given twice")
        self._state = state
        self._n_cells = state.n_cells
        self._arrays = {f.name: state._arrays[f.name].copy() for f in declared}
        self._empty = {
            f.name: f.dtype.type(0) if f.default is None else f.default for f in declared
        }

    def __getitem__[T: np.generic](self, field: Field[T]) -> NDArray[T]:
        name = self._open()._schema._declared(field).name
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(f"field {name!r} is not in this edit: {list(self._arrays)}") from None

    def move(self, a: SupportsIndex, b: SupportsIndex) -> None:
        """Move every edited field's values from cell ``a`` to cell ``b``, and clear ``a``.

        Parameters
        ----------
        a, b : int
            Cell ids. ``b``'s values are overwritten; moving a cell onto itself clears it.

        Raises
        ------
        TypeError
            If a cell is not an integer.
        IndexError
            If a cell is not a cell of the space.
        RuntimeError
            If the edit is frozen.
        """
        self._open()
        a, b = check_cell(a, self._n_cells), check_cell(b, self._n_cells)
        for name, values in self._arrays.items():
            values[..., b] = values[..., a]
            values[..., a] = self._empty[name]

    def swap(self, a: SupportsIndex, b: SupportsIndex) -> None:
        """Swap every edited field's values between cells ``a`` and ``b``.

        Parameters
        ----------
        a, b : int
            Cell ids.

        Raises
        ------
        TypeError
            If a cell is not an integer.
        IndexError
            If a cell is not a cell of the space.
        RuntimeError
            If the edit is frozen.
        """
        self._open()
        pair = [check_cell(a, self._n_cells), check_cell(b, self._n_cells)]
        for values in self._arrays.values():
            values[..., pair] = values[..., pair[::-1]]

    def clear(self, cell: SupportsIndex) -> None:
        """Set every edited field at ``cell`` to its default (0 for a field without one).

        Parameters
        ----------
        cell : int
            A cell id.

        Raises
        ------
        TypeError
            If ``cell`` is not an integer.
        IndexError
            If ``cell`` is not a cell of the space.
        RuntimeError
            If the edit is frozen.
        """
        self._open()
        cell = check_cell(cell, self._n_cells)
        for name, values in self._arrays.items():
            values[..., cell] = self._empty[name]

    def freeze(self) -> S:
        """Return the state with the edited fields' values, and close the edit.

        Returns
        -------
        State
            A state of the edited state's class, at the same step. The edited fields get new
            versions, whether or not their values changed; the others keep their arrays and
            versions.

        Raises
        ------
        RuntimeError
            If the edit is already frozen.
        """
        state = self._open()
        self._state = None
        return state._with({name: read_only(values) for name, values in self._arrays.items()})

    def _open(self) -> S:
        """Return the state being edited, or raise if the edit is frozen."""
        if self._state is None:
            raise RuntimeError("the edit is frozen: make a new one with State.edit")
        return self._state

    def __repr__(self) -> str:
        status = "frozen" if self._state is None else "open"
        return f"Edit({status}, {', '.join(self._arrays)})"


def _fill(
    state: State,
    space: Space,
    arrays: dict[str, NDArray[Any]],
    versions: dict[str, int],
    step: int,
    derived: dict[Derived[Any], tuple[tuple[int, ...], NDArray[Any]]],
) -> None:
    """Set the attributes of a new state, which refuses ordinary assignment."""
    set_ = object.__setattr__
    set_(state, "_space", space)
    set_(state, "_arrays", arrays)
    set_(state, "_versions", versions)
    set_(state, "_step", step)
    set_(state, "_derived", derived)
    set_(state, "_views", {})


def _initial_arrays(
    schema: _Schema, n_cells: int, values: Mapping[str, ArrayLike]
) -> dict[str, NDArray[Any]]:
    """Return the starting arrays of a schema's fields, from values by name or the defaults.

    Each array is read-only, of shape ``(*field.shape, n_cells)`` and the field's dtype.
    """
    arrays = {}
    for name, field in schema.items():
        if name in values:
            arrays[name] = as_field_array(field, n_cells, values[name])
        elif field.default is not None:
            full = np.full(field.array_shape(n_cells), field.default, field.dtype)
            arrays[name] = read_only(full)
        elif field.static:
            raise ValueError(f"static field {name!r} needs an initial value")
        else:
            arrays[name] = read_only(np.zeros(field.array_shape(n_cells), field.dtype))
    return arrays


def as_field_array(
    field: Field[Any],
    n_cells: int,
    values: ArrayLike,
    *,
    where: ArrayLike | None = None,
    base: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return values as a new read-only array for a field.

    Values are converted without changing what they mean. Floats are never truncated to
    integers, and only booleans become booleans. Integers may go into any integer field if they
    fit its range, so codes computed as int64 can be stored as uint8, but a count that does not
    fit raises instead of wrapping around. Numbers may be stored as floats of lower precision,
    and round, but a finite value beyond the float's range raises instead of becoming infinite.

    Parameters
    ----------
    field : Field
        The field.
    n_cells : int
        Number of cells.
    values : array_like
        New values, broadcast to ``(*field.shape, n_cells)`` and converted to ``field.dtype``.
    where : array_like of bool, optional
        Where to write, broadcast to the field's shape. Elsewhere the result keeps the values
        of ``base``, and values there are not checked. Defaults to everywhere.
    base : ndarray, optional
        The field's current values. Required with ``where``.

    Returns
    -------
    ndarray, shape (*field.shape, n_cells)
        A new read-only array of dtype ``field.dtype``.

    Raises
    ------
    ValueError
        If ``values`` or ``where`` does not broadcast to the field's shape, or values do not
        fit the range of the field's dtype.
    TypeError
        If the values would change kind (such as floats into an integer field), or ``where``
        is not boolean.
    """
    what = f"field {field.name!r}"
    return _convert(what, field.dtype, field.array_shape(n_cells), values, where, base)


def _convert(
    what: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    values: ArrayLike,
    where: ArrayLike | None = None,
    base: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Convert values as `as_field_array` does, naming ``what`` in errors."""
    if (
        where is None
        and type(values) is np.ndarray
        and values.dtype == dtype
        and values.shape == shape
    ):
        return read_only(values.copy())  # every check below would pass: only copy
    source = np.asarray(values)
    mask: NDArray[np.bool_] | bool = True
    if where is not None:
        mask = np.asarray(where)
        if mask.dtype != np.bool_:
            raise TypeError(f"{what}: where must be boolean, got {mask.dtype}")
        if base is None:
            raise ValueError("writing with where needs the field's current values as base")
    if not (_broadcasts(source.shape, shape) and _broadcasts(np.shape(mask), shape)):
        given = f"values of shape {source.shape}"
        if where is not None:
            given += f" and where of shape {np.shape(mask)}"
        raise ValueError(f"{what} has shape {shape}; {given} do not broadcast to it")
    narrowing = _check_conversion(what, dtype, source, mask, shape)
    out = np.empty(shape, dtype) if base is None or where is None else base.copy()
    if not narrowing:
        np.copyto(out, source, casting="unsafe", where=mask)
        return read_only(out)
    try:  # values may round, but must not overflow to infinity
        with np.errstate(over="raise"):
            np.copyto(out, source, casting="unsafe", where=mask)
    except FloatingPointError:
        raise ValueError(
            f"{what}: values beyond +-{np.finfo(dtype).max} do not fit in {dtype}"
        ) from None
    return read_only(out)


def _broadcasts(shape: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """Return whether an array of ``shape`` broadcasts to ``target`` without changing it."""
    try:
        return np.broadcast_shapes(shape, target) == target
    except ValueError:
        return False


def _check_conversion(
    what: str,
    dtype: np.dtype[Any],
    source: NDArray[Any],
    mask: NDArray[np.bool_] | bool,
    shape: tuple[int, ...],
) -> bool:
    """Raise if storing ``source`` in the field would change the meaning of its values.

    Returns whether the field is a float type narrower than ``source``. Such values may still
    overflow, which only the conversion itself detects.
    """
    if np.can_cast(source.dtype, dtype, "safe"):
        return False
    if not (source.dtype.kind in "iu" and dtype.kind in "iu"):
        if np.can_cast(source.dtype, dtype, "same_kind"):
            return dtype.kind == "f"
        raise TypeError(
            f"{what}: {source.dtype} values cannot be stored as {dtype} without changing their "
            "kind; convert them explicitly"
        )
    if mask is not True:
        source = np.broadcast_to(source, shape)[np.broadcast_to(mask, shape)]
    if source.size:
        low, high, info = int(source.min()), int(source.max()), np.iinfo(dtype)
        if low < info.min or high > info.max:
            raise ValueError(
                f"{what}: values from {low} to {high} do not fit in {dtype}, whose range is "
                f"[{info.min}, {info.max}]"
            )
    return False
