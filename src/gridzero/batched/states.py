"""A `core.State`'s fields as JAX arrays: what the rules of a model in arrays read and write.

`Fields` holds the arrays of one state, keyed by the `core.Field`s of its class, as `core.State`
does: ``fields[WorldState.people]``. It is a JAX tree of arrays, so many states stack into one
(``jax.tree.map``) and a function of one runs for all of them under ``jax.vmap``. `fields` makes
one of a state, and `state` a state of one.

Arrays are kept as JAX computes: float32 for floats, int32 for integers. A float64 field
therefore comes back to float32's precision.
"""

from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from gridzero.core import Field, Space, State


@jax.tree_util.register_pytree_node_class
class Fields[S: State]:
    """The fields of one state as arrays, keyed by field (see `fields`).

    A value: `replace` returns a new one. Under ``jax.vmap`` and ``jax.jit`` its arrays are
    traced like any other tree's.

    Parameters
    ----------
    kind : type of State
        The class of the state these are the fields of.
    declared : tuple of Field
        Its fields, in the order the state gives them.
    arrays : tuple of Array
        Their values, each of shape ``(*field.shape, n_cells)``, with any leading axes a batch
        adds.
    step : Array of int32, shape ()
        The step the state is at.

    Attributes
    ----------
    kind : type of State
        The class of the state.
    declared : tuple of Field
        Its fields.
    step : Array of int32, shape ()
        The step the state is at.
    """

    __slots__ = ("_arrays", "declared", "kind", "step")

    def __init__(
        self,
        kind: type[S],
        declared: tuple[Field[Any], ...],
        arrays: tuple[Any, ...],
        step: ArrayLike,
    ) -> None:
        self.kind = kind
        self.declared = declared
        self._arrays = arrays
        self.step: jax.Array = cast("jax.Array", step)  # NumPy's and tracers stand in for JAX's

    def __getitem__(self, field: Field[Any]) -> jax.Array:
        """Return a field's values: Array of shape ``(*field.shape, n_cells)``.

        Raises KeyError if the state's class does not declare the field.
        """
        return jnp.asarray(self._arrays[self._index(field)])

    def replace(self, values: Mapping[Field[Any], ArrayLike], /) -> Fields[S]:
        """Return these fields with some of them replaced.

        Parameters
        ----------
        values : mapping of Field to Array
            New values, each of its field's shape. They are converted to the dtype the field
            is kept in.

        Returns
        -------
        Fields
            A new value, at the same step.

        Raises
        ------
        KeyError
            If the state's class does not declare a field.
        ValueError
            If values do not have their field's shape.
        """
        arrays = list(self._arrays)
        for field, new in values.items():
            index = self._index(field)
            old = arrays[index]
            new = jnp.asarray(new)
            if new.shape != old.shape:
                raise ValueError(
                    f"field {field.name!r} has shape {old.shape}, got values of shape {new.shape}"
                )
            arrays[index] = new.astype(old.dtype)
        return Fields(self.kind, self.declared, tuple(arrays), self.step)

    def at(self, step: ArrayLike) -> Fields[S]:
        """Return these fields at another step.

        Parameters
        ----------
        step : Array of int, shape ()
            The step.

        Returns
        -------
        Fields
            A new value with the same arrays.
        """
        return Fields(self.kind, self.declared, self._arrays, jnp.asarray(step, jnp.int32))

    def _index(self, field: Field[Any]) -> int:
        """Return where a field's array is kept; KeyError if it is not declared."""
        try:
            return self.declared.index(field)
        except ValueError:
            name = getattr(field, "name", field)
            raise KeyError(f"{self.kind.__name__} declares no field {name!r}") from None

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return the arrays and the step as a JAX tree's children, and the rest."""
        return (*self._arrays, self.step), (self.kind, self.declared)

    @classmethod
    def tree_unflatten(cls, rest: tuple[Any, ...], children: tuple[Any, ...]) -> Fields[S]:
        """Rebuild from `tree_flatten`'s parts."""
        return cls(rest[0], rest[1], tuple(children[:-1]), children[-1])

    def __repr__(self) -> str:
        names = ", ".join(field.name for field in self.declared)
        return f"Fields({self.kind.__name__}: {names})"


def fields[S: State](state: S) -> Fields[S]:
    """Return a state's fields as arrays for JAX.

    Parameters
    ----------
    state : State
        A state of any class, on any space.

    Returns
    -------
    Fields
        NumPy arrays, which JAX takes as they are, in the dtypes it computes in: floats as
        float32, signed integers as int32, unsigned ones as uint32, booleans as they are.
        Stack several with ``jax.tree.map(lambda *xs: np.stack(xs), *all_of_them)``.

    Raises
    ------
    TypeError
        If ``state`` is not a `core.State`.
    ValueError
        If an integer does not fit in 32 bits, or a float is too large for float32.
    """
    if not isinstance(state, State):
        raise TypeError(f"state must be a core.State, got {type(state).__name__}")
    declared = tuple(state)
    arrays = tuple(_narrow(field.name, np.asarray(state[field])) for field in declared)
    return Fields(type(state), declared, arrays, np.int32(state.step))


def state[S: State](values: Fields[S], space: Space) -> S:
    """Return one state's fields as a `core.State`: the inverse of `fields`.

    Parameters
    ----------
    values : Fields
        The fields of one state, not a batch.
    space : Space
        The space the state is on.

    Returns
    -------
    State
        A state of the fields' class, at their step, each field in its declared dtype.
        Floats that went through float32 keep that precision.

    Raises
    ------
    TypeError, ValueError
        Whatever the state's class raises for values that do not fit its fields or its space.
    """
    arrays = {field: np.asarray(values[field]).astype(field.dtype) for field in values.declared}
    return values.kind(space, arrays, step=int(values.step))


def _narrow(name: str, array: NDArray[Any]) -> NDArray[Any]:
    """Return a field's array in the dtype JAX computes in, checking that its values fit."""
    kind = array.dtype.kind
    if kind == "b":
        return array
    target = {"f": np.float32, "i": np.int32, "u": np.uint32}.get(kind)
    if target is None:
        raise TypeError(f"field {name!r} holds {array.dtype}, which JAX arrays cannot")
    if array.dtype == target:
        return array
    if kind == "f" and array.dtype.itemsize <= 4:
        return array.astype(target)  # a narrower float: every value fits
    if kind == "f":
        limit = float(np.finfo(np.float32).max)
        if array.size and not (np.abs(array[np.isfinite(array)]) <= limit).all():
            raise ValueError(f"field {name!r} holds values too large for float32")
    else:
        low, high = (-(2**31), 2**31 - 1) if kind == "i" else (0, 2**32 - 1)
        if array.size and (array.min() < low or array.max() > high):
            raise ValueError(f"field {name!r} holds integers that do not fit in 32 bits")
    return array.astype(target)
