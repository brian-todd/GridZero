"""Intents as arrays: `core`'s `Assign`, `Flow`, `Spread`, `Source` and `Claim`, and `resolve`.

A rule in arrays proposes what `core`'s rules propose, under the same names and with the same
meaning, and `resolve` applies it to `Fields` as `core.resolve` does to a state: takers of a
field share each cell's amount, nothing becomes negative, nothing enters a cell outside a
domain, and proposals out of range are clipped.

What differs:

- Amounts are float32, so results equal `core`'s to float32's precision, and a total is
  conserved to that precision.
- Nothing about values is checked, because compiled code cannot raise: values that are not
  finite, or negative amounts in a field, give results that mean nothing. What depends only on
  the fields and the intents is checked as in `core`, when the rule is traced: a field the
  phase may not write, a field assigned twice, an intensive field moved.
- `Flow` and `Spread` take one field, not a nested state.
- An attraction counts as 0 below ``2**-100`` of the largest (`core`: ``2**-500``).
"""

from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from gridzero.batched.operators import Operator, pull, push
from gridzero.batched.states import Fields
from gridzero.core import Field, State


class Assign(NamedTuple):
    """Overwrite a field with new values, as `core.Assign`."""

    field: Field[Any]
    """A dynamic field."""
    values: ArrayLike
    """New values, broadcast to the field's shape ``(*field.shape, n_cells)``."""
    where: ArrayLike | None = None
    """Where to write, broadcast to the field's shape: bool. Everywhere by default."""


class Flow(NamedTuple):
    """Send parts of each cell's amount along edges, conserving the total, as `core.Flow`."""

    field: Field[Any]
    """A dynamic, extensive, floating field."""
    op: Operator
    """The edges amounts move along. Its weights are not used."""
    rate: ArrayLike
    """The fraction of the source's amount sent along each edge, in the order of the
    operator's ``sources`` and ``targets``: float, shape (..., n_edges). Negative rates count
    as 0, and a cell's rates are scaled to add up to 1 where they add up to more."""
    domain: ArrayLike | None = None
    """Cells amounts may move into: bool, shape (n_cells,). What is sent toward another cell
    stays at its source."""


class Spread(NamedTuple):
    """Send a fraction of each cell's amount to its neighbors, as `core.Spread`."""

    field: Field[Any]
    """A dynamic, extensive, floating field."""
    op: Operator
    """Neighbors and weights, which must not be negative."""
    leave: ArrayLike
    """The fraction of each cell's amount that leaves, clipped to [0, 1]: float, broadcast to
    the field's shape."""
    toward: ArrayLike | None = None
    """Each cell's attraction as a destination, negative values counting as 0: float,
    broadcast like ``leave``. 1 everywhere by default."""
    domain: ArrayLike | None = None
    """Cells amounts may move into: bool, shape (n_cells,)."""


class Source(NamedTuple):
    """Add amounts to an extensive field, or remove them, as `core.Source`."""

    field: Field[Any]
    """A dynamic, extensive, floating field."""
    amount: ArrayLike
    """What is added to each cell, broadcast to the field's shape. A negative amount is
    removed, never more than the cell holds."""


class Claim(NamedTuple):
    """Move amounts from one field into another within each cell, as `core.Claim`."""

    field: Field[Any]
    """The resource: a dynamic, extensive, floating field."""
    amount: ArrayLike
    """What each claimant asks for, broadcast to the shape of ``into``. Negative claims
    count as 0."""
    into: Field[Any]
    """The field that receives the grants, of shape ``(*claimants, *field.shape)``."""
    efficiency: ArrayLike = 1.0
    """What each unit granted becomes in ``into``, broadcast to its shape."""


type Intent = Assign | Flow | Spread | Source | Claim
"""Any intent a rule in arrays can return."""

type Intents = Intent | Iterable[Intent] | None
"""What a rule returns: one intent, an iterable of intents, or None for no change."""

_FAINTEST = 2.0**-100
"""Attractions below this fraction of the largest count as 0, far from float32's smallest."""

type _Deliver = Callable[[jax.Array], list[tuple[Field[Any], jax.Array]]]
"""Given the granted fraction of each cell's amount, what a taker adds to which fields."""


def resolve[S: State](
    fields: Fields[S],
    intents: Intents,
    *,
    writes: Collection[Field[Any]] | None = None,
) -> Fields[S]:
    """Apply a rule's intents to the fields it read, as `core.resolve` does to a state.

    Parameters
    ----------
    fields : Fields
        The fields the rule read: one state's.
    intents : intent, iterable of intents, or None
        What the rule returned.
    writes : collection of Field, optional
        The fields the rule may write. Every dynamic field by default.

    Returns
    -------
    Fields
        New fields at the same step, or ``fields`` itself if there is nothing to apply.

    Raises
    ------
    TypeError
        If something is not an intent, or amounts are moved in a field that is not floating.
    KeyError
        If the state's class does not declare an intent's field.
    ValueError
        If an intent writes a static field or one outside ``writes``, a field is assigned
        twice or assigned and also changed by another intent, a field whose amounts move is
        intensive, or a claim's ``into`` does not fit its field.
    """
    assigned: dict[Field[Any], Assign] = {}
    moving: list[Flow | Spread | Source | Claim] = []
    changed: dict[Field[Any], str] = {}
    for intent in _as_intents(intents):
        written = (intent.field, intent.into) if isinstance(intent, Claim) else (intent.field,)
        for field in written:
            _check_writable(fields, field, writes)
        if isinstance(intent, Assign):
            if intent.field in assigned:
                raise ValueError(
                    f"field {intent.field.name!r} is assigned twice; combine the values in one "
                    "Assign"
                )
            assigned[intent.field] = intent
        else:
            moving.append(intent)
            for field in written:
                changed.setdefault(field, type(intent).__name__)
    for field, kind in changed.items():
        if field in assigned:
            raise ValueError(
                f"field {field.name!r} is assigned and also changed by a {kind} in the same phase"
            )
    new: dict[Field[Any], jax.Array] = {}
    for field, intent in assigned.items():
        old = fields[field]
        values = jnp.broadcast_to(jnp.asarray(intent.values), old.shape).astype(old.dtype)
        new[field] = values if intent.where is None else jnp.where(intent.where, values, old)
    new |= _resolve_amounts(fields, moving)
    return fields.replace(new) if new else fields


def _as_intents(intents: Intents) -> tuple[Intent, ...]:
    """Return what a rule returned as a tuple of intents; TypeError for anything else."""
    kinds = (Assign, Flow, Spread, Source, Claim)
    if intents is None:
        return ()
    if isinstance(intents, kinds):
        return (intents,)
    if isinstance(intents, str | Mapping | jax.Array | np.ndarray) or not isinstance(
        intents, Iterable
    ):
        raise TypeError(
            f"a rule must return intents or None, got {type(intents).__name__}; "
            "wrap new values in Assign(field, values)"
        )
    out = tuple(intents)
    for intent in out:
        if not isinstance(intent, kinds):
            raise TypeError(f"expected an intent such as Assign, got {type(intent).__name__}")
    return out


def _check_writable(
    fields: Fields[Any], field: Field[Any], writes: Collection[Field[Any]] | None
) -> None:
    """Raise unless a rule may write a field: declared, dynamic, and among ``writes``."""
    fields[field]  # KeyError if the class does not declare it
    if field.static:
        raise ValueError(f"field {field.name!r} is static, so rules cannot write it")
    if writes is not None and field not in writes:
        names = sorted(each.name for each in writes)
        raise ValueError(f"field {field.name!r} is not among the fields the phase writes: {names}")


def _amounts(fields: Fields[Any], field: Field[Any], what: str) -> jax.Array:
    """Return the amounts of a field that can hold them: floating and extensive."""
    if field.dtype.kind != "f":
        raise TypeError(
            f"{what}: field {field.name!r} holds {field.dtype}, but flows, sources and claims "
            "need a floating field"
        )
    if not field.extensive:
        raise ValueError(
            f"{what}: field {field.name!r} is intensive, but flows, sources and claims move "
            "amounts, so they need an extensive field"
        )
    return fields[field]


def _resolve_amounts(
    fields: Fields[Any], intents: list[Flow | Spread | Source | Claim]
) -> dict[Field[Any], jax.Array]:
    """Apply the intents that move amounts, sharing each cell's amount among its takers."""
    current: dict[Field[Any], jax.Array] = {}
    demands: dict[Field[Any], list[jax.Array]] = {}
    deliveries: dict[Field[Any], list[_Deliver]] = {}
    added: dict[Field[Any], list[jax.Array]] = {}

    def amounts_of(field: Field[Any], what: str) -> jax.Array:
        if field not in current:
            current[field] = _amounts(fields, field, what)
        return current[field]

    for intent in intents:
        field = intent.field
        what = f"{type(intent).__name__} on {field.name!r}"
        x = amounts_of(field, what)
        if isinstance(intent, Source):
            amount = jnp.asarray(intent.amount, x.dtype)
            demands.setdefault(field, []).append(jnp.maximum(-amount, 0.0))
            added.setdefault(field, []).append(jnp.maximum(amount, 0.0))
            continue
        if isinstance(intent, Claim):
            amounts_of(intent.into, what)
            demand, deliver = _claim(intent, x, current[intent.into], what)
        elif isinstance(intent, Flow):
            demand, deliver = _flow(intent, x)
        else:
            demand, deliver = _spread(intent, x)
        demands.setdefault(field, []).append(demand)
        deliveries.setdefault(field, []).append(deliver)
    remaining: dict[Field[Any], jax.Array] = {}
    for field, x in current.items():
        taken = demands.get(field)
        if not taken:
            remaining[field] = x
            continue
        demand = jnp.broadcast_to(sum(taken[1:], start=taken[0]), x.shape)
        short = demand > x
        granted = jnp.where(short, x / jnp.where(short, demand, 1.0), 1.0)
        remaining[field] = jnp.where(short, 0.0, x - demand)
        for deliver in deliveries.get(field, ()):
            for target, values in deliver(granted):
                added.setdefault(target, []).append(values)
    for field, extra in added.items():
        for values in extra:
            remaining[field] = remaining[field] + values
    return remaining


def _flow(intent: Flow, x: jax.Array) -> tuple[jax.Array, _Deliver]:
    """Return what a flow asks of each cell's amounts, and how to deliver what it is granted."""
    op, n_cells = intent.op, x.shape[-1]
    sources, targets = op.sources, op.targets

    def by(cells: jax.Array, values: jax.Array) -> jax.Array:
        """Add values per edge up per cell: (..., n_edges) to (..., n_cells)."""
        return jnp.zeros((*values.shape[:-1], n_cells), values.dtype).at[..., cells].add(values)

    rate = jnp.maximum(jnp.asarray(intent.rate, x.dtype), 0.0)
    rate = jnp.broadcast_to(rate, jnp.broadcast_shapes(rate.shape, sources.shape))
    total = by(sources, rate)
    over = total > 1
    scale = jnp.where(over, 1.0 / jnp.where(over, total, 1.0), 1.0)
    total = jnp.minimum(total, 1.0)
    if intent.domain is not None:  # what was meant for cells outside stays home
        rate = rate * jnp.asarray(intent.domain, jnp.bool_)[targets]
        total = by(sources, rate) * scale
    sending = x * scale

    def deliver(granted: jax.Array) -> list[tuple[Field[Any], jax.Array]]:
        return [(intent.field, by(targets, (granted * sending)[..., sources] * rate))]

    return x * total, deliver


def _spread(intent: Spread, x: jax.Array) -> tuple[jax.Array, _Deliver]:
    """Return what a spread asks of each cell's amounts, and how to deliver what it is granted."""
    op, n_cells = intent.op, x.shape[-1]
    leave = jnp.clip(jnp.asarray(intent.leave, x.dtype), 0.0, 1.0)
    toward = None  # 1 everywhere
    if intent.toward is not None:
        toward = jnp.maximum(jnp.asarray(intent.toward, x.dtype), 0.0)
        toward = jnp.broadcast_to(toward, jnp.broadcast_shapes(toward.shape, (n_cells,)))
        largest = toward.max(axis=-1, keepdims=True)
        toward = toward / jnp.where(largest > 0, largest, 1.0)
        toward = toward * (toward >= _FAINTEST)
    if intent.domain is not None:
        inside = jnp.asarray(intent.domain, jnp.bool_).astype(x.dtype)
        toward = inside if toward is None else toward * inside
    pulled = pull(jnp.ones(n_cells, x.dtype) if toward is None else toward, op)
    somewhere = pulled > 0  # nowhere to go: nothing leaves
    pulled = jnp.where(somewhere, pulled, 1.0)
    demand = x * leave * somewhere

    def deliver(granted: jax.Array) -> list[tuple[Field[Any], jax.Array]]:
        received = push(granted * demand / pulled, op)
        return [(intent.field, received if toward is None else toward * received)]

    return demand, deliver


def _claim(intent: Claim, x: jax.Array, into: jax.Array, what: str) -> tuple[jax.Array, _Deliver]:
    """Return what a claim asks of each cell's amounts, and how to deliver what it is granted."""
    field, target = intent.field, intent.into
    claimants = len(target.shape) - len(field.shape)
    if target == field or claimants < 0 or target.shape[claimants:] != field.shape:
        raise ValueError(
            f"{what}: into must be another field, of shape (*claimants, *{field.shape}); "
            f"{target.name!r} has shape {target.shape}"
        )
    claims = jnp.broadcast_to(jnp.maximum(jnp.asarray(intent.amount, x.dtype), 0.0), into.shape)
    efficiency = jnp.maximum(jnp.asarray(intent.efficiency, x.dtype), 0.0)
    demand = claims.sum(axis=tuple(range(claimants)))

    def deliver(granted: jax.Array) -> list[tuple[Field[Any], jax.Array]]:
        return [(target, claims * granted * efficiency)]

    return demand, deliver
