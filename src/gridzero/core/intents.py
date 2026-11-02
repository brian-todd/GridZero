"""Intents: the changes a phase proposes, and the resolver that applies them to a state.

Phases never write fields themselves. They return intents, and `resolve` applies them, checking
that each phase writes only the dynamic fields it declared:

- `Assign` overwrites a field.
- `Flow` and `Spread` move amounts of an extensive field between neighboring cells. They
  conserve its total. Given a nested state, they move every amount it holds along the
  same rates, such as people with the stores they carry.
- `Source` adds amounts to an extensive field, or removes them.
- `Claim` moves amounts from one extensive field into another within each cell, rationing a
  scarce supply among several claimants.

The resolver guarantees bounds and masks as well as conservation:

- **No amount becomes negative.** Everything that takes from a field in one phase (flows out,
  removals and claims) shares each cell's amount at the start of the phase. When together they
  ask for more than the cell holds, each gets the same fraction of what it asked for.
- **Nothing enters a cell outside a flow's domain.** What a `Flow` sends toward it stays home,
  and a `Spread` splits its leavers over the other neighbors.
- **Proposals out of range are clipped, not rejected.** Negative rates and claims count as 0,
  and a cell never sends more than it holds. A bad rule then makes a dull world rather than a
  broken one. Values that are not finite raise, because no clipping makes them meaningful.

Each phase reads one state, so all of its intents see the same amounts: amounts added by a
`Source` cannot leave in the same phase. To move what was just born, use a later phase.

Flows and rationing have Numba fast paths with bit-identical results, and spreads use the fast
paths of `pull` and `push` (see `gridzero.core.implementation`).
"""

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import _kernels
from gridzero.core._arrays import as_rows, cell_mask, finite_floats, read_only
from gridzero.core.fields import (
    Field,
    NestedState,
    State,
    _as_nested,
    _broadcasts,
    _is_nested_state,
    _written,
    as_field_array,
)
from gridzero.core.implementations import fast
from gridzero.core.operators import (
    NeighborOperator,
    _sum_by_source,
    _sum_by_target,
    pull,
    push,
)


@dataclass(frozen=True, eq=False)
class Assign:
    """Overwrite a field with new values.

    An `Assign` replaces the whole field, so a phase may not combine it with other intents on
    the same field.

    Parameters
    ----------
    field : Field
        A dynamic field.
    values : array_like
        New values, broadcast to the field's shape ``(*spec.shape, n_cells)`` and converted
        to its dtype without changing their meaning (see `as_field_array`): floats are never
        truncated to integers, and integers must fit the field's range.
    where : array_like of bool, optional
        Where to write, broadcast to the field's shape. Elsewhere the field keeps its values.
        Defaults to everywhere.
    """

    field: Field[Any]
    values: ArrayLike
    where: ArrayLike | None = None


@dataclass(frozen=True, eq=False)
class Flow:
    """Send parts of each cell's amount to its neighbors, edge by edge, conserving the total.

    This is the general form of movement: any split of each cell's amount over its out-edges.
    `Spread` covers the common case of a leave rate and attractive destinations more cheaply.

    Parameters
    ----------
    field : Field or nested state
        A dynamic, extensive, floating field. A `NestedState` as nested in a state class, such as
        ``WorldState.settlers``, stands for each of its dynamic, extensive, floating fields:
        each moves along the same rates, as if by a flow of its own.
    op : NeighborOperator
        The edges amounts move along. Its weights are not used.
    rate : array_like of float, shape (..., n_edges)
        Fraction of the source cell's amount sent along each edge ``i -> j``, aligned with
        ``op.graph.indices`` (``op.graph.sources`` holds each edge's source ``i``). Broadcast
        to ``(*spec.shape, n_edges)``, for each field it moves. Negative rates count as 0. If
        a cell's rates add up to more than 1, they are scaled to add up to 1: the cell sends
        all it holds.
    domain : array_like of bool, shape (n_cells,), optional
        Cells amounts may move into. Whatever a cell's rates, scaled if need be, send toward
        another cell stays at its source. Defaults to every cell.
    """

    field: Field[Any] | type[NestedState]
    op: NeighborOperator
    rate: ArrayLike
    domain: ArrayLike | None = None


@dataclass(frozen=True, eq=False)
class Spread:
    """Send a fraction of each cell's amount to its neighbors, split by weight and attraction.

    A fraction ``leave_i`` of cell ``i``'s amount leaves, and is split over its neighbors ``j``
    in proportion to ``w_ij * toward_j``. With a uniform ``toward`` this is diffusion along the
    operator's weights; otherwise it is migration toward attractive cells. It equals a `Flow`
    with rates ``leave_i * w_ij * toward_j / sum_k (w_ik * toward_k)``, but costs two sparse
    products instead of several passes over every edge.

    Parameters
    ----------
    field : Field or nested state
        A dynamic, extensive, floating field. A `NestedState` as nested in a state class, such as
        ``WorldState.settlers``, stands for each of its dynamic, extensive, floating fields:
        each moves with the same ``leave`` and ``toward``, as if by a spread of its own, so the
        ratios between them hold in what leaves each cell and what arrives.
    op : NeighborOperator
        Neighbors and weights ``w_ij``, which must not be negative. Unweighted means 1.
    leave : array_like of float, shape (..., n_cells)
        Fraction of each cell's amount that leaves, broadcast to ``(*spec.shape, n_cells)``
        for each field it moves, and clipped to [0, 1]. Nothing leaves a cell with nowhere to
        go: one whose neighbors all have ``w_ij * toward_j == 0``.
    toward : array_like of float, shape (..., n_cells), optional
        Attraction of each cell as a destination, broadcast like ``leave``. Negative values
        count as 0. Only ratios matter: multiplying ``toward`` by a constant changes nothing.
        Attractions below ``2**-500`` (about 3e-151) of the largest in their component count
        as 0. Defaults to 1 everywhere.
    domain : array_like of bool, shape (n_cells,), optional
        Cells amounts may move into: attraction is 0 everywhere else, so the share of each
        cell's leavers is split over its other neighbors. Defaults to every cell.
    """

    field: Field[Any] | type[NestedState]
    op: NeighborOperator
    leave: ArrayLike
    toward: ArrayLike | None = None
    domain: ArrayLike | None = None


@dataclass(frozen=True, eq=False)
class Source:
    """Add amounts to an extensive field, or remove them: births, deaths, growth, harvest.

    Parameters
    ----------
    field : Field
        A dynamic, extensive, floating field.
    amount : array_like of float, shape (..., n_cells)
        Amount added to each cell, broadcast to ``(*spec.shape, n_cells)``. A negative amount
        is removed, but never more than the cell holds: removals share the cell's amount with
        everything else that takes from the field in the same phase.
    """

    field: Field[Any]
    amount: ArrayLike


@dataclass(frozen=True, eq=False)
class Claim:
    """Move amounts from one extensive field into another within each cell, rationing scarcity.

    Claimants, such as population groups, each ask for an amount of a resource in their own
    cell. If the cell holds enough, every claim is granted. Otherwise each claimant gets the
    same fraction of its claim, and the cell is emptied. Granted amounts leave ``field``, and
    ``efficiency`` times them is added to ``into``: at the default of 1 the sum of the two
    fields is conserved, and below it the rest is lost, as when predators turn half of what
    they eat into young.

    Parameters
    ----------
    field : Field
        The resource: a dynamic, extensive, floating field.
    amount : array_like of float, shape (..., n_cells)
        What each claimant asks for, broadcast to the shape of ``into``,
        ``(*claimants, *field_shape, n_cells)``. Negative claims count as 0.
    into : Field
        The field that receives the grants: dynamic, extensive and floating, with component
        shape ``(*claimants, *field_shape)``. For example, three groups claiming a food field
        of shape ``()`` put what they get into a field of shape ``(3,)``.
    efficiency : array_like of float, default 1.0
        What each unit granted becomes in ``into``, broadcast to the shape of ``into``, so each
        claimant may have its own. Negative values count as 0.
    """

    field: Field[Any]
    amount: ArrayLike
    into: Field[Any]
    efficiency: ArrayLike = 1.0


type Intent = Assign | Flow | Spread | Source | Claim
"""Any intent a phase can return."""

type Intents = Intent | Iterable[Intent] | None
"""What a phase returns: one intent, an iterable of intents, or None for no change."""

_INTENT_TYPES = (Assign, Flow, Spread, Source, Claim)

type _Deliver = Callable[[NDArray[np.float64] | None], list[tuple[str, NDArray[np.float64]]]]
"""Given the granted fraction of each cell's amount, what a taker adds to which fields.

The fraction is None when every taker of the field gets all it asked for.
"""


def resolve[S: State](
    state: S,
    intents: Intents,
    *,
    writes: Field[Any]
    | type[NestedState]
    | Collection[Field[Any] | type[NestedState]]
    | None = None,
) -> S:
    """Apply a phase's intents to the state it read.

    Every intent reads ``state``. Intents that move or remove amounts of the same field share
    each cell's amount, and are scaled down together where they ask for more than there is.

    Parameters
    ----------
    state : State
        The state the phase read.
    intents : intent, iterable of intents, or None
        What the phase returned: `Assign`, `Flow`, `Spread`, `Source` and `Claim` objects.
    writes : Field, nested state, or collection of them, optional
        Fields the phase may write, a nested state standing for its dynamic fields. Defaults to
        every dynamic field.

    Returns
    -------
    State
        A new state of the same class with the changed fields replaced, at the same step. If
        there is nothing to apply, ``state`` itself.

    Raises
    ------
    TypeError
        If an intent is not one of the intent types, a field is not a declared `Field`, its
        values would change kind in the field, it moves amounts of a field that is not
        floating, or an array has the wrong dtype.
    KeyError
        If an intent has a field that the state does not have, or declares differently.
    ValueError
        If an intent writes a static field or one outside ``writes``, a field is assigned
        twice or assigned and changed by other intents, a field it moves amounts of is
        intensive or holds negative or non-finite amounts, its arrays do not broadcast to the
        field's shape or hold NaN or infinite values, its operator is on another number of
        cells or has negative weights (`Spread`), or a result does not fit the field's dtype.
    """
    if writes is None:
        return _resolve(state, intents, None)
    names = frozenset(state._schema._declared(field).name for field in _written(writes))
    return _resolve(state, intents, names)


def _resolve[S: State](state: S, intents: Intents, writes: Collection[str] | None) -> S:
    """Apply a phase's intents, as `resolve` does, given the names of the fields it writes."""
    assigns: dict[str, Assign] = {}
    amounts: list[Flow | Spread | Source | Claim] = []
    changed: dict[str, str] = {}  # field -> kind of the first intent that moves its amounts
    for intent in _per_field(_as_intents(intents)):
        field = _field(intent)
        fields = (field, intent.into) if isinstance(intent, Claim) else (field,)
        names = [_check_writable(state, field, writes) for field in fields]
        if isinstance(intent, Assign):
            if names[0] in assigns:
                raise ValueError(
                    f"field {names[0]!r} is assigned twice; combine the values in one Assign"
                )
            assigns[names[0]] = intent
        else:
            amounts.append(intent)
            for name in names:
                changed.setdefault(name, type(intent).__name__)
    for name, kind in changed.items():
        if name in assigns:
            raise ValueError(
                f"field {name!r} is assigned and also changed by a {kind} in the same phase; an "
                "Assign replaces the whole field, so it cannot be combined with other intents"
            )
    arrays: dict[str, NDArray[Any]] = {
        name: as_field_array(
            intent.field, state.n_cells, intent.values, where=intent.where, base=state._arrays[name]
        )
        for name, intent in assigns.items()
    }
    if amounts:
        arrays |= _resolve_amounts(state, amounts)
    return state._with(arrays) if arrays else state


def _as_intents(intents: Intents) -> tuple[Intent, ...]:
    """Return what a phase returned as a tuple of intents: one, an iterable of them, or None.

    Raises
    ------
    TypeError
        If it is anything else, or an iterable holds something other than an intent. A
        string, a mapping or an array is refused even though it is iterable.
    """
    if intents is None:
        return ()
    if isinstance(intents, _INTENT_TYPES):
        return (intents,)
    if isinstance(intents, str | Mapping | np.ndarray) or not isinstance(intents, Iterable):
        raise TypeError(
            f"a phase must return intents or None, got {type(intents).__name__}; "
            "wrap new values in Assign(field, values)"
        )
    intents = tuple(intents)
    for intent in intents:
        if not isinstance(intent, _INTENT_TYPES):
            raise TypeError(f"expected an intent such as Assign, got {type(intent).__name__}")
    return intents


def _per_field(intents: tuple[Intent, ...]) -> list[Intent]:
    """Return the intents with each flow or spread of a nested state made one per field.

    Raises
    ------
    TypeError
        If an intent other than a flow or a spread is given a nested state, or one as
        declared rather than as nested in a state class.
    ValueError
        If a nested state has no field the flow or spread can move.
    """
    out: list[Intent] = []
    for intent in intents:
        if not _is_nested_state(intent.field):
            out.append(intent)
            continue
        nested = _as_nested(intent.field)
        if not isinstance(intent, Flow | Spread):
            raise TypeError(
                f"{type(intent).__name__} takes a field, got the nested state "
                f"{nested._prefix!r}: only Flow and Spread move a nested state's amounts together"
            )
        moving = [
            field
            for field in nested._schema.values()
            if field.extensive and not field.static and field.dtype.kind == "f"
        ]
        if not moving:
            raise ValueError(
                f"{type(intent).__name__} on nested state {nested._prefix!r}: it has no dynamic, "
                "extensive, floating field to move"
            )
        out.extend(dataclasses.replace(intent, field=field) for field in moving)
    return out


def _field(intent: Intent) -> Field[Any]:
    """Return the field of an intent, once `_per_field` has made it one per field.

    Nested states are gone by then. Anything else that is not a declared field fails where the field
    is looked up, with the reason (a derived field, an undeclared one).
    """
    return cast("Field[Any]", intent.field)


def _check_writable(state: State, field: Field[Any], writes: Collection[str] | None) -> str:
    """Return the name of a field an intent writes, checking that the phase may write it."""
    name = state._schema._declared(field).name
    if field.static:
        raise ValueError(f"field {name!r} is static, so phases cannot write it")
    if writes is not None and name not in writes:
        raise ValueError(
            f"field {name!r} is not among the fields the phase writes: {sorted(writes)}"
        )
    return name


def _resolve_amounts(
    state: State, intents: list[Flow | Spread | Source | Claim]
) -> dict[str, NDArray[Any]]:
    """Apply the intents that move amounts, sharing each cell's amount among its takers."""
    use_kernels = fast()
    current: dict[str, NDArray[np.float64]] = {}
    demands: defaultdict[str, list[NDArray[np.float64]]] = defaultdict(list)
    deliveries: defaultdict[str, list[_Deliver]] = defaultdict(list)
    added: defaultdict[str, list[NDArray[np.float64]]] = defaultdict(list)

    def amounts_of(name: str, what: str) -> tuple[Field[Any], NDArray[np.float64]]:
        """Return a field that holds amounts, and its current amounts, read once per field."""
        spec = _amount_spec(state, name, what)
        if name not in current:
            current[name] = _current_amounts(state, name)
        return spec, current[name]

    try:  # sums of finite amounts may still overflow float64
        with np.errstate(over="raise"):
            for intent in intents:
                name = _field(intent).name
                what = f"{type(intent).__name__} on {name!r}"
                spec, x = amounts_of(name, what)
                if isinstance(intent, Source):
                    amount = _floats(intent.amount, spec.array_shape(state.n_cells), "amount", what)
                    if amount.min(initial=0.0) < 0:
                        demands[name].append(np.maximum(-amount, 0.0))
                        amount = np.maximum(amount, 0.0)
                    added[name].append(amount)
                    continue
                if isinstance(intent, Claim):
                    into, _ = amounts_of(intent.into.name, what)
                    demand, deliver = _claim(intent, spec, into, state.n_cells, what)
                elif isinstance(intent, Flow):
                    demand, deliver = _flow(intent, spec, x, state.n_cells, what, use_kernels)
                else:
                    demand, deliver = _spread(intent, spec, x, state.n_cells, what)
                demands[name].append(demand)
                deliveries[name].append(deliver)
            remaining = {}
            for name, x in current.items():
                taken = demands[name]
                if not taken:
                    remaining[name] = x
                    continue
                demand = sum(taken[1:], start=taken[0])
                granted, remaining[name] = _ration(x, demand, use_kernels)
                for deliver in deliveries[name]:
                    for target, values in deliver(granted):
                        added[target].append(values)
            for name, extra in added.items():
                for values in extra:
                    remaining[name] = remaining[name] + values
                if not remaining[name].max(initial=0.0) < np.inf:  # sparse sums overflow quietly
                    raise FloatingPointError
    except FloatingPointError:
        raise ValueError(
            f"amounts of {sorted(current)} overflow float64; the sources or rates are too large"
        ) from None
    return {name: _store(state, name, values) for name, values in remaining.items()}


def _ration(
    x: NDArray[np.float64], demand: NDArray[np.float64], use_kernels: bool
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64]]:
    """Share each amount among its takers, who together ask for ``demand``.

    Parameters
    ----------
    x : ndarray of float64, shape (*shape, n_cells)
        Amounts of a field, non-negative.
    demand : ndarray of float64, broadcastable to the shape of ``x``
        What all takers of each amount ask for, together.
    use_kernels : bool
        Whether to take the fast path.

    Returns
    -------
    granted : ndarray of float64, the shape of ``x``, or None
        The fraction of its request each taker gets: ``x / demand`` where the demand exceeds
        the amount, and 1 elsewhere. None if every request is met in full.
    remaining : ndarray of float64, the shape of ``x``
        What remains of each amount, exactly non-negative.
    """
    if use_kernels:
        demand = np.ascontiguousarray(np.broadcast_to(demand, x.shape))
        granted, remaining = np.empty(x.shape), np.empty(x.shape)  # C order: views below
        short = _kernels.ration(
            x.reshape(-1), demand.reshape(-1), granted.reshape(-1), remaining.reshape(-1)
        )
        return (granted if short else None), remaining
    short = demand > x
    if not short.any():
        return None, x - demand
    granted = np.ones_like(x)
    np.divide(x, demand, out=granted, where=short)
    return granted, np.where(short, 0.0, x - demand)


def _store(state: State, name: str, values: NDArray[np.float64]) -> NDArray[Any]:
    """Return resolved amounts as a field array, keeping a fresh float64 array without a copy."""
    spec = state._schema[name]
    fresh = values.flags.writeable and values.flags.c_contiguous and values.base is None
    if fresh and values.dtype == spec.dtype and values.shape == spec.array_shape(state.n_cells):
        return read_only(values)
    return as_field_array(spec, state.n_cells, values)


def _amount_spec(state: State, name: str, what: str) -> Field[Any]:
    """Return the field named ``name`` after checking that it can hold amounts.

    Raises
    ------
    TypeError
        If the field is not floating.
    ValueError
        If the field is intensive.
    """
    spec = state._schema[name]
    if spec.dtype.kind != "f":
        raise TypeError(
            f"{what}: field {name!r} holds {spec.dtype}, but flows, sources and claims need a "
            "floating field"
        )
    if not spec.extensive:
        raise ValueError(
            f"{what}: field {name!r} is intensive, but flows, sources and claims move amounts, "
            "so they need an extensive field"
        )
    return spec


def _current_amounts(state: State, name: str) -> NDArray[np.float64]:
    """Return a field's amounts as float64, of shape ``(*shape, n_cells)``.

    Raises
    ------
    ValueError
        If an amount is negative, NaN or infinite.
    """
    x = state._arrays[name].astype(np.float64, copy=False)
    if x.size and not x.min() >= 0:  # also catches NaN
        raise ValueError(
            f"field {name!r} holds negative or NaN amounts; flows, sources and claims need "
            "non-negative amounts"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"field {name!r} holds infinite amounts")
    return x


def _floats(values: ArrayLike, shape: tuple[int, ...], name: str, what: str) -> NDArray[np.float64]:
    """Return ``values`` as finite float64 that broadcast to ``shape``, without broadcasting."""
    array = finite_floats(values, f"{what}: {name}")
    if not _broadcasts(array.shape, shape):
        raise ValueError(
            f"{what}: {name} has shape {array.shape}, which does not broadcast to {shape}"
        )
    return array


def _operator(op: object, n_cells: int, what: str) -> NeighborOperator:
    """Return an intent's ``op`` after checking that it is an operator on ``n_cells`` cells.

    Raises
    ------
    TypeError
        If ``op`` is not a `NeighborOperator`.
    ValueError
        If it has another number of cells.
    """
    if not isinstance(op, NeighborOperator):
        raise TypeError(f"{what}: op must be a NeighborOperator, got {type(op).__name__}")
    if op.n_cells != n_cells:
        raise ValueError(
            f"{what}: the operator has {op.n_cells} cells, but the state has {n_cells}"
        )
    return op


def _domain(domain: ArrayLike | None, n_cells: int, what: str) -> NDArray[np.bool_] | None:
    """Return an intent's domain as a boolean mask of shape ``(n_cells,)``, or None."""
    return None if domain is None else cell_mask(domain, n_cells, f"{what}: domain")


def _flow(
    intent: Flow,
    spec: Field[Any],
    x: NDArray[np.float64],
    n_cells: int,
    what: str,
    use_kernels: bool,
) -> tuple[NDArray[np.float64], _Deliver]:
    """Return what a flow asks of each cell's amounts, and how to deliver what it is granted.

    Rates are clipped at 0, and a cell's rates are scaled down to add up to 1 when they add up
    to more. What is meant for cells outside the domain stays where it is.

    Parameters
    ----------
    intent : Flow
        The flow.
    spec : Field
        The field it moves.
    x : ndarray of float64, shape (*spec.shape, n_cells)
        The field's current amounts.
    n_cells : int
        Number of cells of the state.
    what : str
        What the intent is, for error messages.
    use_kernels : bool
        Whether to take the fast path.

    Returns
    -------
    demand : ndarray of float64, shape (*spec.shape, n_cells)
        The amount that leaves each cell, if granted in full.
    deliver : _Deliver
        Given the fraction granted, what the flow adds to the field.

    Raises
    ------
    TypeError
        If the operator, rates or domain have the wrong type.
    ValueError
        If they do not fit the state, a rate is not finite, or a cell's rates add up beyond
        float64.
    """
    op = _operator(intent.op, n_cells, what)
    rate = _floats(intent.rate, (*spec.shape, op.n_edges), "rate", what)
    rate = np.broadcast_to(rate, np.broadcast_shapes(rate.shape, (op.n_edges,)))
    domain = _domain(intent.domain, n_cells, what)
    edges = _flow_kernels if use_kernels else _flow_reference
    total, kept, send = edges(op, rate, domain)  # each cell's outflow fraction
    if not np.isfinite(total).all():  # sparse sums overflow quietly
        raise ValueError(f"{what}: rate is too large: a cell's rates add up beyond float64")
    sending = x  # what each cell's rates are fractions of
    scale = None
    over = total > 1
    if over.any():  # send at most everything
        scale = np.ones_like(total)
        np.divide(1.0, total, out=scale, where=over)
        total = np.minimum(total, 1.0)
        sending = x * scale
    if kept is not None:  # after scaling, so what was meant for cells outside stays home
        total = kept if scale is None else kept * scale

    def deliver(granted: NDArray[np.float64] | None) -> list[tuple[str, NDArray[np.float64]]]:
        return [(spec.name, send(sending if granted is None else granted * sending))]

    return x * total, deliver


type _Send = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Given what each cell sends rates of, what each cell receives along the flow's edges."""


def _flow_reference(
    op: NeighborOperator, rate: NDArray[np.float64], domain: NDArray[np.bool_] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, _Send]:
    """Add up each cell's rates, clipped at 0: all of them, and those toward the domain.

    ``rate`` has shape ``(..., n_edges)``. Returns both sums, of shape ``(..., n_cells)`` (the
    second None without a domain), and the function that sends amounts along the edges.
    """
    rate = np.maximum(rate, 0.0)
    total = _sum_by_source(rate, op)
    kept = None
    if domain is not None:
        rate = rate * domain[op.graph.indices]
        kept = _sum_by_source(rate, op)

    def send(source: NDArray[np.float64]) -> NDArray[np.float64]:
        return _sum_by_target(np.take(source, op.graph.sources, axis=-1) * rate, op)

    return total, kept, send


def _flow_kernels(
    op: NeighborOperator, rate: NDArray[np.float64], domain: NDArray[np.bool_] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, _Send]:
    """Fast path of `_flow_reference`: each pass over the edges is one kernel.

    Takes and returns the same shapes. The function returned sends amounts of any shape
    ``(..., n_cells)`` to which the rates' leading axes broadcast.
    """
    indptr, indices, _ = op._csr
    shape = (*rate.shape[:-1], op.n_cells)
    rates = as_rows(rate)
    inside = None if domain is None else domain.astype(np.float64)  # 1.0 or 0.0 per cell
    total = np.empty((rates.shape[0], op.n_cells))
    kept = None if inside is None else np.empty_like(total)
    _kernels.flow_rate_sums(indptr, indices, rates, inside, total, kept)

    def send(source: NDArray[np.float64]) -> NDArray[np.float64]:
        sources = as_rows(source)
        each = rates
        if each.shape[0] not in (1, sources.shape[0]):  # spread the rates over the components
            each = as_rows(np.broadcast_to(rate, (*source.shape[:-1], op.n_edges)))
        received = np.empty_like(sources)
        _kernels.flow_send(indptr, indices, each, inside, sources, received)
        return received.reshape(source.shape)

    return total.reshape(shape), None if kept is None else kept.reshape(shape), send


def _spread(
    intent: Spread, spec: Field[Any], x: NDArray[np.float64], n_cells: int, what: str
) -> tuple[NDArray[np.float64], _Deliver]:
    """Return what a spread asks of each cell's amounts, and how to deliver what it is granted.

    A cell's leaving amount is split among its neighbors in proportion to ``w_ij * toward_j``,
    with ``toward`` 0 outside the domain. A cell with nowhere to go keeps everything.

    Parameters
    ----------
    intent : Spread
        The spread.
    spec : Field
        The field it moves.
    x : ndarray of float64, shape (*spec.shape, n_cells)
        The field's current amounts.
    n_cells : int
        Number of cells of the state.
    what : str
        What the intent is, for error messages.

    Returns
    -------
    demand : ndarray of float64, shape (*spec.shape, n_cells)
        The amount that leaves each cell, if granted in full.
    deliver : _Deliver
        Given the fraction granted, what the spread adds to the field.

    Raises
    ------
    TypeError
        If the operator, ``leave``, ``toward`` or the domain have the wrong type.
    ValueError
        If they do not fit the state, a value is not finite, the operator has negative
        weights, or the weighted attractions add up beyond float64.
    """
    op = _operator(intent.op, n_cells, what)
    if not op._nonnegative:
        raise ValueError(f"{what}: the operator has negative weights, so it cannot split amounts")
    shape = spec.array_shape(n_cells)
    leave = np.clip(_floats(intent.leave, shape, "leave", what), 0.0, 1.0)
    domain = _domain(intent.domain, n_cells, what)
    toward = None  # 1 everywhere
    if intent.toward is not None:
        toward = np.maximum(_floats(intent.toward, shape, "toward", what), 0.0)
        toward = _relative(np.broadcast_to(toward, np.broadcast_shapes(toward.shape, (n_cells,))))
    if domain is not None:
        toward = domain.astype(np.float64) if toward is None else toward * domain
    pulled = op.weight_sums if toward is None else pull(toward, op)  # sum_j w_ij * toward_j
    if not np.isfinite(pulled).all():  # SciPy sums overflow quietly
        raise ValueError(
            f"{what}: the operator's weights are too large: w_ij * toward_j adds up beyond float64"
        )
    if not (pulled > 0).all():  # nowhere to go: nothing leaves
        leave = leave * (pulled > 0)
        pulled = np.where(pulled > 0, pulled, 1.0)
    demand = x * leave

    def deliver(granted: NDArray[np.float64] | None) -> list[tuple[str, NDArray[np.float64]]]:
        moved = demand if granted is None else granted * demand
        received = push(moved / pulled, op)  # takes the fast path of push when there is one
        return [(spec.name, received if toward is None else toward * received)]

    return demand, deliver


_FAINTEST = 2.0**-500
"""Attractions below this fraction of the largest count as 0.

Spread divides by ``sum_j w_ij * toward_j`` before its push, so the sum must stay far from
float64's smallest values. Ratios beyond 1e150 mean nothing in a model anyway.
"""


def _relative(toward: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return attractions as fractions of the largest in each component, with faint ones 0.

    ``toward`` is non-negative float64 of shape ``(..., n_cells)``. Returns float64 of the same
    shape, in [0, 1], with 1 at each component's largest unless all of it is 0.
    """
    largest = toward.max(axis=-1, keepdims=True, initial=0.0)
    scaled = toward / np.where(largest > 0, largest, 1.0)
    return scaled * (scaled >= _FAINTEST)


def _claim(
    intent: Claim, spec: Field[Any], into: Field[Any], n_cells: int, what: str
) -> tuple[NDArray[np.float64], _Deliver]:
    """Return what a claim asks of each cell's amounts, and how to deliver what it is granted.

    Parameters
    ----------
    intent : Claim
        The claim.
    spec : Field
        The field claimed from.
    into : Field
        The field the claims go into, of shape ``(*claimants, *spec.shape)``.
    n_cells : int
        Number of cells of the state.
    what : str
        What the intent is, for error messages.

    Returns
    -------
    demand : ndarray of float64, shape (*spec.shape, n_cells)
        What the claimants of each cell ask for together, negative claims counted as 0.
    deliver : _Deliver
        Given the fraction granted, what each claimant adds to ``into``.

    Raises
    ------
    TypeError
        If the amounts claimed are not numbers.
    ValueError
        If ``into`` is the claimed field, its shape does not end with the claimed field's, or
        the amounts are not finite or do not fit its shape.
    """
    n_claimant_axes = len(into.shape) - len(spec.shape)
    if (
        intent.into.name == intent.field.name
        or n_claimant_axes < 0
        or into.shape[n_claimant_axes:] != spec.shape
    ):
        raise ValueError(
            f"{what}: into must be another field, of shape (*claimants, *{spec.shape}); "
            f"{intent.into.name!r} has shape {into.shape}"
        )
    shape = into.array_shape(n_cells)
    claims = np.broadcast_to(np.maximum(_floats(intent.amount, shape, "amount", what), 0.0), shape)
    efficiency = np.maximum(_floats(intent.efficiency, shape, "efficiency", what), 0.0)
    demand = claims.sum(axis=tuple(range(n_claimant_axes)))

    def deliver(granted: NDArray[np.float64] | None) -> list[tuple[str, NDArray[np.float64]]]:
        received = claims if granted is None else claims * granted
        return [(intent.into.name, received * efficiency)]

    return demand, deliver
