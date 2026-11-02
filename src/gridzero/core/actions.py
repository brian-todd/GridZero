"""Action spaces: the choices a rule makes for each cell, laid out the same way for every cell.

A cell's movement choices come from its neighborhood. On the planet a hexagon can keep what it
holds or send it to any of 6 neighbors, and a pentagon to any of 5. A neighbor action space
gives every cell the same row of slots, so one rule or policy works on every cell:

- slot 0 means "stay",
- slots 1 to ``op.size`` are the cell's neighbors, in ascending id order, as in
  `NeighborOperator.query`,
- slots past a cell's degree are invalid.

A domain, the cells that may be moved into, is given to each call that needs one, because it
often changes from step to step (no fox moves into a fire). Slots whose neighbor lies outside
it are invalid in that call. `mask(domain)` gives the valid slots, the mask a policy samples
under, and `values(x)` the value of ``x`` at each slot's cell.

`NeighborAllocation` splits each cell's amount over its slots, and `NeighborChoice` picks one
slot per cell. Both turn actions into a `Flow`, whose resolver enforces conservation, bounds and
the domain. `Box` is a bounded real value per cell, such as a rate a rule controls.

Actions on invalid slots are ignored rather than rejected, so a policy with a fixed number of
outputs works on cells of every degree. `contains` tells whether an action is strictly valid.

Conversions have Numba fast paths with bit-identical results (see
`gridzero.core.implementation`). Random draws always come from NumPy, so a
seed gives the same actions on every path.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol, SupportsIndex, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gridzero.core import _kernels
from gridzero.core._arrays import (
    as_rows,
    cell_mask,
    check_int,
    component_shape,
    finite_floats,
    frozen_copy,
    int_array,
    read_only,
)
from gridzero.core.fields import Field, NestedState
from gridzero.core.implementations import fast
from gridzero.core.intents import Flow
from gridzero.core.operators import NeighborOperator, _cells_last, _fill_value


@runtime_checkable
class ActionSpace(Protocol):
    """What every action space provides: the shape of an action, sampling and a validity check."""

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Shape of one action array."""
        ...

    def sample(self, rng: np.random.Generator) -> NDArray:
        """Return a random valid action.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        ndarray, shape action_shape
            A valid action. Its dtype depends on the space: float64 for allocations and
            boxes, int64 for choices.
        """
        ...

    def contains(self, action: ArrayLike) -> bool:
        """Return whether ``action`` is a valid action.

        Parameters
        ----------
        action : array_like, shape action_shape
            A proposed action, of the dtype `sample` returns or one that converts to it
            exactly.

        Returns
        -------
        bool
            True if ``action`` has shape `action_shape` and is valid in every cell.
        """
        ...


def _cells_and_slots(values: NDArray, n_cells: int, n_slots: int, name: str) -> None:
    """Check that ``values`` has shape ``(..., n_cells, n_slots)``, one row of slots per cell.

    Raises
    ------
    ValueError
        If it does not.
    """
    if values.ndim < 2 or values.shape[-2:] != (n_cells, n_slots):
        raise ValueError(
            f"{name} must have shape (..., {n_cells}, {n_slots}): one row of slots per cell, "
            f"got {values.shape}"
        )


@dataclass(frozen=True, eq=False, init=False)
class _NeighborSlots:
    """Slots shared by the neighbor action spaces: stay, then each neighbor by id.

    Its parameters and attributes are those of `NeighborAllocation` and `NeighborChoice`.
    """

    op: NeighborOperator
    shape: tuple[int, ...]
    targets: NDArray[np.int64] = field(repr=False)
    valid: NDArray[np.bool_] = field(repr=False)

    def __init__(
        self,
        op: NeighborOperator,
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
    ) -> None:
        if not isinstance(op, NeighborOperator):
            raise TypeError(f"op must be a NeighborOperator, got {type(op).__name__}")
        targets, valid, _ = op._stay_slots  # shared by every action space on the operator
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "shape", component_shape(shape))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "valid", valid)

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.op.n_cells

    @property
    def n_slots(self) -> int:
        """Slots per cell: stay, plus the largest number of neighbors of any cell."""
        return self.targets.shape[1]

    def mask(self, domain: ArrayLike | None = None) -> NDArray[np.bool_]:
        """Return which slots are valid: the action mask, under which policies choose.

        Parameters
        ----------
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be moved into. Defaults to every cell.

        Returns
        -------
        ndarray of bool, shape (n_cells, n_slots)
            True for staying, which is always valid, and for each neighbor inside the domain;
            False past a cell's degree. Without a domain, `valid` itself (read-only).

        Raises
        ------
        TypeError
            If ``domain`` is not boolean.
        ValueError
            If ``domain`` does not have one entry per cell.
        """
        if domain is None:
            return self.valid
        inside = cell_mask(domain, self.n_cells, "domain")
        if fast():
            out = np.empty(self.valid.shape, dtype=np.bool_)
            _kernels.slot_mask(self.op._csr[0], self.op._csr[1], inside, out)
            return out
        out = self.valid.copy()
        out[:, 1:] &= np.take(inside, self.targets[:, 1:])  # -1 takes the last; already False
        return out

    def values(
        self, x: ArrayLike, fill: float = 0, domain: ArrayLike | None = None
    ) -> NDArray[Any]:
        """Return the value of ``x`` at each slot's cell, and ``fill`` where the slot is invalid.

        Parameters
        ----------
        x : array_like, shape (..., n_cells)
            Per-cell values, such as food. Leading axes are carried through.
        fill : float, default 0
            Value for invalid slots. It must fit the dtype of ``x``, as in `gather`.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be moved into: slots of neighbors outside it get ``fill`` too.

        Returns
        -------
        ndarray, same dtype as ``x``, shape (..., n_cells, n_slots)
            ``x`` at the cell itself in slot 0, then at each neighbor: scores for `best`, or
            weights for an allocation.

        Raises
        ------
        ValueError
            If the last axis of ``x`` is not ``n_cells`` long, or ``fill`` does not fit its
            dtype.
        TypeError, ValueError
            As for `mask`, for ``domain``.
        """
        x = _cells_last(x, self.n_cells)
        value = _fill_value(fill, x.dtype)
        out = np.take(x, self.targets, axis=-1)
        np.copyto(out, value, where=~self.mask(domain))  # the reference's take, then copyto
        return out

    @cached_property
    def _open(self) -> NDArray[np.float64]:
        """``valid`` as 1.0 and 0.0, to mask by multiplying: float64, (n_cells, n_slots)."""
        return read_only(self.valid.astype(np.float64))

    def _open_in(self, domain: ArrayLike | None) -> NDArray[np.float64]:
        """`mask` as 1.0 and 0.0 for a domain: float64, (n_cells, n_slots)."""
        return self._open if domain is None else self.mask(domain).astype(np.float64)

    @property
    def _edge_slots(self) -> NDArray[np.int64]:
        """Flat position of each edge's slot in a ``(n_cells, n_slots)`` table, in CSR order."""
        return self.op._stay_slots[2]


@dataclass(frozen=True, eq=False, init=False)
class NeighborAllocation(_NeighborSlots):
    """Split each cell's amount over staying and moving to each of its neighbors.

    An action holds a non-negative weight per slot. Each cell's weights over its valid slots
    are normalized into the fractions of its amount that stay and that go to each neighbor.

    Parameters
    ----------
    op : NeighborOperator
        The neighbors each cell may send to.
    shape : int or tuple of int, default ()
        Component shape of an action, such as ``(3,)`` for one allocation per group.

    Attributes
    ----------
    targets : ndarray of int64, shape (n_cells, n_slots)
        The cell each slot points to: the cell itself in slot 0, then its neighbors, then -1.
    valid : ndarray of bool, shape (n_cells, n_slots)
        Whether each slot exists: staying, and each neighbor. `mask` restricts it to a domain.

    Raises
    ------
    TypeError
        If ``op`` is not a `NeighborOperator`, or a shape entry is not an integer.
    ValueError
        If a shape entry is negative.
    """

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Shape of an action: ``(*shape, n_cells, n_slots)``."""
        return (*self.shape, self.n_cells, self.n_slots)

    def fractions(self, weights: ArrayLike, domain: ArrayLike | None = None) -> NDArray[np.float64]:
        """Normalize weights into the fraction of each cell's amount for each slot.

        Parameters
        ----------
        weights : array_like of float, shape (..., n_cells, n_slots)
            Weight of each slot. Negative weights and weights in invalid slots count as 0. A
            cell whose weights are all 0 keeps everything.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be moved into. Slots of neighbors outside it are invalid, so their
            weights count as 0 and the rest share the cell's amount. Defaults to every cell.

        Returns
        -------
        ndarray of float64, shape (..., n_cells, n_slots)
            Non-negative fractions, 0 in invalid slots. Each cell's fractions add up to 1.

        Raises
        ------
        TypeError
            If ``weights`` does not hold numbers, or ``domain`` is not boolean.
        ValueError
            If ``weights`` has the wrong shape or holds NaN or infinite values, or ``domain``
            does not have one entry per cell.
        """
        weights = self._weights(weights)
        open_slots = self._open_in(domain)
        if fast():
            rows = as_rows(weights, keep=2)
            out = np.empty_like(rows)
            _kernels.fractions(rows, open_slots, out)
            return out.reshape(weights.shape)
        weights = np.maximum(weights, 0.0) * open_slots  # 5x faster than np.where(valid, ...)
        total = self._slot_sum(weights)
        huge = np.isinf(total)
        if huge.any():  # weights near the float64 maximum: rescale by each row's largest
            largest = np.where(huge, weights.max(axis=-1), 1.0)
            weights = weights / largest[..., None]
            total = self._slot_sum(weights)
        empty = total == 0
        out = weights / np.where(empty, 1.0, total)[..., None]  # 1 / total overflows if tiny
        out[..., 0] += empty  # nothing proposed: stay
        return out

    def _weights(self, weights: ArrayLike) -> NDArray[np.float64]:
        """Return ``weights`` as float64 of shape ``(..., n_cells, n_slots)``, checked finite."""
        weights = finite_floats(weights, "weights")
        _cells_and_slots(weights, self.n_cells, self.n_slots, "weights")
        return weights

    def _slot_sum(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sum ``(..., n_cells, n_slots)`` weights over slots, infinite where it overflows.

        Slots are added one at a time, in order. Returns float64 of shape ``(..., n_cells)``.
        """
        total = weights[..., 0].copy()
        with np.errstate(over="ignore"):
            for slot in range(1, self.n_slots):  # 2x faster than a sum over the short last axis
                total += weights[..., slot]
        return total

    def flow(
        self,
        field: Field[Any] | type[NestedState],
        weights: ArrayLike,
        domain: ArrayLike | None = None,
    ) -> Flow:
        """Return the flow that moves amounts of a field as the weights allocate them.

        Parameters
        ----------
        field : Field or nested state
            A dynamic, extensive, floating field, or a `NestedState` as nested in a state class.
        weights : array_like of float, shape (..., n_cells, n_slots)
            As for `fractions`. Leading axes broadcast against the field's component shape.
        domain : array_like of bool, shape (n_cells,), optional
            As for `fractions`. The flow is restricted to it too.

        Returns
        -------
        Flow
            A flow along ``op`` with rates of shape (..., n_edges), restricted to the domain.
        """
        if fast():  # the fractions of each edge, without the table of every slot
            weights = self._weights(weights)
            rows = as_rows(weights, keep=2)
            rates = np.empty((rows.shape[0], self.op.n_edges))
            _kernels.allocation_rates(self.op._csr[0], rows, self._open_in(domain), rates)
            rates = rates.reshape(*weights.shape[:-2], self.op.n_edges)
            return Flow(field, self.op, rates, domain)
        moving = self.fractions(weights, domain)
        moving = moving.reshape(*moving.shape[:-2], self.n_cells * self.n_slots)
        return Flow(field, self.op, np.take(moving, self._edge_slots, axis=-1), domain)

    def sample(
        self, rng: np.random.Generator, domain: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Return random fractions, uniform over each cell's valid allocations.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        domain : array_like of bool, shape (n_cells,), optional
            As for `fractions`.

        Returns
        -------
        ndarray of float64, shape action_shape
            Fractions drawn from a flat Dirichlet distribution over each cell's valid slots.
        """
        return self.fractions(rng.exponential(size=self.action_shape), domain)

    def contains(self, action: ArrayLike) -> bool:
        """Return whether ``action`` is a valid action.

        Parameters
        ----------
        action : array_like of float or int, shape action_shape
            Proposed weights.

        Returns
        -------
        bool
            True if ``action`` has shape `action_shape` and holds finite, non-negative
            numbers. Weights in invalid slots are allowed, since they are ignored.
        """
        action = np.asarray(action)
        return (
            action.shape == self.action_shape
            and action.dtype.kind in "iuf"
            and bool(np.all(np.isfinite(action) & (action >= 0)))
        )


@dataclass(frozen=True, eq=False, init=False)
class NeighborChoice(_NeighborSlots):
    """Pick one slot per cell: stay, or one of its neighbors.

    An action holds a slot index per cell. Choosing an invalid slot means staying.

    Parameters
    ----------
    op : NeighborOperator
        The neighbors each cell may choose.
    shape : int or tuple of int, default ()
        Component shape of an action, such as ``(3,)`` for one choice per group.

    Attributes
    ----------
    targets : ndarray of int64, shape (n_cells, n_slots)
        The cell each slot points to: the cell itself in slot 0, then its neighbors, then -1.
    valid : ndarray of bool, shape (n_cells, n_slots)
        Whether each slot exists: staying, and each neighbor. `mask` restricts it to a domain.

    Raises
    ------
    TypeError
        If ``op`` is not a `NeighborOperator`, or a shape entry is not an integer.
    ValueError
        If a shape entry is negative.
    """

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Shape of an action: ``(*shape, n_cells)``."""
        return (*self.shape, self.n_cells)

    @cached_property
    def _chosen(self) -> NDArray[np.int64]:
        """``targets`` with every invalid slot pointing back at its own cell: (n_cells, n_slots)."""
        own = np.arange(self.n_cells)[:, None]
        return read_only(np.where(self.valid, self.targets, own))

    def _is_valid(self, slots: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Whether each cell's slot is valid: 3x faster than ``valid[cells, slots]``.

        ``slots`` is int64 of shape ``(..., n_cells)``, each in range. Returns bool of the same
        shape.
        """
        return np.take(self.valid, np.arange(self.n_cells) * self.n_slots + slots)

    def _slots(self, choice: ArrayLike) -> NDArray[np.int64]:
        """Return ``choice`` as int64 slots of shape ``(..., n_cells)``, each in range.

        Raises
        ------
        TypeError
            If ``choice`` does not hold integers.
        ValueError
            If its last axis is not ``n_cells`` long.
        IndexError
            If a slot is outside ``[0, n_slots)``.
        """
        slots = int_array(choice, "choice")
        if slots.ndim < 1 or slots.shape[-1] != self.n_cells:
            raise ValueError(
                f"choice must have shape (..., {self.n_cells}): one slot per cell, got "
                f"{slots.shape}"
            )
        if np.any((slots < 0) | (slots >= self.n_slots)):
            raise IndexError(f"choice slots must lie in [0, {self.n_slots})")
        return slots

    def cells(self, choice: ArrayLike, domain: ArrayLike | None = None) -> NDArray[np.int64]:
        """Return the cell each choice points to.

        Parameters
        ----------
        choice : array_like of int, shape (..., n_cells)
            Slot chosen by each cell.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be chosen. Defaults to every cell.

        Returns
        -------
        ndarray of int64, shape (..., n_cells)
            The chosen neighbor, or the cell itself if it chose to stay, chose an invalid
            slot, or chose a neighbor outside the domain.

        Raises
        ------
        TypeError
            If ``choice`` does not hold integers, or ``domain`` is not boolean.
        ValueError
            If ``choice`` or ``domain`` does not have one entry per cell.
        IndexError
            If a slot is outside ``[0, n_slots)``.
        """
        slots = self._slots(choice)
        chosen = np.take(self._chosen, np.arange(self.n_cells) * self.n_slots + slots)
        if domain is None:
            return chosen
        inside = cell_mask(domain, self.n_cells, "domain")
        return np.where(np.take(inside, chosen), chosen, np.arange(self.n_cells))

    def flow(
        self,
        field: Field[Any] | type[NestedState],
        choice: ArrayLike,
        rate: ArrayLike = 1.0,
        domain: ArrayLike | None = None,
    ) -> Flow:
        """Return the flow that sends part of each cell's amount to the neighbor it chose.

        Parameters
        ----------
        field : Field or nested state
            A dynamic, extensive, floating field, or a `NestedState` as nested in a state class.
        choice : array_like of int, shape (..., n_cells)
            Slot chosen by each cell. Leading axes broadcast against the field's component
            shape. Cells that stay, or chose an invalid slot, send nothing.
        rate : array_like of float, shape (..., n_cells), default 1.0
            Fraction of its amount each cell sends to its choice, broadcast against
            ``choice``.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be moved into. The flow is restricted to it, so a cell that chose a
            neighbor outside it sends nothing, as if it stayed. Defaults to every cell.

        Returns
        -------
        Flow
            A flow along ``op`` with rates of shape (..., n_edges), restricted to the domain.

        Raises
        ------
        TypeError, ValueError, IndexError
            As for `cells`, and if ``rate`` is not numbers that broadcast against ``choice``.
        """
        slots = self._slots(choice)
        rate = finite_floats(rate, "rate")
        lead = np.broadcast_shapes(slots.shape, rate.shape)[:-1]
        slots = np.broadcast_to(slots, (*lead, self.n_cells))
        rate = np.broadcast_to(rate, (*lead, self.n_cells))
        if fast():
            rows = as_rows(slots)
            rates = np.empty((rows.shape[0], self.op.n_edges))
            _kernels.choice_rates(self.op._csr[0], self.valid, rows, as_rows(rate), rates)
            return Flow(field, self.op, rates.reshape(*lead, self.op.n_edges), domain)
        moves = (slots > 0) & self._is_valid(slots)
        n_edges = self.op.n_edges
        edge = self.op.graph.indptr[:-1] + slots - 1  # the chosen slot's edge, where it moves
        flat = np.arange(int(np.prod(lead)), dtype=np.int64).reshape(*lead, 1) * n_edges + edge
        rates = np.zeros((*lead, n_edges))
        rates.reshape(-1)[flat[moves]] = rate[moves]
        return Flow(field, self.op, rates, domain)

    def best(self, scores: ArrayLike, domain: ArrayLike | None = None) -> NDArray[np.int64]:
        """Return each cell's valid slot with the highest score.

        Parameters
        ----------
        scores : array_like of float or int, shape (..., n_cells, n_slots)
            Score of each slot, such as `values` gives. Scores of invalid slots are ignored.
            Integer scores are compared exactly, without converting them to floats.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be chosen. Slots of neighbors outside it are ignored too. Defaults
            to every cell.

        Returns
        -------
        ndarray of int64, shape (..., n_cells)
            The chosen slots. Ties go to the lowest slot, so staying wins a tie.

        Raises
        ------
        TypeError
            If ``scores`` does not hold numbers, or ``domain`` is not boolean.
        ValueError
            If ``scores`` has the wrong shape or holds NaN, or ``domain`` does not have one
            entry per cell.
        """
        scores = np.asarray(scores)
        if scores.dtype.kind not in "iuf":
            raise TypeError(f"scores must hold numbers, got dtype {scores.dtype}")
        _cells_and_slots(scores, self.n_cells, self.n_slots, "scores")
        if np.isnan(scores).any():
            raise ValueError("scores hold NaN")
        valid = self.mask(domain)
        if fast():
            return self._best(scores, valid)
        # Slot 0 is always valid, so a tie at the lowest score still picks a valid slot.
        lowest = -np.inf if scores.dtype.kind == "f" else np.iinfo(scores.dtype).min
        return np.where(valid, scores, lowest).argmax(axis=-1)

    def _best(self, scores: NDArray, valid: NDArray[np.bool_]) -> NDArray[np.int64]:
        """Fast path of `best`, comparing in float64, int64 or uint64, which keep every order.

        ``scores`` are checked numbers of shape ``(..., n_cells, n_slots)``, without NaN, and
        ``valid`` the mask of shape ``(n_cells, n_slots)``. Returns int64 of shape
        ``(..., n_cells)``.
        """
        if scores.dtype.kind == "f":
            work = np.float64
        else:
            work = np.uint64 if scores.dtype == np.uint64 else np.int64
        rows = as_rows(scores, work, keep=2)
        out = np.empty(rows.shape[:2], dtype=np.int64)
        _kernels.best_slots(rows, valid, out)
        return out.reshape(scores.shape[:-1])

    def sample(
        self, rng: np.random.Generator, domain: ArrayLike | None = None
    ) -> NDArray[np.int64]:
        """Return a random valid slot per cell, uniform over each cell's valid slots.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be chosen. Defaults to every cell.

        Returns
        -------
        ndarray of int64, shape action_shape
            The chosen slots.
        """
        draws = rng.random((*self.action_shape, self.n_slots))
        valid = self.mask(domain)
        if fast():
            return self._best(draws, valid)
        return np.where(valid, draws, -1.0).argmax(axis=-1)

    def contains(self, action: ArrayLike, domain: ArrayLike | None = None) -> bool:
        """Return whether ``action`` is a valid action.

        Parameters
        ----------
        action : array_like of int, shape action_shape
            Proposed slots.
        domain : array_like of bool, shape (n_cells,), optional
            Cells that may be chosen. Defaults to every cell.

        Returns
        -------
        bool
            True if ``action`` has shape `action_shape`, holds integers, and picks a valid
            slot in every cell: staying, or a neighbor inside the domain.
        """
        action = np.asarray(action)
        if action.shape != self.action_shape or action.dtype.kind not in "iu":
            return False
        if np.any((action < 0) | (action >= self.n_slots)):
            return False
        if not np.all(self._is_valid(action)):
            return False
        if domain is None:
            return True
        return bool(np.all(self.cells(action, domain) == self.cells(action)))


@dataclass(frozen=True, eq=False, init=False)
class Box:
    """A bounded real value per cell and component, such as a rate a rule controls.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    low, high : array_like of float
        Bounds, broadcast to ``(*shape, n_cells)``: per cell, per component or both. They must
        be finite, with ``low <= high``.
    shape : int or tuple of int, default ()
        Component shape of an action.

    Raises
    ------
    TypeError
        If ``n_cells`` or a shape entry is not an integer, or a bound does not hold numbers.
    ValueError
        If ``n_cells`` or a shape entry is negative, or a bound does not broadcast to the
        action shape, is not finite, or has ``low > high``.
    """

    n_cells: int
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    shape: tuple[int, ...]

    def __init__(
        self,
        n_cells: SupportsIndex,
        low: ArrayLike,
        high: ArrayLike,
        shape: int | np.integer[Any] | Sequence[SupportsIndex] = (),
    ) -> None:
        n_cells = check_int(n_cells, "n_cells", minimum=0)
        shape = component_shape(shape)
        action_shape = (*shape, n_cells)
        bounds = []
        for name, given in (("low", low), ("high", high)):
            bound = finite_floats(given, name)
            try:
                bound = np.broadcast_to(bound, action_shape)
            except ValueError:
                raise ValueError(
                    f"{name} has shape {bound.shape}, which does not broadcast to {action_shape}"
                ) from None
            bounds.append(frozen_copy(bound, np.float64))
        low_, high_ = bounds
        if np.any(low_ > high_):
            raise ValueError("low must not exceed high")
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "low", low_)
        object.__setattr__(self, "high", high_)

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Shape of an action: ``(*shape, n_cells)``."""
        return (*self.shape, self.n_cells)

    def clip(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return values clipped into the bounds.

        Parameters
        ----------
        values : array_like of float, shape action_shape
            Proposed values, such as a policy's raw outputs.

        Returns
        -------
        ndarray of float64, shape action_shape
            The values, moved into ``[low, high]``.

        Raises
        ------
        TypeError
            If ``values`` does not hold numbers.
        ValueError
            If ``values`` has the wrong shape or holds NaN.
        """
        values = np.asarray(values)
        if values.dtype.kind not in "iuf":
            raise TypeError(f"values must hold numbers, got dtype {values.dtype}")
        if values.shape != self.action_shape:
            raise ValueError(f"values must have shape {self.action_shape}, got {values.shape}")
        if np.isnan(values).any():
            raise ValueError("values hold NaN")
        return np.clip(values.astype(np.float64, copy=False), self.low, self.high)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return values drawn uniformly between the bounds.

        Bounds as wide as float64 allows work: ``high - low`` may overflow, so the draw
        interpolates between them instead of adding a multiple of their difference to ``low``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        ndarray of float64, shape action_shape
            Values in ``[low, high]``.
        """
        u = rng.random(self.action_shape)
        with np.errstate(over="ignore"):  # rounding may pass float64's maximum; clip fixes it
            values = self.low * (1.0 - u) + self.high * u
        return np.clip(values, self.low, self.high)

    def contains(self, action: ArrayLike) -> bool:
        """Return whether ``action`` is a valid action.

        Parameters
        ----------
        action : array_like of float or int, shape action_shape
            Proposed values.

        Returns
        -------
        bool
            True if ``action`` has shape `action_shape`, holds numbers, and lies within the
            bounds.
        """
        action = np.asarray(action)
        return (
            action.shape == self.action_shape
            and action.dtype.kind in "iuf"
            and bool(np.all((action >= self.low) & (action <= self.high)))
        )
