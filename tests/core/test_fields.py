import copy
import math
import pickle
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from demo import (
    NESTED_CALLS,
    Biome,
    Continent,
    DemoState,
    EcosystemState,
    PlanetState,
    Species,
    WorldState,
    fields,
)
from gridzero.core import (
    Assign,
    Context,
    Derived,
    Field,
    LatticeSpace,
    MeshSpace,
    Nested,
    NestedState,
    Simulation,
    State,
    derived,
    phase,
    resolve,
)

DTYPES: list[type[np.generic]] = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
]
DTYPES += [np.uint64, np.float16, np.float32, np.float64]


class LifeState(State):
    alive = Field(bool)
    pop = Field(np.float64, (2,), extensive=True)


class TerrainState(State):
    terrain = Field(np.uint8, static=True)


class BoardState(LifeState, TerrainState):
    """The fields of `DemoState`, in its order, declared by two classes."""


class FoodState(State):
    food = Field(np.float64, static=True)


class PieceState(State):
    kind = Field(np.int8, default=-1)
    hp = Field(np.int16)
    ground = Field(np.uint8, static=True, default=3)


# Field.


def test_a_field_normalizes_its_dtype_and_shape():
    field = Field("float32", 3)
    assert isinstance(field.dtype, np.dtype)
    assert field.dtype == np.float32
    assert field.shape == (3,)
    assert field.array_shape(10) == (3, 10)
    assert Field(float, (2, np.int64(4))).shape == (2, 4)


def test_a_field_takes_the_name_of_its_attribute():
    assert (DemoState.alive.name, DemoState.pop.name, DemoState.terrain.name) == (
        "alive",
        "pop",
        "terrain",
    )
    assert Field(float).name == ""


@pytest.mark.parametrize("name", ["", "2pop", "pop size"])
def test_a_field_name_must_be_an_identifier(name):
    with pytest.raises(ValueError, match="identifier"):
        type("Bad", (State,), {name: Field(float)})


@pytest.mark.parametrize("dtype", [object, str, np.complex128, "datetime64[s]"])
def test_a_field_rejects_a_dtype_that_is_not_numeric(dtype):
    with pytest.raises(TypeError, match="boolean, integer or floating"):
        Field(dtype)


def test_a_field_needs_a_dtype():
    # np.dtype(None) is float64, so a dtype left unset would quietly make a float field.
    with pytest.raises(TypeError, match="dtype is required"):
        Field(None)  # ty: ignore[no-matching-overload]


@pytest.mark.parametrize("shape", [-1, (2, -1)])
def test_a_field_rejects_a_negative_shape(shape):
    with pytest.raises(ValueError, match="shape entry must be non-negative"):
        Field(float, shape)


@pytest.mark.parametrize("shape", [(2.0,), (True,), ((2,),), 2.0, None], ids=repr)
def test_a_field_rejects_a_shape_that_is_not_integers(shape):
    with pytest.raises(TypeError, match="shape"):
        Field(float, shape)


def test_a_flag_cannot_be_extensive():
    with pytest.raises(ValueError, match="amounts"):
        Field(bool, extensive=True)


def test_fields_are_values(make_lattice):
    AgainState = fields(pop=Field(np.float64, (2,), extensive=True))
    assert AgainState.pop == DemoState.pop
    assert {DemoState.pop: 1}[AgainState.pop] == 1
    assert AgainState.pop != fields(pop=Field(np.float64, (2,))).pop
    assert AgainState.pop != fields(pop=Field(np.float64, (2,), extensive=True, default=1.0)).pop
    assert AgainState.pop != fields(food=Field(np.float64, (2,), extensive=True)).food
    assert pickle.loads(pickle.dumps(DemoState.pop)) == DemoState.pop
    assert copy.copy(DemoState.pop) is DemoState.pop
    assert copy.deepcopy(DemoState.pop) is DemoState.pop
    # A state reads a field declared the same way by another class.
    state = AgainState(make_lattice(), {AgainState.pop: 1.0})
    assert state[DemoState.pop] is state[AgainState.pop]


def test_fields_are_immutable():
    with pytest.raises(AttributeError, match="immutable"):
        DemoState.pop.shape = (3,)


def test_fields_print_their_declaration():
    assert repr(DemoState.pop) == "Field('pop', float64, shape=(2,), extensive)"
    assert repr(Field(np.int8, static=True, default=-1)) == (
        "Field(<undeclared>, int8, static, default=-1)"
    )


# Field: defaults.


class OwnedState(State):
    owner = Field(np.int8, default=-1)
    height = Field(np.float32, static=True, default=0.5)
    seen = Field(bool)


def test_fields_left_out_start_at_their_default(make_lattice):
    state = OwnedState(make_lattice())
    np.testing.assert_array_equal(state[OwnedState.owner], np.full(25, -1, np.int8))
    np.testing.assert_array_equal(state[OwnedState.height], np.full(25, 0.5, np.float32))
    assert not state[OwnedState.seen].any()
    assert state[OwnedState.owner].dtype == np.int8
    assert not state[OwnedState.owner].flags.writeable
    given = OwnedState(make_lattice(), {OwnedState.owner: 2})
    np.testing.assert_array_equal(given[OwnedState.owner], np.full(25, 2))


def test_a_default_covers_every_component(make_lattice):
    Groups = fields(pop=Field(np.float64, (2, 3), extensive=True, default=1.5))
    np.testing.assert_array_equal(Groups(make_lattice())[Groups.pop], np.full((2, 3, 25), 1.5))


def test_a_default_is_stored_in_the_fields_dtype():
    default = Field(np.uint8, default=7).default
    assert default == 7
    assert type(default) is np.uint8
    assert Field(float).default is None


@pytest.mark.parametrize(
    ("dtype", "default", "error", "message"),
    [
        (np.uint8, 300, ValueError, "values from 300 to 300 do not fit in uint8"),
        (np.uint8, -1, ValueError, "do not fit in uint8"),
        (np.int8, 0.5, TypeError, "without changing their kind"),
        (bool, 1, TypeError, "without changing their kind"),
        (np.float16, 1e10, ValueError, "do not fit in float16"),
        (np.float64, [1.0, 2.0], ValueError, "a default is one value"),
    ],
    ids=["too-large", "negative-unsigned", "float-to-int", "int-to-bool", "float16", "array"],
)
def test_a_default_must_keep_its_meaning_in_the_field(dtype, default, error, message):
    with pytest.raises(error, match=message):
        Field(dtype, default=default)


# State: construction.


def test_state_stores_every_field_cells_last(make_lattice):
    space = make_lattice()
    state = DemoState(space, {DemoState.terrain: 1})
    assert list(state) == [DemoState.alive, DemoState.pop, DemoState.terrain]
    assert (state[DemoState.alive].shape, state[DemoState.alive].dtype) == ((25,), np.bool_)
    assert (state[DemoState.pop].shape, state[DemoState.pop].dtype) == ((2, 25), np.float64)
    assert (state[DemoState.terrain].shape, state[DemoState.terrain].dtype) == ((25,), np.uint8)
    assert (state.space, state.n_cells, state.step, len(state)) == (space, 25, 0, 3)


def test_a_state_reads_its_fields_as_attributes(make_state):
    state = make_state(pop=1.0, alive=True)
    assert state.pop is state[DemoState.pop]
    assert state.alive is state[DemoState.alive]
    assert state.terrain is state[DemoState.terrain]
    assert DemoState.pop.name == "pop"  # on the class, the attribute is the field


def test_a_state_takes_its_step_by_keyword_only(make_lattice):
    space = make_lattice()
    assert DemoState(space, {DemoState.terrain: 1}, step=2).step == 2
    with pytest.raises(TypeError, match="positional argument"):
        DemoState(space, {DemoState.terrain: 1}, 2)  # ty: ignore[too-many-positional-arguments]


def test_dynamic_fields_start_at_zero(make_state):
    state = make_state()
    np.testing.assert_array_equal(state[DemoState.alive], np.zeros(25, bool))
    np.testing.assert_array_equal(state[DemoState.pop], np.zeros((2, 25)))


def test_static_fields_need_initial_values(make_lattice):
    with pytest.raises(ValueError, match="static field 'terrain' needs an initial value"):
        DemoState(make_lattice())


def test_initial_values_broadcast_over_components_and_cells(make_state):
    per_cell = np.arange(25.0)
    np.testing.assert_array_equal(make_state(pop=per_cell)[DemoState.pop], [per_cell, per_cell])
    per_group = np.array([[1.0], [2.0]])
    np.testing.assert_array_equal(make_state(pop=per_group)[DemoState.pop][:, 7], [1.0, 2.0])
    np.testing.assert_array_equal(make_state(alive=True)[DemoState.alive], np.ones(25, bool))


def test_initial_values_are_copied_into_read_only_arrays(make_state):
    terrain = np.zeros(25, np.uint8)
    state = make_state(terrain=terrain)
    terrain[0] = 9
    assert state[DemoState.terrain][0] == 0
    with pytest.raises(ValueError, match="read-only"):
        state[DemoState.terrain][0] = 1


@pytest.mark.parametrize(
    ("name", "values"),
    [("terrain", np.full(25, 2.0)), ("alive", np.ones(25)), ("alive", np.ones(25, np.int8))],
    ids=["float-to-int", "float-to-bool", "int-to-bool"],
)
def test_initial_values_cannot_change_kind(make_state, name, values):
    with pytest.raises(TypeError, match=f"field '{name}': .* without changing their kind"):
        make_state(**{name: values})


def test_initial_values_may_be_stored_at_lower_precision(make_lattice):
    XY = fields(x=Field(np.float32), y=Field(np.float64))
    state = XY(make_lattice(), {XY.x: np.full(25, 0.1), XY.y: np.arange(25)})
    assert state[XY.x].dtype == np.float32
    assert state[XY.x][0] == np.float32(0.1)
    np.testing.assert_array_equal(state[XY.y], np.arange(25.0))


def test_integer_values_may_go_into_any_integer_field_they_fit(make_state):
    codes = np.arange(25, dtype=np.int64) % 3
    np.testing.assert_array_equal(make_state(terrain=codes)[DemoState.terrain], codes)
    assert make_state(terrain=255)[DemoState.terrain][0] == 255


@pytest.mark.parametrize("values", [np.full(25, -1), np.arange(250, 275), 256, -1])
def test_integer_values_that_do_not_fit_raise_instead_of_wrapping(make_state, values):
    with pytest.raises(ValueError, match=r"'terrain': values from .* do not fit in uint8"):
        make_state(terrain=values)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.float16, 70000),
        (np.float16, np.float64(1e300)),
        (np.float32, 1e300),
        (np.float32, -1e300),
        (np.float16, np.iinfo(np.int64).max),
    ],
    ids=["int-to-float16", "float64-to-float16", "to-float32", "negative-to-float32", "int64-max"],
)
def test_finite_values_beyond_a_float_fields_range_raise(make_lattice, dtype, values):
    X = fields(x=Field(dtype))
    state = X(make_lattice())
    with pytest.raises(ValueError, match=r"'x': values beyond \+-.* do not fit"):
        state.replace({X.x: values})


def test_infinite_and_missing_values_are_stored_as_they_are(make_lattice):
    values = np.resize([np.inf, -np.inf, np.nan, 0.1], 25)
    X = fields(x=Field(np.float32))
    stored = X(make_lattice(), {X.x: values})[X.x]
    np.testing.assert_array_equal(stored[:4], np.array([np.inf, -np.inf, np.nan, 0.1], np.float32))


def test_values_outside_where_are_not_checked_for_float_overflow(make_lattice):
    X = fields(x=Field(np.float32))
    state = X(make_lattice())
    where = np.arange(25) < 5
    new = resolve(state, Assign(X.x, np.where(where, 1.5, 1e300), where=where))
    np.testing.assert_array_equal(new[X.x], np.where(where, 1.5, 0.0))


@pytest.mark.parametrize("source", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("target", DTYPES, ids=lambda dtype: dtype.__name__)
def test_storing_the_limits_of_a_dtype_keeps_their_meaning_or_raises(make_lattice, source, target):
    # The smallest and largest values of `source`, with 0 and 1 between them.
    if source is np.bool_:
        values = np.array([False, True], dtype=source)
    else:
        info = np.finfo(source) if np.dtype(source).kind == "f" else np.iinfo(source)
        values = np.array([info.min, 0, 1, info.max], dtype=source)
    space = make_lattice((2, 2), periodic=False)
    values = np.resize(values, space.n_cells)
    X = fields(x=Field(target))
    try:
        stored = X(space, {X.x: values})[X.x]
    except TypeError, ValueError:
        # Refusing is fine, unless NumPy calls the conversion safe: then it is exact.
        assert not np.can_cast(source, target, "safe")
        return
    if np.dtype(target).kind == "f":  # floats may round, but stay finite and close
        assert np.all(np.isfinite(stored))
        np.testing.assert_allclose(stored, values.astype(np.float64), rtol=np.finfo(target).eps)
    else:  # integers and flags are exact: no wrapping around
        assert stored.tolist() == values.tolist()


def test_initial_values_must_broadcast_to_the_field_shape(make_state):
    with pytest.raises(ValueError, match=r"'pop' has shape \(2, 25\); values of shape \(2,\)"):
        make_state(pop=[1.0, 2.0])


def test_initial_values_must_be_fields_of_the_class(make_lattice):
    with pytest.raises(KeyError, match="no field named 'food'"):
        DemoState(make_lattice(), {DemoState.terrain: 1, FoodState.food: 1.0})
    with pytest.raises(TypeError, match="Field attributes of a State class, got 'terrain'"):
        DemoState(make_lattice(), {"terrain": 1})  # ty: ignore[invalid-argument-type]


def test_step_must_be_a_non_negative_integer(make_state):
    assert make_state(step=np.int64(4)).step == 4
    with pytest.raises(ValueError, match="non-negative"):
        make_state(step=-1)


def test_step_must_be_an_integer(make_state, not_an_integer):
    with pytest.raises(TypeError, match="step must be an integer"):
        make_state(step=not_an_integer)


# State: looking up fields.


def test_lookup_of_a_field_the_state_lacks_lists_the_fields(make_state):
    state = make_state()
    with pytest.raises(KeyError, match=r"no field named 'food'; the fields are \['alive'"):
        state[FoodState.food]
    assert FoodState.food not in state
    assert DemoState.pop in state


def test_fields_are_looked_up_by_field_not_by_name(make_state):
    state = make_state()
    for key in ("pop", 0, None):
        with pytest.raises(TypeError, match="Field attributes of a State class"):
            state[key]
        assert key not in state
    with pytest.raises(TypeError, match="not declared on a State class"):
        state[Field(np.float64, (2,), extensive=True)]


def test_a_field_declared_differently_is_not_the_states_field(make_state):
    state = make_state()
    other = fields(pop=Field(np.float32, (2,), extensive=True)).pop
    assert other not in state
    for lookup in (state.__getitem__, state.version, state.total, state.density):
        with pytest.raises(KeyError, match=r"field 'pop' is declared as .*float64.*, not as"):
            lookup(other)


# State: changes and versions.


def test_replace_returns_a_new_state_and_leaves_the_old_one_unchanged(make_state):
    state = make_state(step=3)
    new = state.replace({DemoState.alive: True})
    assert new is not state
    assert new[DemoState.alive].all()
    assert not state[DemoState.alive].any()
    assert new[DemoState.pop] is state[DemoState.pop]
    assert new.step == 3


def test_replace_can_set_static_fields(make_state):
    replaced = make_state().replace({DemoState.terrain: 7})
    np.testing.assert_array_equal(replaced[DemoState.terrain], np.full(25, 7))


def test_replace_checks_values_like_the_constructor(make_state):
    state = make_state()
    with pytest.raises(KeyError, match="food"):
        state.replace({FoodState.food: 1.0})
    with pytest.raises(TypeError, match="'terrain'"):
        state.replace({DemoState.terrain: 0.5})
    with pytest.raises(TypeError, match="Field attributes of a State class"):
        state.replace({"alive": True})


def test_replace_copies_arrays_that_already_have_the_fields_dtype_and_shape(make_state):
    state = make_state()
    pop = np.arange(50, dtype=np.float64).reshape(2, 25)
    new = state.replace({DemoState.pop: pop, DemoState.alive: state[DemoState.alive]})
    pop[0, 0] = -1.0
    assert new[DemoState.pop][0, 0] == 0.0
    assert new[DemoState.alive] is not state[DemoState.alive]
    assert not new[DemoState.pop].flags.writeable
    assert not new[DemoState.alive].flags.writeable


def test_replace_stores_strided_and_byte_swapped_arrays_as_plain_ones(make_state):
    strided = np.arange(100, dtype=np.float64).reshape(2, 50)[:, ::2]
    swapped = np.arange(50, dtype=">f8").reshape(2, 25)
    for values in (strided, swapped):
        stored = make_state().replace({DemoState.pop: values})[DemoState.pop]
        np.testing.assert_array_equal(stored, values)
        assert stored.dtype == np.dtype(np.float64)
        assert stored.flags.c_contiguous


def test_replace_gives_new_versions_only_to_the_replaced_fields(make_state):
    state = make_state()
    new = state.replace({DemoState.alive: True})
    assert new.version(DemoState.alive) != state.version(DemoState.alive)
    assert new.version(DemoState.pop) == state.version(DemoState.pop)
    assert new.version(DemoState.terrain) == state.version(DemoState.terrain)


def _pieces():
    state = PieceState(LatticeSpace((4,), periodic=False))
    return state.replace({PieceState.kind: [0, -1, 2, -1], PieceState.hp: [5, 0, 7, 0]})


def test_edit_writes_go_into_the_frozen_state_and_leave_the_old_one_unchanged():
    state = _pieces()
    work = state.edit(PieceState.hp)
    work[PieceState.hp][2] -= 3
    new = work.freeze()
    np.testing.assert_array_equal(new.hp, [5, 0, 4, 0])
    np.testing.assert_array_equal(state.hp, [5, 0, 7, 0])
    assert type(new) is PieceState
    assert new.kind is state.kind
    assert not new.hp.flags.writeable


def test_edit_gives_new_versions_only_to_the_edited_fields(make_state):
    state = make_state(step=4)
    new = state.edit(DemoState.alive, DemoState.terrain).freeze()
    assert new.version(DemoState.alive) != state.version(DemoState.alive)
    assert new.version(DemoState.terrain) != state.version(DemoState.terrain)
    assert new.version(DemoState.pop) == state.version(DemoState.pop)
    assert new.step == 4


def test_edit_copies_the_fields_once_into_writable_arrays():
    state = _pieces()
    work = state.edit(PieceState.kind)
    kind = work[PieceState.kind]
    assert kind.flags.writeable
    assert work[PieceState.kind] is kind
    assert not np.shares_memory(kind, state.kind)


def test_move_takes_every_edited_field_to_the_target_and_clears_the_source_to_defaults():
    work = _pieces().edit(PieceState.kind, PieceState.hp, PieceState.ground)
    work.move(2, 3)
    new = work.freeze()
    np.testing.assert_array_equal(new.kind, [0, -1, -1, 2])
    np.testing.assert_array_equal(new.hp, [5, 0, 0, 7])
    np.testing.assert_array_equal(new.ground, [3, 3, 3, 3])


def test_move_onto_a_full_cell_overwrites_it_and_onto_itself_clears_it():
    work = _pieces().edit(PieceState.kind, PieceState.hp)
    work.move(0, 2)
    work.move(2, 2)
    new = work.freeze()
    np.testing.assert_array_equal(new.kind, [-1, -1, -1, -1])
    np.testing.assert_array_equal(new.hp, [0, 0, 0, 0])


def test_swap_exchanges_every_edited_field_and_leaves_others():
    work = _pieces().edit(PieceState.kind)
    work.swap(0, 1)
    work.swap(3, 3)
    new = work.freeze()
    np.testing.assert_array_equal(new.kind, [-1, 0, 2, -1])
    np.testing.assert_array_equal(new.hp, [5, 0, 7, 0])


def test_clear_sets_defaults_and_zero_for_fields_without_one():
    work = _pieces().edit(PieceState.kind, PieceState.hp)
    work.clear(0)
    new = work.freeze()
    np.testing.assert_array_equal(new.kind, [-1, -1, 2, -1])
    np.testing.assert_array_equal(new.hp, [0, 0, 7, 0])


def test_editing_a_nested_state_edits_all_its_fields_and_their_components(make_lattice):
    space = make_lattice((3, 3))
    U = WorldState.settlers
    state = WorldState(
        space,
        {
            U.people: np.arange(18.0).reshape(2, 9),
            U.home: 2.0,
            U.count: np.arange(9),
            WorldState.water: 5.0,
        },
    )
    work = state.edit(U)
    work.move(1, 4)
    new = work.freeze()
    np.testing.assert_array_equal(new.settlers.people[:, 4], [1.0, 10.0])
    np.testing.assert_array_equal(new.settlers.people[:, 1], [0.0, 0.0])
    assert new.settlers.home[1] == 1.0  # the field's default
    assert new.settlers.count[4] == 1
    assert new.water is state.water


@pytest.mark.parametrize("dtype", DTYPES)
def test_edit_keeps_extreme_values_of_every_dtype_exactly(dtype):
    if dtype is np.bool_:
        values = [True, False, True]
    elif np.dtype(dtype).kind == "f":
        info = np.finfo(dtype)
        values = [info.min, info.max, info.smallest_subnormal]
    else:
        info = np.iinfo(dtype)
        values = [info.min, info.max, 1]
    cls = type("ExtremeState", (State,), {"x": Field(dtype)})
    state = cls(LatticeSpace((3,), periodic=False), {cls.x: np.array(values, dtype)})
    work = state.edit(cls.x)
    work.swap(0, 1)
    work.move(2, 0)
    new = work.freeze()
    expected = np.array([values[2], values[0], 0], dtype)
    assert new.x.dtype == np.dtype(dtype)
    assert new.x.tobytes() == expected.tobytes()


def test_a_frozen_edit_refuses_to_be_used_and_its_arrays_are_read_only():
    work = _pieces().edit(PieceState.hp)
    hp = work[PieceState.hp]
    work.freeze()
    for use in (
        work.freeze,
        lambda: work[PieceState.hp],
        lambda: work.move(0, 1),
        lambda: work.swap(0, 1),
        lambda: work.clear(0),
    ):
        with pytest.raises(RuntimeError, match="frozen"):
            use()
    with pytest.raises(ValueError, match="read-only"):
        hp[0] = 1


def test_edit_rejects_no_fields_and_fields_given_twice():
    state = _pieces()
    with pytest.raises(ValueError, match="at least one"):
        state.edit()
    with pytest.raises(ValueError, match="'hp' is given twice"):
        state.edit(PieceState.hp, PieceState.hp)


def test_edit_rejects_fields_its_state_lacks_and_nested_states_not_as_nested(make_lattice):
    state = _pieces()
    with pytest.raises(KeyError, match="food"):
        state.edit(FoodState.food)
    with pytest.raises(TypeError, match="Field attributes"):
        state.edit("hp")  # a name is not a field
    eco = EcosystemState(make_lattice((3, 3)))
    with pytest.raises(KeyError, match="nested"):
        eco.edit(Species)
    with pytest.raises(ValueError, match="given twice"):
        eco.edit(EcosystemState.foxes, EcosystemState.foxes.population)


def test_reading_a_field_outside_the_edit_raises():
    work = _pieces().edit(PieceState.hp)
    with pytest.raises(KeyError, match="not in this edit"):
        work[PieceState.kind]


def test_edit_moves_check_their_cells(not_an_integer):
    work = _pieces().edit(PieceState.hp)
    for use in (work.move, work.swap):
        with pytest.raises(TypeError):
            use(0, not_an_integer)
        with pytest.raises(IndexError):
            use(0, 4)
        with pytest.raises(IndexError):
            use(-5, 0)
    with pytest.raises(TypeError):
        work.clear(not_an_integer)
    with pytest.raises(IndexError):
        work.clear(4)


def test_versions_identify_arrays_not_contents(make_state):
    # Equal contents from separate writes still get different versions.
    a, b = make_state(), make_state()
    assert a.version(DemoState.alive) != b.version(DemoState.alive)
    assert a.replace({DemoState.alive: False}).version(DemoState.alive) != a.version(
        DemoState.alive
    )


def test_version_of_a_field_the_state_lacks_raises(make_state):
    with pytest.raises(KeyError, match="no field named 'food'"):
        make_state().version(FoodState.food)


def test_states_compare_by_identity(make_state):
    a, b = make_state(), make_state()
    assert a == a
    assert a != b
    assert {a: 1}[a] == 1


def test_states_copy_as_themselves(make_state):
    state = make_state()
    assert copy.copy(state) is state
    assert copy.deepcopy(state) is state


def test_states_pickle_with_their_values_and_step(make_state):
    state = make_state(pop=np.arange(50.0).reshape(2, 25), alive=True, step=7)
    restored = pickle.loads(pickle.dumps(state))
    assert type(restored) is DemoState
    assert restored.step == 7
    assert list(restored) == list(state)
    for field in state:
        np.testing.assert_array_equal(restored[field], state[field])
        assert restored[field].dtype == state[field].dtype
        assert not restored[field].flags.writeable


def test_states_of_classes_made_in_a_function_do_not_pickle(make_lattice):
    # Pickle finds a class by its import path, which a class made at run time does not have.
    Local = fields(x=Field(np.int16, default=3))
    with pytest.raises(pickle.PicklingError, match="Some"):
        pickle.dumps(Local(make_lattice()))


# State: more fields, and what one cell holds.


class MoreState(State):
    road = Field(bool)
    food = Field(np.float64, static=True)


class FarmState(DemoState, MoreState):
    """`DemoState`'s fields, then `MoreState`'s."""


def test_extend_adds_the_fields_of_a_subclass_and_shares_the_rest(make_state):
    state = make_state(step=5, pop=1.0)
    farm = state.extend(FarmState, {FarmState.food: 0.5, DemoState.alive: True})
    assert type(farm) is FarmState
    assert list(farm) == [*DemoState._schema.values(), MoreState.road, MoreState.food]
    assert farm.space is state.space
    assert farm.step == 5
    assert farm.pop is state.pop
    assert farm.version(DemoState.pop) == state.version(DemoState.pop)
    assert farm.version(DemoState.alive) != state.version(DemoState.alive)
    assert farm.alive.all()
    np.testing.assert_array_equal(farm.road, False)
    np.testing.assert_array_equal(farm.food, 0.5)
    assert not farm.food.flags.writeable
    assert list(state) == list(DemoState._schema.values())  # unchanged
    assert not state.alive.any()
    assert len({farm.version(field) for field in farm}) == 5


def test_extend_keeps_the_fields_the_state_already_has(make_state):
    state = make_state(pop=1.0)

    class FedState(DemoState, LifeState):  # LifeState's fields are DemoState's too
        fed = Field(np.float64, (2,), extensive=True)

    fed = state.extend(FedState)
    assert list(fed) == [*DemoState._schema.values(), FedState.fed]
    assert fed.pop is state.pop
    same = state.extend(DemoState)
    assert same is not state
    assert type(same) is DemoState
    assert all(same[field] is state[field] for field in state)


def test_extend_checks_values_like_the_constructor(make_state):
    state = make_state()
    with pytest.raises(ValueError, match="static field 'food' needs an initial value"):
        state.extend(FarmState)
    with pytest.raises(KeyError, match=r"no field named 'height'"):
        state.extend(FarmState, {FarmState.food: 1.0, fields(height=Field(float)).height: 1.0})
    with pytest.raises(TypeError, match="'road'"):
        state.extend(FarmState, {FarmState.food: 1.0, FarmState.road: 0.5})
    with pytest.raises(TypeError, match="'terrain'"):
        state.extend(FarmState, {FarmState.food: 1.0, DemoState.terrain: 0.5})


@pytest.mark.parametrize("cls", [MoreState, State, MoreState.road, "FarmState"], ids=repr)
def test_extend_needs_a_subclass_of_the_states_class(make_state, cls):
    with pytest.raises(TypeError, match="a DemoState extends to a subclass of DemoState"):
        make_state().extend(cls)


def test_values_at_returns_what_one_cell_holds(make_state):
    pop = np.arange(50.0).reshape(2, 25)
    held = make_state(pop=pop, alive=np.arange(25) % 2 == 1).values_at(7)
    assert list(held) == [DemoState.alive, DemoState.pop, DemoState.terrain]
    assert held[DemoState.alive] == np.True_
    assert isinstance(held[DemoState.alive], np.bool_)
    assert held[DemoState.terrain] == 7 % 3
    assert type(held[DemoState.terrain]) is np.uint8
    np.testing.assert_array_equal(held[DemoState.pop], [7.0, 32.0])
    assert held[DemoState.pop].dtype == np.float64
    assert not held[DemoState.pop].flags.writeable
    Grid = fields(grid=Field(np.int64, (2, 3)))
    values = np.arange(24).reshape(2, 3, 4)
    state = Grid(LatticeSpace((2, 2), periodic=False), {Grid.grid: values})
    np.testing.assert_array_equal(state.values_at(1)[Grid.grid], values[..., 1])
    assert State(LatticeSpace((2, 2), periodic=False)).values_at(0) == {}


def test_values_at_needs_a_cell_of_the_space(make_state, not_an_integer):
    state = make_state()
    with pytest.raises(TypeError, match="cell id"):
        state.values_at(not_an_integer)
    for cell in (-1, 25):
        with pytest.raises(IndexError, match="out of range"):
            state.values_at(cell)


# State: amounts and densities.


class PopState(State):
    pop = Field(np.float64, (2,), extensive=True)


def test_density_divides_amounts_by_cell_measures(make_icosphere):
    space = make_icosphere(2)
    state = PopState(space, {PopState.pop: space.measures * np.array([[1.0], [3.0]])})
    expected = np.broadcast_to([[1.0], [3.0]], (2, space.n_cells))
    np.testing.assert_allclose(state.density(PopState.pop), expected, rtol=1e-12)


def test_total_of_amounts_is_their_sum(make_icosphere):
    space = make_icosphere(2)
    state = PopState(space, {PopState.pop: space.measures * np.array([[1.0], [3.0]])})
    np.testing.assert_allclose(state.total(PopState.pop), [4 * np.pi, 12 * np.pi], rtol=1e-12)


class ClimateState(State):
    temp = Field(float)
    land = Field(bool)


def test_intensive_fields_are_already_densities(make_icosphere):
    state = ClimateState(make_icosphere(2), {ClimateState.temp: 2.0})
    assert state.density(ClimateState.temp) is state[ClimateState.temp]


def test_total_of_an_intensive_field_is_its_integral_over_the_cells(make_icosphere):
    space = make_icosphere(2)
    land = np.arange(space.n_cells) % 4 == 0
    state = ClimateState(space, {ClimateState.temp: 2.0, ClimateState.land: land})
    np.testing.assert_allclose(state.total(ClimateState.temp), 8 * np.pi, rtol=1e-12)
    np.testing.assert_allclose(
        state.total(ClimateState.land), space.measures[land].sum(), rtol=1e-12
    )


def test_totals_of_float32_amounts_are_accumulated_in_float64(make_icosphere, rng):
    # Like pull and push, so that totals before and after a move compare at float64 precision.
    space = make_icosphere(8)
    amounts = rng.random(space.n_cells).astype(np.float32) * 1000
    Small = fields(pop=Field(np.float32, extensive=True))
    total = Small(space, {Small.pop: amounts}).total(Small.pop)
    assert total.dtype == np.float64
    np.testing.assert_allclose(total, math.fsum(amounts.astype(np.float64)), rtol=1e-14)


def test_totals_of_integer_amounts_are_exact(make_lattice):
    Count = fields(count=Field(np.uint8, extensive=True))
    assert Count(make_lattice(), {Count.count: 255}).total(Count.count) == 25 * 255


# State classes.


def test_a_state_class_holds_its_fields_in_declaration_order(make_lattice):
    assert list(LifeState(make_lattice())) == [LifeState.alive, LifeState.pop]
    assert list(State(make_lattice())) == []


def test_state_classes_combine_their_base_classes_fields_first(make_lattice):
    board = BoardState(make_lattice(), {BoardState.terrain: 1})
    assert list(board) == list(DemoState._schema.values())

    class DiamondState(BoardState, TerrainState):  # TerrainState's field is inherited twice
        food = Field(np.float64, extensive=True)

    state = DiamondState(make_lattice(), {DiamondState.terrain: 1})
    assert [field.name for field in state] == ["alive", "pop", "terrain", "food"]


def test_a_state_of_a_subclass_is_a_state_of_its_base_classes(make_lattice):
    board = BoardState(make_lattice(), {BoardState.terrain: 1})
    assert isinstance(board, LifeState)
    assert isinstance(board, TerrainState)
    assert board.terrain is board[TerrainState.terrain]
    assert board.pop is board[LifeState.pop]


def test_state_classes_reject_two_different_fields_with_one_name():
    with pytest.raises(ValueError, match="TwiceState: two different fields are named 'pop'"):

        class TwiceState(LifeState):
            pop = Field(np.float32, (2,), extensive=True)  # ty: ignore[invalid-attribute-override]


def test_a_state_class_attribute_cannot_hide_a_field():
    with pytest.raises(TypeError, match=r"HiddenState\.pop hides the field"):

        class HiddenState(LifeState):
            pop = 3


@pytest.mark.parametrize(
    "name", ["space", "step", "n_cells", "replace", "extend", "total", "keys", "_arrays", "_schema"]
)
def test_a_field_cannot_take_the_name_of_a_member_of_state(name):
    # Fields and the state's own members share one namespace: a field named step would hide it.
    with pytest.raises(ValueError, match=f"the name is taken by State.{name}"):
        type("Bad", (State,), {name: Field(float)})


def test_a_field_is_declared_once_on_a_state_class():
    with pytest.raises(TypeError, match="subclass of State"):

        class Plain:
            pop = Field(float)

    shared = Field(float)

    class FirstState(State):
        a = shared

    class AgainState(State):  # the same field, under its own name
        a = FirstState.a

    assert AgainState.a is FirstState.a
    with pytest.raises(ValueError, match="already declared as 'a'"):

        class SecondState(State):
            b = shared


# Immutability.


def test_a_state_refuses_every_attribute_assignment(make_lattice):
    state = DemoState(make_lattice(), {DemoState.terrain: 1})  # typed, so ty checks the lines below
    with pytest.raises(AttributeError, match="DemoState is immutable"):
        state.pop = np.zeros((2, 25))  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError, match="declare a Field on the class"):
        state.mountains = np.zeros(25, bool)  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError, match="immutable"):
        state._step = 3  # ty: ignore[invalid-assignment]
    assert state.step == 0
    assert not hasattr(state, "mountains")


# Derived fields.

CALLS: Counter[str] = Counter()
"""How many times each derived field of `TerrainBoardState` was computed."""


class TerrainBoardState(State):
    height = Field(np.float64, static=True)
    owner = Field(np.int8)

    @derived(height)
    def land(self) -> NDArray[np.bool_]:
        CALLS["land"] += 1
        return self.height > 0

    @derived(height, land)
    def peaks(self) -> NDArray[np.bool_]:
        CALLS["peaks"] += 1
        return self.land & (self.height >= np.quantile(self.height[self.land], 0.75))

    @derived(owner)
    def claimed(self) -> NDArray[np.bool_]:
        CALLS["claimed"] += 1
        return self.owner > 0


HEIGHT = np.linspace(-1.0, 1.0, 25)


@pytest.fixture
def board(make_lattice):
    """A typed `TerrainBoardState` on 25 cells, its heights rising from -1 to 1."""
    CALLS.clear()
    return TerrainBoardState(make_lattice(), {TerrainBoardState.height: HEIGHT})


def test_a_derived_field_reads_like_a_field(board):
    np.testing.assert_array_equal(board.land, HEIGHT > 0)
    assert board.land.dtype == np.bool_
    assert board[TerrainBoardState.land] is board.land
    assert not board.land.flags.writeable
    assert isinstance(TerrainBoardState.land, Derived)
    assert TerrainBoardState.land.name == "land"


def test_a_derived_field_may_depend_on_another(board):
    height = HEIGHT
    peak = np.quantile(height[height > 0], 0.75)
    np.testing.assert_array_equal(board.peaks, (height > 0) & (height >= peak))
    assert TerrainBoardState.peaks.fields() == (TerrainBoardState.height,)


def test_a_derived_field_is_computed_once_while_its_fields_keep_their_versions(board):
    first = board.land
    assert board.land is first
    claimed = board.replace({TerrainBoardState.owner: 1})  # a field land does not depend on
    assert claimed.land is first
    assert CALLS["land"] == 1
    assert claimed.claimed.all()
    assert not board.claimed.any()  # the old state keeps its own values
    raised = board.replace({TerrainBoardState.height: HEIGHT + 0.5})
    np.testing.assert_array_equal(raised.land, HEIGHT + 0.5 > 0)
    assert CALLS["land"] == 2
    np.testing.assert_array_equal(board.land, HEIGHT > 0)  # still right on the old state


def test_a_derived_field_of_static_fields_is_computed_once_per_run(board):
    @phase(writes=TerrainBoardState.owner)
    def claim(s: TerrainBoardState, ctx: Context) -> Assign:
        return Assign(TerrainBoardState.owner, s.owner + s.land)

    sim = Simulation(board, [claim], seed=0)
    sim.run(5)
    np.testing.assert_array_equal(sim.state.owner, np.where(HEIGHT > 0, 5, 0))
    assert CALLS["land"] == 1
    sim.state.claimed  # noqa: B018 - reads it, on the last state
    assert CALLS["claimed"] == 1


def test_a_derived_field_cannot_read_a_field_it_does_not_depend_on(make_lattice):
    class SneakyState(TerrainBoardState):
        @derived(TerrainBoardState.owner)
        def high(self) -> NDArray[np.bool_]:
            return self.height > 0.5  # height is not among its dependencies

        @derived(TerrainBoardState.owner)
        def keyed(self) -> NDArray[np.bool_]:
            return self[TerrainBoardState.height] > 0.5

    state = SneakyState(make_lattice(), {SneakyState.height: HEIGHT})
    for read in (lambda: state.high, lambda: state.keyed, lambda: state[SneakyState.high]):
        with pytest.raises(KeyError, match=r"'(high|keyed)'") as caught:
            read()
        assert "reads field 'height', which it does not depend on" in str(caught.value)


def test_a_derived_field_may_read_derived_fields_of_the_fields_it_depends_on(make_lattice):
    class ShoreState(TerrainBoardState):
        @derived(TerrainBoardState.height)  # land is computed from height alone
        def low_land(self) -> NDArray[np.bool_]:
            return self.land & (self.height < 0.5)

    state = ShoreState(make_lattice(), {ShoreState.height: HEIGHT})
    np.testing.assert_array_equal(state.low_land, (HEIGHT > 0) & (HEIGHT < 0.5))


def test_derived_fields_are_not_stored(board):
    assert list(board) == [TerrainBoardState.height, TerrainBoardState.owner]
    assert len(board) == 2
    assert TerrainBoardState.land not in board
    assert list(board.values_at(3)) == [TerrainBoardState.height, TerrainBoardState.owner]


@pytest.mark.parametrize(
    "write",
    [
        lambda s: s.replace({TerrainBoardState.land: True}),
        lambda s: resolve(s, Assign(TerrainBoardState.land, True)),  # ty: ignore[invalid-argument-type]
        lambda s: s.version(TerrainBoardState.land),
        lambda s: s.total(TerrainBoardState.land),
        lambda s: s.density(TerrainBoardState.land),
        lambda s: TerrainBoardState(s.space, {TerrainBoardState.land: True}),  # ty: ignore[invalid-argument-type]
    ],
    ids=["replace", "assign", "version", "total", "density", "constructor"],
)
def test_a_derived_field_cannot_be_written(board, write):
    with pytest.raises(TypeError, match="computed from other fields, so it is not stored"):
        write(board)


def test_a_state_of_another_class_has_no_derived_field_of_this_one(make_state):
    with pytest.raises(KeyError, match="DemoState has no derived field"):
        make_state()[TerrainBoardState.land]


def test_subclasses_and_extended_states_share_derived_fields_and_their_cache(board):
    first = board.land

    class GameState(TerrainBoardState):
        moves = Field(np.uint8)

    game = board.extend(GameState)
    assert game.land is first
    assert CALLS["land"] == 1


def test_a_state_pickles_without_its_cache_and_computes_again(board):
    first = board.land
    restored = pickle.loads(pickle.dumps(board))
    np.testing.assert_array_equal(restored.land, first)
    assert restored.land is not first
    assert CALLS["land"] == 2
    assert pickle.loads(pickle.dumps(TerrainBoardState.land)) is TerrainBoardState.land
    assert copy.deepcopy(TerrainBoardState.land) is TerrainBoardState.land


def test_a_derived_field_returning_a_field_array_returns_that_array(make_lattice):
    class AliasState(TerrainBoardState):
        @derived(TerrainBoardState.height)
        def elevation(self) -> NDArray[np.float64]:
            return self.height

    state = AliasState(make_lattice(), {AliasState.height: HEIGHT})
    assert state.elevation is state.height


def test_a_derived_field_keeps_the_cells_last(make_lattice):
    class GroupsState(TerrainBoardState):
        @derived(TerrainBoardState.height)
        def bands(self) -> NDArray[np.float64]:
            return np.stack([self.height, -self.height])  # two components per cell

        @derived(TerrainBoardState.height)
        def mean(self) -> NDArray[np.float64]:
            return self.height.mean()  # ty: ignore[invalid-return-type]

        @derived(TerrainBoardState.height)
        def per_row(self) -> NDArray[np.float64]:
            return self.height.reshape(5, 5)[:, :4]

    state = GroupsState(make_lattice(), {GroupsState.height: HEIGHT})
    assert state.bands.shape == (2, 25)
    for name in ("mean", "per_row"):
        with pytest.raises(ValueError, match=rf"derived field '{name}' must have the cells last"):
            getattr(state, name)


def test_an_error_in_a_derived_field_names_it(make_lattice):
    class BrokenState(TerrainBoardState):
        @derived(TerrainBoardState.height)
        def broken(self) -> NDArray[np.float64]:
            raise ZeroDivisionError("no")

    state = BrokenState(make_lattice(), {BrokenState.height: HEIGHT})
    with pytest.raises(ZeroDivisionError) as caught:
        state.broken  # noqa: B018 - reading it computes it
    assert "computing derived field 'broken'" in caught.value.__notes__


def test_a_derived_field_must_depend_on_fields_of_its_class():
    with pytest.raises(KeyError, match=r"OrphanState\.food_left depends on 'food': no field named"):

        class OrphanState(TerrainBoardState):
            @derived(FoodState.food)
            def food_left(self) -> NDArray[np.float64]:
                return self.food  # ty: ignore[unresolved-attribute]


def test_a_derived_field_is_declared_with_fields_in_parentheses():
    with pytest.raises(TypeError, match=r"with @derived\(field, ...\), with parentheses"):
        derived(lambda s: s)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="takes the fields a method depends on, got 'height'"):
        derived("height")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["step", "total", "space"])
def test_a_derived_field_cannot_take_the_name_of_a_member_of_state(name):
    with pytest.raises(ValueError, match=f"the name is taken by State.{name}"):
        type("Bad", (TerrainBoardState,), {name: derived()(lambda s: s.height)})


def test_a_derived_field_is_declared_on_a_state_class():
    with pytest.raises(TypeError, match="declared on a subclass of State"):

        class Plain:
            @derived()
            def nothing(self) -> NDArray[np.float64]:
                return np.zeros(1)


# Nested states.


def test_a_nested_state_stores_its_fields_flat_under_qualified_names(make_lattice):
    state = EcosystemState(make_lattice())
    names = [field.name for field in state]
    assert names == [
        "grass",
        "rabbits.population",
        "rabbits.eaten",
        "foxes.population",
        "foxes.eaten",
    ]
    assert set(state.values_at(0)) == set(state)


def test_a_nested_states_view_reads_the_states_arrays(make_lattice):
    state = EcosystemState(make_lattice(), {EcosystemState.foxes.population: np.arange(25.0)})
    assert state.foxes.population is state[EcosystemState.foxes.population]
    np.testing.assert_array_equal(state.foxes.population, np.arange(25.0))
    assert not state.rabbits.population.any()
    assert isinstance(state.foxes, Species)
    assert state.foxes.space is state.space


def test_one_nested_state_nested_twice_gives_two_sets_of_fields():
    foxes, rabbits = EcosystemState.foxes.population, EcosystemState.rabbits.population
    assert (foxes.name, rabbits.name) == ("foxes.population", "rabbits.population")
    assert foxes != rabbits
    assert issubclass(EcosystemState.foxes, Species)
    assert Species.population.name == "population"  # the class as declared is unchanged


def test_a_state_makes_each_view_once_and_new_states_their_own(make_lattice):
    state = EcosystemState(make_lattice())
    assert state.foxes is state.foxes
    fed = state.replace({EcosystemState.foxes.population: 1.0})
    assert fed.foxes is not state.foxes
    assert fed.foxes.population.all()
    assert not state.foxes.population.any()


def test_a_nested_states_view_is_immutable(make_lattice):
    state = EcosystemState(make_lattice())
    with pytest.raises(AttributeError, match="view of an immutable state"):
        state.foxes.population = np.ones(25)  # ty: ignore[invalid-assignment] - the test's point


def test_a_nested_state_is_not_made_directly():
    with pytest.raises(TypeError, match="nest it in a state class"):
        Species()


@pytest.mark.parametrize(
    "declared", [DemoState, int, "Species", EcosystemState.foxes, Biome.foxes, PlanetState.north]
)
def test_nested_takes_a_nested_state_as_declared(declared):
    with pytest.raises(TypeError, match="a NestedState subclass as declared"):
        Nested(declared)


def test_nested_is_declared_on_a_state_or_a_nested_state():
    with pytest.raises(TypeError, match="declared on a subclass of State or NestedState"):
        type("Pack", (), {"members": Nested(Species)})


@pytest.mark.parametrize("name", ["space", "step", "total", "replace"])
def test_a_nested_state_cannot_take_the_name_of_a_member_of_state(name):
    with pytest.raises(ValueError, match=f"taken by State.{name}"):
        type("Some", (State,), {name: Nested(Species)})


def test_a_field_of_a_nested_state_cannot_take_the_name_of_its_member():
    with pytest.raises(ValueError, match=r"taken by NestedState\.space"):
        type("Some", (NestedState,), {"space": Field(np.float64)})


def test_a_nested_states_own_field_is_named_through_the_state_class(make_lattice):
    state = EcosystemState(make_lattice())
    with pytest.raises(KeyError, match=r"declared on a nested state.*'rabbits\.population'"):
        state[Species.population]


def test_a_derived_field_of_a_nested_state_is_computed_and_cached_for_each_nesting(make_lattice):
    NESTED_CALLS.clear()
    state = EcosystemState(make_lattice(), {EcosystemState.foxes.population: np.arange(25.0)})
    np.testing.assert_array_equal(state.foxes.present, np.arange(25.0) > 0)
    assert not state.rabbits.present.any()
    assert state.foxes.present is state.foxes.present
    assert state[EcosystemState.foxes.present] is state.foxes.present
    assert EcosystemState.foxes.present.name == "foxes.present"
    assert NESTED_CALLS == {"foxes": 1, "rabbits": 1}
    fed = state.replace({EcosystemState.foxes.eaten: 1.0})  # a field present does not read
    assert fed.foxes.present is state.foxes.present
    gone = state.replace({EcosystemState.foxes.population: 0.0})
    assert not gone.foxes.present.any()
    assert NESTED_CALLS["foxes"] == 2


def test_a_derived_field_of_a_nested_state_cannot_read_a_field_it_does_not_depend_on(make_lattice):
    class Sloppy(NestedState):
        a = Field(np.float64)
        b = Field(np.float64)

        @derived(a)
        def both(self) -> NDArray[np.float64]:
            return self.a + self.b

    cls = fields(g=Nested(Sloppy))
    with pytest.raises(KeyError, match=r"reads field 'g\.b', which it does not depend on"):
        _ = cls(make_lattice()).g.both


def test_a_derived_field_may_depend_on_a_whole_nested_state(make_lattice):
    values = {EcosystemState.foxes.population: 2.0, EcosystemState.foxes.eaten: [1.0] * 24 + [3.0]}
    state = EcosystemState(make_lattice(), values)
    np.testing.assert_array_equal(state.hunting, [True] * 24 + [False])
    expected = (EcosystemState.foxes.population, EcosystemState.foxes.eaten)
    assert EcosystemState.hunting.fields() == expected


def test_nested_states_combine_by_inheritance(make_lattice):
    class Predators(Species):
        dens = Field(np.int8)

    cls = fields(foxes=Nested(Predators))
    state = cls(make_lattice(), {cls.foxes.population: 3.0})
    assert [field.name for field in state] == ["foxes.population", "foxes.eaten", "foxes.dens"]
    assert state.foxes.present.all()  # the base class's derived field


class GrassState(State):
    grass = Field(np.float64, extensive=True)


class GrassAndFoxesState(GrassState):
    foxes = Nested(Species)


def test_extend_adds_a_nested_states_fields_and_subclasses_inherit_them(make_lattice):
    meadow = GrassState(make_lattice(), {GrassState.grass: 1.0})
    wild = meadow.extend(GrassAndFoxesState, {GrassAndFoxesState.foxes.population: 2.0})
    assert wild.grass is meadow.grass
    assert (wild.foxes.population == 2.0).all()
    assert not wild.foxes.eaten.any()
    sub = type("Sub", (GrassAndFoxesState,), {})
    assert sub.foxes is GrassAndFoxesState.foxes


def test_a_class_attribute_cannot_hide_a_nested_state():
    with pytest.raises(TypeError, match=r"foxes\.population hides the field"):
        type("Some", (EcosystemState,), {"foxes": 1})


def test_states_with_nested_states_pickle_and_so_do_their_classes_and_views(make_lattice):
    state = EcosystemState(make_lattice(), {EcosystemState.foxes.population: np.arange(25.0)})
    loaded = pickle.loads(pickle.dumps(state))
    np.testing.assert_array_equal(loaded.foxes.population, state.foxes.population)
    assert pickle.loads(pickle.dumps(EcosystemState.foxes)) is EcosystemState.foxes
    assert pickle.loads(pickle.dumps(EcosystemState.foxes.present)) is EcosystemState.foxes.present
    view = pickle.loads(pickle.dumps(state.foxes))
    np.testing.assert_array_equal(view.population, state.foxes.population)


# Nested states nested in one another.


def test_nested_states_nest_at_any_depth_and_store_their_fields_flat(make_lattice):
    names = [field.name for field in PlanetState(make_lattice())]
    assert names == [
        "grass",
        *(f"{biome}.{name}" for biome in ("north", "south", "europe.north") for name in BIOME),
    ]


BIOME = ("rain", "rabbits.population", "rabbits.eaten", "foxes.population", "foxes.eaten")


def test_a_view_of_a_view_reads_the_states_arrays(make_lattice):
    deep = PlanetState.europe.north.foxes.population
    state = PlanetState(make_lattice(), {deep: np.arange(25.0)})
    population: NDArray[np.float64] = state.europe.north.foxes.population  # typed, for ty
    field: Field[np.float64] = PlanetState.europe.north.foxes.population
    assert population is state[field]
    np.testing.assert_array_equal(population, np.arange(25.0))
    assert not state.north.foxes.population.any()
    assert (state.europe.north.rain == 1.0).all()
    assert isinstance(state.europe.north, Biome)
    assert isinstance(state.europe.north.foxes, Species)
    assert state.europe.north.foxes.space is state.space


def test_a_state_makes_each_view_at_depth_once_and_new_states_their_own(make_lattice):
    state = PlanetState(make_lattice())
    assert state.europe.north.foxes is state.europe.north.foxes
    assert state.europe.north.foxes is not state.north.foxes
    fed = state.replace({PlanetState.europe.north.foxes.population: 1.0})
    assert fed.europe.north.foxes is not state.europe.north.foxes
    assert fed.europe.north.foxes.population.all()
    assert not state.europe.north.foxes.population.any()


def test_a_view_at_depth_is_immutable(make_lattice):
    state = PlanetState(make_lattice())
    with pytest.raises(AttributeError, match="view of an immutable state"):
        state.europe.north.rain = np.ones(25)  # ty: ignore[invalid-assignment] - the test's point


def test_one_nested_state_under_two_paths_gives_two_sets_of_fields():
    deep, shallow = PlanetState.europe.north.foxes.population, PlanetState.north.foxes.population
    assert (deep.name, shallow.name) == ("europe.north.foxes.population", "north.foxes.population")
    assert deep != shallow
    assert issubclass(PlanetState.europe.north.foxes, Species)
    assert issubclass(PlanetState.europe.north, Biome)


def test_the_fields_of_a_nested_state_as_declared_are_named_relative_to_it(make_lattice):
    assert Biome.foxes.population.name == "foxes.population"
    assert Continent.north.foxes.population.name == "north.foxes.population"
    assert issubclass(Biome.foxes, Species)
    state = PlanetState(make_lattice())
    with pytest.raises(KeyError, match=r"declared on a nested state.*'north\.foxes\.population'"):
        state[Biome.foxes.population]


def test_derived_fields_at_depth_are_computed_and_cached_for_each_path(make_lattice):
    NESTED_CALLS.clear()
    values = {PlanetState.europe.north.foxes.population: np.arange(25.0)}
    state = PlanetState(make_lattice(), values)
    np.testing.assert_array_equal(state.europe.north.foxes.present, np.arange(25.0) > 0)
    assert not state.north.foxes.present.any()
    assert state.europe.north.foxes.present is state.europe.north.foxes.present
    assert state[PlanetState.europe.north.foxes.present] is state.europe.north.foxes.present
    assert PlanetState.europe.north.foxes.present.name == "europe.north.foxes.present"
    assert NESTED_CALLS == {"europe.north.foxes": 1, "north.foxes": 1}


def test_a_derived_field_of_a_nested_state_may_depend_on_a_state_nested_in_it(make_lattice):
    values = {PlanetState.europe.north.foxes.population: [1.0] * 24 + [0.0]}
    state = PlanetState(make_lattice(), values)
    np.testing.assert_array_equal(state.europe.north.hunted, [True] * 24 + [False])
    assert not state.north.hunted.any()
    expected = ("europe.north.foxes.population", "europe.north.foxes.eaten", "europe.north.rain")
    assert [field.name for field in PlanetState.europe.north.hunted.fields()] == list(expected)
    dry = state.replace({PlanetState.europe.north.rain: 0.0})
    assert not dry.europe.north.hunted.any()


def test_a_derived_field_of_a_state_reads_through_views_of_views(make_lattice):
    values = {PlanetState.europe.north.foxes.population: [0.0] + [2.0] * 24}
    state = PlanetState(make_lattice(), values)
    np.testing.assert_array_equal(state.european_foxes, [False] + [True] * 24)
    assert len(PlanetState.european_foxes.fields()) == len(BIOME)


def test_a_nested_state_in_a_nested_state_cannot_take_the_name_of_its_member():
    with pytest.raises(ValueError, match=r"taken by NestedState\.space"):
        type("Some", (NestedState,), {"space": Nested(Species)})


def test_nested_states_at_depth_combine_by_inheritance(make_lattice):
    class Predators(Species):
        dens = Field(np.int8)

    class Forest(Biome):
        foxes = Nested(Predators)  # replaces the base's, keeping its fields
        wolves = Nested(Species)

    cls = fields(woods=Nested(Forest))
    state = cls(make_lattice(), {cls.woods.foxes.dens: 1, cls.woods.rabbits.population: 2.0})
    assert [field.name for field in state] == [
        *(f"woods.{name}" for name in BIOME),
        "woods.foxes.dens",
        "woods.wolves.population",
        "woods.wolves.eaten",
    ]
    assert isinstance(state.woods.foxes, Predators)
    assert (state.woods.foxes.dens == 1).all()
    assert state.woods.rabbits.present.all()  # inherited from Biome, bound under the new path
    assert not state.woods.hunted.any()  # Biome's derived field, now on Predators' foxes


def test_a_class_attribute_cannot_hide_a_field_at_depth():
    with pytest.raises(TypeError, match=r"foxes\.population hides the field"):
        type("Some", (Biome,), {"foxes": 1})


class GrassAndContinentState(GrassState):
    europe = Nested(Continent)


def test_extend_adds_the_fields_of_nested_states_at_depth(make_lattice):
    meadow = GrassState(make_lattice(), {GrassState.grass: 1.0})
    values = {GrassAndContinentState.europe.north.foxes.population: 2.0}
    wild = meadow.extend(GrassAndContinentState, values)
    assert wild.grass is meadow.grass
    assert (wild.europe.north.foxes.population == 2.0).all()
    assert (wild.europe.north.rain == 1.0).all()


def test_states_with_nested_states_at_depth_pickle_and_so_do_their_classes_and_views(
    make_lattice,
):
    values = {PlanetState.europe.north.foxes.population: np.arange(25.0)}
    state = PlanetState(make_lattice(), values)
    loaded = pickle.loads(pickle.dumps(state))
    np.testing.assert_array_equal(loaded.europe.north.foxes.population, np.arange(25.0))
    for named in (
        PlanetState.europe.north.foxes,
        PlanetState.europe.north.foxes.present,
        PlanetState.europe.north.hunted,
        Biome.foxes,
    ):
        assert pickle.loads(pickle.dumps(named)) is named
    view = pickle.loads(pickle.dumps(state.europe.north.foxes))
    np.testing.assert_array_equal(view.population, np.arange(25.0))


@pytest.mark.parametrize(
    ("cls", "field"),
    [
        (EcosystemState, EcosystemState.foxes.present),
        (PlanetState, PlanetState.north.hunted),
        (PlanetState, PlanetState.europe.north.foxes.present),
    ],
    ids=["depth 1", "depth 1, of a biome", "depth 3"],
)
def test_a_derived_field_of_a_nested_state_read_first_by_its_name_reads_the_view(
    make_lattice, cls, field
):
    space = make_lattice()
    assert cls(space)[field].shape == (space.n_cells,)


# States that say which kind of space their rules need.


class FlatState(State):
    space: LatticeSpace
    height = Field(np.float64)


class AnySurfaceState(State):
    space: LatticeSpace | MeshSpace
    height = Field(np.float64)


class FlatFromAnyState(AnySurfaceState):
    space: LatticeSpace


class FlatPiecesState(FlatState):
    owner = Field(np.int8)


class WidenedState(FlatState):
    space: LatticeSpace | MeshSpace  # ty: ignore[invalid-attribute-override] - refused, as tested


if TYPE_CHECKING:
    from decimal import Decimal  # for ty alone: not defined when the tests run


class DeferredState(State):
    space: LatticeSpace
    later: Decimal  # only the space annotation is evaluated


class QuotedState(State):  # as under `from __future__ import annotations`
    space: "LatticeSpace"  # noqa: UP037 (quoted on purpose, as said above)
    later: "Decimal"  # noqa: UP037 (quoted on purpose, as said above)


def test_a_state_is_made_only_on_the_kind_of_space_its_class_annotates(
    make_lattice, make_icosphere
):
    assert FlatState(make_lattice()).space.dim == 2
    with pytest.raises(TypeError, match="FlatState is made on a space of kind LatticeSpace"):
        FlatState(make_icosphere())


def test_a_union_annotation_takes_any_of_its_kinds(make_lattice, make_icosphere):
    AnySurfaceState(make_lattice())
    AnySurfaceState(make_icosphere())


def test_subclasses_keep_and_may_narrow_the_annotation(make_lattice, make_icosphere):
    with pytest.raises(TypeError, match="FlatPiecesState"):
        FlatPiecesState(make_icosphere())
    with pytest.raises(TypeError, match="FlatFromAnyState"):
        FlatFromAnyState(make_icosphere())
    FlatFromAnyState(make_lattice())


def test_a_subclass_cannot_widen_the_annotation_of_its_base(make_icosphere):
    with pytest.raises(TypeError, match="WidenedState is made on a space of kind LatticeSpace"):
        WidenedState(make_icosphere())


def test_extending_to_a_narrower_class_checks_the_space(make_lattice, make_icosphere):
    on_mesh = AnySurfaceState(make_icosphere())
    with pytest.raises(TypeError, match="FlatFromAnyState"):
        on_mesh.extend(FlatFromAnyState)
    assert isinstance(AnySurfaceState(make_lattice()).extend(FlatFromAnyState), FlatFromAnyState)


@pytest.mark.parametrize("cls", [DeferredState, QuotedState])
def test_the_space_annotation_is_evaluated_alone(cls, make_lattice, make_icosphere):
    cls(make_lattice())
    with pytest.raises(TypeError, match=cls.__name__):
        cls(make_icosphere())


def test_a_space_annotation_naming_nothing_raises_when_the_class_is_made():
    with pytest.raises(NameError, match="Nowhere"):

        class LostState(State):
            # Undefined on purpose: ruff and ty would both say so.
            space: Nowhere  # noqa: F821  # ty: ignore[unresolved-reference]


def test_a_pickled_state_keeps_its_space_check(make_lattice):
    state = FlatState(make_lattice(), {FlatState.height: 2.0})
    again = pickle.loads(pickle.dumps(state))
    np.testing.assert_array_equal(again.height, state.height)
    assert FlatState._space_kinds == (LatticeSpace,)


def test_a_space_annotation_must_name_classes():
    with pytest.raises(TypeError, match="a class or a union of classes"):

        class Bad(State):
            space: list[int]  # ty: ignore[invalid-property-type-override] - raises, as checked
