"""`batched`'s dice: a key per run, a stream per step, and rolls that are fair."""

import jax
import numpy as np
import pytest

from gridzero import batched


def test_the_same_seed_gives_the_same_key_and_other_seeds_other_keys():
    assert batched.root(7).dtype == np.uint32
    assert batched.root(7).shape == (2,)
    assert batched.root(7).tolist() == batched.root(7).tolist()
    keys = {tuple(batched.root(seed).tolist()) for seed in (0, 1, 2, 2**32, 2**32 + 1, 2**63 - 1)}
    assert len(keys) == 6  # the high bits count too


def test_a_seed_in_two_halves_gives_the_seeds_key_compiled_too():
    for seed in (0, 7, 2**32 + 5, 2**63 - 1):
        halves = np.array([seed & 0xFFFFFFFF, seed >> 32], dtype=np.uint32)
        assert batched.root(halves).tolist() == batched.root(seed).tolist()
        assert jax.jit(batched.root)(halves).tolist() == batched.root(seed).tolist()
    with pytest.raises(ValueError, match="shape"):
        batched.root(np.zeros(3, np.uint32))


def test_streams_differ_by_every_part_of_their_name_and_repeat():
    key = batched.root(3)
    names = [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 0, 0)]
    streams = {tuple(batched.stream(key, *name).tolist()) for name in names}
    assert len(streams) == len(names)
    assert tuple(key.tolist()) not in streams
    assert batched.stream(key, 1, 5).tolist() == batched.stream(key, 1, 5).tolist()
    assert batched.stream(key).tolist() == key.tolist()
    assert batched.stream(key, np.int32(4)).tolist() == batched.stream(key, 4).tolist()


def test_rolls_repeat_for_a_key_and_fill_a_shape():
    key = batched.stream(batched.root(0), 2)
    rolls = batched.d(key, 20, (3, 4))
    assert rolls.dtype == np.int32
    assert rolls.shape == (3, 4)
    assert rolls.tolist() == batched.d(key, 20, (3, 4)).tolist()
    assert batched.d(key, 20).shape == ()
    assert batched.d(key, 1, (5,)).tolist() == [1] * 5


def test_a_d20_rolls_every_face_about_as_often():
    rolls = np.asarray(batched.d(batched.root(11), 20, (40_000,)))
    assert rolls.min() == 1
    assert rolls.max() == 20
    counts = np.bincount(rolls, minlength=21)[1:]
    chi2 = ((counts - 2000) ** 2 / 2000).sum()
    assert chi2 < 43.8  # 19 degrees of freedom: p = 0.001


def test_each_step_of_each_game_of_a_batch_rolls_its_own_dice():
    keys = np.stack([np.asarray(batched.root(seed)) for seed in range(64)])
    plies = np.arange(64, dtype=np.int32) % 8
    roll = jax.jit(jax.vmap(lambda key, ply: batched.d(batched.stream(key, 1, ply), 20, (3,))))
    rolls = np.asarray(roll(keys, plies))
    for row in (0, 9, 63):  # a row of the batch is that game alone
        alone = batched.d(batched.stream(keys[row], 1, int(plies[row])), 20, (3,))
        assert rolls[row].tolist() == alone.tolist()
    assert len({tuple(r) for r in rolls.tolist()}) > 60
    again = np.asarray(roll(keys, plies + 1))
    assert (rolls != again).any(axis=1).mean() > 0.9


def test_a_permutation_holds_each_number_once_in_every_order_about_as_often():
    key = batched.root(5)
    order = batched.permutation(key, 9)
    assert order.dtype == np.int32
    assert sorted(order.tolist()) == list(range(9))
    assert order.tolist() == batched.permutation(key, 9).tolist()
    assert batched.permutation(key, 0).shape == (0,)
    shuffle = jax.jit(jax.vmap(lambda k: batched.permutation(batched.stream(key, k), 3)))
    orders = np.asarray(shuffle(np.arange(6000)))
    counts = np.unique(orders @ [9, 3, 1], return_counts=True)[1]
    assert counts.size == 6
    assert ((counts - 1000) ** 2 / 1000).sum() < 20.5  # 5 degrees of freedom: p = 0.001


def test_bad_seeds_and_sides_raise(not_an_integer):
    with pytest.raises(TypeError):
        batched.root(not_an_integer)
    with pytest.raises(ValueError, match="seed"):
        batched.root(-1)
    with pytest.raises(ValueError, match="2\\*\\*63"):
        batched.root(2**63)
    with pytest.raises(TypeError):
        batched.d(batched.root(0), not_an_integer)
    with pytest.raises(ValueError, match="sides"):
        batched.d(batched.root(0), 0)
    with pytest.raises(TypeError):
        batched.permutation(batched.root(0), not_an_integer)
    with pytest.raises(ValueError, match="n must"):
        batched.permutation(batched.root(0), -1)
