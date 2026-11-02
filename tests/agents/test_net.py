import numpy as np
import pytest
import torch

from agents_demo import PLACE, Patches
from gridzero.agents import AlphaZeroPlayer, UniformPlayer, self_play
from gridzero.agents.net import (
    CellConv,
    LatticeConv,
    NetEvaluator,
    NetPlayer,
    PolicyNet,
    _Block,
    judgements,
    observations,
    read,
    unpack,
)
from gridzero.core import LatticeSpace, icosphere
from gridzero.game import play

GAME = Patches()


def small_net(game=GAME, seed=0):
    torch.manual_seed(seed)
    return PolicyNet(game, channels=8, blocks=2)


@pytest.mark.parametrize(("tiling", "kind"), [("hex", "facet"), ("hypercubic", "vertex")])
def test_a_lattice_convolution_reads_each_cell_and_its_neighbors(tiling, kind):
    space = LatticeSpace((6, 7), tiling=tiling, periodic=False)
    conv = LatticeConv(space, 1, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    graph = space.adjacency(kind)
    for cell in (0, 9, 20, 41):
        x = torch.zeros(1, 1, 6, 7)
        x.view(-1)[cell] = 1.0
        y = conv(x).view(-1).detach().numpy()
        near = np.zeros(space.n_cells, dtype=np.float32)
        near[graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]]] = 1.0
        near[cell] = 1.0
        np.testing.assert_array_equal(y, near)  # the kernel is symmetric: pull equals push


def test_a_hex_convolution_ignores_the_corner_weights():
    torch.manual_seed(1)
    conv = LatticeConv(LatticeSpace((5, 5), tiling="hex", periodic=False), 2, 3)
    x = torch.randn(4, 2, 5, 5)
    before = conv(x)
    with torch.no_grad():
        conv.weight[:, :, 0, 0] = 100.0
        conv.weight[:, :, 2, 2] = -100.0
    torch.testing.assert_close(conv(x), before)


@pytest.mark.parametrize(
    "space",
    [
        LatticeSpace((6, 4), tiling="triangle", periodic=False),
        LatticeSpace((5, 5), periodic=True),
        LatticeSpace((3, 3, 3), periodic=False),
        LatticeSpace((9,), periodic=False),
    ],
)
def test_lattice_convolutions_refuse_lattices_they_dont_support(space):
    with pytest.raises(ValueError, match="squares or hexes"):
        LatticeConv(space, 1, 1)


@pytest.mark.parametrize(
    "space", [icosphere(1), icosphere(2), LatticeSpace((4, 5), tiling="hex", periodic=False)]
)
def test_a_cell_convolution_reads_each_cell_and_the_mean_of_its_neighbors(space, rng):
    torch.manual_seed(2)
    conv = CellConv(space, 3, 2)
    x = rng.standard_normal((4, 3, space.n_cells)).astype(np.float32)
    graph = space.adjacency()
    w_own = conv.own.weight.detach().numpy()[:, :, 0]
    w_near = conv.near.weight.detach().numpy()[:, :, 0]
    assert conv.own.bias is not None
    bias = conv.own.bias.detach().numpy()
    want = np.empty((4, 2, space.n_cells), dtype=np.float32)
    for cell in range(space.n_cells):
        near = graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]]
        mean = x[:, :, near].mean(axis=2)
        want[:, :, cell] = x[:, :, cell] @ w_own.T + mean @ w_near.T + bias
    got = conv(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_a_cell_convolution_moves_with_the_cells_it_reads():
    """Neighbors are alike: a value on a pentagon spreads equally to its five neighbors."""
    space = icosphere(2)
    conv = CellConv(space, 1, 1, bias=False)
    with torch.no_grad():
        conv.own.weight.zero_()
        conv.near.weight.fill_(1.0)
    pentagon = int(np.flatnonzero(np.diff(space.adjacency().indptr) == 5)[0])
    x = torch.zeros(1, 1, space.n_cells)
    x[0, 0, pentagon] = 1.0
    y = conv(x)[0, 0].detach().numpy()
    graph = space.adjacency()
    near = graph.indices[graph.indptr[pentagon] : graph.indptr[pentagon + 1]]
    np.testing.assert_allclose(y[near], 1.0 / np.diff(graph.indptr)[near], rtol=1e-6)
    assert (np.delete(y, near) == 0).all()


def test_a_new_block_of_cell_convolutions_passes_its_input_through():
    space = icosphere(1)
    block = _Block(lambda i, o, b: CellConv(space, i, o, b), 4)
    x = torch.rand(2, 4, space.n_cells)
    torch.testing.assert_close(block(x), x)


def test_a_net_on_squares_scores_every_action_and_values_in_minus_one_to_one():
    net = small_net()
    obs = torch.rand(5, len(GAME.planes), GAME.space.n_cells)
    logits, value = net(obs)
    assert logits.shape == (5, GAME.n_verbs, GAME.space.n_cells)
    assert value.shape == (5,)
    assert (value.abs() <= 1).all()
    assert net.shape == GAME.space.shape  # 3 by 4: rows and columns are not mixed up


def test_a_net_sees_the_board_as_its_lattice_lays_it_out():
    # The stem's 3x3 window around cell (0, 1) of the 3x4 board: rows 0 and 1, columns 0 to 2.
    net = PolicyNet(GAME, channels=4, blocks=0)
    torch.manual_seed(0)
    x = torch.zeros(1, len(GAME.planes) + 1, 3, 4)
    x[0, 0, 0, 1] = 1.0
    with torch.no_grad():
        y = net.stem(x) - net.stem(torch.zeros_like(x))
    reached = (y.abs().sum(dim=1)[0] > 0).numpy()
    np.testing.assert_array_equal(reached, [[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]])


def test_a_net_player_plays_whole_games_and_maps_its_choice_back_to_the_board():
    net = small_net()
    record = play(GAME, [NetPlayer(net), UniformPlayer()], GAME.start(1))
    assert record.final.over
    position = GAME.step(GAME.start(0), (PLACE, 0))  # player 1 to move
    legal = GAME.legal(position)
    decision = NetPlayer(net).decide(GAME, position, legal, np.random.default_rng(0))
    verb, cell = divmod(decision.index, GAME.space.n_cells)
    assert decision.player == 1
    assert decision.action == GAME.frame(1).action_back((verb, cell))
    assert legal[decision.action]
    np.testing.assert_array_equal(decision.legal, GAME.frame(1).into(legal).ravel())


def test_reading_a_position_gives_the_observation_and_scores_in_the_players_frame():
    net = small_net()
    position = GAME.step(GAME.start(0), (PLACE, 0))
    legal = GAME.legal(position)
    reading = read(net, GAME, position, legal)
    np.testing.assert_array_equal(reading.obs, GAME.observe(position, 1))
    with torch.no_grad():
        scores, value = net(torch.from_numpy(reading.obs)[None])
    np.testing.assert_array_equal(reading.scores, scores[0].flatten().double().numpy())
    assert reading.value == float(value[0])


def test_a_net_evaluator_gives_log_priors_on_the_board_and_player_zeros_values():
    net = small_net()
    positions = [GAME.start(0), GAME.step(GAME.start(0), (PLACE, 0))]
    legal = [GAME.legal(p) for p in positions]
    logits, values = NetEvaluator(net)(GAME, positions, legal)
    players, obs = observations(GAME, positions)
    assert players == [0, 1]
    with torch.no_grad():
        scores, value = net(torch.from_numpy(obs).float() / 255)
    want_logits, want_values = judgements(
        GAME, scores.double().numpy(), value.double().numpy(), legal, players
    )
    for got, want, mask in zip(logits, want_logits, legal, strict=True):
        assert got.shape == (mask.sum(),)
        assert np.exp(got).sum() == pytest.approx(1.0)
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(values, want_values)
    assert values[1] == pytest.approx((1 - float(value[1])) / 2)  # player 1's value, turned


def test_a_net_evaluator_guides_self_play_on_squares():
    player = AlphaZeroPlayer(NetEvaluator(small_net()), simulations=4, noise=True)
    experience, played = self_play(GAME, player, [0, 1], keep=1.0)
    assert len(experience) == sum(p.kept for p in played) > 0
    assert all(p.final.over for p in played)


def test_unpack_reads_bits_packed_by_numpy(rng):
    bits = rng.random((3, 21)) < 0.5
    packed = torch.from_numpy(np.packbits(bits, axis=1))
    np.testing.assert_array_equal(unpack(packed, 21).numpy(), bits)
