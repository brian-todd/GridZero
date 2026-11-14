"""Tests of Colonies' learned players: weights specs."""

import pytest
import torch

from examples.games.colonies.learned import BLOCKS, CHANNELS, network, player
from gridzero.agents.muzero_net import NetModel


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "net.pt"
    torch.manual_seed(0)
    torch.save(network().state_dict(), path)
    return str(path)


def test_a_new_network_has_the_shared_size():
    net = network()
    assert (net.channels, net.blocks) == (CHANNELS, BLOCKS)


def test_a_bare_path_is_the_file_searched_without_noise(weights):
    name, searcher = player(weights, 12)
    assert name == "net"
    assert isinstance(searcher.model, NetModel)
    assert (searcher.simulations, searcher.noise) == (12, False)


def test_a_spec_sets_the_simulations(weights):
    name, searcher = player(f"{weights}@64", 12)
    assert name == "net@64"
    assert searcher.simulations == 64
    loaded = searcher.model.net.state_dict()  # ty: ignore[unresolved-attribute]  # NetModel's net
    for key, value in network(weights).state_dict().items():
        torch.testing.assert_close(loaded[key], value)


def test_anything_but_a_number_after_the_path_is_refused(weights):
    with pytest.raises(ValueError, match="a number of simulations"):
        player(f"{weights}@root")


def test_weights_saved_with_the_retired_projection_load_and_play_the_same(weights, tmp_path):
    saved = torch.load(weights)
    saved["projection.0.weight"] = torch.zeros(CHANNELS, CHANNELS, 1)
    saved["projection_head.2.bias"] = torch.zeros(CHANNELS)
    path = tmp_path / "older.pt"
    torch.save(saved, path)
    loaded = network(str(path)).state_dict()
    assert loaded.keys() == network(weights).state_dict().keys()
    for key, value in loaded.items():
        torch.testing.assert_close(value, saved[key])


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_weights_that_dont_fit_are_refused(weights, tmp_path, change):
    saved = torch.load(weights)
    if change == "missing":
        del saved["stem.own.weight"]
    else:
        saved["encoder_out.weight"] = torch.zeros(6, CHANNELS)  # a network with chance codes
    path = tmp_path / "broken.pt"
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="don't fit"):
        network(str(path))
