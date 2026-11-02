"""Tests of Colonies' learned players: weights specs."""

import pytest
import torch

from examples.games.colonies.learned import BLOCKS, CHANNELS, network, player
from gridzero.agents.muzero_net import NetModel, StochasticNetModel


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "net.pt"
    torch.manual_seed(0)
    torch.save(network().state_dict(), path)
    return str(path)


def test_a_new_network_has_the_shared_size():
    net = network()
    assert (net.channels, net.blocks, net.codes) == (CHANNELS, BLOCKS, 0)


def test_weights_without_codes_play_through_plain_dynamics(weights):
    assert isinstance(player(weights)[1].model, NetModel)


def test_a_bare_path_is_searched_below_the_root_over_every_action(weights):
    name, searcher = player(weights, 12)
    assert name == "net"
    assert (searcher.simulations, searcher.below, searcher.noise) == (12, "all", False)


def test_tags_set_the_old_way_and_the_simulations(weights):
    name, searcher = player(f"{weights}@root@64", 12)
    assert name == "net@root@64"
    assert (searcher.simulations, searcher.below) == (64, "root")
    loaded = searcher.model.net.state_dict()  # ty: ignore[unresolved-attribute]  # NetModel's net
    for key, value in network(weights).state_dict().items():
        torch.testing.assert_close(loaded[key], value)


def test_an_unknown_tag_is_refused(weights):
    with pytest.raises(ValueError, match="a tag is root or a number"):
        player(f"{weights}@fast")


def test_weights_saved_before_the_projection_load_and_play_the_same(weights, tmp_path):
    old = {k: v for k, v in torch.load(weights).items() if not k.startswith("projection")}
    path = tmp_path / "old.pt"
    torch.save(old, path)
    loaded = network(str(path)).state_dict()
    for key, value in old.items():
        torch.testing.assert_close(loaded[key], value)


def test_weights_missing_more_than_the_projection_are_refused(weights, tmp_path):
    saved = torch.load(weights)
    del saved["stem.own.weight"]
    path = tmp_path / "broken.pt"
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="don't fit"):
        network(str(path))


def test_weights_with_codes_load_with_them_and_play_through_chance_nodes(tmp_path):
    torch.manual_seed(0)
    saved = network(codes=6)
    path = tmp_path / "chance.pt"
    torch.save(saved.state_dict(), path)
    loaded = network(str(path))
    assert loaded.codes == 6
    for key, value in saved.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    _, searcher = player(str(path))
    assert isinstance(searcher.model, StochasticNetModel)
