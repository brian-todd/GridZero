import numpy as np
import pytest
import torch

from examples.games.tactics import (
    Action,
    RandomPlayer,
    Verb,
)
from gridzero.agents.net import (
    NetEvaluator,
    NetPlayer,
    PolicyNet,
    as_input,
    masked_logits,
    observations,
    read,
)
from gridzero.agents.zero import AlphaZeroPlayer
from gridzero.game import play
from tactics_demo import FOOT, SMALL, TINY, piece


def small_net(game=TINY, seed=0):
    torch.manual_seed(seed)
    return PolicyNet(game, channels=8, blocks=2)


def test_the_net_scores_every_action_and_values_in_minus_one_to_one():
    net = small_net()
    obs = torch.rand(5, len(TINY.planes), 81)
    logits, value = net(obs)
    assert logits.shape == (5, Verb.PLACE + len(TINY.units), 81)
    assert value.shape == (5,)
    assert torch.all(value.abs() <= 1)
    assert torch.isfinite(logits).all()


def test_the_first_layer_reads_a_plane_of_ones_to_tell_the_edge_from_empty_tiles():
    net = small_net()
    seen = []
    net.stem.register_forward_hook(lambda module, args, out: seen.append(args[0]))
    obs = torch.rand(2, len(TINY.planes), 81)
    net(obs)
    (x,) = seen
    assert x.shape == (2, len(TINY.planes) + 1, 9, 9)
    torch.testing.assert_close(x[:, :-1].reshape(2, -1, 81), obs)
    assert torch.all(x[:, -1] == 1)


def test_new_blocks_pass_their_input_through():
    torch.manual_seed(2)
    deep = PolicyNet(TINY, channels=8, blocks=3)
    shallow = PolicyNet(TINY, channels=8, blocks=0)
    shallow.load_state_dict(
        {k: v for k, v in deep.state_dict().items() if not k.startswith("body.")}
    )
    obs = torch.rand(3, len(TINY.planes), 81)
    for a, b in zip(deep(obs), shallow(obs), strict=True):
        torch.testing.assert_close(a, b)


def test_net_sizes_must_be_integers_in_range(not_an_integer):
    with pytest.raises(TypeError, match="channels"):
        PolicyNet(TINY, channels=not_an_integer)
    with pytest.raises(ValueError, match="channels"):
        PolicyNet(TINY, channels=0)
    with pytest.raises(TypeError, match="blocks"):
        PolicyNet(TINY, blocks=not_an_integer)
    with pytest.raises(ValueError, match="blocks"):
        PolicyNet(TINY, blocks=-1)


def test_masked_logits_keep_legal_scores_and_rule_out_the_rest():
    logits = torch.arange(12.0).reshape(2, 3, 2)
    legal = torch.tensor([[[True, False], [False, False], [False, True]]] * 2)
    out = masked_logits(logits, legal)
    assert out.shape == (2, 6)
    assert out[0].tolist() == [0.0, -np.inf, -np.inf, -np.inf, -np.inf, 5.0]
    assert out[1, 0] == 6.0


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_a_net_player_plays_whole_games_on_either_side(temperature):
    player = NetPlayer(small_net(), temperature=temperature)
    for players in ((player, RandomPlayer()), (RandomPlayer(), player)):
        record = play(TINY, players, TINY.start(3))  # raises on an illegal action
        assert record.final.over


def test_a_net_player_replays_the_same_game_from_the_same_seed():
    player = NetPlayer(small_net(), temperature=1.0)
    first = play(TINY, (player, player), TINY.start(5))
    second = play(TINY, (player, player), TINY.start(5))
    assert first.events == second.events


def test_a_net_player_plays_its_best_action_mapped_back_to_the_board(make_battle, at):
    net = PolicyNet(SMALL, channels=4, blocks=0)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.policy_bias.bias[Verb.FACE + 2] = 1.0  # face direction 2 in the player's frame
    player = NetPlayer(net)
    for team, facing in ((0, 2), (1, 5)):
        pieces = [piece(1, 1, FOOT, team, facing=0), piece(6, 6, FOOT, 1 - team, facing=0)]
        position = make_battle(pieces)
        action = player(
            SMALL, SMALL.player(position), position, SMALL.legal(position), np.random.default_rng(0)
        )
        assert action == Action(Verb.FACE + facing, at(1, 1))


def test_a_net_player_breaks_ties_at_random():
    net = PolicyNet(TINY, channels=4, blocks=0)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    player = NetPlayer(net)
    position = TINY.start(0)
    legal = TINY.legal(position)
    picks = {
        player(TINY, TINY.player(position), position, legal, np.random.default_rng(s))
        for s in range(20)
    }
    assert len(picks) > 1
    assert all(legal[verb, cell] for verb, cell in picks)


def test_a_net_player_reads_its_own_teams_frame():
    net = small_net()
    player = NetPlayer(net)
    position = TINY.start(2)
    position = TINY.step(
        position,
        player(
            TINY, TINY.player(position), position, TINY.legal(position), np.random.default_rng(0)
        ),
    )
    assert TINY.player(position) == 1
    legal = TINY.legal(position)
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(TINY.observe(position, 1))[None])
    mask = torch.from_numpy(TINY.frame(1).into(legal))[None]
    verb, cell = divmod(int(masked_logits(logits, mask).argmax()), 81)
    got = player(TINY, TINY.player(position), position, legal, np.random.default_rng(0))
    assert got == TINY.frame(1).action_back(Action(verb, cell))


def test_a_net_players_temperature_is_a_real_number_of_at_least_zero(not_a_real):
    with pytest.raises(ValueError, match="temperature"):
        NetPlayer(small_net(), temperature=-0.5)
    with pytest.raises((TypeError, ValueError), match="temperature"):
        NetPlayer(small_net(), temperature=not_a_real)


def test_a_decision_keeps_its_log_probability_and_value():
    net = small_net()
    position = TINY.start(4)
    legal = TINY.legal(position)
    decision = NetPlayer(net, temperature=1.0).decide(
        TINY, position, legal, np.random.default_rng(1)
    )
    team = TINY.player(position)
    with torch.no_grad():
        logits, value = net(torch.from_numpy(decision.obs)[None])
    mask = torch.from_numpy(TINY.frame(team).into(legal))[None]
    logp = torch.log_softmax(masked_logits(logits, mask).double(), dim=1)[0]
    assert decision.logp == pytest.approx(float(logp[decision.index]), abs=1e-6)
    assert decision.value == pytest.approx(float(value[0]))
    assert decision.player == team
    np.testing.assert_array_equal(decision.legal, TINY.frame(team).into(legal).ravel())
    verb, cell = divmod(decision.index, 81)
    assert decision.action == TINY.frame(team).action_back(Action(verb, cell))


def test_a_best_choice_has_the_log_probability_of_its_ties():
    net = PolicyNet(TINY, channels=4, blocks=0)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    position = TINY.start(0)
    legal = TINY.legal(position)
    decision = NetPlayer(net).decide(TINY, position, legal, np.random.default_rng(0))
    assert decision.logp == pytest.approx(-np.log(legal.sum()))


class Bytes(PolicyNet):
    """A network that rounds its input to bytes first, as observations are stored."""

    def forward(self, obs):
        return super().forward(torch.round(obs * 255) / 255)


def alone(net, position, legal):
    """Return the policy on the board and team 0's value in [0, 1], of one position alone."""
    team, _, mask, logits, value = read(net, TINY, position, legal)
    probs = np.exp(np.where(mask, logits - logits[mask].max(), -np.inf))
    probs /= probs.sum()
    board = TINY.frame(team).back(probs.reshape(legal.shape))
    return board, (1.0 + (value if team == 0 else -value)) / 2.0


def test_a_net_evaluator_judges_a_batch_as_each_position_is_judged_alone_in_bytes():
    net = small_net()
    rounded = Bytes(TINY, channels=8, blocks=2)
    rounded.load_state_dict(net.state_dict())
    rng = np.random.default_rng(1)
    positions = []
    position = TINY.start(7)
    while not position.over and len(positions) < 40:
        positions.append(position)
        position = TINY.step(
            position,
            RandomPlayer()(TINY, TINY.player(position), position, TINY.legal(position), rng),
        )
    legal = [TINY.legal(p) for p in positions]
    logits, values = NetEvaluator(net)(TINY, positions, legal)
    assert {TINY.player(p) for p in positions} == {0, 1}
    assert any(p.deploying for p in positions)
    assert not all(p.deploying for p in positions)
    assert values.shape == (len(positions),)
    for p, mask, row, value in zip(positions, legal, logits, values, strict=True):
        probs, expected = alone(rounded, p, mask)
        assert row.shape == (np.count_nonzero(mask),)
        np.testing.assert_allclose(np.exp(row), probs[mask], rtol=1e-5, atol=1e-7)
        assert value == pytest.approx(expected, abs=1e-6)


def test_a_net_evaluator_sees_observations_as_they_are_stored():
    positions = [TINY.start(3), TINY.start(4)]
    teams, obs = observations(TINY, positions)
    assert obs.dtype == np.uint8
    for p, t, o in zip(positions, teams, obs, strict=True):
        assert t == TINY.player(p)
        np.testing.assert_array_equal(o, np.rint(TINY.observe(p, t) * 255))
    np.testing.assert_array_equal(as_input(torch.from_numpy(obs)).numpy(), obs / np.float32(255))


def test_a_net_evaluator_guides_an_alphazero_player_through_whole_games():
    player = AlphaZeroPlayer(NetEvaluator(small_net()), simulations=4, considered=4)
    for players in ((player, RandomPlayer()), (RandomPlayer(), player)):
        assert play(TINY, players, TINY.start(1)).final.over
