import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from agents_demo import Lowest, Patches
from gridzero.agents import UniformPlayer
from gridzero.agents.muzero import LatentSearch, MuZeroPlayer, Trajectories, self_play, trajectories
from gridzero.game import Action, play, replay, series

GAME = Patches()


@pytest.fixture(scope="module")
def records():
    return series(GAME, [("uniform", UniformPlayer()), ("lowest", Lowest())], 3)


@pytest.fixture(scope="module")
def data(records):
    return trajectories(GAME, records)


def test_each_player_of_each_game_has_a_trajectory_of_its_decisions_in_order(records, data):
    assert data.n_trajectories == 2 * len(records)
    lengths = np.diff(data.bounds)
    for i, record in enumerate(records):
        plies = len(record.actions)
        assert lengths[2 * i] == (plies + 1) // 2  # player 0 moves first
        assert lengths[2 * i + 1] == plies // 2
    np.testing.assert_array_equal(data.round, np.concatenate([np.arange(n) for n in lengths]))


def test_rows_hold_each_decisions_observation_reveal_legal_actions_and_action(records, data):
    record = records[1]
    positions = list(replay(GAME, record))
    row = data.bounds[2 + 1]  # player 1's first decision in the second game
    position = positions[1]
    frame = GAME.frame(1)
    obs = np.rint(GAME.observe(position, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(data.obs[row], obs)
    np.testing.assert_array_equal(data.revealed[row], obs)  # Patches reveals its observation
    legal = frame.into(GAME.legal(position)).ravel()
    np.testing.assert_array_equal(np.unpackbits(data.legal[row])[: legal.size], legal)
    verb, cell = frame.action_into(record.actions[1])
    assert data.action[row] == verb * GAME.space.n_cells + cell


def test_the_policy_is_all_on_the_action_taken_among_the_legal_ones(data):
    n_bits = GAME.n_verbs * GAME.space.n_cells
    for row in range(len(data)):
        legal = np.flatnonzero(np.unpackbits(data.legal[row])[:n_bits])
        policy = data.policy[data.start[row] : data.start[row + 1]]
        assert policy.size == legal.size
        np.testing.assert_array_equal(legal[policy == 1.0], [data.action[row]])
        assert policy.sum() == 1.0


def test_the_outcome_is_each_players_result(records, data):
    for i, record in enumerate(records):
        for p in (0, 1):
            rows = slice(data.bounds[2 * i + p], data.bounds[2 * i + p + 1])
            want = (
                0.0 if record.final.winner is None else (1.0 if record.final.winner == p else -1.0)
            )
            assert (data.outcome[rows] == want).all()


def test_remaining_counts_the_decisions_left_in_each_trajectory(data):
    remaining = data.remaining()
    ends = data.bounds[1:] - 1
    assert (remaining[ends] == 0).all()
    np.testing.assert_array_equal(remaining[data.bounds[:-1]], np.diff(data.bounds) - 1)


def test_joined_trajectories_are_those_of_all_the_games_in_order(records, data):
    joined = Trajectories.join([trajectories(GAME, records[:2]), trajectories(GAME, records[2:])])
    for name in ("obs", "revealed", "legal", "action", "policy", "start", "outcome", "round"):
        np.testing.assert_array_equal(getattr(joined, name), getattr(data, name))
    np.testing.assert_array_equal(joined.bounds, data.bounds)


def test_trajectories_need_a_record():
    with pytest.raises(ValueError, match="at least one record"):
        trajectories(GAME, [])


@dataclass(frozen=True)
class Oracle:
    """A model whose latent state is the actions taken: winning means first taking ``target``."""

    target: int
    n_actions: int = GAME.n_verbs * GAME.space.n_cells
    prior: tuple[float, ...] = ()

    def _judged(self, latents):
        logits = np.zeros((len(latents), self.n_actions))
        if self.prior:
            logits[:] = self.prior
        values = np.array(
            [0.0 if not path else (1.0 if path[0] == self.target else -1.0) for path in latents]
        )
        return list(latents), logits, values

    def initial(self, obs):
        return self._judged([() for _ in obs])

    def recurrent(self, latents, actions):
        return self._judged([(*path, int(a)) for path, a in zip(latents, actions, strict=True)])


def root(game=GAME):
    position = game.start(0)
    return game.observe(position, 0), game.legal(position)


def test_the_search_finds_the_action_its_model_says_wins():
    obs, legal = root()
    player = MuZeroPlayer(Oracle(target=7), simulations=48)
    result = player.search(obs, legal, np.random.default_rng(0))
    assert result.action == Action(0, 7)
    assert result.policy.flat[7] == result.policy.max()
    assert result.policy.sum() == pytest.approx(1.0)
    assert result.visits.sum() == 48
    assert result.q.flat[7] == 1.0
    assert np.isnan(result.q[~legal]).all()


def test_without_simulations_the_player_plays_its_priors_best():
    obs, legal = root()
    prior = np.zeros(legal.size)
    prior[13] = 5.0
    player = MuZeroPlayer(Oracle(target=7, prior=tuple(prior)), simulations=0)
    result = player.search(obs, legal, np.random.default_rng(0))
    assert result.action == Action(1, 1)  # flat 13 = verb 1, cell 1 of 12
    assert result.visits.sum() == 0


def test_below_the_root_the_search_tries_actions_illegal_at_the_root():
    obs, legal = root()
    legal[:, 5:] = False
    model = Oracle(target=0)
    player = MuZeroPlayer(model, simulations=60, considered=2)
    search = LatentSearch(player, obs, legal, np.random.default_rng(1))
    deeper = set()
    while (leaf := search.leaf()) is not None:
        if leaf.parent is None:
            latents, logits, values = model.initial([leaf.obs])
        else:
            action = leaf.parent.actions[leaf.index]
            if leaf.parent is not search.root:
                deeper.add(int(action))
            latents, logits, values = model.recurrent([leaf.parent.latent], np.array([action]))
        search.expand(leaf, latents[0], logits[0], float(values[0]))
    assert set(np.flatnonzero(search.result().visits)) <= set(np.flatnonzero(legal))
    assert deeper - set(np.flatnonzero(legal).tolist())  # the player can't know what will be legal


def test_the_same_generator_searches_the_same_way_and_noise_changes_the_choice():
    obs, legal = root()
    noisy = MuZeroPlayer(Oracle(target=7), simulations=4, considered=4, noise=True)
    a = noisy.search(obs, legal, np.random.default_rng(3))
    b = noisy.search(obs, legal, np.random.default_rng(3))
    assert a.action == b.action
    np.testing.assert_array_equal(a.visits, b.visits)
    chosen = {noisy.search(obs, legal, np.random.default_rng(s)).action for s in range(10)}
    assert len(chosen) > 1


def test_the_player_decides_in_its_frame_and_plays_whole_games():
    """Player 1 sees the board turned half a turn: its winning cell is the board's opposite."""
    player = MuZeroPlayer(Oracle(target=0), simulations=24)
    position = GAME.step(GAME.start(0), Action(0, 5))
    legal = GAME.legal(position)
    action = player(GAME, 1, position, legal, np.random.default_rng(0))
    assert action == GAME.frame(1).action_back(Action(0, 0))
    record = play(GAME, [player, UniformPlayer()], GAME.start(1))
    assert record.final.over


def test_a_single_legal_action_is_played_without_asking_the_model():
    class Silent:
        def initial(self, obs):
            raise AssertionError("asked")

        def recurrent(self, latents, actions):
            raise AssertionError("asked")

    position = GAME.start(0)
    legal = np.zeros((2, 12), dtype=np.bool_)
    legal[1, 4] = True
    action = MuZeroPlayer(Silent())(GAME, 0, position, legal, np.random.default_rng(0))
    assert action == Action(1, 4)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"logits": np.zeros(5)}, "scores for"),
        ({"logits": np.full(24, np.nan)}, "not finite"),
        ({"value": 1.5}, "outside"),
    ],
)
def test_the_search_refuses_bad_answers_from_its_model(answer, message):
    obs, legal = root()
    player = MuZeroPlayer(Oracle(target=0))
    search = LatentSearch(player, obs, legal, np.random.default_rng(0))
    leaf = search.leaf()
    assert leaf is not None
    with pytest.raises(ValueError, match=message):
        search.expand(leaf, (), answer.get("logits", np.zeros(24)), answer.get("value", 0.0))


def test_the_search_needs_a_legal_action():
    obs, legal = root()
    with pytest.raises(ValueError, match="legal action"):
        LatentSearch(MuZeroPlayer(Oracle(target=0)), obs, legal & False, np.random.default_rng(0))


@pytest.mark.parametrize(
    "settings",
    [
        {"simulations": -1},
        {"considered": 0},
        {"noise": 1},
        {"c_visit": -1.0},
        {"c_scale": "x"},
    ],
)
def test_the_player_refuses_bad_settings(settings):
    with pytest.raises((TypeError, ValueError)):
        MuZeroPlayer(Oracle(target=0), **settings)


def test_since_keeps_the_later_trajectories_renumbered(data):
    later = data.since(3)
    assert later.n_trajectories == data.n_trajectories - 3
    row = data.bounds[3]
    np.testing.assert_array_equal(later.action, data.action[row:])
    np.testing.assert_array_equal(later.bounds, data.bounds[3:] - row)
    for i in range(len(later)):
        np.testing.assert_array_equal(
            later.policy[later.start[i] : later.start[i + 1]],
            data.policy[data.start[row + i] : data.start[row + i + 1]],
        )
    assert len(data.since(data.n_trajectories)) == 0
    with pytest.raises(ValueError, match="at most"):
        data.since(data.n_trajectories + 1)


def test_self_play_keeps_every_decision_of_every_player_with_the_searchs_policy():
    player = MuZeroPlayer(Oracle(target=0), simulations=8, noise=True)
    data, played = self_play(GAME, player, [3, 4])
    assert [p.seed for p in played] == [3, 4]
    assert data.n_trajectories == 4
    assert len(data) == sum(p.plies for p in played) == sum(p.kept for p in played)
    n_bits = GAME.n_verbs * GAME.space.n_cells
    for row in range(len(data)):
        legal = np.unpackbits(data.legal[row])[:n_bits].astype(bool)
        policy = data.policy[data.start[row] : data.start[row + 1]]
        assert policy.size == legal.sum()
        assert policy.sum() == pytest.approx(1.0, abs=1e-6)
        assert legal[data.action[row]]
    for p, final in zip((0, 1), (played[0].final, played[0].final), strict=True):
        rows = slice(data.bounds[p], data.bounds[p + 1])
        want = 0.0 if final.winner is None else (1.0 if final.winner == p else -1.0)
        assert (data.outcome[rows] == want).all()


def test_self_play_replays_each_game_alike_whatever_runs_beside_it():
    player = MuZeroPlayer(Oracle(target=0), simulations=8, noise=True)
    alone, _ = self_play(GAME, player, [4])
    together, _ = self_play(GAME, player, [3, 4])
    later = together.since(2)
    for name in ("obs", "action", "policy", "outcome"):
        np.testing.assert_array_equal(getattr(alone, name), getattr(later, name))


def test_self_play_needs_a_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        self_play(GAME, MuZeroPlayer(Oracle(target=0)), [])


def test_values_in_the_tree_are_the_deciding_players_mapped_into_zero_to_one():
    obs, legal = root()
    result = MuZeroPlayer(Oracle(target=7), simulations=48).search(
        obs, legal, np.random.default_rng(0)
    )
    visited = result.visits > 0
    assert visited.sum() > 1
    np.testing.assert_array_equal(result.q[visited & (np.arange(24).reshape(2, 12) != 7)], 0.0)
    assert 0.0 < result.value < 1.0  # the root's own 0.5 and wins and losses averaged


def test_the_improved_policy_moves_toward_the_actions_the_search_found_good():
    obs, legal = root()
    result = MuZeroPlayer(Oracle(target=7), simulations=48).search(
        obs, legal, np.random.default_rng(0)
    )
    assert result.policy.flat[7] > 4 / legal.sum()  # the prior is uniform


def test_the_search_plays_only_among_the_actions_halving_kept():
    """Action 1 is never considered, though its completed value would outscore action 0's."""
    obs, legal = root()
    prior = np.full(legal.size, -50.0)
    prior[0], prior[1] = 3.0, 0.0
    player = MuZeroPlayer(Oracle(target=1, prior=tuple(prior)), simulations=1, considered=1)
    result = player.search(obs, legal, np.random.default_rng(0))
    assert result.visits.flat[0] == 1
    assert result.action == Action(0, 0)


def test_below_the_root_the_search_visits_what_falls_furthest_short_of_its_policy():
    obs, legal = root()
    player = MuZeroPlayer(Oracle(target=0))
    search = LatentSearch(player, obs, legal, np.random.default_rng(0))
    leaf = search.leaf()
    assert leaf is not None
    search.expand(leaf, (), np.zeros(legal.size), 0.0)
    node = search.root
    assert node is not None
    node.n[:] = 3
    node.n[5] = 0  # the prior is uniform: the one action with no visits falls furthest short
    node.w[:] = 1.5
    assert search._below(node) == 5


def test_the_model_reads_the_observation_rounded_to_bytes_as_stored_for_training():
    seen = []

    @dataclass(frozen=True)
    class Recording(Oracle):
        def initial(self, obs):
            seen.extend(obs)
            return super().initial(obs)

    obs, legal = root()
    obs = obs + np.float32(0.3 / 255)  # between two byte values
    MuZeroPlayer(Recording(target=0), simulations=2).search(obs, legal, np.random.default_rng(0))
    np.testing.assert_array_equal(seen[0], (np.rint(obs * 255) / 255).astype(np.float32))


def test_only_the_root_stretches_a_vanishing_gap_between_tried_and_untried_actions():
    obs, legal = root()
    player = MuZeroPlayer(Oracle(target=0))
    search = LatentSearch(player, obs, legal, np.random.default_rng(0))
    leaf = search.leaf()
    assert leaf is not None
    search.expand(leaf, (), np.zeros(legal.size), -0.2)  # 0.4 in the tree
    node = search.root
    assert node is not None
    node.n[0] = 1000
    node.w[0] = 600.0
    below = dataclasses.replace(node)  # the same statistics at a node that isn't the root
    # the whole scale: (c_visit + most visits) * c_scale
    assert search._sigma(node)[0] - search._sigma(node)[1] == pytest.approx(1050 * player.c_scale)
    assert search._sigma(below)[0] - search._sigma(below)[1] < 0.03
