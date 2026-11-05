import dataclasses

import numpy as np
import pytest

from agents_demo import GAMBLE, PLACE, Patches, PatchesPosition, PatchesState
from gridzero.agents import AlphaZeroPlayer, UniformPlayer, self_play
from gridzero.agents.zero import sigma
from gridzero.game import series, tally

GAME = Patches()


@dataclasses.dataclass
class Uniform:
    """An evaluator with a uniform prior and even values, which keeps the positions it judged."""

    calls: list[PatchesPosition] = dataclasses.field(default_factory=list)

    def __call__(self, game, positions, legal):
        self.calls += positions
        return [np.zeros(int(m.sum())) for m in legal], np.full(len(positions), 0.5)


def last_move():
    """Player 1 to move on the last free cell, 11: placing it wins, gambling wins half the time.

    The board is ``0110 / 1001 / 101.`` by rows of 4: with cell 11, player 1's groups tie
    player 0's (3 each) and it has more stones; without it, player 0's group of 3 beats 2.
    """
    owner = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, -1])
    start = GAME.start(0)
    state = start.state.replace({PatchesState.owner: owner})
    return dataclasses.replace(start, state=state, ply=11)


def test_the_search_places_rather_than_gambles_on_a_won_last_move():
    result = AlphaZeroPlayer(Uniform(), simulations=16, considered=2).search(
        GAME, last_move(), np.random.default_rng(0)
    )
    assert result.action == (PLACE, 11)
    assert result.q[PLACE, 11] == 0.0  # player 0's value: player 1 wins
    assert 0.0 < result.q[GAMBLE, 11] < 1.0  # the coin went both ways
    assert result.policy[PLACE, 11] > result.policy[GAMBLE, 11]


@dataclasses.dataclass(frozen=True)
class Favor:
    """An evaluator with even values and a prior that favors gambling by ``logit``."""

    logit: float

    def __call__(self, game, positions, legal):
        logits = [np.where(np.nonzero(m)[0] == GAMBLE, self.logit, 0.0) for m in legal]
        return logits, np.full(len(positions), 0.5)


def test_an_extra_visit_in_a_round_cut_short_doesnt_choose_the_action():
    # 3 simulations of 2 actions: a round of one visit each, then the gamble's second visit
    # (its logit 1.5 beats the placement's 0 plus sigma, scaled by 1 visit at most), and the
    # round ends there. Scaled by the gamble's 2 visits, the placement's win now beats it.
    player = AlphaZeroPlayer(Favor(1.5), simulations=3, considered=2, c_visit=0.0, c_scale=1.0)
    result = player.search(GAME, last_move(), np.random.default_rng(0))
    assert result.visits[GAMBLE, 11] == 2
    assert result.visits[PLACE, 11] == 1
    assert result.q[GAMBLE, 11] > 0.0  # player 0's value: player 1 lost a coin
    assert result.action == (PLACE, 11)


@dataclasses.dataclass(frozen=True)
class Lure:
    """Player 0 to move on cells 0-5: placing on 0 or 1 looks best and loses, 2 wins.

    At the root the prior favors placing on 0 and 1 by ``logit`` over 2, and the other cells
    and gambles not at all; below it, placing on 3-5 only. Player 0's values: 1 in the subtree
    of placing on 2, 0.8 one ply below placing on 0 or 1 and 0 deeper, 0 elsewhere.
    """

    logit: float = 3.0
    ply: int = 6

    def position(self):
        owner = np.array([-1] * 6 + [0, 1] * 3)
        start = GAME.start(0)
        state = start.state.replace({PatchesState.owner: owner})
        return dataclasses.replace(start, state=state, ply=self.ply)

    def __call__(self, game, positions, legal):
        logits, values = [], []
        for position, mask in zip(positions, legal, strict=True):
            verbs, cells = np.nonzero(mask)
            if position.ply == self.ply:
                prior = np.select([cells < 2, cells == 2], [self.logit, 0.0], -100.0)
                logits.append(np.where(verbs == GAMBLE, -1000.0, prior))
            else:
                logits.append(np.where((verbs == PLACE) & (cells >= 3), 0.0, -100.0))
            lured = position.state.owner[:2] == 0
            if position.state.owner[2] == 0 and not lured.any():
                values.append(1.0)
            elif position.ply == self.ply + 1 and lured.any():
                values.append(0.8)
            else:
                values.append(0.0)
        return logits, np.array(values)


@pytest.mark.parametrize("seed", range(5))
def test_after_a_whole_round_actions_it_dropped_are_not_chosen(seed):
    # 12 legal actions, fewer than the 16 considered: the rounds visit 12 actions, 6, 3, then
    # two at a time. The 23rd simulation ends the first round of two (cells 0 and 1, 4 visits
    # each), where cell 2's win, dropped with 3 visits, would now outscore them.
    lure = Lure()
    player = AlphaZeroPlayer(lure, simulations=23, c_visit=0.0, c_scale=1.0)
    result = player.search(GAME, lure.position(), np.random.default_rng(seed))
    assert result.visits[PLACE, :3].tolist() == [4, 4, 3]
    assert result.policy[PLACE, 2] > result.policy[PLACE, :2].max()
    assert result.action.verb == PLACE
    assert result.action.cell in (0, 1)


def test_results_are_distributions_over_legal_actions():
    position = GAME.step(GAME.start(3), (PLACE, 5))
    legal = GAME.legal(position)
    result = AlphaZeroPlayer(Uniform(), simulations=12).search(
        GAME, position, np.random.default_rng(1)
    )
    assert result.policy.shape == result.visits.shape == (2, 12)
    assert result.policy.sum() == pytest.approx(1.0)
    assert (result.policy[~legal] == 0).all()
    assert result.visits.sum() == 12
    assert np.isnan(result.q[~legal]).all()
    assert legal[result.action]


def test_children_are_judged_once_and_chance_outcomes_once_each():
    # The tree doesn't merge transpositions, but one ply below the root every position is a
    # child of its own: kept after a placement, one per coin outcome after a gamble.
    evaluator = Uniform()
    position = GAME.step(GAME.start(3), (PLACE, 5))
    AlphaZeroPlayer(evaluator, simulations=60, considered=4).search(
        GAME, position, np.random.default_rng(2)
    )
    children = [p.key() for p in evaluator.calls if p.ply == position.ply + 1]
    assert len(children) == len(set(children)) > 4


@dataclasses.dataclass(frozen=True)
class Asked(Patches):
    """`Patches` that keeps the positions it was asked for the legal actions of."""

    asked: list[PatchesPosition] = dataclasses.field(default_factory=list, compare=False)

    def legal(self, position):
        self.asked.append(position)
        return super().legal(position)


def test_a_player_searches_from_the_legal_actions_it_was_given():
    game, position = Asked(), GAME.step(GAME.start(3), (PLACE, 5))
    player = AlphaZeroPlayer(Uniform(), simulations=12)
    action = player(
        game, game.player(position), position, GAME.legal(position), np.random.default_rng(4)
    )
    assert all(p.key() != position.key() for p in game.asked)  # only positions below it
    assert action == player.search(GAME, position, np.random.default_rng(4)).action


def test_the_search_never_reads_the_games_seed_and_follows_its_generator():
    player = AlphaZeroPlayer(Uniform(), simulations=24, noise=True)
    position = GAME.step(GAME.start(3), (GAMBLE, 5))

    def run(seed, rng):
        result = player.search(GAME, dataclasses.replace(position, seed=seed), rng)
        return result.action, result.policy.tobytes(), result.visits.tobytes()

    assert run(1, np.random.default_rng(7)) == run(99, np.random.default_rng(7))
    assert len({run(1, np.random.default_rng(s))[2] for s in range(5)}) > 1


def test_the_search_needs_two_players_and_a_game_not_over():
    class Three(Patches):
        @property
        def n_players(self):
            return 3

    player = AlphaZeroPlayer(Uniform(), simulations=2)
    with pytest.raises(ValueError, match="two players"):
        player.search(Three(), Three().start(0), np.random.default_rng(0))
    over = GAME.step(last_move(), (PLACE, 11))
    with pytest.raises(ValueError, match="not over"):
        player.search(GAME, over, np.random.default_rng(0))


@dataclasses.dataclass(frozen=True)
class Rollout:
    """An evaluator with a uniform prior, valuing a position by one uniform playout from it."""

    def __call__(self, game, positions, legal):
        values = []
        for position in positions:
            rng = np.random.default_rng(list(position.key()))
            while not position.over:
                legal_now = game.legal(position)
                position = game.step(
                    position, UniformPlayer()(game, game.player(position), position, legal_now, rng)
                )
            values.append(0.5 if position.winner is None else float(position.winner == 0))
        return [np.zeros(int(m.sum())) for m in legal], np.array(values)


@pytest.mark.slow
def test_the_search_beats_a_uniform_player_from_either_seat():
    # Patches is noisy (uniform players split 200-172-28 over 400 games), and the search wins
    # from both seats: 150-41-9 at 512 simulations, 135-53-12 at 128 (2026-11-05).
    player = AlphaZeroPlayer(Rollout(), simulations=128)
    records = series(GAME, [("search", player), ("uniform", UniformPlayer())], 40, seed=0)
    for seat in (0, 1):
        mine = [r for r in records if r.names[seat] == "search"]
        assert sum(r.winner == "search" for r in mine) > sum(r.winner == "uniform" for r in mine)
    wins = tally(records).wins
    assert wins["search"] >= 2 * wins["uniform"]


def test_self_play_keeps_decisions_in_the_deciding_players_frame():
    player = AlphaZeroPlayer(Uniform(), simulations=8, noise=True)
    experience, played = self_play(GAME, player, range(6), keep=1.0)
    assert len(experience) == sum(p.kept for p in played) > 0
    n_cells = GAME.space.n_cells
    legal = np.unpackbits(experience.legal, axis=1)[:, : 2 * n_cells].reshape(-1, 2, n_cells)
    free = experience.obs[:, GAME.planes.index("free")] == 255
    np.testing.assert_array_equal(legal[:, PLACE], free)  # both in the same frame
    np.testing.assert_array_equal(legal[:, GAMBLE], free)
    lengths = np.diff(experience.start)
    np.testing.assert_array_equal(lengths, legal.reshape(len(experience), -1).sum(axis=1))
    for i in range(len(experience)):
        policy = experience.policy[experience.start[i] : experience.start[i + 1]]
        assert policy.sum() == pytest.approx(1.0, abs=1e-6)
    turned = [p for p in experience.ply if p % 2 == 1]
    assert turned, "player 1 decided too"


def test_self_play_reports_each_game_with_its_last_position():
    player = AlphaZeroPlayer(Uniform(), simulations=4)
    experience, played = self_play(GAME, player, [2, 9], keep=0.0)
    assert len(experience) == 0
    assert experience.obs.shape[1:] == (len(GAME.planes), GAME.space.n_cells)
    assert [p.seed for p in played] == [2, 9]
    for p in played:
        assert p.final.over
        assert (p.plies, p.winner) == (p.final.ply, p.final.winner)
        assert p.kept == 0 < p.searched <= p.plies


def test_sigma_rescales_over_the_node_s_actions_only_when_asked():
    q = np.array([0.2, 0.3, 0.0])
    n = np.array([1, 1, 0], dtype=np.int64)
    logits = np.log(np.array([0.5, 0.25, 0.25]))
    # The unvisited action's value: the raw 0.4 mixed over 2 visits with the prior's mean of q.
    mixed = (0.4 + 2 * (0.5 * 0.2 + 0.25 * 0.3) / 0.75) / 3
    plain = sigma(q, 0.4, n, logits, 10.0, 0.5, rescale=False)
    np.testing.assert_allclose(plain, 5.5 * np.array([0.2, 0.3, mixed]))
    full = sigma(q, 0.4, n, logits, 10.0, 0.5, rescale=True)
    np.testing.assert_allclose(full, 5.5 * (plain / 5.5 - 0.2) / 0.1)


def test_sigma_keeps_a_vanishing_gap_small_below_the_root():
    # One action tried many times: the mix for the others is just under its mean.
    q = np.array([0.6, 0.0, 0.0])
    n = np.array([1000, 0, 0], dtype=np.int64)
    logits = np.log(np.full(3, 1 / 3))
    plain = sigma(q, 0.4, n, logits, 50.0, 0.1, rescale=False)
    assert plain[0] - plain[1] < 0.03
    assert sigma(q, 0.4, n, logits, 50.0, 0.1, rescale=True)[0] > 100.0
