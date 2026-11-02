import dataclasses

import numpy as np
import pytest

from examples.games.tactics import (
    Died,
    Ended,
    GreedyPlayer,
    Placed,
    Plan,
    SearchPlayer,
    Verb,
    material,
)
from examples.games.tactics.search import _Node
from gridzero.core import steps
from gridzero.game import play
from tactics_demo import ARCHER, CAPTAIN, FOOT, SMALL, TINY, mirrored, piece, play_random

QUICK = SearchPlayer(simulations=12, plans=4, depth=3, rollout_turns=0)
"""A small search, without rollouts, for tests of behavior rather than strength."""


def act(player, game, position, seed=0):
    return player(
        game, game.player(position), position, game.legal(position), np.random.default_rng(seed)
    )


def worth(kind, hp=None):
    stats = SMALL.units[kind]
    return (stats.hp + stats.atk) * (stats.hp if hp is None else hp) / stats.hp


def test_material_is_team_0s_share_of_hit_points_and_atk_weighted_by_health(make_battle):
    position = make_battle(
        [piece(1, 1, FOOT, 0), piece(2, 1, CAPTAIN, 0, hp=3), piece(7, 7, ARCHER, 1, hp=2)]
    )
    ours = worth(FOOT) + worth(CAPTAIN, hp=3)
    assert material(SMALL, position) == pytest.approx(ours / (ours + worth(ARCHER, hp=2)))


def test_material_is_even_between_equal_sides_and_turns_over_with_the_board(make_battle):
    even = make_battle([piece(1, 1, FOOT, 0), piece(7, 7, FOOT, 1)])
    assert material(SMALL, even) == 0.5
    uneven = make_battle([piece(1, 1, FOOT, 0, hp=2), piece(7, 7, CAPTAIN, 1)])
    assert material(SMALL, mirrored(SMALL, uneven)) == pytest.approx(1 - material(SMALL, uneven))


def test_search_picks_legal_actions_everywhere():
    rng = np.random.default_rng(2)
    for position, _ in play_random(TINY, 2, rng):
        if position.over or rng.random() > 0.15:
            continue
        verb, cell = act(dataclasses.replace(QUICK), TINY, position, int(rng.integers(100)))
        assert TINY.legal(position)[verb, cell]


def test_search_never_reads_the_games_seed():
    # The seed fixes every future roll: a search that read it would know which attacks hit.
    rng = np.random.default_rng(4)
    checked = 0
    for position, _ in play_random(TINY, 4, rng):
        if position.over or position.deploying or rng.random() > 0.2:
            continue
        plans = {
            QUICK.search(TINY, dataclasses.replace(position, seed=seed), np.random.default_rng(7))
            for seed in (1, 2, 3)
        }
        assert len(plans) == 1, position.ply
        checked += 1
    assert checked >= 10


def test_search_with_the_same_generator_gives_the_same_plan():
    position = play(TINY, [GreedyPlayer(), GreedyPlayer()], TINY.start(3)).final
    rng = np.random.default_rng(3)
    for battle, _ in play_random(TINY, 3, rng):
        if not battle.over and not battle.deploying:
            position = battle
            break
    first = QUICK.search(TINY, position, np.random.default_rng(5))
    assert QUICK.search(TINY, position, np.random.default_rng(5)) == first


def test_candidates_are_the_greedy_scores_best_first(make_battle, at):
    position = make_battle([piece(1, 4, FOOT, 0), piece(7, 4, FOOT, 1)])
    player = SearchPlayer(plans=5)
    found = player.candidates(SMALL, position, SMALL.legal(position), np.random.default_rng(0))
    assert len(found) == 5
    scores = [score for _, score in found]
    assert scores == sorted(scores, reverse=True)
    greedy = act(GreedyPlayer(), SMALL, position)
    assert found[0][0] == Plan(greedy.cell, False)


def test_candidates_attack_first_only_when_moving_after(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(5, 4, FOOT, 1)])
    found = SearchPlayer(plans=100).candidates(
        SMALL, position, SMALL.legal(position), np.random.default_rng(0)
    )
    assert any(plan.attack_first for plan, _ in found)
    assert Plan(at(4, 4), True) not in [plan for plan, _ in found]


def battles(game, seed, share):
    """Yield some positions in battle from a random game."""
    rng = np.random.default_rng(seed)
    for position, _ in play_random(game, seed, rng):
        if not position.over and not position.deploying and rng.random() < share:
            yield position


def test_candidates_score_a_mirrored_board_the_same():
    for position in battles(TINY, 6, 0.3):
        turned = mirrored(TINY, position)
        m = position.state.space.opposite()
        rng = np.random.default_rng(0)
        here = QUICK.candidates(TINY, position, TINY.legal(position), rng)
        there = QUICK.candidates(TINY, turned, TINY.legal(turned), rng)
        assert [s for _, s in here] == pytest.approx([s for _, s in there])
        if len(here) == 1 or here[0][1] > here[1][1]:
            assert here[0][0] == Plan(int(m[there[0][0].tile]), there[0][0].attack_first)


def test_ties_at_the_top_are_broken_at_random_not_by_cell(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(0, 0, FOOT, 1)])  # tiles tie around it
    legal = SMALL.legal(position)
    tops = set()
    for seed in range(12):
        found = QUICK.candidates(SMALL, position, legal, np.random.default_rng(seed))
        assert found[0][1] == found[1][1]
        tops.add(found[0][0])
    assert len(tops) > 1


def test_once_moved_a_unit_plans_only_where_it_stands(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0, moved=2), piece(5, 4, FOOT, 1)])
    found = SearchPlayer().candidates(
        SMALL, position, SMALL.legal(position), np.random.default_rng(0)
    )
    assert [plan.tile for plan, _ in found] == [at(4, 4)]


def test_search_keeps_out_of_reach_where_greedy_without_risk_steps_in(make_battle, at):
    position = make_battle([piece(1, 4, FOOT, 0, hp=1), piece(6, 4, FOOT, 1)])
    careless = GreedyPlayer(risk=0.0)
    verb, cell = act(careless, SMALL, position)
    dist = steps(position.state.space, at(6, 4))
    assert verb == Verb.MOVE
    assert dist[cell] <= 3  # the enemy moves 2 and attacks next
    player = SearchPlayer(simulations=60, depth=2, greedy=careless)
    plan = player.search(SMALL, position, np.random.default_rng(0))
    assert dist[plan.tile] > 3


def test_the_player_follows_its_plan_over_the_turn(make_battle, at):
    position = make_battle([piece(1, 4, FOOT, 0), piece(6, 4, FOOT, 1)])
    plan = QUICK.search(SMALL, position, np.random.default_rng(0))
    assert plan.tile != at(1, 4)
    player = dataclasses.replace(QUICK)  # a player of its own, with no plan yet
    verbs = []
    while not verbs or verbs[-1] != Verb.END:
        verb, cell = act(player, SMALL, position)
        if verb == Verb.MOVE:
            assert cell == plan.tile
        verbs.append(verb)
        position = SMALL.step(position, (verb, cell))
    assert verbs[0] == Verb.MOVE


def test_search_plays_whole_games_with_rollouts():
    player = SearchPlayer(simulations=6, plans=3, depth=2, rollout_turns=2)
    record = play(TINY, [player, GreedyPlayer()], TINY.start(5))
    assert record.final.over


def test_search_needs_a_position_in_battle():
    with pytest.raises(ValueError, match="battle"):
        QUICK.search(TINY, TINY.start(0), np.random.default_rng(0))
    final = play(TINY, [GreedyPlayer(), GreedyPlayer()], TINY.start(1)).final
    with pytest.raises(ValueError, match="battle"):
        QUICK.search(TINY, final, np.random.default_rng(0))


def test_search_deploys_as_greedy_does():
    position = TINY.start(1)
    assert act(QUICK, TINY, position) == act(GreedyPlayer(), TINY, position)


def test_search_parameters_are_checked(not_an_integer):
    for name in ("simulations", "plans", "depth", "rollout_turns"):
        with pytest.raises(TypeError, match=name):
            dataclasses.replace(QUICK, **{name: not_an_integer})
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(QUICK, **{name: -1})
    with pytest.raises(ValueError, match="exploration"):
        SearchPlayer(exploration=-0.1)
    with pytest.raises(ValueError, match="temperature"):
        SearchPlayer(temperature=0.0)
    with pytest.raises(TypeError, match="greedy"):
        SearchPlayer(greedy="greedy")  # ty: ignore[invalid-argument-type]


def counts(n, w, children=()):
    node = _Node(n, w)
    for key, child in children:
        node.children[key] = child
    return node


def test_selection_follows_the_prior_between_plans_alike():
    low, high = Plan(1, False), Plan(2, False)
    node = counts(10, 5.0, [((0, low), counts(5, 2.5)), ((0, high), counts(5, 2.5))])
    plan, new = QUICK._select(node, [(low, 0.0), (high, 3.0)], uid=0, team=0)
    assert (plan, new) == (high, False)


def test_team_1_selects_for_its_own_value():
    worse, better = Plan(1, False), Plan(2, False)  # for team 1: team 0's value 0.9 and 0.2
    node = counts(10, 5.0, [((0, worse), counts(5, 4.5)), ((0, better), counts(5, 1.0))])
    player = SearchPlayer(exploration=0.0)
    assert player._select(node, [(worse, 0.0), (better, 0.0)], uid=0, team=1)[0] == better
    assert player._select(node, [(worse, 0.0), (better, 0.0)], uid=0, team=0)[0] == worse


def test_an_untried_plan_counts_at_the_nodes_value_for_the_team_choosing():
    tried, untried = Plan(1, False), Plan(2, False)
    node = counts(10, 8.0, [((0, tried), counts(10, 5.0))])  # the node: 0.8 for team 0
    player = SearchPlayer(exploration=0.0)
    candidates = [(tried, 0.0), (untried, 0.0)]
    assert player._select(node, candidates, uid=0, team=0) == (untried, True)  # 0.8 > 0.5
    assert player._select(node, candidates, uid=0, team=1) == (tried, False)  # 0.5 > 0.2


def test_each_simulation_visits_the_root_and_one_of_its_plans(make_battle):
    position = make_battle([piece(1, 4, FOOT, 0), piece(6, 4, FOOT, 1)])
    root = _Node()
    rng = np.random.default_rng(0)
    for _ in range(7):
        QUICK._simulate(SMALL, root, dataclasses.replace(position, seed=int(rng.integers(99))), rng)
    assert root.n == 7
    assert sum(child.n for child in root.children.values()) == 7
    assert 0.0 <= root.w <= 7.0


def test_a_plan_to_attack_first_attacks_before_moving(make_battle, at):
    position = make_battle([piece(4, 4, FOOT, 0), piece(5, 4, FOOT, 1)])
    legal, rng = SMALL.legal(position), np.random.default_rng(0)
    first = QUICK._next(SMALL, position, legal, Plan(at(2, 4), True), rng)
    assert first == (Verb.ATTACK, at(5, 4))
    assert QUICK._next(SMALL, position, legal, Plan(at(2, 4), False), rng) == (Verb.MOVE, at(2, 4))


def test_finished_games_are_worth_their_result(make_battle):
    position = make_battle([piece(1, 4, FOOT, 0), piece(6, 4, FOOT, 1)])
    rng = np.random.default_rng(0)
    for winner, value in ((None, 0.5), (0, 1.0), (1, 0.0)):
        final = dataclasses.replace(position, winner=winner, over=True)
        assert QUICK._evaluate(SMALL, final, rng) == value


@dataclasses.dataclass(frozen=True)
class CountingSearch(SearchPlayer):
    """A search player that notes the ply of each search it runs."""

    plies: list[int] = dataclasses.field(default_factory=list)

    def search(self, game, position, rng):
        self.plies.append(position.ply)
        return super().search(game, position, rng)


def test_the_player_searches_once_per_turn():
    player = CountingSearch(simulations=4, plans=3, depth=2)
    record = play(TINY, [player, GreedyPlayer()], TINY.start(6))
    team = {e.uid: e.team for e in record.events if isinstance(e, Placed)}
    ours = [e for e in record.events if isinstance(e, Ended | Died) and team[e.uid] == 0]
    assert 0 < len(player.plies) <= len(ours)  # a turn ends by its end or its unit's death
