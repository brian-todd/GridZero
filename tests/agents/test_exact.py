"""Tests of exact values of small deterministic games: `agents.exact`."""

import dataclasses

import numpy as np
import pytest

from agents_demo import GAMBLE, PLACE, Patches, PatchesPosition, PatchesState, Took
from gridzero.agents.exact import optimal, solve


@dataclasses.dataclass(frozen=True)
class Sure(Patches):
    """`Patches` whose gamble always lands heads: the cell and its free neighbor of lowest id.

    No step draws, so the game has exact values, with two verbs.
    """

    def step(self, position, action):
        verb, cell = int(action[0]), int(action[1])
        if verb != GAMBLE:
            return super().step(position, action)
        graph = self.space.adjacency()
        owner = position.state.owner.copy()
        free = [
            n for n in graph.indices[graph.indptr[cell] : graph.indptr[cell + 1]] if owner[n] < 0
        ]
        taken = (cell, *(int(n) for n in free[:1]))
        player = position.ply % 2
        owner[list(taken)] = player
        state = position.state.replace({PatchesState.owner: owner})
        over = bool((owner >= 0).all())
        return PatchesPosition(
            state,
            position.seed,
            ply=position.ply + 1,
            over=over,
            winner=self._winner(state) if over else None,
            events=(Took(player, taken),),
        )


STRIP = Sure(shape=(1, 3))


def actions(mask):
    """Return the True actions of a mask as a set of (verb, cell)."""
    return {(int(v), int(c)) for v, c in np.argwhere(mask)}


def test_the_first_player_wins_a_strip_of_three_by_the_middle_or_any_gamble():
    # Placing on 0 or 2 lets player 1 gamble on 1 and take 1 and the other end: a group of 2.
    # Placing on 1, or gambling anywhere (two cells at once), leaves player 0 the larger group.
    values = solve(STRIP)
    start = STRIP.start(0)
    assert values[start.key()] == 1.0
    assert actions(optimal(STRIP, start, values)) == {
        (PLACE, 1),
        (GAMBLE, 0),
        (GAMBLE, 1),
        (GAMBLE, 2),
    }


def test_player_one_keeps_its_own_best_value():
    # After player 0 places on 0, player 1 wins by gambling on 1 or 2 (taking both), and loses
    # by placing: player 0 then ties the groups and has more stones.
    values = solve(STRIP)
    after = STRIP.step(STRIP.start(0), (PLACE, 0))
    assert values[after.key()] == 0.0
    assert actions(optimal(STRIP, after, values)) == {(GAMBLE, 1), (GAMBLE, 2)}


def test_values_are_results_at_the_end_and_the_movers_best_child_before():
    game = Sure(shape=(2, 3))
    values = solve(game)
    seen, frontier = set(), [game.start(0)]
    while frontier:
        position = frontier.pop()
        if position.key() in seen:
            continue
        seen.add(position.key())
        if position.over:
            expected = 0.5 if position.winner is None else float(position.winner == 0)
        else:
            children = [game.step(position, a) for a in actions(game.legal(position))]
            frontier += children
            child = [values[c.key()] for c in children]
            expected = max(child) if game.player(position) == 0 else min(child)
        assert values[position.key()] == expected
    assert seen == set(values)
    assert {0.0, 0.5, 1.0} == set(values.values())


def test_solving_from_a_later_position_gives_the_values_below_it():
    game = Sure(shape=(2, 3))
    later = game.step(game.step(game.start(0), (GAMBLE, 4)), (PLACE, 0))
    below = solve(game, later)
    everything = solve(game)
    assert later.key() in below
    assert len(below) < len(everything)
    assert all(everything[key] == value for key, value in below.items())


def test_solve_refuses_chance_and_other_numbers_of_players():
    with pytest.raises(ValueError, match="drew"):
        solve(Patches(shape=(1, 3)))

    @dataclasses.dataclass(frozen=True)
    class Three(Sure):
        @property
        def n_players(self):
            return 3

    with pytest.raises(ValueError, match="two players"):
        solve(Three(shape=(1, 3)))


def test_optimal_needs_a_position_not_over_whose_children_were_solved():
    values = solve(STRIP)
    over = STRIP.step(STRIP.start(0), (GAMBLE, 1))
    over = STRIP.step(over, (PLACE, 2))
    assert over.over
    with pytest.raises(ValueError, match="over"):
        optimal(STRIP, over, values)
    with pytest.raises(KeyError):
        optimal(STRIP, STRIP.start(0), {})
