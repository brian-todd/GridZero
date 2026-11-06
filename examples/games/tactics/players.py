"""Players of Tactics: who picks an action from the legal ones.

A player is a `game.Player`: a callable taking the game, the player it is, what that player is
shown, the mask of legal actions and a random generator, and returning an action. The game has
full information, so what a player is shown (`Tactics.view`) is the position itself; a network
would call `Tactics.observe` on it. Players are frozen dataclasses whose fields are their
parameters, like the rules, so they can be varied too.
"""

import functools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gridzero.core import check_int, check_real, hops, sight, snap, steps
from gridzero.game import Action

from . import objective
from .board import home_corners
from .game import Tactics
from .rules import default_roster
from .state import TacticsPosition, TacticsState, Verb


@dataclass(frozen=True)
class RandomPlayer:
    """Picks a legal action at random, ending a turn only sometimes when it could act.

    Parameters
    ----------
    end_share : float, default 0.1
        The chance of ending the unit's turn when other actions are legal, in [0, 1]. Otherwise
        it picks one of the other legal actions, all equally likely. Were `Verb.END` one action
        among them, a unit with nothing to do but turn would end at once half the time.

    Raises
    ------
    TypeError, ValueError
        If ``end_share`` is not a real number in [0, 1].
    """

    end_share: float = 0.1

    def __post_init__(self) -> None:
        check_real(self.end_share, "end_share", 0.0, 1.0)

    def __call__(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return a random legal action (see `game.Player`)."""
        choices = np.argwhere(legal)
        ending = choices[:, 0] == Verb.END
        if ending.any() and (ending.all() or rng.random() < self.end_share):
            choices = choices[ending]
        else:
            choices = choices[~ending]
        verb, cell = choices[rng.integers(len(choices))]
        return Action(int(verb), int(cell))


@dataclass(frozen=True)
class GreedyPlayer:
    """Looks one action ahead: the best attack now, or the best tile to move to.

    On its turn a unit attacks the target of most value now, unless a move first would let it
    attack better; otherwise it moves to the tile that scores best, then defers its attack
    if it is an archer (onto the point if an enemy could walk onto it, else onto the open tile
    nearest an enemy), turns to face the nearest enemy, and ends its turn. It moves once per
    turn, before or after attacking. The value of an attack is its chance to hit times the damage,
    plus a bonus for a kill worth the target's hit points and Atk. An engineer works on an adjacent
    tile instead when that is worth more than attacking: how much the change would delay enemies'
    walks to the point against friends', and the lines of fire it opens for friends and cuts for
    enemies. A tile scores by how close it is to the nearest enemy (archers want to
    stand at their range), the Atk of the enemies that could reach and attack it next round,
    friends beside footsoldiers, friends in a captain's aura, and the walk to the point, if its
    side doesn't own it and no friend stands there (`objective.wanted`). The nearest enemy is the
    nearest by a walk around walls and rubble (units aside), and threats count steps on the open
    board, ignoring walls.

    In deployment it fields ``roster``, placing knights and footsoldiers nearest the enemy, then
    the captain a row behind, then the ranged units a row further.

    Parameters
    ----------
    attack : float, default 1.0
        Weight of the expected damage of an attack.
    kill : float, default 1.0
        Weight of the chance to kill, times the target's hit points and Atk.
    risk : float, default 0.1
        Weight of the Atk of the enemies that could attack a tile next round.
    advance : float, default 0.5
        Weight of each step closer to the nearest enemy.
    formation : float, default 0.3
        Weight of each friend that would give a footsoldier its formation bonus.
    aura : float, default 0.3
        Weight of each friend in a captain's aura, or of standing in a friendly aura.
    point : float, default 1.0
        Weight of each step closer to the point. At 0 it ignores the objective: a pure fighter.
    work : float, default 1.0
        Weight of each step of delay, or line of fire, a change of terrain gives its side. At 0
        it never builds or breaks.
    roster : tuple of int, optional
        How many of each kind to field. By default `rules.default_roster` of the game's caps. A
        roster that doesn't fill the team under the rules is clipped: past it, the next legal
        kind by rank.

    Raises
    ------
    TypeError, ValueError
        If a weight is not a finite real number, or ``roster`` is not a tuple of non-negative
        integers.
    """

    attack: float = 1.0
    kill: float = 1.0
    risk: float = 0.1
    advance: float = 0.5
    formation: float = 0.3
    aura: float = 0.3
    point: float = 1.0
    work: float = 1.0
    roster: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("attack", "kill", "risk", "advance", "formation", "aura", "point", "work"):
            check_real(getattr(self, name), name)
        if self.roster is not None:
            if not isinstance(self.roster, tuple):
                raise TypeError(f"roster must be a tuple of ints, got {self.roster!r}")
            for count in self.roster:
                check_int(count, "each roster entry", minimum=0)

    def __call__(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return the greedy action (see `game.Player`)."""
        if position.deploying:
            return self._deploy(game, player, position, legal, rng)
        return self._act(game, position, legal, rng)

    def _deploy(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Place the roster's next kind in order, in the row its kind belongs to."""
        state = position.state
        space = state.space
        roster = self.roster or default_roster(game.caps, game.team_size)
        kinds = range(len(game.units))
        placed = [
            np.count_nonzero((state.units.kind == k) & (state.units.team == player)) for k in kinds
        ]
        legal_kinds = [k for k in kinds if legal[Verb.PLACE + k].any()]
        wanted = [k for k in legal_kinds if k < len(roster) and placed[k] < roster[k]]
        kind = min(wanted or legal_kinds, key=lambda k: _deploy_rank(game, k))
        stats = game.units[kind]
        row = 2 if stats.reach > 1 else 1 if stats.aura else 0
        cells = np.flatnonzero(legal[Verb.PLACE + kind])
        corner = home_corners(space)[1 - player]
        dist = hops(space, cells, corner)
        score = -np.abs(dist - (dist.min() + row)).astype(np.float64)
        return Action(Verb.PLACE + kind, int(cells[_argmax(score, rng)]))

    def _act(
        self,
        game: Tactics,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Attack, move, defer, turn or end: the first that is worth doing."""
        a = game.active(position)
        now_value, now_target = -np.inf, -1
        if legal[Verb.ATTACK].any():
            now_value, now_target = self._best_attack(
                game, position, a, a, np.flatnonzero(legal[Verb.ATTACK])
            )
        best, best_attack = a, -np.inf
        if legal[Verb.MOVE].any() and position.state.units.moved[a] == 0:
            cells, score, attacks = self._moves(game, position, legal, a)
            total = score + np.where(np.isfinite(attacks), attacks, 0.0)
            i = _argmax(total, rng)
            best, best_attack = int(cells[i]), attacks[i]
        work_value, work = self._best_work(game, position, legal, a)
        if now_target >= 0 and now_value >= best_attack and now_value >= work_value:
            return Action(Verb.ATTACK, now_target)
        if work is not None and work_value >= best_attack:
            return work
        if best != a:
            return Action(Verb.MOVE, best)
        return self._wrap_up(game, position, legal, work, rng)

    def _moves(
        self, game: Tactics, position: TacticsPosition, legal: NDArray[np.bool_], a: int
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """Score the tiles the unit on ``a`` may move to, and its own.

        Returns the tiles, shape (n,), how good each is to stand on, and the value of the best
        attack from each, or -inf where none would be possible (both float64, shape (n,)).
        """
        state = position.state
        u = state.units
        me = game.units[u.kind[a]]
        enemies = np.flatnonzero(state.occupied & (u.team != u.team[a]))
        cells = np.append(np.flatnonzero(legal[Verb.MOVE]), a)
        score = self._tile_scores(game, position, a, cells)
        attacks = np.full(cells.size, -np.inf)
        if not u.attacked[a] and enemies.size:
            # Sight is symmetric: the tiles each enemy sees within reach are those it can be
            # attacked from, as the game decides it.
            seen = sight(state.space, enemies, state.wall, me.reach, separate=True)[:, cells]
            if me.double_move:  # a knight moving past its move can't attack
                seen[:, hops(state.space, cells, a) > me.move] = False
            for i in np.flatnonzero(seen.any(axis=0)):
                targets = enemies[seen[:, i]]
                attacks[i] = self._best_attack(game, position, a, int(cells[i]), targets)[0]
        return cells, score, attacks

    def _wrap_up(
        self,
        game: Tactics,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        work: Action | None,
        rng: np.random.Generator,
    ) -> Action:
        """End a turn: ``work`` if any, defer as the class says, face an enemy, end."""
        state = position.state
        u = state.units
        a = game.active(position)
        if work is not None:
            return work
        xy = state.space.coords
        enemies = np.flatnonzero(state.occupied & (u.team != u.team[a]))
        if legal[Verb.DEFER].any() and enemies.size:
            tiles = np.flatnonzero(legal[Verb.DEFER])
            for tile in tiles[state.point[tiles] & ~state.occupied[tiles]]:
                walk = [_walk_range(game, int(k)) for k in u.kind[enemies]]
                if (hops(state.space, enemies, tile) <= walk).any():
                    return Action(Verb.DEFER, int(tile))
            nearest = hops(state.space, tiles[:, None], enemies).min(axis=1).astype(np.float64)
            return Action(Verb.DEFER, int(tiles[_argmax(-nearest, rng)]))
        if enemies.size:
            target = enemies[np.argmin(hops(state.space, enemies, a))]
            d = snap(state.space, xy[target] - xy[a], u.facing[a])
            if legal[Verb.FACE + d, a]:
                return Action(Verb.FACE + d, a)
        return Action(Verb.END, a)

    def _best_work(
        self, game: Tactics, position: TacticsPosition, legal: NDArray[np.bool_], a: int
    ) -> tuple[float, Action | None]:
        """Return the value of the best change of terrain and its action, if any is worth doing."""
        if self.work == 0 or not (legal[Verb.BUILD].any() or legal[Verb.BREAK].any()):
            return -np.inf, None
        state = position.state
        team = int(state.units.team[a])
        walks = _Walks(game, state, team)
        best, action = 0.0, None
        for verb in (Verb.BUILD, Verb.BREAK):
            for tile in np.flatnonzero(legal[verb]):
                value = self.work * walks.gain(int(tile), verb == Verb.BUILD)
                if value > best:
                    best, action = value, Action(verb, int(tile))
        return (best, action) if action is not None else (-np.inf, None)

    def _best_attack(
        self, game: Tactics, position: TacticsPosition, a: int, at: int, targets: object
    ) -> tuple[float, int]:
        """Return the value of the best attack from ``at`` on one of ``targets``, and its cell."""
        u = position.state.units
        best, cell = -np.inf, -1
        for t in np.asarray(targets, dtype=np.int64):
            odds = game.odds(position, a, int(t), at=at)
            hp = int(u.hp[t])
            stats = game.units[u.kind[t]]
            lethal = odds.atk >= hp
            value = odds.chance * (
                self.attack * min(odds.atk, hp) + self.kill * lethal * (stats.hp + stats.atk)
            )
            if value > best:
                best, cell = value, int(t)
        return best, cell

    def _tile_scores(
        self, game: Tactics, position: TacticsPosition, a: int, cells: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Return how good each tile is to stand on, attacks aside."""
        state = position.state
        u = state.units
        me = game.units[u.kind[a]]
        team = u.team[a]
        enemies = np.flatnonzero(state.occupied & (u.team != team))
        friends = np.flatnonzero(state.occupied & (u.team == team))
        friends = friends[friends != a]
        score = np.zeros(cells.size)
        if enemies.size:
            # The walk around walls and rubble, not hops: across a wall, hops would hold both
            # sides still, a step around it scoring as a step away.
            nearest = steps(state.space, enemies, blocked=~state.passable)[cells]
            want = me.reach if me.reach > 1 else 1
            score -= self.advance * np.abs(nearest - want)
            dist = hops(state.space, cells[:, None], enemies)
            reach = np.array(
                [_threat_range(game, int(k)) for k in u.kind[enemies]], dtype=np.float64
            )
            atk = np.array([game.units[k].atk for k in u.kind[enemies]], dtype=np.float64)
            score -= self.risk * ((dist <= reach) * atk).sum(axis=1)
        if friends.size:
            dist = hops(state.space, cells[:, None], friends)
            if me.formation:
                formed = np.array([game.units[k].formation for k in u.kind[friends]])
                count = ((dist == 1) & formed).sum(axis=1)
                score += self.formation * np.minimum(count, game.formation_cap)
            if me.aura:
                score += self.aura * (dist <= game.aura_radius).sum(axis=1)
            else:
                givers = np.array([game.units[k].aura for k in u.kind[friends]])
                score += self.aura * ((dist <= game.aura_radius) & givers).any(axis=1)
        if self.point != 0:
            tiles = objective.wanted(state.point, position.hold, u.team, int(team), a)
            if tiles.size:
                score -= self.point * steps(state.space, tiles, blocked=~state.passable)[cells]
        return score


class _Walks:
    """What a side gains from changing a tile: delays to walks, and lines of fire."""

    def __init__(self, game: Tactics, state: TacticsState, team: int) -> None:
        self.game, self.state, self.team = game, state, team
        self.space = state.space
        u = state.units
        self.point = np.flatnonzero(state.point)
        self.friends = np.flatnonzero(state.occupied & (u.team == team))
        self.enemies = np.flatnonzero(state.occupied & (u.team != team))
        self.blocked = ~state.passable

    @functools.cached_property
    def walks(self) -> float:
        """The side's lead in steps to the point, as things stand."""
        return self._walks(self.blocked)

    @functools.cached_property
    def lines(self) -> int:
        """The side's lead in lines of fire, as things stand."""
        return self._lines(self.state.wall)

    def _walks(self, blocked: NDArray[np.bool_]) -> float:
        """Return the enemies' steps to the point, summed, minus the friends'."""
        if not self.point.size:
            return 0.0
        distance = steps(self.space, self.point, blocked=blocked)
        return float(distance[self.enemies].sum() - distance[self.friends].sum())

    def _lines(self, wall: NDArray[np.bool_]) -> int:
        """Return the friends' lines of fire on enemies minus the enemies' on friends."""
        return self._fire(wall, self.friends, self.enemies) - self._fire(
            wall, self.enemies, self.friends
        )

    def _fire(
        self, wall: NDArray[np.bool_], shooters: NDArray[np.intp], targets: NDArray[np.intp]
    ) -> int:
        """Return how many (ranged shooter, target) pairs see each other within reach."""
        kinds = self.state.units.kind
        total = 0
        for k, stats in enumerate(self.game.units):
            cells = shooters[kinds[shooters] == k]
            if stats.reach > 1 and cells.size and targets.size:
                seen = sight(self.space, cells, wall, stats.reach, separate=True)
                total += int(np.count_nonzero(seen[:, targets]))
        return total

    def gain(self, tile: int, build: bool) -> float:
        """Return what moving ``tile`` one step of the ladder is worth to the side."""
        state = self.state
        value = 0.0
        if state.passable[tile] or (state.rubble[tile] and not build):
            blocked = self.blocked.copy()
            blocked[tile] = build
            value += self._walks(blocked) - self.walks
        if state.wall[tile] or (state.rubble[tile] and build):
            wall = state.wall.copy()
            wall[tile] = build
            value += self._lines(wall) - self.lines
        return value


@dataclass(frozen=True)
class SitterPlayer:
    """Deploys as greedy does, then never moves: it strikes what comes in reach and ends its turn.

    A yardstick for the objective: turtling should lose.

    Parameters
    ----------
    greedy : GreedyPlayer, default GreedyPlayer()
        Deploys, and chooses between targets.

    Raises
    ------
    TypeError
        If ``greedy`` is not a `GreedyPlayer`.
    """

    greedy: GreedyPlayer = field(default_factory=GreedyPlayer)

    def __post_init__(self) -> None:
        if not isinstance(self.greedy, GreedyPlayer):
            raise TypeError(f"greedy must be a GreedyPlayer, got {type(self.greedy).__name__}")

    def __call__(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> Action:
        """Return its deployment, its best attack, or the end of its turn (see `game.Player`)."""
        if position.deploying:
            return self.greedy(game, player, position, legal, rng)
        a = game.active(position)
        if legal[Verb.ATTACK].any():
            targets = np.flatnonzero(legal[Verb.ATTACK])
            return Action(Verb.ATTACK, self.greedy._best_attack(game, position, a, a, targets)[1])
        return Action(Verb.END, a)


def _deploy_rank(game: Tactics, kind: int) -> tuple[bool, bool, int, int]:
    """Return the order kinds deploy in: melee first, most hit points first, auras, then ranged."""
    stats = game.units[kind]
    return (stats.reach > 1, stats.aura, -stats.hp, kind)


def _walk_range(game: Tactics, kind: int) -> int:
    """Return how far a unit of ``kind`` can walk in a turn."""
    stats = game.units[kind]
    return (2 if stats.double_move else 1) * stats.move


def _threat_range(game: Tactics, kind: int) -> int:
    """Return how far a unit of ``kind`` can strike from in a turn: its movement plus reach."""
    return _walk_range(game, kind) + int(game.units[kind].reach)


def _argmax(score: NDArray[np.float64], rng: np.random.Generator) -> int:
    """Return the index of the highest score, ties broken at random."""
    best = np.flatnonzero(score == score.max())
    return int(best[rng.integers(best.size)]) if best.size > 1 else int(best[0])
