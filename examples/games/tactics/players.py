"""Players of Tactics: who picks an action from the legal ones.

A player is a `game.Player`: a callable taking the game, the player it is, what that player is
shown, the mask of legal actions and a random generator, and returning an action. The game has
full information, so what a player is shown (`Tactics.view`) is the position itself; a network
would call `Tactics.observe` on it. Players are frozen dataclasses whose fields are their
parameters, like the rules, so they can be varied too.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gridzero.core import check_real, hops, sight, snap, steps
from gridzero.game import Action

from .board import home_corners
from .game import Tactics
from .state import TacticsPosition, Verb


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
    onto the open tile nearest an enemy if it is an archer, turns to face the nearest enemy,
    and ends its turn. It moves once per turn, before or after attacking. The value of an
    attack is its chance to hit times the damage, plus a bonus for a kill worth the target's
    hit points and Atk. A tile scores by how close it is to the nearest enemy (archers want to
    stand at their range), the Atk of the enemies that could reach and attack it next round,
    friends beside footsoldiers, and friends in a captain's aura. The nearest enemy is the
    nearest by a walk around walls and rubble (units aside), and threats count steps on the open
    board, ignoring walls.

    In deployment it places knights and footsoldiers nearest the enemy, then the captain a row
    behind, then the archers a row further.

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

    Raises
    ------
    TypeError, ValueError
        If a weight is not a finite real number.
    """

    attack: float = 1.0
    kill: float = 1.0
    risk: float = 0.1
    advance: float = 0.5
    formation: float = 0.3
    aura: float = 0.3

    def __post_init__(self) -> None:
        for name in ("attack", "kill", "risk", "advance", "formation", "aura"):
            check_real(getattr(self, name), name)

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
        """Place the next kind in order, in the row its kind belongs to."""
        space = position.state.space
        left = [k for k in range(len(game.units)) if legal[Verb.PLACE + k].any()]
        kind = min(left, key=lambda k: _deploy_rank(game, k))
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
        if now_target >= 0 and now_value >= best_attack:
            return Action(Verb.ATTACK, now_target)
        if best != a:
            return Action(Verb.MOVE, best)
        return self._wrap_up(game, position, legal, rng)

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
        rng: np.random.Generator,
    ) -> Action:
        """End a turn: defer onto the open tile nearest an enemy, face the nearest enemy, end."""
        state = position.state
        u = state.units
        a = game.active(position)
        xy = state.space.coords
        enemies = np.flatnonzero(state.occupied & (u.team != u.team[a]))
        if legal[Verb.DEFER].any() and enemies.size:
            tiles = np.flatnonzero(legal[Verb.DEFER])
            nearest = hops(state.space, tiles[:, None], enemies).min(axis=1).astype(np.float64)
            return Action(Verb.DEFER, int(tiles[_argmax(-nearest, rng)]))
        if enemies.size:
            target = enemies[np.argmin(hops(state.space, enemies, a))]
            d = snap(state.space, xy[target] - xy[a], u.facing[a])
            if legal[Verb.FACE + d, a]:
                return Action(Verb.FACE + d, a)
        return Action(Verb.END, a)

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
        return score


def _deploy_rank(game: Tactics, kind: int) -> tuple[bool, bool, int, int]:
    """Return the order kinds deploy in: melee first, most hit points first, auras, then ranged."""
    stats = game.units[kind]
    return (stats.reach > 1, stats.aura, -stats.hp, kind)


def _threat_range(game: Tactics, kind: int) -> int:
    """Return how far a unit of ``kind`` can strike from in a turn: its movement plus reach."""
    stats = game.units[kind]
    return (2 if stats.double_move else 1) * stats.move + int(stats.reach)


def _argmax(score: NDArray[np.float64], rng: np.random.Generator) -> int:
    """Return the index of the highest score, ties broken at random."""
    best = np.flatnonzero(score == score.max())
    return int(best[rng.integers(best.size)]) if best.size > 1 else int(best[0])
