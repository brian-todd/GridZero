"""A search player: open-loop Monte Carlo tree search over plans for whole unit turns.

A unit's turn is several decisions (move, attack, turn, end), and a move has dozens of tiles,
so the search chooses between a few **plans** for a whole turn instead: a tile to move to,
whether to attack before moving or after, and for a unit that builds, a tile to work on instead of
attacking. The greedy player's scores give the candidates and
their priors, and its choices fill in the rest of a turn (which target, where to defer, which
way to face). Both sides plan the same way inside the search.

Chance (the d20 of each attack, and the order of turns when it is rolled) is sampled, not
enumerated: every simulation plays from the root with a fresh seed drawn from the player's own
generator, so the search never reads the game's seed, which fixes every future roll. Nodes are
keyed by the plans chosen to reach them (open loop), so one node gathers many outcomes, and a
plan is a child only where the same unit acts again.
"""

import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from gridzero.core import check_int, check_real
from gridzero.game import Action

from .game import Tactics
from .players import GreedyPlayer
from .state import TacticsPosition, Verb


class Plan(NamedTuple):
    """What a unit does with its turn: where it moves, and whether it attacks first."""

    tile: int
    """The cell to move to; the unit's own to stay."""
    attack_first: bool
    """Attack from where it stands before moving, rather than after."""
    work: Action | None = None
    """Work on a tile after moving, in place of attacking; skipped if no longer legal."""


class Judgment(NamedTuple):
    """What a search made of a unit's turn."""

    plans: tuple[Plan, ...]
    """The plans tried at the root."""
    visits: tuple[int, ...]
    """The simulations through each."""
    values: tuple[float, ...]
    """Team 0's mean value through each, in [0, 1]."""
    value: float
    """Team 0's mean value over every simulation, in [0, 1]."""


@dataclass(frozen=True)
class SearchPlayer:
    """Picks each unit's turn by open-loop Monte Carlo tree search over plans.

    At the start of a unit's turn it runs ``simulations`` simulations from the position, each
    with a fresh seed. A simulation walks down the tree, choosing plans by PUCT (a node's
    value, plus a bonus for plans the greedy scores favor and few visits), plays each chosen
    plan out to the end of its unit's turn, and stops after ``depth`` unit turns, at the end of
    the game, or at a plan not tried before. It then plays ``rollout_turns`` unit turns more as
    the greedy player, and scores the position by its standing (see `standing`). The plan tried
    most at the root is played, one action at a time, over the turn. Deployment is the greedy
    player's.

    Parameters
    ----------
    simulations : int, default 100
        Simulations per unit's turn, at least 1.
    plans : int, default 8
        Candidate plans per turn, at least 1: the best by the greedy's scores.
    work_plans : int, default 4
        Further candidates for a unit that builds, at least 0: changes to the terrain around the
        tiles it scores best. They have a place of their own because a change rarely scores above
        0 by the greedy's reckoning, however much it would matter.
    depth : int, default 4
        Unit turns a simulation plays before scoring, the root's included, at least 1.
    exploration : float, default 0.3
        Weight of the prior and visit bonus against values in [0, 1], at least 0.
    temperature : float, default 1.0
        Softness of the priors, a softmax of the plans' scores divided by it; positive.
    rollout_turns : int, default 16
        Unit turns the greedy player plays after the last plan, before scoring.
    hold_weight : float, default 0.25
        Share of a position's value that is the point's clock rather than material, in [0, 1].
    greedy : GreedyPlayer, default GreedyPlayer()
        Scores the plans, fills in each turn, deploys, and plays the rollouts.

    Raises
    ------
    TypeError, ValueError
        If a parameter has the wrong type or is out of range.
    """

    simulations: int = 100
    plans: int = 8
    work_plans: int = 4
    depth: int = 4
    exploration: float = 0.3
    temperature: float = 1.0
    rollout_turns: int = 16
    hold_weight: float = 0.25
    greedy: GreedyPlayer = field(default_factory=GreedyPlayer)
    _chosen: dict[tuple[int, int, int], Plan] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    """The plan of the turn under way, keyed by the game's seed, the round and the unit."""

    def __post_init__(self) -> None:
        check_int(self.simulations, "simulations", minimum=1)
        check_int(self.plans, "plans", minimum=1)
        check_int(self.work_plans, "work_plans", minimum=0)
        check_int(self.depth, "depth", minimum=1)
        check_int(self.rollout_turns, "rollout_turns", minimum=0)
        check_real(self.exploration, "exploration", 0.0)
        check_real(self.hold_weight, "hold_weight", 0.0, 1.0)
        if not check_real(self.temperature, "temperature") > 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
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
        """Return the next action of this turn's plan, searching for one first."""
        return self.decide(game, player, position, legal, rng)[0]

    def decide(
        self,
        game: Tactics,
        player: int,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[Action, Judgment | None]:
        """Return what the player does, and what its search made of the turn if it searched here.

        Parameters
        ----------
        game : Tactics
            The rules.
        player : int
            The player it is.
        position : TacticsPosition
            The position.
        legal : ndarray of bool, shape (n_verbs, n_cells)
            ``game.legal(position)``.
        rng : numpy.random.Generator
            The source of every seed and tie.

        Returns
        -------
        action : Action
            The action.
        judgment : Judgment or None
            The search's, at the first decision of a unit's turn with a choice in it; None when
            this decision followed a plan already chosen, was deployment, or had one legal action.
        """
        if position.deploying or np.count_nonzero(legal) == 1:
            return self.greedy(game, player, position, legal, rng), None
        a = game.active(position)
        key = (position.seed, position.round, int(position.state.units.uid[a]))
        judgment = None
        if key not in self._chosen:
            self._chosen.clear()
            judgment = self.judge(game, position, rng)
            self._chosen[key] = self._pick(judgment, rng)
        return self._next(game, position, legal, self._chosen[key], rng), judgment

    def search(self, game: Tactics, position: TacticsPosition, rng: np.random.Generator) -> Plan:
        """Return the plan the search prefers for the acting unit's turn.

        Parameters
        ----------
        game : Tactics
            The rules.
        position : TacticsPosition
            A position in battle, not over.
        rng : numpy.random.Generator
            The source of every seed and tie the search draws.

        Returns
        -------
        Plan
            The plan tried most at the root; of plans tried equally, one drawn at random.

        Raises
        ------
        ValueError
            If the game is over or still in deployment.
        """
        return self._pick(self.judge(game, position, rng), rng)

    def judge(self, game: Tactics, position: TacticsPosition, rng: np.random.Generator) -> Judgment:
        """Return what the search makes of the acting unit's turn.

        Parameters
        ----------
        game : Tactics
            The rules.
        position : TacticsPosition
            A position in battle, not over.
        rng : numpy.random.Generator
            The source of every seed and tie the search draws.

        Returns
        -------
        Judgment
            The plans tried at the root, their visits and values, and the root's value.

        Raises
        ------
        ValueError
            If the game is over or still in deployment.
        """
        if position.over or position.deploying:
            raise ValueError("search needs a position in battle, not over")
        root = _Node()
        for _ in range(self.simulations):
            sampled = replace(position, seed=int(rng.integers(2**63)))
            self._simulate(game, root, sampled, rng)
        uid = int(position.state.units.uid[game.active(position)])
        tried = [(plan, child) for (who, plan), child in root.children.items() if who == uid]
        return Judgment(
            tuple(plan for plan, _ in tried),
            tuple(child.n for _, child in tried),
            tuple(child.w / child.n for _, child in tried),
            root.w / root.n,
        )

    def _pick(self, judgment: Judgment, rng: np.random.Generator) -> Plan:
        """Return the plan tried most; of plans tried equally, one drawn at random."""
        most = max(judgment.visits)
        best = [p for p, n in zip(judgment.plans, judgment.visits, strict=True) if n == most]
        return best[int(rng.integers(len(best)))]

    def candidates(
        self,
        game: Tactics,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> list[tuple[Plan, float]]:
        """Return the candidate plans for the acting unit's turn, best first, with scores.

        A plan to move to a tile scores as the greedy player scores the tile, plus the value
        of the best attack from it. A plan to attack first scores the best attack from where
        the unit stands plus the tile. Once the unit has moved, its only tile is its own. A unit
        that builds also gets `work_plans`, in random order, after the others.

        Parameters
        ----------
        game : Tactics
            The rules.
        position : TacticsPosition
            A position in battle, not over.
        legal : ndarray of bool, shape (n_verbs, n_cells)
            ``game.legal(position)``.
        rng : numpy.random.Generator
            Breaks ties between plans that score the same. Ties are common (about 1 position in
            7 has one at the top), and breaking them by cell id would send the two teams, whose
            boards are mirror images, consistently toward opposite ends.

        Returns
        -------
        list of (Plan, float)
            At most ``plans`` plans, in decreasing score, ties in random order, then any plans
            to work.
        """
        a = game.active(position)
        u = position.state.units
        if legal[Verb.MOVE].any() and u.moved[a] == 0:
            cells, score, attacks = self.greedy._moves(game, position, legal, a)
        else:
            cells, score, attacks = np.array([a]), np.zeros(1), np.full(1, -np.inf)
        scored = [
            (Plan(int(c), False), float(s + (t if np.isfinite(t) else 0.0)))
            for c, s, t in zip(cells, score, attacks, strict=True)
        ]
        if legal[Verb.ATTACK].any():
            now, _ = self.greedy._best_attack(game, position, a, a, legal[Verb.ATTACK].nonzero()[0])
            scored += [
                (Plan(int(c), True), float(s + now))
                for c, s in zip(cells, score, strict=True)
                if c != a  # attacking, then staying, is the plan to stay
            ]
        best = self._best(scored, rng)
        if game.units[u.kind[a]].builds and self.work_plans and not u.attacked[a]:
            best += self._work_plans(position, cells, score, rng)
        return best

    def _work_plans(
        self,
        position: TacticsPosition,
        cells: NDArray[np.intp],
        score: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> list[tuple[Plan, float]]:
        """Return up to ``work_plans`` plans to change a tile around the unit's two best tiles.

        Each scores as its tile does. The tiles it may turn out to work on are those next to where
        it would stand; whether a change is legal is left until then (`_next`).
        """
        state = position.state
        space = state.space
        near = space.radius_graph(1.0)
        scored = []
        for i in np.argsort(-score, kind="stable")[:2]:
            tile = int(cells[i])
            around = near.indices[near.indptr[tile] : near.indptr[tile + 1]]
            for target in around[~state.occupied[around] & ~state.point[around]]:
                cell = int(target)
                if not state.wall[cell]:
                    work = Action(Verb.BUILD, cell)
                    scored.append((Plan(tile, False, work), float(score[i])))
                if not state.passable[cell]:
                    work = Action(Verb.BREAK, cell)
                    scored.append((Plan(tile, False, work), float(score[i])))
        shuffled = [scored[k] for k in rng.permutation(len(scored))]
        return shuffled[: self.work_plans]

    def _best(
        self, scored: list[tuple[Plan, float]], rng: np.random.Generator
    ) -> list[tuple[Plan, float]]:
        """Return the ``plans`` best plans, in decreasing score, ties in random order."""
        shuffled = [scored[i] for i in rng.permutation(len(scored))]
        shuffled.sort(key=lambda item: -item[1])  # stable: ties stay shuffled
        return shuffled[: self.plans]

    def _simulate(
        self, game: Tactics, root: _Node, position: TacticsPosition, rng: np.random.Generator
    ) -> None:
        """Walk down from ``root`` once, play out the chosen plans, and back the value up."""
        path = [root]
        node = root
        for _ in range(self.depth):
            if position.over:
                break
            legal = game.legal(position)
            a = game.active(position)
            uid = int(position.state.units.uid[a])
            team = int(position.state.units.team[a])
            candidates = self.candidates(game, position, legal, rng)
            plan, new = self._select(node, candidates, uid, team)
            child = node.children.setdefault((uid, plan), _Node())
            position = self._play(game, position, legal, plan, rng)
            path.append(child)
            node = child
            if new:
                break
        value = self._evaluate(game, position, rng)
        for visited in path:
            visited.n += 1
            visited.w += value

    def _select(
        self, node: _Node, candidates: list[tuple[Plan, float]], uid: int, team: int
    ) -> tuple[Plan, bool]:
        """Choose a plan by PUCT for ``team``; also return whether it was never tried here.

        A plan not tried yet counts at the node's own value. Priors are a softmax of the
        candidates' scores.
        """
        scores = np.array([s for _, s in candidates])
        prior = np.exp((scores - scores.max()) / self.temperature)
        prior /= prior.sum()
        own = node.w / node.n if node.n else 0.5
        if team == 1:
            own = 1.0 - own
        bonus = self.exploration * math.sqrt(max(node.n, 1))
        best, best_value, best_new = candidates[0][0], -math.inf, True
        for (plan, _), p in zip(candidates, prior, strict=True):
            child = node.children.get((uid, plan))
            if child is None or child.n == 0:
                q, n = own, 0
            else:
                q = child.w / child.n
                q, n = (1.0 - q if team == 1 else q), child.n
            value = q + bonus * p / (1 + n)
            if value > best_value:
                best, best_value, best_new = plan, value, n == 0
        return best, best_new

    def _play(
        self,
        game: Tactics,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        plan: Plan,
        rng: np.random.Generator,
    ) -> TacticsPosition:
        """Play ``plan`` to the end of the acting unit's turn: its end, its death, or the game's."""
        a = game.active(position)
        uid = position.state.units.uid[a]
        while True:
            action = self._next(game, position, legal, plan, rng)
            position = game.step(position, action)
            if action.verb == Verb.END or position.over or not _acting(game, position, uid):
                return position
            legal = game.legal(position)

    def _next(
        self,
        game: Tactics,
        position: TacticsPosition,
        legal: NDArray[np.bool_],
        plan: Plan,
        rng: np.random.Generator,
    ) -> Action:
        """Return the next action of ``plan`` from where the turn stands.

        A plan with work moves to its tile, then works. Otherwise attack first if the plan says so
        and an attack is legal, then move to its tile if that is still legal and the unit hasn't
        moved, then attack if it can. Then defer, turn and end as the greedy player does. A step
        the dice made impossible is skipped.
        """
        a = game.active(position)
        u = position.state.units
        if plan.work is not None:
            if plan.tile != a and u.moved[a] == 0 and legal[Verb.MOVE, plan.tile]:
                return Action(Verb.MOVE, plan.tile)
            if legal[plan.work.verb, plan.work.cell]:
                return plan.work
        if legal[Verb.ATTACK].any() and (plan.attack_first or plan.tile == a or u.moved[a]):
            return self._attack(game, position, legal, a)
        if plan.tile != a and u.moved[a] == 0 and legal[Verb.MOVE, plan.tile]:
            return Action(Verb.MOVE, plan.tile)
        if legal[Verb.ATTACK].any():
            return self._attack(game, position, legal, a)
        _, work = self.greedy._best_work(game, position, legal, a)
        return self.greedy._wrap_up(game, position, legal, work, rng)

    def _attack(
        self, game: Tactics, position: TacticsPosition, legal: NDArray[np.bool_], a: int
    ) -> Action:
        """Return the greedy's best attack among the legal ones."""
        _, target = self.greedy._best_attack(game, position, a, a, legal[Verb.ATTACK].nonzero()[0])
        return Action(Verb.ATTACK, target)

    def _evaluate(
        self, game: Tactics, position: TacticsPosition, rng: np.random.Generator
    ) -> float:
        """Return team 0's value of a position: its result, or its standing after the rollout."""
        ended = 0
        while not position.over and ended < self.rollout_turns:
            legal = game.legal(position)
            action = self.greedy(game, game.player(position), position, legal, rng)
            position = game.step(position, action)
            ended += action.verb == Verb.END
        if position.over:
            return 0.5 if position.winner is None else float(position.winner == 0)
        return standing(game, position, self.hold_weight)


def standing(game: Tactics, position: TacticsPosition, hold_weight: float) -> float:
    """Return team 0's value of a position still going: material, and how near a side is to winning.

    Parameters
    ----------
    game : Tactics
        The rules.
    position : TacticsPosition
        A position in battle.
    hold_weight : float
        Share of the value given to the point, in [0, 1].

    Returns
    -------
    float
        ``(1 - hold_weight) * material + hold_weight * clock``, in [0, 1], where ``clock`` runs
        from 0.5, with no owner, to 1 as team 0 nears ``hold_rounds``, or to 0 as team 1 does.
    """
    owner, clock = position.hold
    progress = 0.5
    if owner >= 0:
        share = 0.5 * min(1.0, clock / game.hold_rounds)
        progress += share if owner == 0 else -share
    return (1.0 - hold_weight) * material(game, position) + hold_weight * progress


def material(game: Tactics, position: TacticsPosition) -> float:
    """Return team 0's share of the material on the board, in [0, 1].

    Each unit counts its kind's hit points plus Atk (what the greedy player's kill bonus
    weighs), times the share of its hit points left.

    Parameters
    ----------
    game : Tactics
        The rules.
    position : TacticsPosition
        A position in battle.

    Returns
    -------
    float
        ``m0 / (m0 + m1)``, 0.5 when both sides have the same, or when neither has any.
    """
    u = position.state.units
    cells = position.state.occupied.nonzero()[0]
    worth = np.array([game.units[k].hp + game.units[k].atk for k in range(len(game.units))])
    full = np.array([game.units[k].hp for k in range(len(game.units))])
    kind = u.kind[cells]
    value = worth[kind] * u.hp[cells] / full[kind]
    team = u.team[cells]
    total = value.sum()
    return float(value[team == 0].sum() / total) if total > 0 else 0.5


def _acting(game: Tactics, position: TacticsPosition, uid: np.integer) -> bool:
    """Return whether the unit ``uid`` is still the one acting."""
    a = game.active(position)
    return a >= 0 and position.state.units.uid[a] == uid


@dataclass(eq=False)
class _Node:
    """Visits of a node of the search, and the sum of team 0's values over them."""

    n: int = 0
    w: float = 0.0
    children: dict[tuple[int, Plan], _Node] = field(default_factory=dict)
