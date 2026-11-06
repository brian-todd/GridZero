"""A code player of Tactics: an evaluator written as code, for the AlphaZero search.

A code player is `agents.AlphaZeroPlayer` on an evaluator that is plain code: named features of a
position and of each legal action, and constants. It needs no network, so it can ship, and it
still searches. `HAND_PICKED` holds constants picked by hand, not fitted.

A search asks for one position at a time, hundreds of times a second, so features stay cheap: walks
to the point (about 8 us) and distances between units (about 14 us), but no lines of fire (about
100 us), where distance stands in for them. Moves ignore walls when they count who could reach whom.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from examples.games.fitting import sigmoid, weighted
from gridzero.core import check_real, hops, snap, steps

from . import objective
from .game import Tactics
from .state import TacticsPosition, TacticsState, Verb

KINDS = ("footsoldier", "archer", "knight", "captain", "engineer")
"""The kinds of unit these features are written for, in the order of the rules' units."""

VALUE_FEATURES = (
    *(f"hp_{kind}" for kind in KINDS),
    "to_place",
    "owner",
    "clock",
    "on_point",
    "walk_nearest",
    "walk_mean",
    "near_point",
    "threat",
    "to_act",
    "deadline",
)
"""The features of a position, in the order of `value_features`, each player 0's way, so the higher
the better for player 0: units of each kind, counted by their share of a full unit's hit points,
player 0's minus player 1's; units still to place, player 0's minus player 1's; 1 if player 0 owns
the point and -1 if player 1 does; the owner's clock as a share of the rounds that win, signed the
same way; 1 if a unit of player 0 stands on the point, -1 for player 1; how much nearer player 0's
nearest unit is to the point than player 1's, and the same for the means, as a share of the board's
longer side, between -1 and 1; units that could walk onto the point in a turn, player 0's minus
player 1's; the Atk that could reach an enemy within a turn, ignoring walls, summed over units and
enemies, player 0's minus player 1's; units yet to take this round's turn, player 0's minus player
1's; and, while nobody has captured, how much of the time to do so has gone, which favors player 0,
since player 1 loses at the deadline. The last seven are 0 during deployment."""

MOVE_FEATURES = (
    "is_move",
    "is_attack",
    "is_defer",
    "is_swap",
    "is_face",
    "is_end",
    "is_build",
    "is_break",
    "damage",
    "kill",
    "damage_after",
    "kill_after",
    "closer_point",
    "closer_enemy",
    "exposure",
    "formation",
    "aura",
    "faces_enemy",
    "defer_on_point",
    "defer_away",
    "work_walks",
    *(f"place_{kind}" for kind in KINDS),
    "place_ahead_melee",
    "place_ahead_ranged",
)
"""The features of an action, in the order of `move_features`, for the player taking it: what verb
it is; for an attack, the expected damage, and the chance of a kill times the target's hit points
and Atk; for a move, the same two for the best attack from the tile; the steps it saves on the walk
to the point, if the unit has reason to go there (`objective.wanted`); the steps it comes closer to
its range of the nearest enemy; the enemies' Atk that could reach the tile next turn, over the
unit's tile now; friends giving it a formation; friends in its aura, or being in a friend's; for a
turn, whether it faces the nearest enemy; for a deferred attack, whether it is onto the point, and
how many steps its tile is from the nearest enemy; for work, the steps it adds to the enemies'
walks to the point less the friends'; for a placement, the kind, and for melee and ranged kinds
apart, how many steps nearer the point its tile is than the farthest legal placement."""


class _Tables(NamedTuple):
    """What features read of the rules, per kind of unit, shape (n_kinds,)."""

    hp: NDArray[np.float64]
    """Full hit points."""
    atk: NDArray[np.float64]
    """Atk."""
    walk: NDArray[np.int64]
    """Tiles it can walk in a turn."""
    strike: NDArray[np.int64]
    """Tiles from which it can strike in a turn: its walk plus its reach."""
    forms: NDArray[np.bool_]
    """Whether it has `formation`."""
    gives: NDArray[np.bool_]
    """Whether it gives an aura."""


@cache
def _tables(game: Tactics) -> _Tables:
    """Return what features read of the rules, after checking they apply to them."""
    if tuple(stats.name for stats in game.units) != KINDS:
        raise ValueError(f"these features are for the kinds {KINDS}")
    walk = [(2 if stats.double_move else 1) * stats.move for stats in game.units]
    return _Tables(
        np.array([stats.hp for stats in game.units], dtype=np.float64),
        np.array([stats.atk for stats in game.units], dtype=np.float64),
        np.array(walk, dtype=np.int64),
        np.array(
            [w + int(stats.reach) for w, stats in zip(walk, game.units, strict=True)],
            dtype=np.int64,
        ),
        np.array([stats.formation for stats in game.units]),
        np.array([stats.aura for stats in game.units]),
    )


def _walks(
    state: TacticsState, sources: NDArray[np.bool_] | NDArray[np.intp]
) -> NDArray[np.float64]:
    """Return each tile's steps to the nearest source, shape (n_cells,), around walls and rubble.

    A tile no source reaches counts as ``n_cells`` steps, so differences stay finite.
    """
    return np.minimum(steps(state.space, sources, blocked=~state.passable), state.n_cells)


def value_features(game: Tactics, positions: Sequence[TacticsPosition]) -> NDArray[np.float64]:
    """Return the features of positions (`VALUE_FEATURES`), each player 0's way.

    Parameters
    ----------
    game : Tactics
        The rules, with the default kinds of unit.
    positions : sequence of TacticsPosition
        Positions not over.

    Returns
    -------
    ndarray of float64, shape (n_positions, len(VALUE_FEATURES))
        The features.

    Raises
    ------
    ValueError
        If the rules have other kinds of unit.
    """
    tables = _tables(game)
    features = np.zeros((len(positions), len(VALUE_FEATURES)))
    for row, position in zip(features, positions, strict=True):
        _value_row(game, tables, position, row)
    return features


def _value_row(
    game: Tactics, tables: _Tables, position: TacticsPosition, row: NDArray[np.float64]
) -> None:
    """Write a position's features into ``row``, float64 of shape (len(VALUE_FEATURES),)."""
    state = position.state
    u = state.units
    cells = state.occupied.nonzero()[0]
    team, kinds = u.team[cells], u.kind[cells]
    sign = 1.0 - 2.0 * team
    n_kinds = len(KINDS)
    row[:n_kinds] = np.bincount(
        kinds, weights=sign * u.hp[cells] / tables.hp[kinds], minlength=n_kinds
    )
    owner, clock = position.hold
    owned = 0.0 if owner < 0 else 1.0 - 2.0 * owner
    row[n_kinds + 1] = owned
    row[n_kinds + 2] = owned * clock / game.hold_rounds
    if position.deploying:
        row[n_kinds] = -sign.sum()  # placed so far, so what is left to place is the other way
        return
    standing = u.team[state.point]
    if standing.size and standing[0] >= 0:
        row[n_kinds + 3] = 1.0 - 2.0 * standing[0]
    walks = _walks(state, state.point)[cells]
    mine, theirs = team == 0, team == 1
    longest = float(max(game.shape))
    row[n_kinds + 4] = max(-1.0, min(1.0, (walks[theirs].min() - walks[mine].min()) / longest))
    row[n_kinds + 5] = max(-1.0, min(1.0, (walks[theirs].mean() - walks[mine].mean()) / longest))
    row[n_kinds + 6] = (sign * (walks <= tables.walk[kinds])).sum()
    apart = hops(state.space, cells[:, None], cells)  # (n_units, n_units)
    enemies = team[:, None] != team
    within = (apart <= tables.strike[kinds][:, None]) & enemies
    row[n_kinds + 7] = (sign * tables.atk[kinds] * within.sum(axis=1)).sum()
    row[n_kinds + 8] = (sign * ~u.done[cells]).sum()
    if owner < 0:
        row[n_kinds + 9] = position.round / game.capture_by


_VERB_FEATURE = np.array(
    [
        MOVE_FEATURES.index(name)
        for name in (
            "is_move",
            "is_attack",
            "is_defer",
            "is_swap",
            *["is_face"] * (Verb.END - Verb.FACE),
            "is_end",
            "is_build",
            "is_break",
        )
    ],
    dtype=np.int64,
)
"""The column of each verb below `Verb.PLACE`, shape (Verb.PLACE,)."""


_COLUMN = {name: j for j, name in enumerate(MOVE_FEATURES)}
"""The column of each name in `MOVE_FEATURES`."""


def move_features(
    game: Tactics, position: TacticsPosition, legal: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Return the features of a position's legal actions (`MOVE_FEATURES`).

    Parameters
    ----------
    game : Tactics
        The rules, with the default kinds of unit.
    position : TacticsPosition
        A position not over.
    legal : ndarray of bool, shape (n_verbs, n_cells)
        ``game.legal(position)``.

    Returns
    -------
    ndarray of float64, shape (n_legal, len(MOVE_FEATURES))
        The features, in the order of ``np.flatnonzero(legal)``.

    Raises
    ------
    ValueError
        If the rules have other kinds of unit.
    """
    tables = _tables(game)
    verbs, tiles = legal.nonzero()  # in the order of flatnonzero, row by row
    features = np.zeros((verbs.size, len(MOVE_FEATURES)))
    if position.deploying:
        _place_columns(game, position, verbs, tiles, features)
        return features
    features[np.arange(verbs.size), _VERB_FEATURE[verbs]] = 1.0
    state = position.state
    u = state.units
    a = game.active(position)
    enemies = (state.occupied & (u.team != u.team[a])).nonzero()[0]
    for i in (verbs == Verb.ATTACK).nonzero()[0]:
        features[i, _COLUMN["damage"] : _COLUMN["damage"] + 2] = _attack(
            game, position, a, a, int(tiles[i])
        )
    moves = (verbs == Verb.MOVE).nonzero()[0]
    if moves.size:
        _move_columns(game, tables, position, a, enemies, moves, tiles[moves], features)
    if enemies.size:
        _turn_columns(position, a, enemies, verbs, tiles, features)
    work = ((verbs == Verb.BUILD) | (verbs == Verb.BREAK)).nonzero()[0]
    if work.size:
        _work_column(position, a, enemies, verbs, tiles, work, features)
    return features


def _attack(
    game: Tactics, position: TacticsPosition, a: int, at: int, target: int
) -> tuple[float, float]:
    """Return an attack's expected damage, and its chance of a kill times what the kill is worth."""
    u = position.state.units
    odds = game.odds(position, a, target, at=at)
    hp = int(u.hp[target])
    stats = game.units[u.kind[target]]
    kill = odds.chance * (stats.hp + stats.atk) if odds.atk >= hp else 0.0
    return odds.chance * min(odds.atk, hp), kill


def _place_columns(
    game: Tactics,
    position: TacticsPosition,
    verbs: NDArray[np.intp],
    tiles: NDArray[np.intp],
    features: NDArray[np.float64],
) -> None:
    """Write the columns of placements: the kind, and how far ahead the tile is."""
    state = position.state
    kinds = verbs - Verb.PLACE
    features[np.arange(verbs.size), _COLUMN["place_" + KINDS[0]] + kinds] = 1.0
    walks = _walks(state, state.point)[tiles]
    ahead = walks.max() - walks
    ranged = np.array([game.units[k].reach > 1 for k in range(len(game.units))])[kinds]
    features[:, _COLUMN["place_ahead_melee"]] = ahead * ~ranged
    features[:, _COLUMN["place_ahead_ranged"]] = ahead * ranged


def _move_columns(
    game: Tactics,
    tables: _Tables,
    position: TacticsPosition,
    a: int,
    enemies: NDArray[np.intp],
    moves: NDArray[np.intp],
    cells: NDArray[np.intp],
    features: NDArray[np.float64],
) -> None:
    """Write the columns of moves, given their rows ``moves`` and tiles ``cells``, shape (n,)."""
    state = position.state
    u = state.units
    me = game.units[u.kind[a]]
    side = int(u.team[a])
    goals = objective.wanted(state.point, position.hold, u.team, side, a)
    if goals.size:
        walks = _walks(state, goals)
        features[moves, _COLUMN["closer_point"]] = walks[a] - walks[cells]
    if enemies.size:
        nearest = _walks(state, enemies)
        want = int(me.reach)
        features[moves, _COLUMN["closer_enemy"]] = np.abs(nearest[a] - want) - np.abs(
            nearest[cells] - want
        )
        apart = hops(state.space, cells[:, None], enemies)  # (n, n_enemies)
        kinds = u.kind[enemies]
        reach, atk = tables.strike[kinds], tables.atk[kinds]
        here = ((hops(state.space, a, enemies) <= reach) * atk).sum()
        features[moves, _COLUMN["exposure"]] = ((apart <= reach) * atk).sum(axis=1) - here
        if not u.attacked[a]:
            _attack_after(game, position, a, enemies, moves, cells, apart, features)
    friends = (state.occupied & (u.team == side)).nonzero()[0]
    friends = friends[friends != a]
    if friends.size:
        apart = hops(state.space, cells[:, None], friends)
        kinds = u.kind[friends]
        if me.formation:
            formed = ((apart == 1) & tables.forms[kinds]).sum(axis=1)
            features[moves, _COLUMN["formation"]] = np.minimum(formed, game.formation_cap)
        near = apart <= game.aura_radius
        features[moves, _COLUMN["aura"]] = (
            near.sum(axis=1) if me.aura else (near & tables.gives[kinds]).any(axis=1)
        )


def _attack_after(
    game: Tactics,
    position: TacticsPosition,
    a: int,
    enemies: NDArray[np.intp],
    moves: NDArray[np.intp],
    cells: NDArray[np.intp],
    apart: NDArray[np.int64],
    features: NDArray[np.float64],
) -> None:
    """Write the two columns of the best attack from each tile, ignoring walls."""
    state = position.state
    me = game.units[state.units.kind[a]]
    within = apart <= me.reach  # (n, n_enemies)
    if me.double_move:  # a knight moving past its move can't attack
        within[hops(state.space, cells, a) > me.move] = False
    column = _COLUMN["damage_after"]
    rows, targets = within.nonzero()
    for i, e in zip(rows, targets, strict=True):
        damage, kill = _attack(game, position, a, int(cells[i]), int(enemies[e]))
        if damage + kill > features[moves[i], column] + features[moves[i], column + 1]:
            features[moves[i], column : column + 2] = damage, kill


def _turn_columns(
    position: TacticsPosition,
    a: int,
    enemies: NDArray[np.intp],
    verbs: NDArray[np.intp],
    tiles: NDArray[np.intp],
    features: NDArray[np.float64],
) -> None:
    """Write the columns of turning and of deferred attacks."""
    state = position.state
    space = state.space
    turns = (verbs >= Verb.FACE) & (verbs < Verb.END)
    if turns.any():
        target = enemies[hops(space, enemies, a).argmin()]
        toward = snap(space, space.coords[target] - space.coords[a], state.units.facing[a])
        features[turns, _COLUMN["faces_enemy"]] = verbs[turns] - Verb.FACE == toward
    defers = (verbs == Verb.DEFER).nonzero()[0]
    if defers.size:
        cells = tiles[defers]
        features[defers, _COLUMN["defer_on_point"]] = state.point[cells]
        features[defers, _COLUMN["defer_away"]] = hops(space, cells[:, None], enemies).min(axis=1)


def _work_column(
    position: TacticsPosition,
    a: int,
    enemies: NDArray[np.intp],
    verbs: NDArray[np.intp],
    tiles: NDArray[np.intp],
    work: NDArray[np.intp],
    features: NDArray[np.float64],
) -> None:
    """Write the column of work: the steps added to the enemies' walks, less the friends'."""
    state = position.state
    if not state.point.any():
        return
    friends = (state.occupied & (state.units.team == state.units.team[a])).nonzero()[0]

    def lead(blocked: NDArray[np.bool_]) -> float:
        walks = np.minimum(steps(state.space, state.point, blocked=blocked), state.n_cells)
        return float(walks[enemies].sum() - walks[friends].sum())

    blocked = ~state.passable
    now = lead(blocked)
    for i in work:
        tile = int(tiles[i])
        build = verbs[i] == Verb.BUILD
        if state.passable[tile] if build else state.rubble[tile]:  # the rest leave walks alone
            changed = blocked.copy()
            changed[tile] = build
            features[i, _COLUMN["work_walks"]] = lead(changed) - now


@dataclass(frozen=True)
class CodeEvaluator:
    """Judges positions by named features and constants, for the AlphaZero search.

    The value of a position for player 0 is ``sigmoid(intercept + sum_j value[j] * f[j])`` over
    `value_features`. The prior's logit of an action is ``sum_j prior[j] * g[j]`` over
    `move_features`.

    Parameters
    ----------
    value : tuple of float
        A weight per name in `VALUE_FEATURES`, in logits.
    intercept : float
        The value's score when every feature is 0, in logits: the lean of the seats.
    prior : tuple of float
        A weight per name in `MOVE_FEATURES`, in logits.

    Raises
    ------
    TypeError, ValueError
        If a constant is not a finite real, or a tuple has the wrong length.
    """

    value: tuple[float, ...]
    intercept: float
    prior: tuple[float, ...]

    def __post_init__(self) -> None:
        for weights, names, name in (
            (self.value, VALUE_FEATURES, "value"),
            (self.prior, MOVE_FEATURES, "prior"),
        ):
            if not isinstance(weights, tuple):
                raise TypeError(f"{name} must be a tuple, got {type(weights).__name__}")
            if len(weights) != len(names):
                raise ValueError(f"{name} needs {len(names)} weights, got {len(weights)}")
            for weight in weights:
                check_real(weight, f"a weight of {name}")
        check_real(self.intercept, "intercept")

    def __call__(
        self,
        game: Tactics,
        positions: Sequence[TacticsPosition],
        legal: Sequence[NDArray[np.bool_]],
        /,
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the prior's logits and player 0's value of each position (see `Evaluator`).

        Raises
        ------
        ValueError
            If the rules have other kinds of unit, or a score overflows.
        """
        values = sigmoid(weighted(value_features(game, positions), self.value, self.intercept))
        logits = [
            weighted(move_features(game, position, mask), self.prior, 0.0)
            for position, mask in zip(positions, legal, strict=True)
        ]
        return logits, values

    def save(self, path: str | Path) -> None:
        """Write the constants as JSON, by name, so they can be read and ported.

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        Path(path).write_text(
            json.dumps(
                {
                    "value": dict(zip(VALUE_FEATURES, self.value, strict=True)),
                    "intercept": self.intercept,
                    "prior": dict(zip(MOVE_FEATURES, self.prior, strict=True)),
                },
                indent=2,
            )
        )

    @staticmethod
    def load(path: str | Path) -> CodeEvaluator:
        """Read constants written by `save`.

        Parameters
        ----------
        path : str or Path
            The file to read.

        Returns
        -------
        CodeEvaluator
            The evaluator.

        Raises
        ------
        KeyError
            If a feature's weight is missing: the file is of another set of features.
        """
        data = json.loads(Path(path).read_text())
        return CodeEvaluator(
            tuple(float(data["value"][name]) for name in VALUE_FEATURES),
            float(data["intercept"]),
            tuple(float(data["prior"][name]) for name in MOVE_FEATURES),
        )


def _by_name(names: tuple[str, ...], **weights: float) -> tuple[float, ...]:
    """Return ``weights`` in the order of ``names``, 0 for those not given.

    Raises
    ------
    ValueError
        If a weight names no feature.
    """
    unknown = set(weights) - set(names)
    if unknown:
        raise ValueError(f"no such features: {sorted(unknown)}")
    return tuple(weights.get(name, 0.0) for name in names)


HAND_PICKED = CodeEvaluator(
    value=_by_name(
        VALUE_FEATURES,
        hp_footsoldier=0.5,
        hp_archer=0.4,
        hp_knight=0.8,
        hp_captain=1.0,
        hp_engineer=0.4,
        to_place=0.5,
        owner=0.3,
        clock=2.0,
        on_point=0.2,
        walk_nearest=1.0,
        walk_mean=1.0,
        near_point=0.1,
        threat=0.01,
        to_act=0.1,
        deadline=0.5,
    ),
    intercept=0.0,
    prior=_by_name(
        MOVE_FEATURES,
        is_move=0.5,
        is_attack=0.5,
        is_defer=-0.5,
        is_swap=-2.0,
        is_face=-1.0,
        is_end=-1.5,
        is_build=-1.0,
        is_break=-1.0,
        damage=0.5,
        kill=0.3,
        damage_after=0.5,
        kill_after=0.3,
        closer_point=0.5,
        closer_enemy=0.25,
        exposure=-0.05,
        formation=0.15,
        aura=0.15,
        faces_enemy=2.0,
        defer_on_point=1.0,
        defer_away=-0.2,
        work_walks=0.25,
        place_footsoldier=0.5,
        place_archer=0.3,
        place_knight=1.5,
        place_captain=2.0,
        place_ahead_melee=0.5,
    ),
)
"""Constants picked by hand, not fitted: a footsoldier's lead is worth half a logit, and the point
about as much as a captain and a knight once nearly held. Moves lean the way `GreedyPlayer` does,
about half as strongly, so that the search can still overrule them."""
