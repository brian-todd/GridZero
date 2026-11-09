"""The units of Tactics and the arithmetic of attacks: stats as data, and the d20 rule.

Every number is a parameter, so that self-play can change the rules between games. A kind of
unit is its index into the game's tuple of `UnitStats`. The passives are flags on the stats, not
tied to a kind, so a variant may give any unit any passive.
"""

from dataclasses import dataclass

from gridzero.core import check_int, check_real


@dataclass(frozen=True)
class UnitStats:
    """What one kind of unit is: its numbers and its passives.

    Parameters
    ----------
    name : str
        The kind's name, such as ``"archer"``.
    hp : int
        Hit points when deployed, at least 1.
    atk : int
        Attack, at least 0: added to hit chances, and the damage of a hit.
    defense : int
        Defense, at least 0: subtracted from attackers' hit chances.
    move : int
        Tiles it may move in a turn, at least 0.
    reach : float
        Attack range, between centers, positive: 1 for adjacent units only. On hexes, a reach
        of 3 or less is the same as that many steps.
    formation : bool, default False
        Gets ``Tactics.formation_bonus`` Def for each adjacent friendly unit with `formation`,
        up to ``Tactics.formation_cap``. Footsoldiers have it.
    defers : bool, default False
        May defer its attack onto a tile instead of attacking. Archers have it.
    double_move : bool, default False
        Moves twice ``move`` in a turn when it doesn't attack, and can't attack once it has
        moved more than ``move``. No default kind has it.
    aura : bool, default False
        Gives friendly units within ``Tactics.aura_radius``, itself included,
        ``Tactics.aura_bonus`` Atk and Def. Auras don't stack. Captains have it.
    builds : bool, default False
        May spend its attack on building or breaking an adjacent tile. Engineers have it.
    charge : int, default 0
        Atk added, before flanking multiplies it, to an attack made after moving at least ``move``
        tiles this round, at least 0. A charge. Knights have it.
    braces : bool, default False
        When a friendly unit with `formation` stands adjacent and it isn't engaged (see
        ``Tactics.engaged_until_turn``), units that charge can't flank it from any side, and
        their charge is stopped: the charger loses its bonus and is struck first, by this unit's
        Atk, unflanked. Others flank it as usual. Footsoldiers have it.
    places : int, default 1
        How many of a team's ``Tactics.team_size`` places the first unit of this kind takes, at
        least 1.
    extra_places : int, default 0
        How many more places each further unit of this kind takes than the one before, at least
        0. Knights have 1: a second knight takes 2 places.

    Raises
    ------
    TypeError
        If a number is not an integer (``reach``: a real number), or a flag not a bool.
    ValueError
        If a number is out of range.
    """

    name: str
    hp: int
    atk: int
    defense: int
    move: int
    reach: float
    formation: bool = False
    defers: bool = False
    double_move: bool = False
    aura: bool = False
    builds: bool = False
    charge: int = 0
    braces: bool = False
    places: int = 1
    extra_places: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {self.name!r}")
        check_int(self.hp, "hp", minimum=1)
        check_int(self.atk, "atk", minimum=0)
        check_int(self.defense, "defense", minimum=0)
        check_int(self.move, "move", minimum=0)
        check_int(self.charge, "charge", minimum=0)
        check_int(self.places, "places", minimum=1)
        check_int(self.extra_places, "extra_places", minimum=0)
        if check_real(self.reach, "reach") <= 0:
            raise ValueError(f"reach must be positive, got {self.reach}")
        for flag in ("formation", "defers", "double_move", "aura", "builds", "braces"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a bool, got {getattr(self, flag)!r}")

    def places_for(self, count: int) -> int:
        """Return the places that ``count`` units of this kind take together.

        Parameters
        ----------
        count : int
            How many, at least 0.

        Returns
        -------
        int
            ``count * places + extra_places * count * (count - 1) // 2``.
        """
        return count * self.places + self.extra_places * count * (count - 1) // 2


FOOTSOLDIER = UnitStats(
    "footsoldier", hp=5, atk=4, defense=2, move=2, reach=1, formation=True, braces=True
)
ARCHER = UnitStats("archer", hp=3, atk=2, defense=1, move=2, reach=3, defers=True)
KNIGHT = UnitStats("knight", hp=7, atk=2, defense=2, move=3, reach=1, charge=2, extra_places=1)
CAPTAIN = UnitStats("captain", hp=9, atk=4, defense=3, move=2, reach=1, aura=True)
ENGINEER = UnitStats("engineer", hp=3, atk=2, defense=1, move=2, reach=3, builds=True)
DEFAULT_UNITS = (FOOTSOLDIER, ARCHER, KNIGHT, CAPTAIN, ENGINEER)
"""The kinds of the default rules, in the order of their kind numbers 0 to 4."""
DEFAULT_CAPS = (6, 3, 2, 1, 1)
"""The most units of each kind of `DEFAULT_UNITS` a side may field."""


def default_roster(
    caps: tuple[int, ...], team_size: int, units: tuple[UnitStats, ...] | None = None
) -> tuple[int, ...]:
    """Return a team that fields every kind it can, then follows the caps.

    One of each kind with a cap, in order while it fits, then one unit at a time to the kind
    with the most of its spare cap per unit already added (``spare / (added + 1)``, ties to the
    lower kind), among the kinds whose next unit still fits. With caps of 6/3/2/1/1 and a team
    of 8 that is 3/2/1/1/1.

    Parameters
    ----------
    caps : tuple of int
        The most of each kind a side may field, per kind.
    team_size : int
        How many places to fill, at most what the caps can field.
    units : tuple of UnitStats, optional
        The kinds, for the places their units take (`UnitStats.places_for`); a place a unit
        by default.

    Returns
    -------
    tuple of int
        How many of each kind, each within its cap, taking at most ``team_size`` places: all of
        them unless no kind fits in what is left.
    """

    def next_places(kind: int, have: int) -> int:
        if units is None:
            return 1
        return units[kind].places_for(have + 1) - units[kind].places_for(have)

    roster = [0] * len(caps)
    left = team_size
    for kind, cap in enumerate(caps):
        if cap > 0 and next_places(kind, 0) <= left:
            roster[kind] = 1
            left -= next_places(kind, 0)
    spare = [cap - have for cap, have in zip(caps, roster, strict=True)]
    added = [0] * len(caps)
    while True:
        fits = [
            k
            for k in range(len(caps))
            if added[k] < spare[k] and next_places(k, roster[k] + added[k]) <= left
        ]
        if not fits:
            return tuple(have + more for have, more in zip(roster, added, strict=True))
        kind = max(fits, key=lambda k: (spare[k] / (added[k] + 1), -k))
        left -= next_places(kind, roster[kind] + added[kind])
        added[kind] += 1


def roll_needed(atk: int, defense: int, base: int = 8, step: int = 2) -> int:
    """Return the d20 roll an attack needs to hit, before the rules for 1 and 20.

    Parameters
    ----------
    atk, defense : int
        The attacker's effective Atk and the target's effective Def.
    base : int, default 8
        The roll needed when they are equal: 8 or more is a 65% chance.
    step : int, default 2
        How much each point of Atk over Def lowers it: 2 faces, or 10%.

    Returns
    -------
    int
        ``base - step * (atk - defense)``. It may lie outside 1 to 20.
    """
    return base - step * (atk - defense)


def hits(roll: int, needed: int) -> bool:
    """Return whether a d20 roll hits: a 1 always misses and a 20 always hits.

    Parameters
    ----------
    roll : int
        The roll, 1 to 20.
    needed : int
        The roll needed, from `roll_needed`.

    Returns
    -------
    bool
        True on a hit.
    """
    return roll == 20 or (roll != 1 and roll >= needed)


def hit_chance(atk: int, defense: int, base: int = 8, step: int = 2) -> float:
    """Return the chance that an attack hits, from 0.05 to 0.95.

    Parameters
    ----------
    atk, defense : int
        The attacker's effective Atk and the target's effective Def.
    base, step : int
        As for `roll_needed`.

    Returns
    -------
    float
        The share of the 20 rolls that hit.
    """
    needed = roll_needed(atk, defense, base, step)
    return sum(hits(roll, needed) for roll in range(1, 21)) / 20
