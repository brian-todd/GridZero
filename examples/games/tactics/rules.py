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
        moved more than ``move``. Knights have it.
    aura : bool, default False
        Gives friendly units within ``Tactics.aura_radius``, itself included,
        ``Tactics.aura_bonus`` Atk and Def. Auras don't stack. Captains have it.

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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {self.name!r}")
        check_int(self.hp, "hp", minimum=1)
        check_int(self.atk, "atk", minimum=0)
        check_int(self.defense, "defense", minimum=0)
        check_int(self.move, "move", minimum=0)
        if check_real(self.reach, "reach") <= 0:
            raise ValueError(f"reach must be positive, got {self.reach}")
        for flag in ("formation", "defers", "double_move", "aura"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a bool, got {getattr(self, flag)!r}")


FOOTSOLDIER = UnitStats("footsoldier", hp=5, atk=4, defense=2, move=2, reach=1, formation=True)
ARCHER = UnitStats("archer", hp=3, atk=2, defense=1, move=2, reach=3, defers=True)
KNIGHT = UnitStats("knight", hp=7, atk=3, defense=2, move=3, reach=1, double_move=True)
CAPTAIN = UnitStats("captain", hp=9, atk=4, defense=3, move=2, reach=1, aura=True)
DEFAULT_UNITS = (FOOTSOLDIER, ARCHER, KNIGHT, CAPTAIN)
"""The four kinds of the first rules, in the order of their kind numbers 0 to 3."""
DEFAULT_COUNTS = (6, 3, 2, 1)
"""How many units of each kind of `DEFAULT_UNITS` each side has: 12 in all."""


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
