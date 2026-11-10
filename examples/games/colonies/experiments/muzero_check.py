"""Measure a saved world model of Colonies: its network alone, and searched by a MuZero player.

Loads the weights a spec names (`examples.games.colonies.learned`), then plays, on the games
players are scored on (`EVAL_SEED`), on every core, under fog:

- the network alone (`MuZeroPlayer` with no simulations: the best legal order by its policy)
  against the uniform player, the settler and the raider;
- the MuZero player (its search over the network's latent states, without noise) against the
  same three;
- the MuZero player against the network alone.

Neither ever reads more than its view. If the search beats the network alone, planning on the
learned model improves on what the network learned. The network runs on the CPU, in each
worker.

Run with ``uv run python run.py colonies.muzero_check weights [starts] [simulations]``: by
default 25 starts against each rival, and 32 simulations a decision (unless the spec sets
them).
"""

import dataclasses
import time

from examples.games import CORES
from examples.games.colonies import EVAL_SEED, Colonies, Raider, Settler
from examples.games.colonies.learned import player
from gridzero.agents import UniformPlayer
from gridzero.game import ladder


def main(weights: str, starts: str = "25", simulations: str = "32") -> None:
    """Play the network alone and searched against the scripted players and each other.

    Parameters
    ----------
    weights : str
        A spec of saved weights (`examples.games.colonies.learned`).
    starts : str, default "25"
        Starts against each rival, each played once per rotation of the seats.
    simulations : str, default "32"
        The MuZero player's simulations a decision.
    """
    game = Colonies()
    _, agent = player(weights, int(simulations))
    simulations = str(agent.simulations)
    alone = dataclasses.replace(agent, simulations=0)
    scripted = {"uniform": UniformPlayer(), "settler": Settler(), "raider": Raider()}
    for label, who, rivals in (
        ("network alone", alone, scripted),
        (f"MuZero, {simulations} simulations", agent, {**scripted, "network alone": alone}),
    ):
        start = time.perf_counter()
        scores = ladder(game, who, rivals, int(starts), EVAL_SEED, workers=CORES)
        text = ", ".join(f"{name} {s.won}-{s.lost}-{s.drawn}" for name, s in scores.items())
        print(
            f"{label} against {text} (wins-losses-draws), {time.perf_counter() - start:.0f} s.",
            flush=True,
        )
