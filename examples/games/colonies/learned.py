"""Colonies' learned players: the size of network the experiments save, and weights as players.

The weights don't record the network's size nor how it was trained, so the experiments share
one size. A weights file is named by a **spec**: its path, then optionally ``@`` and its
simulations a decision. ``runs/muzero/colonies_scratch_all_gen_028.pt@64`` is that network,
searched with 64 simulations. This module imports Torch, so the package does not import it.
"""

from pathlib import Path

import torch
from torch import Tensor

from gridzero.agents.muzero import MuZeroPlayer
from gridzero.agents.muzero_net import NetModel, WorldNet

from .game import Colonies

CHANNELS, BLOCKS = 64, 4
"""The network every Colonies experiment trains and loads."""

_RETIRED = ("projection.", "projection_head.")
"""Parameters that only a loss removed on 2026-11-09 used, which saved weights may still hold."""


def network(path: str | None = None) -> WorldNet:
    """Return a new network of `CHANNELS` and `BLOCKS`, or the one saved at ``path``.

    Parameters
    ----------
    path : str, optional
        Saved weights, loaded on the CPU. A spec's simulations are ignored.

    Returns
    -------
    WorldNet
        On the CPU, in evaluation mode when loaded. Weights saved from 2026-10-30 to 2026-11-09
        also hold a projection that only training used: it is dropped.

    Raises
    ------
    RuntimeError
        If the weights don't fit the network, the projection aside: those of a network with
        chance codes (Stochastic MuZero, removed on 2026-11-09) among them.
    """
    net = WorldNet(Colonies(), channels=CHANNELS, blocks=BLOCKS)
    if path is not None:
        saved = torch.load(path.split("@")[0], map_location="cpu")
        kept = {k: v for k, v in saved.items() if not k.startswith(_RETIRED)}
        missing, unexpected = net.load_state_dict(kept, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"weights at {path} don't fit: missing {missing}, unexpected {unexpected}"
            )
        net.eval()
    return net


def player(spec: str, simulations: int = 32) -> tuple[str, MuZeroPlayer[Tensor]]:
    """Return the MuZero player a spec names, on the CPU, and its name.

    Parameters
    ----------
    spec : str
        A path, then optionally ``@`` and its simulations a decision (see the module).
    simulations : int, default 32
        Its simulations a decision, unless the spec sets them.

    Returns
    -------
    name : str
        The file's name without its suffix, then the simulations if the spec gave them.
    player : MuZeroPlayer
        Without noise.

    Raises
    ------
    ValueError
        If what follows an ``@`` is not a number.
    """
    path, *tags = spec.split("@")
    for tag in tags:
        if not tag.isdigit():
            raise ValueError(f"after @ comes a number of simulations, got {tag!r} in {spec!r}")
        simulations = int(tag)
    name = "@".join([Path(path).stem, *tags])
    return name, MuZeroPlayer(NetModel(network(path)), simulations)
