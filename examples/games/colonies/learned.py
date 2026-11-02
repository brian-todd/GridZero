"""Colonies' learned players: the size of network the experiments save, and weights as players.

The weights don't record the network's size (finding E2) nor how it was trained, so the
experiments share one size; a network's chance codes (Stochastic MuZero) are read from the
encoder its weights hold. A weights file is named by a **spec**: its path, then tags each
after an ``@``: ``root`` for a network trained before 2026-10-30, whose policy only knew legal
actions, so that its search keeps below the root the actions legal at the root
(`MuZeroPlayer`'s ``below="root"``), and a number for its simulations a decision.
``runs/muzero/colonies_scratch_gen_029.pt@root@64`` is that network, searched the old way, 64
simulations a decision. This module imports Torch, so the package does not import it.
"""

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from gridzero.agents.muzero import MuZeroPlayer
from gridzero.agents.muzero_net import WorldNet, world_model

from .game import Colonies

CHANNELS, BLOCKS = 64, 4
"""The network every Colonies experiment trains and loads."""


def network(path: str | None = None, codes: int = 0) -> WorldNet:
    """Return a new network of `CHANNELS` and `BLOCKS`, or the one saved at ``path``.

    Parameters
    ----------
    path : str, optional
        Saved weights, loaded on the CPU. A spec's tags are ignored.
    codes : int, default 0
        A new network's chance codes (`WorldNet`'s ``codes``); saved weights tell their own.

    Returns
    -------
    WorldNet
        On the CPU, in evaluation mode when loaded. Weights saved before the consistency loss
        (2026-10-30) have no projection, which only training uses: it is left new.

    Raises
    ------
    RuntimeError
        If the weights don't fit the network, the projection aside.
    """
    saved = None if path is None else torch.load(path.split("@")[0], map_location="cpu")
    if saved is not None and "encoder_out.weight" in saved:
        codes = saved["encoder_out.weight"].shape[0]
    net = WorldNet(Colonies(), channels=CHANNELS, blocks=BLOCKS, codes=codes)
    if saved is not None:
        missing, unexpected = net.load_state_dict(saved, strict=False)
        missing = [k for k in missing if not k.startswith(("projection.", "projection_head."))]
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
        A path, then tags each after an ``@`` (see the module).
    simulations : int, default 32
        Its simulations a decision, unless a tag sets them.

    Returns
    -------
    name : str
        The file's name without its suffix, then the tags as given.
    player : MuZeroPlayer
        Without noise.

    Raises
    ------
    ValueError
        If a tag is neither ``root`` nor a number.
    """
    path, *tags = spec.split("@")
    below: Literal["all", "root"] = "all"
    for tag in tags:
        if tag == "root":
            below = "root"
        elif tag.isdigit():
            simulations = int(tag)
        else:
            raise ValueError(f"a tag is root or a number of simulations, got {tag!r} in {spec!r}")
    name = "@".join([Path(path).stem, *tags])
    return name, MuZeroPlayer(world_model(network(path)), simulations, below=below)
