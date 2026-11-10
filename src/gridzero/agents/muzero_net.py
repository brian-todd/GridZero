"""The network of a MuZero player: a latent state, its dynamics, and predictions from it.

`WorldNet` has MuZero's three functions, over cells of any space (`agents.net.CellConv`):

- ``represent``: a player's observation (`agents.Observable.observe`) to a latent state, one
  vector per cell;
- ``dynamics``: a latent state and the player's action to the latent state at its next
  decision, with whatever the world did in between (the other players, the dice);
- ``predict``: from a latent state, a policy over actions, the value for the player, and a
  belief: the whole position as the player would see it with nothing hidden
  (`agents.Revealable.reveal`), which play never reads but which shows what the latent holds.

Every latent state is scaled into [0, 1] over its channels and cells, as in MuZero, so that
those from ``represent`` and from ``dynamics`` share one range. `NetModel` is the network as an
`agents.muzero.WorldModel`, for `agents.muzero.MuZeroPlayer`.

Its convolutions treat a cell's neighbors alike, which a planet needs, having no directions. On
a lattice that leaves it unable to tell them: on Tactics' hexes (2026-11-09) it chose the
direction a unit turns to as a code player did 44% of the time, where that player's own move
scores did 83%. For a game on a lattice that hides nothing, `agents.net.PolicyNet` sees
directions.

This module imports Torch, so the package does not import it: import it as
``gridzero.agents.muzero_net``.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from gridzero.agents.frame import Revealable
from gridzero.agents.net import CellConv, _Block
from gridzero.core._arrays import check_int


class Prediction(NamedTuple):
    """What `WorldNet.predict` gives for a batch of latent states."""

    logits: Tensor
    """A score per action, float32 of shape (batch, n_verbs, n_cells), before masking."""
    value: Tensor
    """The value for the player, float32 of shape (batch,), in [-1, 1]."""
    belief: Tensor
    """The logits of the revealed planes, float32 of shape (batch, n_revealed, n_cells): their
    sigmoid guesses ``game.reveal(...)``."""


class WorldNet(nn.Module):
    """MuZero's representation, dynamics and prediction, on the cells of a game's space.

    Each function is a convolution in, then residual blocks of `CellConv` (each sees one cell
    further). The policy scores each verb on each cell, plus a bias per verb from the mean over
    cells, as `agents.net.PolicyNet` does; the value pools the mean and maximum over cells.
    An action goes into ``dynamics`` as ``2 * n_verbs`` planes: for each verb, a one on the
    action's cell if it is that verb, and a plane of ones if it is, so that an action naming
    one cell can change all of them.

    Parameters
    ----------
    game : Revealable
        The rules: they fix the planes, the revealed planes, the verbs and the space.
    channels : int, default 64
        Channels of the latent state, at least 1.
    blocks : int, default 4
        Residual blocks of the representation and of the dynamics, at least 0.

    Raises
    ------
    TypeError, ValueError
        If ``channels`` or ``blocks`` is not an integer in range.
    """

    def __init__(self, game: Revealable[Any, Any], channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        channels = check_int(channels, "channels", minimum=1)
        blocks = check_int(blocks, "blocks", minimum=0)
        space = game.space
        self.n_cells = space.n_cells
        self.n_planes = len(game.planes)
        self.n_revealed = len(game.revealed)
        self.n_verbs = game.n_verbs
        self.channels = channels
        self.blocks = blocks
        conv = functools.partial(CellConv, space)
        self.stem = CellConv(space, self.n_planes, channels)
        self.body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
        self.dynamics_stem = CellConv(space, channels + 2 * self.n_verbs, channels)
        self.dynamics_body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
        self.policy_conv = CellConv(space, channels, channels)
        self.policy_out = nn.Conv1d(channels, self.n_verbs, kernel_size=1)
        self.policy_bias = nn.Linear(channels, self.n_verbs)
        half = channels // 2 or 1
        self.value_conv = nn.Conv1d(channels, half, kernel_size=1)
        self.value_hidden = nn.Linear(2 * half, channels)
        self.value_out = nn.Linear(channels, 1)
        self.belief_conv = CellConv(space, channels, channels)
        self.belief_out = nn.Conv1d(channels, self.n_revealed, kernel_size=1)

    def represent(self, obs: Tensor) -> Tensor:
        """Return the latent states of observations.

        Parameters
        ----------
        obs : Tensor of float32, shape (batch, n_planes, n_cells)
            Observations (`agents.Observable.observe`), each in its player's frame.

        Returns
        -------
        Tensor of float32, shape (batch, channels, n_cells)
            Latent states, each scaled into [0, 1].
        """
        return _scaled(self.body(F.relu(self.stem(obs))))

    def dynamics(self, latent: Tensor, action: Tensor) -> Tensor:
        """Return the latent states at the player's next decisions, after its actions.

        Parameters
        ----------
        latent : Tensor of float32, shape (batch, channels, n_cells)
            Latent states.
        action : Tensor of int64, shape (batch,)
            The player's action in each, in its frame: ``verb * n_cells + cell``.

        Returns
        -------
        Tensor of float32, shape (batch, channels, n_cells)
            The next latent states, each scaled into [0, 1].
        """
        planes = self.action_planes(action).to(latent.dtype)
        x = F.relu(self.dynamics_stem(torch.cat([latent, planes], dim=1)))
        return _scaled(self.dynamics_body(x))

    def action_planes(self, action: Tensor) -> Tensor:
        """Return actions as the dynamics read them.

        Parameters
        ----------
        action : Tensor of int64, shape (batch,)
            Actions in the player's frame: ``verb * n_cells + cell``.

        Returns
        -------
        Tensor of float32, shape (batch, 2 * n_verbs, n_cells)
            Plane ``v`` is 1 on the action's cell if its verb is ``v``; plane ``n_verbs + v``
            is 1 everywhere if its verb is ``v``. The rest are 0.
        """
        batch = action.shape[0]
        rows = torch.arange(batch, device=action.device)
        verb = torch.div(action, self.n_cells, rounding_mode="floor")
        planes = torch.zeros(batch, 2 * self.n_verbs, self.n_cells, device=action.device)
        planes[rows, verb, action % self.n_cells] = 1.0
        planes[rows, self.n_verbs + verb] = 1.0
        return planes

    def predict(self, latent: Tensor) -> Prediction:
        """Return the policy's scores, the value and the belief of latent states.

        Parameters
        ----------
        latent : Tensor of float32, shape (batch, channels, n_cells)
            Latent states.

        Returns
        -------
        Prediction
            Scores, values and the belief's logits.
        """
        mean = latent.mean(dim=2)
        p = self.policy_out(F.relu(self.policy_conv(latent)))
        logits = p + self.policy_bias(mean)[:, :, None]
        v = F.relu(self.value_conv(latent))
        pooled = torch.cat([v.mean(dim=2), v.amax(dim=2)], dim=1)
        value = torch.tanh(self.value_out(F.relu(self.value_hidden(pooled))))[:, 0]
        belief = self.belief_out(F.relu(self.belief_conv(latent)))
        return Prediction(logits, value, belief)

    def forward(self, obs: Tensor) -> Prediction:  # ty: ignore[invalid-mutable-override]  # see net._Block
        """Return `predict` of `represent`: the predictions of observations."""
        return self.predict(self.represent(obs))


def _scaled(x: Tensor) -> Tensor:
    """Return each state of a batch scaled into [0, 1] over all its channels and cells."""
    flat = x.flatten(1)
    low = flat.min(dim=1).values[:, None, None]
    high = flat.max(dim=1).values[:, None, None]
    return (x - low) / torch.clamp(high - low, min=1e-5)


@dataclass(frozen=True, eq=False)
class NetModel:
    """A `WorldNet` as an `agents.muzero.WorldModel`: states are its tensors.

    It runs where the network's parameters are, without gradients. A player in a series sent to
    other processes carries it there, so keep the network on the CPU for that.

    Parameters
    ----------
    net : WorldNet
        The network, in evaluation mode.
    """

    net: WorldNet

    def initial(
        self, obs: Sequence[NDArray[np.float32]]
    ) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states of observations, their scores and values (see the protocol)."""
        device = next(self.net.parameters()).device
        with torch.no_grad():
            latent = self.net.represent(torch.from_numpy(np.stack(obs)).to(device))
            return _judged(self.net, latent)

    def recurrent(
        self, latents: Sequence[Tensor], actions: NDArray[np.int64]
    ) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states after actions, their scores and values (see the protocol)."""
        with torch.no_grad():
            latent = torch.stack(list(latents))
            action = torch.from_numpy(np.asarray(actions, dtype=np.int64)).to(latent.device)
            return _judged(self.net, self.net.dynamics(latent, action))


def _judged(
    net: WorldNet, latent: Tensor
) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
    """Return the states one by one, with their scores and values as float64 arrays."""
    pred = net.predict(latent)
    logits = pred.logits.flatten(1).double().cpu().numpy()
    return list(latent), logits, pred.value.double().cpu().numpy()
