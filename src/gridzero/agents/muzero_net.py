"""The network of a MuZero player: a latent state, its dynamics, and predictions from it.

`WorldNet` has MuZero's three functions, over cells of any space (`agents.net.CellConv`):

- ``represent``: a player's observation (`agents.Observable.observe`) to a latent state, one
  vector per cell;
- ``dynamics``: a latent state and the player's action to the latent state at its next
  decision, with whatever the world did in between (the other players, the dice);
- ``predict``: from a latent state, a policy over actions, the value for the player, and a
  belief: the whole position as the player would see it with nothing hidden
  (`agents.Revealable.reveal`), which play never reads but which shows what the latent holds.

With ``codes`` (Stochastic MuZero; Antonoglou et al., "Planning in stochastic environments with
a learned model", ICLR 2022), the dynamics go in two steps, through an **afterstate**, the
state just after the player's action and before the world answers:

- ``afterstate``: a latent state and the player's action to the afterstate;
- ``predict_afterstate``: from an afterstate, ``sigma``, scores of the codes (what the world
  will do), and ``Q``, its value for the player;
- ``chance``: an afterstate and a code to the latent state at the player's next decision;
- ``encode``: the player's observations at a decision and at its next one to scores of the
  code of what happened, whose best training takes (VQ-VAE, with one-hot codes).

Training alone uses ``project``, EfficientZero's projection of a latent state, with which
`agents.muzero_train` holds the states the dynamics reach close to those ``represent`` makes of
the observations there (its consistency loss).

Every latent state is scaled into [0, 1] over its channels and cells, as in MuZero, so that
those from ``represent`` and from ``dynamics`` share one range; afterstates are scaled too.
`NetModel` is a network without codes as an `agents.muzero.WorldModel`, `StochasticNetModel` one
with codes as an `agents.muzero.StochasticModel`, and `world_model` either, for
`agents.muzero.MuZeroPlayer`.

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


class AfterPrediction(NamedTuple):
    """What `WorldNet.predict_afterstate` gives for a batch of afterstates."""

    sigma: Tensor
    """A score per code, float32 of shape (batch, codes): what the world will do."""
    value: Tensor
    """The afterstate's value for the player (``Q``), float32 of shape (batch,), in [-1, 1]."""


class WorldNet(nn.Module):
    """MuZero's representation, dynamics and prediction, on the cells of a game's space.

    Each function is a convolution in, then residual blocks of `CellConv` (each sees one cell
    further). The policy scores each verb on each cell, plus a bias per verb from the mean over
    cells, as `agents.net.PolicyNet` does; the value pools the mean and maximum over cells.
    An action goes into ``dynamics`` as ``2 * n_verbs`` planes: for each verb, a one on the
    action's cell if it is that verb, and a plane of ones if it is, so that an action naming
    one cell can change all of them.

    With ``codes``, ``dynamics`` gives way to ``afterstate`` (an action read as ``dynamics``
    reads it) and ``chance`` (a code read as one plane per code, its weight everywhere), each a
    convolution in and residual blocks as ``dynamics`` is; ``predict_afterstate`` pools as the
    value does, and ``encode`` is a convolution over both observations' planes, one residual
    block, and the same pooling.

    Parameters
    ----------
    game : Revealable
        The rules: they fix the planes, the revealed planes, the verbs and the space.
    channels : int, default 64
        Channels of the latent state, at least 1.
    blocks : int, default 4
        Residual blocks of the representation and of each step of the dynamics, at least 0.
    codes : int, default 0
        Chance codes, at least 0: 0 for MuZero's deterministic dynamics, more for Stochastic
        MuZero's afterstates and codes.

    Raises
    ------
    TypeError, ValueError
        If ``channels``, ``blocks`` or ``codes`` is not an integer in range.
    """

    def __init__(
        self, game: Revealable[Any, Any], channels: int = 64, blocks: int = 4, codes: int = 0
    ) -> None:
        super().__init__()
        channels = check_int(channels, "channels", minimum=1)
        blocks = check_int(blocks, "blocks", minimum=0)
        codes = check_int(codes, "codes", minimum=0)
        space = game.space
        self.n_cells = space.n_cells
        self.n_planes = len(game.planes)
        self.n_revealed = len(game.revealed)
        self.n_verbs = game.n_verbs
        self.channels = channels
        self.blocks = blocks
        self.codes = codes
        conv = functools.partial(CellConv, space)
        self.stem = CellConv(space, self.n_planes, channels)
        self.body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
        if codes == 0:
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
        self.projection = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.projection_head = nn.Sequential(
            nn.Conv1d(channels, half, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(half, channels, kernel_size=1),
        )
        if codes > 0:
            self.afterstate_stem = CellConv(space, channels + 2 * self.n_verbs, channels)
            self.afterstate_body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
            self.after_conv = nn.Conv1d(channels, half, kernel_size=1)
            self.after_hidden = nn.Linear(2 * half, channels)
            self.sigma_out = nn.Linear(channels, codes)
            self.q_out = nn.Linear(channels, 1)
            self.chance_stem = CellConv(space, channels + codes, channels)
            self.chance_body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
            self.encoder_stem = CellConv(space, 2 * self.n_planes, channels)
            self.encoder_body = _Block(conv, channels)
            self.encoder_conv = nn.Conv1d(channels, half, kernel_size=1)
            self.encoder_out = nn.Linear(2 * half, codes)

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

        Raises
        ------
        ValueError
            If the network has codes: its dynamics are ``afterstate``, then ``chance``.
        """
        if self.codes > 0:
            raise ValueError("a network with codes steps by afterstate, then chance")
        planes = self.action_planes(action).to(latent.dtype)
        x = F.relu(self.dynamics_stem(torch.cat([latent, planes], dim=1)))
        return _scaled(self.dynamics_body(x))

    def afterstate(self, latent: Tensor, action: Tensor) -> Tensor:
        """Return the afterstates of the player's actions: before the world answers.

        Parameters
        ----------
        latent : Tensor of float32, shape (batch, channels, n_cells)
            Latent states.
        action : Tensor of int64, shape (batch,)
            The player's action in each, in its frame: ``verb * n_cells + cell``.

        Returns
        -------
        Tensor of float32, shape (batch, channels, n_cells)
            The afterstates, each scaled into [0, 1].

        Raises
        ------
        ValueError
            If the network has no codes.
        """
        self._needs_codes()
        planes = self.action_planes(action).to(latent.dtype)
        x = F.relu(self.afterstate_stem(torch.cat([latent, planes], dim=1)))
        return _scaled(self.afterstate_body(x))

    def predict_afterstate(self, afterstate: Tensor) -> AfterPrediction:
        """Return the codes' scores and the value of afterstates.

        Parameters
        ----------
        afterstate : Tensor of float32, shape (batch, channels, n_cells)
            Afterstates.

        Returns
        -------
        AfterPrediction
            ``sigma`` and ``Q``.

        Raises
        ------
        ValueError
            If the network has no codes.
        """
        self._needs_codes()
        a = F.relu(self.after_conv(afterstate))
        hidden = F.relu(self.after_hidden(torch.cat([a.mean(dim=2), a.amax(dim=2)], dim=1)))
        return AfterPrediction(self.sigma_out(hidden), torch.tanh(self.q_out(hidden))[:, 0])

    def chance(self, afterstate: Tensor, code: Tensor) -> Tensor:
        """Return the latent states at the player's next decisions, after the world's answers.

        Parameters
        ----------
        afterstate : Tensor of float32, shape (batch, channels, n_cells)
            Afterstates.
        code : Tensor of float32, shape (batch, codes)
            What the world did, as weights over the codes: one-hot in play, and in training the
            encoder's one-hot code with the gradient passed straight through.

        Returns
        -------
        Tensor of float32, shape (batch, channels, n_cells)
            The next latent states, each scaled into [0, 1].

        Raises
        ------
        ValueError
            If the network has no codes.
        """
        self._needs_codes()
        planes = code.to(afterstate.dtype)[:, :, None].expand(-1, -1, self.n_cells)
        x = F.relu(self.chance_stem(torch.cat([afterstate, planes], dim=1)))
        return _scaled(self.chance_body(x))

    def encode(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        """Return the scores of the code of what happened between two decisions.

        Parameters
        ----------
        obs, next_obs : Tensor of float32, shape (batch, n_planes, n_cells)
            The player's observations at a decision and at its next one.

        Returns
        -------
        Tensor of float32, shape (batch, codes)
            A score per code.

        Raises
        ------
        ValueError
            If the network has no codes.
        """
        self._needs_codes()
        x = self.encoder_body(F.relu(self.encoder_stem(torch.cat([obs, next_obs], dim=1))))
        e = F.relu(self.encoder_conv(x))
        return self.encoder_out(torch.cat([e.mean(dim=2), e.amax(dim=2)], dim=1))

    def _needs_codes(self) -> None:
        """Raise ValueError unless the network has codes."""
        if self.codes == 0:
            raise ValueError("a network without codes has no afterstates: use dynamics")

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

    def project(self, latent: Tensor, head: bool = True) -> Tensor:
        """Return EfficientZero's projection of latent states, for the consistency loss.

        Each cell's vector goes through the projection, then, with ``head``, the prediction
        head: SimSiam's two branches, the one with the head learning to match the one without,
        whose gradient is stopped.

        Parameters
        ----------
        latent : Tensor of float32, shape (batch, channels, n_cells)
            Latent states.
        head : bool, default True
            Whether to apply the prediction head after the projection.

        Returns
        -------
        Tensor of float32, shape (batch, channels, n_cells)
            The projections.
        """
        projected = self.projection(latent)
        return self.projection_head(projected) if head else projected

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
    """A `WorldNet` without codes as an `agents.muzero.WorldModel`: states are its tensors.

    It runs where the network's parameters are, without gradients. A player in a series sent to
    other processes carries it there, so keep the network on the CPU for that.

    Parameters
    ----------
    net : WorldNet
        The network, without codes, in evaluation mode.

    Raises
    ------
    ValueError
        If the network has codes.
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

    def __post_init__(self) -> None:
        if self.net.codes > 0:
            raise ValueError("a network with codes is a StochasticNetModel")


@dataclass(frozen=True, eq=False)
class StochasticNetModel:
    """A `WorldNet` with codes as an `agents.muzero.StochasticModel`: states are its tensors.

    It runs where the network's parameters are, without gradients, as `NetModel` does.

    Parameters
    ----------
    net : WorldNet
        The network, with codes, in evaluation mode.

    Raises
    ------
    ValueError
        If the network has no codes.
    """

    net: WorldNet

    def __post_init__(self) -> None:
        if self.net.codes == 0:
            raise ValueError("a network without codes is a NetModel")

    def initial(
        self, obs: Sequence[NDArray[np.float32]]
    ) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states of observations, their scores and values (see the protocol)."""
        device = next(self.net.parameters()).device
        with torch.no_grad():
            latent = self.net.represent(torch.from_numpy(np.stack(obs)).to(device))
            return _judged(self.net, latent)

    def afterstates(
        self, latents: Sequence[Tensor], actions: NDArray[np.int64]
    ) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
        """Return the afterstates of actions, the codes' probabilities and ``Q`` (the protocol)."""
        with torch.no_grad():
            latent = torch.stack(list(latents))
            action = torch.from_numpy(np.asarray(actions, dtype=np.int64)).to(latent.device)
            after = self.net.afterstate(latent, action)
            pred = self.net.predict_afterstate(after)
            probs = torch.softmax(pred.sigma.double(), dim=1).cpu().numpy()
            return list(after), probs, pred.value.double().cpu().numpy()

    def chance(
        self, afterstates: Sequence[Tensor], codes: NDArray[np.int64]
    ) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
        """Return the latent states after codes, their scores and values (see the protocol)."""
        with torch.no_grad():
            after = torch.stack(list(afterstates))
            code = torch.from_numpy(np.asarray(codes, dtype=np.int64)).to(after.device)
            one_hot = F.one_hot(code, self.net.codes).to(after.dtype)
            return _judged(self.net, self.net.chance(after, one_hot))


def world_model(net: WorldNet) -> NetModel | StochasticNetModel:
    """Return a network as the model it is: a `StochasticNetModel` with codes, else a `NetModel`.

    Parameters
    ----------
    net : WorldNet
        The network, in evaluation mode.

    Returns
    -------
    NetModel or StochasticNetModel
        The network as a model for `agents.muzero.MuZeroPlayer`.
    """
    return StochasticNetModel(net) if net.codes > 0 else NetModel(net)


def _judged(
    net: WorldNet, latent: Tensor
) -> tuple[list[Tensor], NDArray[np.float64], NDArray[np.float64]]:
    """Return the states one by one, with their scores and values as float64 arrays."""
    pred = net.predict(latent)
    logits = pred.logits.flatten(1).double().cpu().numpy()
    return list(latent), logits, pred.value.double().cpu().numpy()
