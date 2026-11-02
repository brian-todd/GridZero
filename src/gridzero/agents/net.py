"""A network that plays any learnable game: a policy and a value over its observations.

`PolicyNet` reads a player's observation (`agents.frame.Learnable.observe`) and returns a score
per action, in the player's frame, and the value of the position for the player, from -1
(lost) to 1 (won). Its body is a stack of residual blocks of lattice convolutions
(`LatticeConv`), so it sees what is near each cell the way the board connects them, wherever
the cell is: squares and hexes. `NetPlayer` plays with it, and `NetEvaluator` judges positions
for the AlphaZero player (`agents.zero.AlphaZeroPlayer`), many at a time. It was first written
for Tactics (T7), whose saved weights it still loads. `CellConv` convolves over the neighbors
of any space, such as a planet, for networks of games off the lattice (`agents.muzero_net`).

This module imports Torch, so the package does not import it: import it as
``gridzero.agents.net``.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from gridzero.agents.frame import Learnable
from gridzero.core import Adjacency, LatticeSpace, Space, State
from gridzero.core._arrays import check_int, check_real
from gridzero.game import Action, Position


class LatticeConv(nn.Conv2d):
    """A convolution over each cell and the cells around it, on a 2-D lattice laid out as an image.

    A lattice of shape ``(Q, R)`` is an image of that shape, ``x[..., q, r]``, and the kernel is
    the 3x3 window of offsets around a cell, masked to the tiling's neighbors:

    - squares: the whole window, the 8 cells that share a side or a corner,
    - hexes (axial coordinates): neighbors at ``(+-1, 0)``, ``(0, +-1)``, ``(1, -1)`` and ``(-1,
      1)``, so the corners at ``(-1, -1)`` and ``(1, 1)`` are held at zero.

    Cells beyond the lattice read as zero. Triangles, whose two orientations have different
    neighbors, lattices that wrap, and lattices that aren't 2-D aren't supported yet.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of squares or hexes that doesn't wrap.
    in_channels, out_channels : int
        Channels in and out.
    bias : bool, default True
        Whether to add a learned bias per output channel.

    Raises
    ------
    ValueError
        If the lattice is not one of those.
    """

    mask: Tensor

    def __init__(
        self, space: LatticeSpace, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        if space.dim != 2 or space.tiling not in ("hex", "hypercubic") or any(space.periodic):
            raise ValueError(
                "lattice convolutions need a 2-D lattice of squares or hexes that doesn't wrap, "
                f"got a {space.dim}-D {space.tiling} lattice, periodic {space.periodic}"
            )
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        mask = torch.ones(3, 3)
        if space.tiling == "hex":
            mask[0, 0] = mask[2, 2] = 0.0  # kernel slot (i, j) reads the offset (i - 1, j - 1)
        self.register_buffer("mask", mask)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, input: Tensor) -> Tensor:
        """Return the convolution of ``input``, shape (batch, in_channels, Q, R)."""
        return F.conv2d(input, self.weight * self.mask, self.bias, padding=1)


class CellConv(nn.Module):
    """A convolution over each cell and the mean of its neighbors, on any space.

    ``out[:, :, i] = W_own @ x[:, :, i] + W_near @ mean(x[:, :, j] for j adjacent to i) + b``,
    with the neighbors of `core.Adjacency` (cells that share a facet). It treats every neighbor
    alike, so it needs no directions: on a sphere, which has none, it is the convolution to use.
    A cell with no neighbors reads zero for their mean. The means are a product with a dense
    matrix of ``n_cells**2`` float32s, fast on a GPU for up to a few thousand cells (2,562 at
    frequency 16 take 26 MB).

    Parameters
    ----------
    space : Space
        Any space: a mesh or a lattice.
    in_channels, out_channels : int
        Channels in and out.
    bias : bool, default True
        Whether to add a learned bias per output channel.
    """

    means: Tensor

    def __init__(
        self, space: Space, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super().__init__()
        graph = Adjacency().compile(space).graph
        degree = np.diff(graph.indptr)
        means = np.zeros((space.n_cells, space.n_cells), dtype=np.float32)
        target = np.repeat(np.arange(space.n_cells), degree)
        means[graph.indices, target] = (1.0 / degree)[target]  # column i averages i's neighbors
        self.register_buffer("means", torch.from_numpy(means), persistent=False)
        self.own = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.near = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, input: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]  # see _Block
        """Return the convolution of ``input``, shape (batch, in_channels, n_cells)."""
        return self.own(input) + self.near(input) @ self.means.to(input.dtype)


class _Block(nn.Module):
    """Two convolutions and a skip: ``relu(x + second(relu(first(x))))``.

    The second convolution starts at zero, with no bias, so a new block passes its input
    through unchanged, and a deep stack trains without normalization layers, which would behave
    differently in training and in play. ``conv(in_channels, out_channels, bias)`` makes either
    convolution: `LatticeConv` or `CellConv` on a space.
    """

    def __init__(self, conv: Callable[[int, int, bool], nn.Module], channels: int) -> None:
        super().__init__()
        self.first = conv(channels, channels, True)
        self.second = conv(channels, channels, False)
        for weight in self.second.parameters():
            nn.init.zeros_(weight)

    # Torch declares `Module.forward` as a writable attribute of type `Callable[..., Any]`, so
    # ty reads every typed override as narrowing what may be written to it.
    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return the block's output, the shape of ``x``."""
        return F.relu(x + self.second(F.relu(self.first(x))))


class PolicyNet(nn.Module):
    """A residual network of lattice convolutions with a policy head and a value head.

    The input is a batch of observations (`Learnable.observe`) laid out on the board, with a
    plane of ones added so that the network can tell the board's edge from empty cells. The
    policy head scores each verb on each cell with a convolution, plus a bias per verb from the
    whole board's mean, since some verbs (such as ending a turn in Tactics) depend on more than
    one cell. The value head pools the board's mean and maximum.

    Parameters
    ----------
    game : Learnable
        The rules: they fix the planes, the verbs and the board (`Learnable.space`), squares or
        hexes.
    channels : int, default 64
        Channels of the body, at least 1.
    blocks : int, default 6
        Residual blocks, at least 0. Each sees one cell further, so the body sees ``1 + 2 *
        blocks`` cells from each cell.

    Raises
    ------
    TypeError, ValueError
        If ``channels`` or ``blocks`` is not an integer in range, or the board isn't one
        `LatticeConv` supports.
    """

    def __init__[P: Position[State, object]](
        self, game: Learnable[P], channels: int = 64, blocks: int = 6
    ) -> None:
        super().__init__()
        channels = check_int(channels, "channels", minimum=1)
        blocks = check_int(blocks, "blocks", minimum=0)
        space = game.space
        self.shape = (space.shape[0], space.shape[1])
        self.n_planes = len(game.planes)
        self.n_verbs = game.n_verbs
        self.channels = channels
        self.blocks = blocks
        self.stem = LatticeConv(space, self.n_planes + 1, channels)
        conv = functools.partial(LatticeConv, space)
        self.body = nn.Sequential(*(_Block(conv, channels) for _ in range(blocks)))
        self.policy_conv = LatticeConv(space, channels, channels)
        self.policy_out = nn.Conv2d(channels, self.n_verbs, kernel_size=1)
        self.policy_bias = nn.Linear(channels, self.n_verbs)
        self.value_conv = nn.Conv2d(channels, channels // 2 or 1, kernel_size=1)
        self.value_hidden = nn.Linear(2 * (channels // 2 or 1), channels)
        self.value_out = nn.Linear(channels, 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:  # ty: ignore[invalid-mutable-override]  # see _Block
        """Return the scores of actions and the values of a batch of observations.

        Parameters
        ----------
        obs : Tensor of float32, shape (batch, n_planes, n_cells)
            Observations (`Learnable.observe`), each in its player's frame.

        Returns
        -------
        logits : Tensor of float32, shape (batch, n_verbs, n_cells)
            A score per action, in the player's frame, before masking illegal ones.
        value : Tensor of float32, shape (batch,)
            The position's value for the player, in [-1, 1].
        """
        batch = obs.shape[0]
        x = obs.reshape(batch, self.n_planes, *self.shape)
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        x = self.body(F.relu(self.stem(x)))
        mean = x.mean(dim=(2, 3))
        p = self.policy_out(F.relu(self.policy_conv(x)))
        logits = p.reshape(batch, self.n_verbs, -1) + self.policy_bias(mean)[:, :, None]
        v = F.relu(self.value_conv(x))
        pooled = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)
        value = torch.tanh(self.value_out(F.relu(self.value_hidden(pooled))))[:, 0]
        return logits, value


def masked_logits(logits: Tensor, legal: Tensor) -> Tensor:
    """Return scores with illegal actions at -inf, flattened over verbs and cells.

    Parameters
    ----------
    logits : Tensor of float, shape (batch, n_verbs, n_cells)
        Scores from `PolicyNet`.
    legal : Tensor of bool, shape (batch, n_verbs, n_cells)
        Legal actions in the same frame, at least one per row.

    Returns
    -------
    Tensor of float, shape (batch, n_verbs * n_cells)
        The scores of legal actions, and -inf elsewhere. Action ``verb * n_cells + cell``.
    """
    return logits.flatten(1).masked_fill(~legal.flatten(1), -torch.inf)


@dataclass(frozen=True, eq=False)
class NetPlayer:
    """Plays with a `PolicyNet`: the best legal action, or one drawn from its policy.

    The network runs on the device its parameters are on, one decision at a time, without
    gradients. A player in other processes of a `series` travels by pickling, so give it a
    network on the CPU there.

    Parameters
    ----------
    net : PolicyNet
        The network. It is put in evaluation mode on each call.
    temperature : float, default 0.0
        0 plays the highest score, ties broken at random. Above 0, actions are drawn with
        probability proportional to ``exp(score / temperature)``.

    Raises
    ------
    TypeError, ValueError
        If ``temperature`` is not a finite real number of at least 0.
    """

    net: PolicyNet
    temperature: float = 0.0

    def __post_init__(self) -> None:
        check_real(self.temperature, "temperature", 0.0)

    def __call__[P: Position[State, object]](
        self,
        game: Learnable[P],
        player: int,
        position: P,
        legal: NDArray[np.bool_],
        rng: np.random.Generator,
        /,
    ) -> Action:
        """Return the network's action (see `game.Player`)."""
        return self.decide(game, position, legal, rng).action

    def decide[P: Position[State, object]](
        self, game: Learnable[P], position: P, legal: NDArray[np.bool_], rng: np.random.Generator
    ) -> Decision:
        """Return the network's action with what led to it, for learning from play.

        Parameters
        ----------
        game, position, legal, rng
            As for `game.Player`.

        Returns
        -------
        Decision
            The action, and the observation, legal actions, choice, its log-probability and
            the value, in the player's frame.
        """
        player, obs, mask, logits, value = read(self.net, game, position, legal)
        scores = logits[mask]
        choices = np.flatnonzero(mask)
        if self.temperature == 0.0:
            best = np.flatnonzero(scores == scores.max())
            pick = int(best[rng.integers(best.size)])
            logp = -float(np.log(best.size))
        else:
            z = (scores - scores.max()) / self.temperature
            log_total = float(np.log(np.exp(z).sum()))
            pick = int(rng.choice(choices.size, p=np.exp(z - log_total)))
            logp = float(z[pick]) - log_total
        index = int(choices[pick])
        n_cells = legal.shape[1]
        action = game.frame(player).action_back(Action(index // n_cells, index % n_cells))
        return Decision(action, player, obs, mask, index, logp, value)


@dataclass(frozen=True, eq=False)
class NetEvaluator:
    """Judges positions for the AlphaZero player with a `PolicyNet` (`agents.zero.Evaluator`).

    The positions go through the network as one batch, on the device its parameters are on,
    without gradients, as bytes (`observations`); on a GPU in bfloat16, as training runs,
    so results there depend a little on the batch (on the CPU they are exact, float32). As
    for `NetPlayer`, give it a network on the CPU to send it to other processes.

    Parameters
    ----------
    net : PolicyNet
        The network. It is put in evaluation mode on each call.
    """

    net: PolicyNet

    def __call__[P: Position[State, object]](
        self, game: Learnable[P], positions: Sequence[P], legal: Sequence[NDArray[np.bool_]], /
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the log-softmax of the network's scores over legal actions, and the values.

        Parameters
        ----------
        game, positions, legal
            As for `agents.zero.Evaluator`.

        Returns
        -------
        logits : list of ndarray of float64, shape (n_legal,)
            The log-probabilities of each position's legal actions, mapped back to the board,
            in the order of ``np.flatnonzero(legal)``.
        values : ndarray of float64, shape (n_positions,)
            ``(1 + v) / 2`` for the network's value ``v`` of the acting player, as player 0's.
        """
        players, obs = observations(game, positions)
        device = next(self.net.parameters()).device
        self.net.eval()
        gpu = device.type == "cuda"
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=gpu):
            scores, value = self.net(as_input(torch.from_numpy(obs).to(device)))
        return judgements(
            game, scores.double().cpu().numpy(), value.double().cpu().numpy(), legal, players
        )


def observations[P: Position[State, object]](
    game: Learnable[P], positions: Sequence[P]
) -> tuple[list[int], NDArray[np.uint8]]:
    """Return the acting player of each position and its observation, as bytes (`as_input`).

    The AlphaZero player's network sees observations as `agents.zero.self_play` stores them for
    training: ``round(255 * game.observe(...))``.

    Parameters
    ----------
    game : Learnable
        The rules.
    positions : sequence of P
        Positions not over.

    Returns
    -------
    players : list of int
        The acting player of each position.
    obs : ndarray of uint8, shape (n_positions, n_planes, n_cells)
        Each position as its acting player sees it, ``round(255 * game.observe(...))``.
    """
    players = [game.player(p) for p in positions]
    n_planes, n_cells = len(game.planes), game.space.n_cells
    obs = np.empty((len(positions), n_planes, n_cells), dtype=np.uint8)
    for i, (p, t) in enumerate(zip(positions, players, strict=True)):
        obs[i] = np.rint(game.observe(p, t) * 255)
    return players, obs


def as_input(obs: Tensor) -> Tensor:
    """Return observations stored as bytes as a `PolicyNet`'s input.

    Parameters
    ----------
    obs : Tensor of uint8, shape (batch, n_planes, n_cells)
        Observations, ``round(255 * game.observe(...))``.

    Returns
    -------
    Tensor of float32, shape (batch, n_planes, n_cells)
        ``obs / 255``, on the device of ``obs``.
    """
    return obs.float().div_(255.0)


def judgements[P: Position[State, object]](
    game: Learnable[P],
    scores: NDArray[np.float64],
    value: NDArray[np.float64],
    legal: Sequence[NDArray[np.bool_]],
    players: Sequence[int],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Turn a `PolicyNet`'s outputs into what a `zero.Evaluator` returns.

    Parameters
    ----------
    game : Learnable
        The rules.
    scores : ndarray of float64, shape (n_positions, n_verbs, n_cells)
        The network's scores, each in its acting player's frame.
    value : ndarray of float64, shape (n_positions,)
        The network's values, each for its acting player, in [-1, 1].
    legal : sequence of ndarray of bool, shape (n_verbs, n_cells)
        The legal actions of each position, on the board.
    players : sequence of int
        The acting player of each position.

    Returns
    -------
    logits : list of ndarray of float64, shape (n_legal,)
        The log-probabilities of each position's legal actions, mapped back to the board, in
        the order of ``np.flatnonzero(legal)``.
    values : ndarray of float64, shape (n_positions,)
        ``(1 + v) / 2`` for each value ``v``, as player 0's.
    """
    logits = []
    for i, (mask, player) in enumerate(zip(legal, players, strict=True)):
        x = game.frame(player).back(scores[i].reshape(scores.shape[1:]))[mask]
        z = x - x.max()
        logits.append(z - np.log(np.exp(z).sum()))
    sign = np.where(np.array(players) == 0, 1.0, -1.0)
    return logits, (1.0 + sign * value) / 2.0


class Reading(NamedTuple):
    """What a `PolicyNet` makes of one position, as `read` returns it."""

    player: int
    """The acting player."""
    obs: NDArray[np.float32]
    """Its observation, shape (n_planes, n_cells)."""
    legal: NDArray[np.bool_]
    """The legal actions in its frame, flat over verbs and cells, shape (n_verbs * n_cells,)."""
    scores: NDArray[np.float64]
    """The network's scores in its frame, flat as ``legal``."""
    value: float
    """The network's value of the position for the player, in [-1, 1]."""


def read[P: Position[State, object]](
    net: PolicyNet, game: Learnable[P], position: P, legal: NDArray[np.bool_]
) -> Reading:
    """Run a network on the acting player's observation of one position, without gradients.

    The network runs on the device its parameters are on, in float32, and is put in evaluation
    mode.

    Parameters
    ----------
    net : PolicyNet
        The network.
    game : Learnable
        The rules.
    position : P
        A position not over.
    legal : ndarray of bool, shape (n_verbs, n_cells)
        ``game.legal(position)``.

    Returns
    -------
    Reading
        The player, its observation, and the legal actions, scores and value in its frame.
    """
    player = game.player(position)
    mask = game.frame(player).into(legal).ravel()
    device = next(net.parameters()).device
    obs = game.observe(position, player)
    net.eval()
    with torch.no_grad():
        logits, value = net(torch.from_numpy(obs).to(device)[None])
    return Reading(player, obs, mask, logits[0].flatten().double().cpu().numpy(), float(value[0]))


class Decision(NamedTuple):
    """One decision of a `NetPlayer`, with what led to it."""

    action: Action
    """The action, on the board."""
    player: int
    """The deciding player."""
    obs: NDArray[np.float32]
    """Its observation, shape (n_planes, n_cells)."""
    legal: NDArray[np.bool_]
    """The legal actions in its frame, flat over verbs and cells, shape (n_verbs * n_cells,)."""
    index: int
    """The action in its frame, ``verb * n_cells + cell``."""
    logp: float
    """The log-probability of drawing the action: at temperature 0, minus the log of the ties."""
    value: float
    """The network's value of the position for the player, in [-1, 1]."""


def channels_last(net: PolicyNet) -> torch.device:
    """Lay out a network on a GPU for its tensor cores (channels last); return its device.

    Training with this layout, in bfloat16, more than doubled the speed on an RTX 3080 (T7).

    Parameters
    ----------
    net : PolicyNet
        The network, laid out in place when it is on a GPU.

    Returns
    -------
    torch.device
        The device of its parameters.
    """
    device = next(net.parameters()).device
    if device.type == "cuda":
        net.to(memory_format=torch.channels_last)
    return device


def unpack(packed: Tensor, n_bits: int) -> Tensor:
    """Return bits packed by `numpy.packbits` (most significant first), such as legal actions.

    Parameters
    ----------
    packed : Tensor of uint8, shape (batch, n_bytes)
        Rows of bits, packed.
    n_bits : int
        Bits per row, at most ``8 * n_bytes``.

    Returns
    -------
    Tensor of bool, shape (batch, n_bits)
        The bits, on the device of ``packed``.
    """
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed[:, :, None] >> shifts) & 1
    return bits.flatten(1)[:, :n_bits].bool()


def outputs(
    net: PolicyNet, obs: NDArray[np.uint8], legal: NDArray[np.uint8], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a network on stored decisions: masked scores, values and legal actions.

    Parameters
    ----------
    net : PolicyNet
        The network, on ``device``. On a GPU it runs in bfloat16 (see `channels_last`), and
        its outputs are returned as float32.
    obs : ndarray of uint8, shape (n, n_planes, n_cells)
        Observations, ``round(255 * game.observe(...))``.
    legal : ndarray of uint8, shape (n, ceil(n_verbs * n_cells / 8))
        Legal actions, flattened over verbs and cells and packed by `numpy.packbits`.
    device : torch.device
        Where the network is.

    Returns
    -------
    scores : Tensor of float32, shape (n, n_verbs * n_cells)
        The network's scores, -inf where illegal (`masked_logits`).
    value : Tensor of float32, shape (n,)
        Its values, in [-1, 1].
    legal : Tensor of bool, shape (n, n_verbs * n_cells)
        The legal actions, unpacked.
    """
    n = obs.shape[0]
    x = as_input(torch.from_numpy(obs).to(device))
    mask = unpack(torch.from_numpy(legal).to(device), net.n_verbs * x.shape[-1])
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, value = net(x)
    return masked_logits(logits.float(), mask.reshape(n, net.n_verbs, -1)), value.float(), mask
