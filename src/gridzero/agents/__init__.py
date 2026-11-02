"""Agents: what decides, for any game of `gridzero.game`.

Players that read a game only through its protocol, so that they play games they were not
written for. This package needs only NumPy and the standard library; the modules that need
Torch are not imported here.

Players
-------
UniformPlayer
    A legal action, uniformly at random: the baseline for any game.

Exact play
----------
exact.solve, exact.optimal
    Player 0's value of every position of a small deterministic game of two players, and the
    actions that keep the best value: the exact yardstick, as on tic-tac-toe.

Learning
--------
AlphaZeroPlayer, Evaluator, SearchResult, self_play, Experience, Played
    Gumbel AlphaZero tree search over single decisions, with chance edges, for any game of two
    players whose view is its position, judged by an evaluator such as a network
    (`agents.net`, which imports Torch); many games of it at once, with the decisions to
    learn from.
Observable, Revealable, Learnable, Frame
    What a learner needs from a game beyond `game.Game`: observations of views as planes per
    cell in each player's frame; the whole position as a player would see it with nothing
    hidden; a game whose view is its position, on a 2-D lattice; and the frame, which maps
    actions between the board and it.
net, zero_train, imitate, ppo
    The network and its training (Torch, not imported here): AlphaZero's loop, imitating a
    teacher's decisions, and PPO self-play, all on `Experience`.

World models
------------
muzero.MuZeroPlayer, muzero.WorldModel, muzero.StochasticModel, muzero.self_play
    MuZero: the same Gumbel search over the latent states of a learned model of the game,
    which plays from what a player is shown, so games with hidden information too (NumPy
    only, not imported here).
muzero_net, muzero_train
    The world model's network and its training (Torch, not imported here).

Language models
---------------
llm.LLMPlayer, llm.Menu, llm.Option
    A player that asks a language model to pick from a numbered menu of the legal actions, which
    each game writes (`Menu`).
llm.Chat, llm.LlamaServer, llm.Message, llm.parse_choice
    The model behind it: any chat that answers to a JSON schema, and llama.cpp's server.
"""

from gridzero.agents.frame import Frame, Learnable, Observable, Revealable
from gridzero.agents.random import UniformPlayer
from gridzero.agents.zero import (
    AlphaZeroPlayer,
    Evaluator,
    Experience,
    Played,
    SearchResult,
    self_play,
)

__all__ = [
    "AlphaZeroPlayer",
    "Evaluator",
    "Experience",
    "Frame",
    "Learnable",
    "Observable",
    "Played",
    "Revealable",
    "SearchResult",
    "UniformPlayer",
    "self_play",
]
