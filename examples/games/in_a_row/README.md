# In a row

k in a row: a family of two-player games of lines, from tic-tac-toe to Connect Four. It is the
second game, built on `game` and `agents` unchanged to find what they lacked.

## The rules

Players take turns placing stones on a square board, and the first with `length` stones in a
straight line wins. `InARow()` is Connect Four: 7 columns of 6, four in a row, each stone falling
to the lowest free cell. `InARow((3, 3), 3, gravity=False)` is tic-tac-toe. A full board without
a line is a draw.

`InARow` is a `game.Game` and an `agents.Learnable`: its planes are the player's stones, the
other's, and who moved first. So the generic AlphaZero player, network and training play it
as they are.

## Players

- `ThreatPlayer`: wins when it can, blocks the other's win, and never plays a move that hands
  over a win. It is the scripted yardstick.
- Through `agents`: `UniformPlayer`; `NetPlayer` and `AlphaZeroPlayer` on a `PolicyNet`; and
  `agents.exact.solve`, the value under perfect play of every position of a small board.

## Experiments

| Command | What it does |
|---|---|
| `in_a_row.zero_scratch <minutes> [prefix] [starts] [seed]` | AlphaZero on Connect Four from scratch, in a time box; after each generation, the network alone against uniform and `ThreatPlayer`, the agent with its search against `ThreatPlayer`, and its opening |
| `in_a_row.zero_check <weights> [starts] [simulations]` | The same measures, for saved weights |

Run each as `uv run python run.py <command>`. There is no `watch`, since the boards are drawn as
text (`board_text`).

## What we found

- **Exact values** (2026-10-28): tic-tac-toe is a draw (solved in 0.16 s), and three in a row
  on 4x3 and 4x4 boards is a first player's win. The number of positions by ply matches OEIS
  A212693.
- **Agent A** (2026-10-28, seed 0, 25 minutes, 65 generations): the network alone beat uniform
  100-0 and `ThreatPlayer` 74-22-4. With its search (96 simulations) it beat `ThreatPlayer`
  96-2-2. It opens in the center column, the only winning first move.
- **Agent B** (2026-10-29, seed 1, 25 minutes, 47 generations): the network alone beat
  `ThreatPlayer` 82-17-1, and with its search 96-4-0. It opens in the center column too.
- **Self-play drifts into draws** (finding G11): by the end of agent A's run, 190 of 256
  self-play games were drawn. A study on a small board against the exact solver is still open.
- **A weakness in `agents.zero`** (finding G8): with an uneven last round of sequential halving,
  the search could play a lost move over a drawn one. Exact values found it, and it was fixed on
  2026-10-29.

The weights and logs are in `runs/in_a_row/` (not in git).

## Files

| File | What it holds |
|---|---|
| `game.py` | `InARow`: the rules, the lines, and the observation planes |
| `state.py` | `InARowState`, `InARowPosition`, `Placed` |
| `players.py` | `ThreatPlayer` |
| `match.py` | `EVAL_SEED`, and `opening`: what an evaluator makes of the empty board |
| `text.py` | `board_text` |
