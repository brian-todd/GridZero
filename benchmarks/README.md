# Benchmarks

`bench.py` times the core operations, intents, action spaces, queries and whole models on the
reference path (NumPy/SciPy) and on the Numba fast paths, and prints Markdown tables headed by
the machine and library versions. `tactics.py` times the game of Tactics: new boards, masks of
legal actions, steps and random games. `zero_selfplay.py` times AlphaZero self-play on Tactics
with the GPU: games an hour, leaves judged a second, and the GPU's memory and use.

```bash
taskset -c 4 uv run python -m benchmarks.bench           # everything, about 2 minutes
taskset -c 4 uv run python -m benchmarks.bench models    # or: operations, intents, actions, queries
taskset -c 4 uv run python -m benchmarks.tactics         # the game of Tactics, about 10 s
uv run python -m benchmarks.zero_selfplay weights.pt     # self-play on the GPU, every core
```

- **Pin to one performance core.** On the i9-12900K, cores 0–15 are performance cores
  (5.1–5.2 GHz) and 16–23 efficiency cores (3.9 GHz). Unpinned runs vary with where the
  scheduler puts them.
- **Expect up to 30% between processes** for the same kernel and data, from where arrays land
  in memory. Compare runs of the same script, not single figures.
- Each figure is the best of 7 repeats after a warm-up that compiles the kernels.

## Target

A 3-phase population model (logistic growth, a spread toward free
room, random shocks every 10 steps) must run at least **1,000 steps/s at f=32** (10,242 cells)
on one core. The script ends by checking it.

## Latest results

2026-10-19, i9-12900K pinned to core 4, Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0.
Per step:

| Model | Reference (µs) | Numba (µs) | Steps/s |
|---|---|---|---|
| 3-phase population, f=16 | 162 | 146 | 6,853 |
| 3-phase population, f=32 | 411 | 340 | **2,940** |
| 3-phase population, f=64 | 1,459 | 1,113 | 898 |
| foragers (choice, claim, allocation), f=32 | 3,958 | 2,481 | 403 |
| Life, 256x256 lattice | 235 | 179 | 5,578 |
| Life, 1024x1024 lattice | 7,736 | 5,380 | 186 |

Queries, per call:

| Case | Reference (µs) | Numba (µs) |
|---|---|---|
| `steps`, unit costs, 200x200 hexes | 2,745 | 177 |
| `steps`, costs, 200x200 hexes | 2,961 | 2,571 |
| `reach`, budget 5, costs | 250 | 20 |
| `path` to a cell 7 steps away, costs | 2,966 | 18 |
| `sight` within 10, 64x64 squares / hexes / triangles, 15% walls | 33,046 / 42,599 / 24,485 | 207 / 237 / 205 |
| `visible` across a 64x64 board | 1,144-1,479 | 15 |

### Tactics

2026-10-24 (after milestone T3), same machine and versions, pinned to core 4. The default rules
on 32x32 boards, means over every decision of 5 games, in µs:

| Call | Random players | Greedy players |
|---|---|---|
| `start` (a new board) | 8,498 | 8,537 |
| `legal` | 175 | 188 |
| the player's choice | 36 | 206 |
| `step` | 144 | 171 |
| decisions per game | 2,510 | 658 |
| a game | 0.90 s | 0.38 s |

`sight`'s set-up in Python is the largest part of a decision.
