# Tactics

A two-player skirmish on a 24x24 hex board with walls and rubble: the first game, and the one
the `game` and `agents` layers were first written for.

## The rules

Each side deploys 6 footsoldiers, 3 archers, 2 knights and a captain in its home corner, one
unit at a time, alternately. Then every round each living unit gets one turn, in a shuffled
order: it may move up to its allowance and attack once, in any order (move, attack, move), turn
to face any direction, and end its turn. A side wins by killing every enemy unit. After the
round cap, the side with more hit points left wins.

- **Units:** footsoldiers gain Def from adjacent footsoldiers (a formation), archers attack at
  range where they can see and may defer an attack onto a tile, knights move twice as far when
  they don't attack, and captains lift the Atk and Def of friends near them.
- **Attacks** hit on a d20 roll that gets easier as the attacker's Atk exceeds the target's Def,
  and a hit takes Atk from the target's hit points.
- **Flanking** triples Atk: attacking from outside the three sides a target faces, or with two
  enemies adjacent to it. A unit attacked this round is engaged, and later attackers get +1 Atk.

The rules are data: `Tactics` and `UnitStats` (`rules.py`) hold every number. The boards are
symmetric under the half turn (`make_board`), and team 1 sees the board turned half a turn, so a
network plays both sides the same way.

## Players

- `RandomPlayer`: a legal action at random, ending a unit's turn now and then.
- `GreedyPlayer`: one ply of heuristics (attack value, kills, risk, advance, formation, aura).
- `SearchPlayer`: open-loop Monte Carlo tree search over plans for whole unit turns (a tile, and
  whether to attack before moving), with greedy's scores as priors and 16 greedy unit turns of
  rollout.
- Through `agents`: `UniformPlayer`, `NetPlayer` and `AlphaZeroPlayer` on a `PolicyNet`, since
  `Tactics` is an `agents.Learnable`.

## Experiments

| Command | What it does |
|---|---|
| `tactics.tournament [starts] [prefix]` | Greedy against random, and greedy against itself (is the game even by side?); a picture of a game's end |
| `tactics.search [starts] [budgets...]` | The search player against greedy at a few budgets of simulations |
| `tactics.watch [start] [prefix] [players...]` | One game, decision by decision, in a window or a GIF (random, greedy, search) |
| `tactics.zero_scratch <minutes> [prefix] ...` | AlphaZero from scratch on a 9x9 board, in a time box |
| `tactics.zero_check <weights> [starts] [simulations] [rivals...]` | The AlphaZero player on saved weights against the network alone, greedy or search |

Run each as `uv run python run.py <command>`; `--help` prints an experiment's docstring.
`benchmarks/tactics.py` times the game itself, and `benchmarks/zero_selfplay.py` times
self-play on the GPU.

## What we found

- **2026-11-01, `tactics.tournament` at its defaults:** greedy beat random 20-0. Greedy against
  greedy, each game with dice of its own, went 8-12, with team 0 winning 10 of 20 decided
  games: no edge beyond a fair game's noise. Games run about 12 rounds. (The 2026-10-31 run
  rolled the same dice in both games of a board, which counted each game twice.)
- **The search player** (T5, 2026-10-25) beat greedy 41-5 (89%), and the search without
  rollouts 67-18.
- **Rule tuning** (T6, 2026-10-25) settled today's numbers: footsoldier Atk 4, knight move 3,
  flanking x3, the 24x24 board, and the round cap decided by hit points.
- **Networks** (T7, 2026-10-25/26): imitating greedy matched 79% of its decisions; PPO
  self-play then beat greedy 59-37, but lost to the search player 19-29.
- **AlphaZero** (T8, 2026-10-26/27): the search beat its own network 32-15. Training from a
  PPO network didn't improve it. From scratch on 9x9 it beat random 41-4 after one generation,
  then self-play drifted into draws and it never beat greedy. The loop learns, but Tactics'
  incentives stall it. Closed as a proof of concept.

The T6 tuning experiments, the T7 network stages and the LLM player (T4) were retired on
2026-10-31, their work done. Their library pieces stay: `agents.imitate`, `agents.ppo` and
`agents.llm`. The hill climb T6 ran on (`gridzero.tuning`) and the board and unit tables as
text (`board_text`, `units_text`) were removed on 2026-11-01, unused, and so was
`tactics.zero_train`, which trained from the PPO stage's weights (`runs/ppo100.pt`), which no
experiment can make any more. `zero_check` still plays saved full-size weights.

## Files

| File | What it holds |
|---|---|
| `game.py` | `Tactics`: the rules, `start`, `legal`, `step`, `view`, `odds`, and the observation planes for networks |
| `rules.py` | `UnitStats`, the default kinds and counts, and the d20 rule |
| `board.py` | Symmetric boards with walls, rubble and homes |
| `state.py` | `TacticsState`, `Unit`, `TacticsPosition`, `Verb` |
| `events.py` | What a step did: `Placed`, `Moved`, `Attacked`, `Died`, ... |
| `players.py` | `RandomPlayer`, `GreedyPlayer` |
| `search.py` | `SearchPlayer`, `Plan`, `material` |
| `match.py` | `EVAL_SEED`, `summarize`, `balance` |
| `text.py`, `pictures.py` | Events and summaries as text; boards drawn as hexes |
