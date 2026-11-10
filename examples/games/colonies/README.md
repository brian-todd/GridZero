# Colonies

Orders on a living planet under fog of war: the fourth game, and the first to run `core`'s loop
inside a step. It tests `game` where the other games don't: hidden information, moves made at
once, players knocked out, and a simulation in a step.

## The rules

Each round, every player still in gives one order in turn: `GATHER` its people toward a land
cell it can see, or `HOLD`. The orders are hidden from the others. The last order of a round
resolves them all together, as a few steps of a `core.Simulation`: people march toward their
targets, fight where players meet, and grow toward what the land supports. A player sees only
the land within `vision` moves of its own people (fog of war; `fog=False` shows everything). A
player left with too few people is out. The last one in wins; otherwise the most people after
the last round wins.

People are one field with a component per player, up to `MAX_PLAYERS` = 4.

## Players

- `Settler`: grows and spreads, never attacking.
- `Raider`: attacks enemies it outnumbers, and otherwise settles.
- Both read only their view. Through `agents`: `UniformPlayer`, and `MuZeroPlayer` on a
  `WorldNet`, which plans on a latent model learned from views alone (`learned.player` loads
  one from a weights spec).

## Experiments

| Command | What it does |
|---|---|
| `colonies.tournament [starts] [prefix] [players]` | The scripted players against uniform and each other, under fog and without, and the raider in every seat; a globe of a game's end |
| `colonies.watch [start] [prefix] [players...]` | One game, round by round: the board or one player's view, each side's people below; a window or a GIF |
| `colonies.muzero_scratch <minutes> [prefix] [starts] [seed] [belief]` | A MuZero player trained from scratch, in a time box; the belief's weight is its one option |
| `colonies.muzero_check <weights> [starts] [simulations]` | A saved network alone and searched, against the scripted players and each other |
| `colonies.muzero_ladder [starts] [simulations] [weights...]` | Elo ratings of saved MuZero players and the scripted players, from every pair's results |

Run each as `uv run python run.py <command>`. `games.curves` draws any run's
`_generations.jsonl`, and refreshes while the run trains.

## What we found

- **2026-10-31, `colonies.tournament` at its defaults** (10 starts, 2 players). Under fog, the
  raider beat uniform 16-4, the settler beat uniform 15-5, and the raider beat the settler 16-4.
  Without fog, the settler did no better than uniform (9-11). Neither seat has an edge.
- **The belief** (W1-W2, 2026-10-30) finds enemies under fog: 1.42 people a cell off, against
  3.71 for guessing "nothing there", ranking fogged cells at an AUC of 0.96. The dynamics beat
  the view carried ahead unchanged at 1 to 3 rounds.
- **MuZero from scratch** (W4, 2026-10-30) in 25 minutes beat uniform, settler and raider 100-0,
  100-0 and 99-1 over 50 starts.
- **Every improvement since has hit the same ceiling**, about 650-700 on `muzero_ladder`'s Elo
  scale, where uniform is 0:
  planning over every action below the root, more simulations (flat past 16), EfficientZero's
  consistency loss, Reanalyze, and Stochastic MuZero (worse, 546). The privileged belief does
  help as an auxiliary loss. An exploiter trained against the best player found no weakness.
  The consistency loss, Reanalyze, Stochastic MuZero and the exploiter's games were removed from
  the code on 2026-11-09, and with them the search of networks trained before 2026-10-30, whose
  policy only knew legal actions.

The belief experiments (`colonies.belief`, `belief_check`) and the exploiter
(`colonies.muzero_exploit`) were retired on 2026-11-01, their findings above; the belief lives
on as `muzero_scratch`'s auxiliary loss.

The weights, logs and figures are in `runs/muzero/` (not in git).

## Files

| File | What it holds |
|---|---|
| `game.py` | `Colonies`: the rules, the round's simulation, views, and the planes for world models |
| `players.py` | `Settler`, `Raider` |
| `match.py` | `EVAL_SEED` |
| `learned.py` | `player(spec)`: a MuZero player from saved weights, `path` or `path@64` |
| `picture.py` | `colors`, `view_colors`, `PLAYERS`, `FOG`, for globes and maps |
