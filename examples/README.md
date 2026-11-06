# Examples

Whole models built on `gridzero`, outside the package, which never imports them. Each one
tests the package somewhere the others don't, and each has a `README.md`: what it is, its model
or rules, its players, how to run its experiments, and what they found.

## The examples

| Example | What it is | What it tests |
|---|---|---|
| [`simulations/competition`](simulations/competition) | Two groups competing for one food supply | Conserved flows, checked against the R* rule of ecology |
| [`simulations/migration`](simulations/migration) | Herds following the rains around a planet | Seven ways to move, and the ideal free distribution |
| [`simulations/predators`](simulations/predators) | Grass, rabbits, foxes and wildfire on a board of triangles | A trophic cascade, and an arrival during a run |
| [`simulations/biomes`](simulations/biomes) | Grassland and forest on a planet, each with its grazers, and wolves that hunt in both | Nested states, two levels deep, and apparent competition |
| [`simulations/settlers`](simulations/settlers) | Farmers and herders settling a world of seasons, droughts, raids and worn-out soil | Eight phases over thousands of steps |
| [`simulations/population_evolution`](simulations/population_evolution) | One founder species on an empty planet, adapting to its climates and splitting into species | A heritable trait as a component axis, against exact theory |
| [`games/tactics`](games/tactics) | A two-player skirmish on a 25x25 hex board: pick a squad, fight over one point, and change the terrain with an engineer | Dice, a choice of pieces, an objective built to be reworked, search and AlphaZero on a large board |
| [`games/in_a_row`](games/in_a_row) | k in a row, from tic-tac-toe to Connect Four | AlphaZero from scratch, against exact values; code players fit to an agent; a page to play them on |
| [`games/voronoi`](games/voronoi) | The Voronoi game on a planet: stones on land, the most land nearest one's stones wins | Any number of players, off a lattice |
| [`games/colonies`](games/colonies) | Orders on a planet under fog of war, each round resolved as a simulation | Hidden information, moves made at once, players knocked out, and MuZero |

## What they have shown

Each README dates its findings; these are the main ones.

- **MuZero learns a game it can't see** (Colonies, 2026-10-30): trained from scratch for 25
  minutes on one GPU, from its own views under fog, it beat uniform, the settler and the raider
  100-0, 100-0 and 99-1. EfficientZero's consistency loss, Reanalyze and Stochastic MuZero all
  stayed at the same ceiling, about 650-700 Elo above uniform.
- **AlphaZero learns Connect Four from scratch** (in_a_row, 2026-10-28/29): in 25 minutes, with
  its search, it beat a scripted player that wins when it can and blocks the other's wins
  96-2-2, and it opens in the center column, the only winning first move. Exact values of small
  boards found a flaw in the search, since fixed.
- **A search can stop looking** (in_a_row, 2026-11-05): rescaling values at every node, as `mctx`
  does, locked nodes below the root onto the first action that looked better than the node's own
  value. Rescaling at the root only beat the old search 77-22-1 with fitted constants at 1024
  simulations, and made budgets count: a flat top of the ladder spread out.
- **Search beats heuristics on Tactics** (2026-10-25): open-loop tree search beat the greedy
  player 89% of the time. AlphaZero learned on a 9x9 board, but its self-play drifted into
  draws and never beat greedy: Tactics' incentives stall it. So Tactics now has an objective and
  no draws (2026-11-05): a greedy that seeks the point beats a sitter 240-0 and a pure fighter
  196-44, and search beats greedy 73%. AlphaZero doesn't beat greedy on these rules yet, even
  after imitating it. A network taught by the search player holds level with greedy once it
  searches. An engineer played by search works about 3.7 tiles a game, and is worth about an
  archer so far.
  A code player with constants picked by hand beats random 96-0 and loses to greedy 31-65.
- **Self-play drifts into draws** in both two-player board games. A study on a small board
  against the exact solver is open.
- **The simulations match their theory:** R*, the ideal free distribution, a trophic cascade,
  apparent competition, and Kimura's balance of mutation against selection.

## Running an experiment

Every experiment runs one way, through `run.py` at the repository's root, whose `EXPERIMENTS`
table names them all. A simulation has one experiment, named after it, that runs it, prints
its checks against theory and saves its pictures; a game's are named `<game>.<experiment>`, and
those over any game's output `games.<experiment>` (`games.curves` draws training runs, and
refreshes while they train).

```bash
uv run python run.py                                   # list every experiment
uv run python run.py predators 100                     # 100 years, two pictures
uv run python run.py tactics.tournament 10             # greedy and random, 10 starts
uv run python run.py tactics.zero_check --help         # one experiment's arguments
```

Experiments write their files in the working directory by default, named after the experiment
(`predators_map.png`); a prefix argument puts them elsewhere. The training experiments need an
NVIDIA GPU, and each runs in a time box given in minutes.

## Writing one

A simulation is a model written as in the top README's "A first simulation", in its own package
with an `experiment.py` beside it. Its experiment has one shape:

```text
SEED = 1                                  # module constant; a comment says what it sets
main(length, prefix, *varied)             # length in the model's unit, then the prefix,
                                          # then any parameters the run varies
model = Model(...)                        # held under its own name: settlement, evolution
Simulation(model.start(space, SEED), model.phases(), seed=SEED)   # one per run
{prefix}_<what>.png                       # every picture, drawn from the states
```

A game is written as in "A first game", in its own package with an `experiments/` folder beside
it. Players are scored on the game's own `EVAL_SEED` starts in every experiment, so their scores
compare, and play is counted in starts, each played once per rotation of the seats (twice, the
sides swapped, with two players):

```text
SEED = 10_000_000                         # module constant, for what else it draws (self-play
                                          # starts, first weights); a comment says what
main(*inputs, amount, prefix, *varied)    # inputs such as weights; the amount in its own unit
                                          # (starts, minutes of a time box, games); the prefix
                                          # if it writes files; then what the run varies
game = Tactics(...)                       # the rules, built in main and passed to helpers
series(game, [a, b], starts, EVAL_SEED, workers=CORES)   # or ladder for one against rivals
{prefix}_<what>.<ext>                     # every file: pictures, weights, figures
```

Watching a game is one experiment per game, `<game>.watch [start] [prefix] [players...]`,
which steps through a game in a window or saves a GIF. The full conventions are in
[`AGENTS.md`](../AGENTS.md), under "Design decisions".

## Language models

An LLM player (`gridzero.agents.llm.LLMPlayer`) asks a model served on this machine by
llama.cpp, which `compose.yaml` at the root starts (it needs a GPU with the NVIDIA container
toolkit, and downloads a 4.6 GB model the first time): `docker compose up -d`, then `down`. No
example uses one since Tactics' was retired on 2026-10-31.
