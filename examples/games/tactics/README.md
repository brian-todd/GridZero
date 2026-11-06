# Tactics

A two-player skirmish on a 25x25 hex board with walls and rubble, fought over one tile at the
center: the first game, and the one the `game` and `agents` layers were first written for.

## The rules

**Squad and deployment.** Each side fields 8 units, at most 6 footsoldiers, 3 archers, 2 knights,
a captain and an engineer. Sides alternate placing a unit in their home corner, team 0 first,
choosing its kind and tile together. Nothing is hidden.

**Turns.** Every round each living unit gets one turn, in a shuffled order: it may move up to its
allowance and attack once, in any order (move, attack, move), turn to face any direction, and
end its turn.

| Kind | HP | Atk | Def | Move | Reach | Passive |
|---|---|---|---|---|---|---|
| Footsoldier | 5 | 4 | 2 | 2 | 1 | Gains Def from adjacent footsoldiers (a formation) |
| Archer | 3 | 2 | 1 | 2 | 3 | May defer its attack onto a tile, and fires when an enemy enters it, with +1 Atk |
| Knight | 7 | 3 | 2 | 3 | 1 | Moves twice as far when it doesn't attack |
| Captain | 9 | 4 | 3 | 2 | 1 | Lifts the Atk and Def of friends near it |
| Engineer | 3 | 2 | 1 | 2 | 3 | Builds and breaks terrain |

- **Attacks** hit on a d20 roll that gets easier as the attacker's Atk exceeds the target's Def,
  and a hit takes Atk from the target's hit points.
- **Flanking** triples Atk: attacking from outside the three sides a target faces, or with two
  enemies adjacent to it. A unit attacked this round is engaged, and later attackers get +1 Atk.
  Overwatch's +1 comes after the multiplier, so a flanked arrow hits for 7.
- **Terrain:** a wall blocks movement and sight, and rubble blocks movement only. In place of its
  attack, an engineer may raise an adjacent tile a step (open, rubble, wall) or lower it a step.
  Not on a unit or the point, and a build may not cut any living unit or the point off from the
  rest, counting walls and rubble as blocked and ignoring units.

**The point.** At the end of each round, a unit standing on the center tile takes it for its side,
which keeps it until an enemy stands there at a round's end. So walking through captures nothing,
and killing the unit on it stops nothing: the enemy has to stand there. Each round's end adds 1 to
the owner's clock, the capture included, and a flip restarts it at 1.

**How a game ends.** There are no draws.

| When | Result |
|---|---|
| A side has no units left | The other side wins |
| A side has owned the point for 5 rounds in a row | It wins |
| Nobody has captured the point by the end of round 10 | Team 1 loses: it places last, so it must take the first step |
| The end of round 20 | The point's owner wins |

The rules are data: `Tactics` and `UnitStats` (`rules.py`) hold every number, and every rule about
the point is in `objective.py`, so that it can be reworked and measured with `tactics.rules`. The
boards are symmetric under the half turn (`make_board`), and team 1 sees the board turned half a
turn, so a network plays both sides the same way.

## Players

- `RandomPlayer`: a legal action at random, ending a unit's turn now and then.
- `GreedyPlayer`: one ply of heuristics (attack value, kills, risk, advance, formation, aura, the
  walk to the point, and an engineer's work). It fields a mix of each kind unless given a
  `roster`. `point=0` makes a pure fighter.
- `SitterPlayer`: deploys, then never moves: a yardstick, since turtling should lose.
- `SearchPlayer`: open-loop Monte Carlo tree search over plans for whole unit turns (a tile,
  whether to attack before moving, and for an engineer a tile to work on instead), with greedy's
  scores as priors, 16 greedy unit turns of rollout, and a value that blends material with the
  point's clock (`standing`).
- Through `agents`: `UniformPlayer`, `NetPlayer` and `AlphaZeroPlayer` on a `PolicyNet`, since
  `Tactics` is an `agents.Learnable`.
- A code player: `AlphaZeroPlayer` on `CodeEvaluator`, plain code of named features and constants
  and no network. `HAND_PICKED` holds constants picked by hand.

## Experiments

| Command | What it does |
|---|---|
| `tactics.tournament [starts] [prefix]` | Greedy against random, and greedy against itself (is the game even by side?); a picture of a game's end |
| `tactics.search [starts] [budgets...]` | The search player against greedy at a few budgets of simulations |
| `tactics.code_ladder [starts] [size] [simulations...]` | The code player against random and greedy, and what a decision costs |
| `tactics.rules [starts] [prefix] [hold_rounds...]` | How games end, pure styles, the engineer against an archer, and rosters, as tables and `{prefix}_rules.json`, to compare versions of the objective |
| `tactics.watch [start] [prefix] [players...]` | One game, decision by decision, in a window or a GIF (random, greedy, sitter, search) |
| `tactics.zero_scratch <minutes> [prefix] ...` | AlphaZero on a 15x15 board, from scratch or after imitating greedy, in a time box, with endings, picks and work by generation |
| `tactics.zero_check <weights> [starts] [simulations] [rivals...]` | The AlphaZero player on saved weights against the network alone, greedy or search |

Run each as `uv run python run.py <command>`; `--help` prints an experiment's docstring.
`benchmarks/tactics.py` times the game itself, and `benchmarks/zero_selfplay.py` times
self-play on the GPU.

## What we found

**Today's rules** (2026-11-06, 240 games a line unless it says otherwise, greedy players but for the
search, rerun once greedy went back for an owned point an enemy stands on; the results dated
2026-11-05 predate that). Greedy understates units whose worth is in how well they are used.

- **The objective does its job:** a point-seeker beat a sitter 240-0, a pure fighter 196-44, and a
  pure capper 214-26, and two sitters end at the deadline. Greedy against itself runs 7.5 rounds
  (longest 12), 46% of games won by holding and 54% by killing, 10.8 of 16 units dead, and the
  first capturer wins 67%. Neither the deadline nor the cap ever fired between point-seekers, so
  a flip in the last round is untested.
- **How long to hold:** 4 rounds gives 68% by holding and the first capturer 70%, 5 gives 46% and
  67%, 6 gives 20% and 60%, 8 gives 1% and 57%.
- **Seats:** team 0 won 43% of decided games at 5 rounds, about two standard deviations under
  half, and 43-45% at every length on the same boards. Team 1 may have a small edge from placing
  last. Greedy against itself over 20 games went 10-10 by team.
- **Rosters:** a slot is worth, over a footsoldier, 18 points of 100 for a captain, 15 for a
  knight, and 2 or 3 for an archer or an engineer (24 games a roster, noisy). Best against
  3/2/1/1/1: 2/3/2/0/1 and 4/0/2/1/1 at 75%, then 2/2/2/1/1 at 71%. Worst: rosters without knights
  or a captain, down to 4%.
- **Overwatch:** 1.2 shots a game, 80% hitting, up from 71% before the +1. Greedy watches little.
- **The engineer isn't answered yet:** greedy's engineer changes about 0.2 tiles a game, far from
  the 3 we hoped for, so the roster with an engineer against the same with an archer is even
  (115-125) and stays so from 4 to 8 rounds to hold. One tile rarely lengthens anyone's shortest
  walk on an open board, and the engineer reaches the center around round 4 or 5. Crediting tiles
  on enemy routes, or closing ways onto an owned point, changed nothing. It takes a better player
  to say what the unit is worth.
- **The search** beat greedy 35-13 (73%, 100 simulations, 48 games), with a quarter of its value on
  the clock. Before it had plans to work, and before that change to greedy, it went 38-10, against
  37-11 with a half on the clock and 30-18 with none.
- `tactics.tournament`: greedy beat random 20-0.
- **AlphaZero doesn't beat greedy yet** (2026-11-05, `tactics.zero_scratch`, about 2 hours on the
  GPU in all, on 15x15, chosen because 9x9 games end by killing 97-100% of the time). From scratch,
  after 11 generations, the network alone still stood at about even with random and 0-48 against
  greedy, with 75% of self-play games ending at the deadline: team 1 never learned the long walk
  to the point. Imitating 800 games of greedy first gave 48-0 and 18-30 at once, and self-play that
  looks like the real game (even seats, no deadlines), but 21 generations of AlphaZero added
  nothing: 17-31 for the network alone, and 9-39 with 64 simulations of search on top (19-29 at
  generation 1). Its value head stayed about as good as guessing, and its search targets grew
  flatter. The search player remains the strongest player.
- **The engineer, with search on both sides** (2026-11-05, 50 simulations, full board): given plans
  to work, the search changes about 3.7 tiles a game, against greedy's 0.2, and it doesn't pay yet.
  With an engineer against the same roster with an archer it goes 22-26 at 4 rounds to hold, 45-51
  at 5 and 24-24 at 8: level at every length, within noise. The same roster with those plans
  against without them went 41-55, so the search doesn't find its work worth the simulations it
  spreads over them, which may only mean 50 simulations are too few.
- **A network taught by search** (2026-11-05, 15x15): imitating 576 games of the search player, which
  beats greedy 34-14 there, matched 72% of its held-out decisions. The network alone lost to greedy
  14-34, but with 64 simulations of search it won 27-21, within noise of even: the first network on
  these rules not to lose to greedy, and its search helps, unlike after AlphaZero's generations.
- **Seats need telling apart:** the deadline makes the seats different, and the frame hides them.
  Without a plane saying which side loses at the deadline, the value's error before training was
  1.5, worse than guessing, and with it 0.7 by the second generation.

**The code player, constants picked by hand** (2026-11-06, `tactics.code_ladder`, 96 games a line on
15x15). It beat random 96-0 at every budget, and lost to greedy: 2-94 on its prior alone, 17-79 at
16 simulations, 31-65 at 32, 30-66 at 128. A decision takes about 14 ms at 32 simulations (about
0.2 ms for greedy, 200 ms for the search player). So the search adds much at first, then little:
more simulations don't make up for its judgment. On the full board at 32 simulations it went 96-0
and 35-61, so it doesn't lean on the small board.

**Under earlier rules** (twelve fixed units, no point, hit points at the round cap), kept for the
history of the agents:

- **2026-11-01, `tactics.tournament` at its defaults:** greedy beat random 20-0. Greedy against
  greedy, each game with dice of its own, went 8-12, with team 0 winning 10 of 20 decided
  games: no edge beyond a fair game's noise. Games run about 12 rounds. (The 2026-10-31 run
  rolled the same dice in both games of a board, which counted each game twice.)
- **The search player** (T5, 2026-10-25) beat greedy 41-5 (89%), and the search without
  rollouts 67-18.
- **Rule tuning** (T6, 2026-10-25) settled the units' numbers still in use: footsoldier Atk 4,
  knight move 3 and flanking x3, with a 24x24 board and the round cap decided by hit points, both
  since replaced.
- **Networks** (T7, 2026-10-25/26): imitating greedy matched 79% of its decisions; PPO
  self-play then beat greedy 59-37, but lost to the search player 19-29.
- **AlphaZero** (T8, 2026-10-26/27): the search beat its own network 32-15. Training from a
  PPO network didn't improve it. From scratch on 9x9 it beat random 41-4 after one generation,
  then self-play drifted into draws and it never beat greedy. The loop learns, but Tactics'
  incentives stall it. Closed as a proof of concept. The point and the end of draws are the
  answer to that stall, not yet tried with a learner. Old weights no longer load: the planes and
  verbs changed.

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
| `rules.py` | `UnitStats`, the default kinds and caps, `default_roster`, and the d20 rule |
| `objective.py` | Every rule about the point: captures, the clock and how a game ends |
| `board.py` | Symmetric boards with walls, rubble, homes and the point |
| `state.py` | `TacticsState`, `Unit`, `Hold`, `TacticsPosition`, `Verb` |
| `events.py` | What a step did: `Placed`, `Moved`, `Built`, `Attacked`, `Captured`, `GameOver`, ... |
| `players.py` | `RandomPlayer`, `GreedyPlayer`, `SitterPlayer` |
| `search.py` | `SearchPlayer`, `Plan`, `standing`, `material` |
| `code_player.py` | `CodeEvaluator`, `HAND_PICKED`, `value_features`, `move_features` |
| `match.py` | `EVAL_SEED`, `small_rules`, `summarize`, `balance` |
| `text.py`, `pictures.py` | Events and summaries as text; boards drawn as hexes |
