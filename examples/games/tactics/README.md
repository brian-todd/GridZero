# Tactics

A two-player skirmish on a 25x25 hex board with walls and rubble, fought over one tile at the
center: the first game, and the one the `game` and `agents` layers were first written for.

## The rules

**Squad and deployment.** Each side fills 8 places, with at most 6 footsoldiers, 3 archers, 2
knights, a captain and an engineer. Every unit takes 1 place, but a second knight takes 2, so a
side fields 7 or 8 units. Sides alternate placing a unit in their home corner, choosing its kind and tile
together: the side with fewer units placed goes next, team 0 on a tie, and a side with no room
left is passed over. Deployment is in the open: the fog (below) falls when the battle starts.

**Turns.** Every round each living unit gets one turn, in an order rolled once when the battle
starts and seen by both sides: the teams alternate, a coin says which starts, each team's units
come in a random order, the side with more units takes the last turns, and the dead drop out. On
its turn a unit may move up to its allowance and attack once, in any order (move, attack, move),
turn to face any direction, and end its turn.

| Kind | HP | Atk | Def | Move | Reach | Sight | Passive |
|---|---|---|---|---|---|---|---|
| Footsoldier | 5 | 4 | 2 | 2 | 1 | 3 | Gains Def from adjacent footsoldiers (a formation), and braces against knights |
| Archer | 3 | 2 | 1 | 2 | 3 | 4 | May defer its attack onto a tile, and fires when an enemy enters it, with +1 Atk |
| Knight | 7 | 2 | 2 | 3 | 1 | 5 | Charges: +2 Atk after moving its full 3 tiles. A second knight takes 2 places |
| Captain | 9 | 4 | 3 | 2 | 1 | 3 | Lifts the Atk and Def of friends near it |
| Engineer | 3 | 2 | 1 | 2 | 3 | 3 | Builds and breaks terrain |

- **Attacks** hit on a d20 roll that gets easier as the attacker's Atk exceeds the target's Def,
  and a hit takes Atk from the target's hit points.
- **Flanking** triples Atk: attacking from outside the three sides a target faces, or with two
  enemies adjacent to it. A unit attacked since its last turn began is engaged, and later
  attackers get +1 Atk. Overwatch's +1 comes after the multiplier, so a flanked arrow hits for 7.
  A charge's +2 comes before it.
- **The wall:** a footsoldier with a footsoldier beside it, and not engaged, can't be flanked by a
  knight from any side, and a knight that charges it loses the +2 and is struck first, unflanked.
  Other units flank it as usual, and once something has hit it, a knight may too: an archer's
  arrow opens the wall for a knight's charge.
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
| A side has owned the point for 4 rounds in a row | It wins |
| Nobody has captured the point by the end of round 10 | Team 1 loses: it places last, so it must take the first step |
| The end of round 20 | The point's owner wins |

**Fog.** Once the battle starts, a side sees the other's units only on tiles within the sight of
one of its own, around walls as a line of fire goes. Deployment, the terrain, the order of turns,
who holds the point and every death stay in the open. `fog=False` hides nothing, as the game was
before 2026-11-10.

- **What a side is shown** (`Tactics.view`) is a belief: a position of its own units, of the enemies
  it sees, and of the rest as ghosts where it last saw them, with the hit points and facing they
  had then. A ghost that isn't where it was, once the side looks there, is believed on the nearest
  tile out of its sight. A unit glimpsed on its way is last seen where it left sight.
- **What a side may do** is what would be legal in its belief, so the legal actions tell it nothing
  more. A move walks the route the mover would plan and stops before the first unit it didn't know
  of, which it then sees; it keeps the movement it has left.
- **A unit always sees what attacks it:** every kind sees at least as far as the furthest reach.
- **A belief plays on as a game that hides nothing,** so a search (the search player, a code player,
  AlphaZero) plans on what its player knows and never steps the position it can't see.

The rules are data: `Tactics` and `UnitStats` (`rules.py`) hold every number, and every rule about
the point is in `objective.py`, so that it can be reworked and measured with `tactics.rules`. The
boards are symmetric under the half turn (`make_board`), and team 1 sees the board turned half a
turn, so a network plays both sides the same way. `earlier_rules()` gives the rules before
2026-11-09 (knights with Atk 3, a double move and a place each, no wall, hold 5, turns shuffled
every round, engaged until the round ends, no fog), which the findings below dated earlier were
found under. `small_rules()` gives quick games on a 15x15 board, with homes 3 rows deep and 3 rounds to
hold: at 4, greedy players ended only 5% of games there by holding, at 3 a quarter.

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
  and no network. `HAND_PICKED` holds constants picked by hand. With `lookahead` it reads its value
  after greedy has played that many unit turns on for both sides, as the constants that play do.
  The experiments have it deploy as greedy does (`Fielded`).

## Experiments

| Command | What it does |
|---|---|
| `tactics.tournament [starts] [prefix]` | Greedy against random, and greedy against itself (is the game even by side?); a picture of a game's end |
| `tactics.search [starts] [budgets...]` | The search player against greedy at a few budgets of simulations |
| `tactics.code_games [games] [prefix] [work_plans] [replica] [minutes]` | The search player against itself on 15x15, saved as `{prefix}_lessons.npz` for code players to fit to |
| `tactics.code_fit <prefix> <lessons...>` | A code player's constants fit to those games, scored on games kept out against simple baselines, and saved as `{prefix}_fit.json`, with what plays as `{prefix}_player.json`: one weight overruled, and the value read 4 unit turns ahead |
| `tactics.code_ladder [weights] [starts] [size] [budgets] [replica] [rivals...]` | The code player against random, greedy or the search player, at equal simulations or equal time, and what a decision and a unit's turn cost |
| `tactics.rules [weights] [starts] [prefix] [simulations] [hold_rounds...]` | How games end, the engineer against an archer, rosters, and pure styles (greedy) or the roster it picks (code player), as tables and `{prefix}_rules.json`, to compare versions of the objective or players |
| `tactics.fog [weights] [starts] [size] [simulations]` | What fog changes: greedy against itself in the open and under fog, a second knight's worth with and without its longer sight, and a code player's record both ways |
| `tactics.watch [start] [prefix] [players...]` | One game, decision by decision, in a window or a GIF (random, greedy, sitter, search) |
| `tactics.zero_scratch <minutes> [prefix] ...` | AlphaZero on a 15x15 board, from scratch or after imitating greedy, in a time box, with endings, picks and work by generation |
| `tactics.ppo <minutes> [prefix] [starts] [games] [taught] [entropy]` | A network warmed up by PPO self-play on 15x15 after imitating greedy, a quarter of its games against greedy, in a time box, scored alone against greedy as it goes |
| `tactics.zero_check <weights> [starts] [simulations] [rivals...]` | The AlphaZero player on saved weights against the network alone, greedy or search |

Run each as `uv run python run.py <command>`; `--help` prints an experiment's docstring.
`benchmarks/tactics.py` times the game itself, and `benchmarks/zero_selfplay.py` times
self-play on the GPU.

## What we found

Everything dated before 2026-11-10 was measured without fog (`fog=False`), which was the game then.

**Fog** (2026-11-10, `tactics.fog` with the fit code player, 15x15, 96 games a line, where about 5
wins is noise; the scouting lines also on 25x25 with 480 games).

- **Fog hides most of the other side:** a side deciding doesn't see 61% of the enemy's units on
  15x15 and 71% on 25x25.
- **It barely changes greedy's games:** 5.3 rounds against 5.4, 33% ended by holding against 27%.
  Greedy walks at where it last saw the enemy, and the armies meet at the point either way.
- **A planner loses its edge:** the code player at 128 simulations beat greedy 69-27 in the open
  and 51-45 under fog, about what it scores with no search to speak of (48-48 at 32 simulations
  both ways). With both sides shown everything, fog's other rules unchanged, it scored 69-27 again:
  the loss is what it doesn't know. Its search plans against ghosts where it last saw them.
- **Greedy gets nothing from a scout:** fielding a second knight for two footsoldiers won 43% in
  the open, 43% under fog and 45% with knights seeing no further than the rest (25x25: 49%, 51% and
  48%). It reads greedy, which doesn't act on what it sees further off, not the knight.

**A code player that looks ahead** (2026-11-11, under fog on 15x15: `tactics.code_games` for 400
games of the search player, `tactics.code_fit`, then `tactics.code_ladder` on starts of their own,
192 games against greedy and 96 against the search player at 50 simulations, which takes 268 ms a
decision and 707 ms a unit's turn). The same constants, their value read as the position stands or
after greedy has played 4 unit turns on:

| Value read | Simulations | ms a decision | ms a unit's turn | v greedy | v search player |
|---|---|---|---|---|---|
| As the position stands | 128 | 97 | 316 | 102-90 | 37-59 |
| 4 unit turns ahead | 32 | 119 | 363 | 118-74 | |
| 4 unit turns ahead | 64 | 226 | 648 | 131-61 | 52-44 |
| 4 unit turns ahead | 128 | 479 | 1428 | 135-57 | 44-52 |

- **What a static value lacks is the next few turns.** A refit under fog, more features, a better
  prior and deeper search had all left the code player near even with greedy; reading the value 4
  unit turns ahead puts it level with the search player it was fit to (96-96 over the two budgets),
  at that player's time.
- **64 simulations are enough,** at about two thirds of a second a unit's turn; 128 double the time
  for nothing measurable. With greedy deploying and on the starts players are scored on, 2 turns
  ahead scored 102-90 against greedy at 64 simulations and 3 turns 131-61 at 128, against 138-54 and
  137-55 for 4: set `lookahead` in the saved constants to try another.
- **The fit stays as it was:** refitting the value to the positions ahead scored 133-59 against
  137-55.
- **It deploys as greedy does.** Deploying by its own search it scored the same against greedy and
  71-121 against the search player over two sets of starts, against 92-100 with greedy's deployment.

**A network warmed up by PPO** (2026-11-11, under fog on 15x15: `tactics.ppo 100 <prefix> 24 288`
on an RTX 3080 and 24 processes; then its last network on the CPU, alone and under the AlphaZero
search at 128 simulations, 192 games a match on the starts players are scored on, greedy deploying
for it, where about 7 wins is noise. The search player has 50 simulations.)

| Player | v the network alone | v greedy | v the search player |
|---|---|---|---|
| The network alone, its best action | | 124-68 | 81-111 |
| The search on it, `c_scale` 0.03 | 125-67 | 123-69 | 94-98 |
| The search on it, `c_scale` 0.1 | 118-74 | 131-61 | 106-86 |

- **PPO plays many games:** 268 iterations of 288 games in 100 minutes, 77,000 games and 13 million
  decisions. An iteration takes 22 seconds, 18 of them play: games here last 5 or 6 rounds.
- **It learns to beat greedy, and then levels off.** After imitating 200 games of greedy the network
  alone scored 9-39. In training's own games against greedy (actions drawn from its policy, deploying
  by itself) it won 28% over the first 20 iterations, 47% over iterations 61-80, and 52-58% from
  iteration 100 on.
- **The search adds to it against planners, not against greedy.** It beats its own network two games
  in three and brings it from 81-111 against the search player to level with it (200-184 over the
  two weights); against greedy it scores what the network alone does.
- **Its strength is its policy, not its value.** On 480 games of the network against itself, its
  value says who wins no better than the code player's formula, fit to 400 games: in the last
  quarter of a game their correlations with the result are 0.68 and 0.74, in the third 0.28 and 0.33,
  and in the first half neither knows (under 0.1). A likely cause, not tested: PPO fits the value to
  its own later values more than to results (`lam` 0.95), and its loss weighs the value at a half.

**The default rules** (2026-11-09, 25x25, starts of their own: a round-robin of six rosters, 480 games a
match for greedy and 96 for the code player at 128 simulations, and two matches for the search player at
50). The rosters: default 3/2/1/1/1, two knights 1/2/2/1/1, no knight 4/2/0/1/1, wall 6/0/1/1/0, archers
3/3/1/1/0, cavalry 3/0/2/1/1.

| Reading | Greedy | Code player | Search player |
|---|---|---|---|
| Won by holding the point, of games | 32% | 41% | |
| Team 0's share of games | 50% | 51% | |
| Round-robin share: archers | 58% | 58% | |
| Round-robin share: two knights | 55% | 56% | |
| Round-robin share: no knight | 54% | 48% | |
| Round-robin share: default | 53% | 53% | |
| Round-robin share: wall | 42% | 46% | |
| Round-robin share: cavalry | 38% | 40% | |
| Two knights v default | 244-236 | 59-37 | 47-49 |
| Wall v cavalry | 274-206 | 54-42 | 55-41 |
| Archers v wall | 304-176 | 63-33 | |

- **The second knight is priced about right:** two knights against the default won 49-62%, where a second
  knight for a single place had won 61-69% for all three players. Over a footsoldier, greedy now values
  a knight at about 3 points of 100, against -4 when every knight took two places.
- **The wall works by being left alone:** of knights' attacks on footsoldiers, about one in ten is on one
  with a footsoldier beside it, and knights are struck first only 1 to 9 times in 480 games. The wall
  beats knights without archers, archers beat the wall, and mixed rosters (two knights, which keeps two
  archers) beat it too: arrows break the wall and knights finish it. Knights without archers are the
  weakest roster, so the triangle is not closed: archers beat cavalry as well.
- **No roster wins everything:** greedy's archers win every match, but those against the default and two
  knights by less than one sigma, and the code player's archers lose to two knights.
- **Holding:** 32-41% of games end by holding for 4 rounds, against 46-53% for 5 under the earlier
  rules. The better player holds more.

**The rules before 2026-11-09** (`earlier_rules()`; 2026-11-06, 240 games a line unless it says otherwise, greedy players but for the
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

**Search against itself** (2026-11-06, `tactics.code_games`, 600 games each on 15x15, 50 simulations):
5.3 rounds a game (longest 10). With plans to work, team 0 won 54% and an engineer changed 3.5 tiles a
game; without them, team 0 won 51% and it changed 0.1. So the lean to team 1 under greedy (43%) doesn't
show between search players, and 600 games only measure a seat to about 4 points either way.

**A code player fit to the search player** (2026-11-06, `tactics.code_fit`, the games above with plans
to work, 120 of 600 kept out). Its value correlates 0.81 with the teacher's (0.72 for the hand-picked
constants, 0.66 for material alone). Its prior picks the teacher's action 69% of the time (42% by hand,
36% by the verb alone), and greedy picks it 79%, ahead on deferred attacks and placement: the teacher's
plans are mostly greedy's. Weights hardly move between the games with and without work, except those of
work itself.

Better imitation didn't all play better. At 32 simulations on 15x15, against greedy on one seed set, 384
games a line, or 192 where marked: hand value and hand prior 55-137 (192), fit value and fit prior 135-249,
hand value and fit prior 206-178. So the prior gains much, and the value loses it, through one weight:
the fit gives a knight's hit points about 0. Taking only that weight from the hand-picked constants gives
195-189, and giving only that one to them gives 117-267; no other group of weights moves the record much.
The teacher's value really is flat in a knight's lead (0.49 a knight up, 0.51 down, other units level),
while the side a knight up goes on to win 65% against 47%. Fitting to who won gives the knight +0.37, but
plays no better (141-243): a value that copies a judge inherits what it misjudges, and being right about
who wins isn't enough for choosing moves. So the constants that play (`{prefix}_player.json`) are the fit's
but for that weight, taken by hand: the one thing in them a person, not the teacher, decided.

**The code player that came of it** (2026-11-06, `tactics.code_ladder`, starts of their own, not those the
choice above was made on; fit on 15x15 only). Records against greedy over 192 games, against the search
player over 96, which at 50 simulations is the player it was fit to:

| Board | Simulations | ms a decision | v random | v greedy | v search player |
|---|---|---|---|---|---|
| 15x15 | 32 | 13 | 192-0 | 91-101 | |
| 15x15 | 50 | 21 | | | 53-43 |
| 15x15 | 64 | 28 | 192-0 | 122-70 | |
| 15x15 | 128 | 57 | 192-0 | 135-57 | 48-48 |
| 15x15 | about 400, the search player's time | 192 | 192-0 | 128-64 | 47-49 |
| 25x25 | 32 | 14 | 192-0 | 117-75 | |
| 25x25 | 50 | 22 | | | 55-41 |
| 25x25 | 64 | 29 | 192-0 | 114-78 | |
| 25x25 | 128 | 61 | 192-0 | 137-55 | 56-40 |

The search player itself beat greedy 130-62 on 15x15 and 57-39 over 96 games on 25x25, at about 170 ms a
decision on 15x15. So the code player clearly beats greedy, matches its teacher on the board it learned on at
a third of the time, and on the board it never saw does at least as well: 111-81 together, where 50 simulations
a turn spread the teacher thin. It is flat past about 128 simulations. All of this is against a teacher at 50
simulations; a stronger one has not been tried.

**What the rules look like to the three players** (2026-11-06, 25x25, `tactics.rules`, and matches on starts
of their own). Greedy, the code player at 128 simulations and the search player at 50 agree on most of it, though they
aren't independent: the code player is fit to the search player, which rolls out with greedy, and all three deploy
as greedy does in these rosters.

| Reading, on the default rules | Greedy | Code player | Search player |
|---|---|---|---|
| Won by holding the point, of games (hold 5) | 46% | 53% | 38% |
| First capturer won (hold 5) | 67% | 65% | 67% |
| Team 0's share of decided games | 44% (960 games) | 49% (768) | 49% (96) |
| Tiles built a game | 0.0 | 7.0 | 7.3 |
| Knights lost, of those placed (hold 5) | 100% | 99% | 100% |
| Second knight for a footsoldier, v default | 331-149 | 118-74 | 63-33 |
| Knight for the engineer, v default | 329-151 | 126-66 | |
| Footsoldier for the knight, v default | 150-330 | 56-136 | |
| Archer for the engineer, v default, all hold lengths | 497-463 | 409-359 | 45-51 (hold 5) |
| The code player's own pick, 5/0/2/1/0, v default | 281-199 | 99-93 | 54-42 |

- **Hold length:** for greedy and the code player, 5 is the only length tried near an even split of holding and
  killing. At 4 holds win about 70%, at 6 about 20%, at 8 almost none. The search player played only 5.
- **Seats:** greedy leans to team 1 (3.6 sigma from even), the two better players don't, though the gap between
  greedy and the code player is only about 2 sigma. No reason yet to change a rule.
- **Knights are what a roster is about:** a slot over a footsoldier is worth 15-19 points of 100 for a knight and 18
  for the captain, 2-3 for an archer or an engineer, for both players that played every roster. Greedy's default roster
  gives up about 10 to 20 points of 100 by fielding one knight. Knights nearly always die, and pay for it: 5 damage
  each a game, against 3 to 4 for others.
- **The engineer is about an archer,** perhaps a little less, even played by those who build.
- **Its own pick:** free to choose, the code player always fields 5 footsoldiers, 2 knights and a captain, no archers
  or engineers, so no shots or tiles in self-play, and deploys them itself in the 99-93 above. This probably follows
  its value's weight on hit points, not tested, and does no harm, perhaps through the second knight.

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
2026-10-31, their work done. Their library pieces stay: `agents.imitate` and `agents.ppo`
(`agents.llm`, the LLM player's, was removed on 2026-11-09, unused since). The hill climb T6
ran on (`gridzero.tuning`) and the board and unit tables as text (`board_text`, `units_text`)
were removed on 2026-11-01, unused, and so was
`tactics.zero_train`, which trained from the PPO stage's weights (`runs/ppo100.pt`), which no
experiment can make any more. `zero_check` still plays saved full-size weights.

## Files

| File | What it holds |
|---|---|
| `game.py` | `Tactics`: the rules, `start`, `legal`, `step`, `view` (a belief under fog), `sees`, `odds`, and the observation planes for networks |
| `rules.py` | `UnitStats`, the default kinds and caps, `default_roster`, and the d20 rule |
| `objective.py` | Every rule about the point: captures, the clock and how a game ends |
| `board.py` | Symmetric boards with walls, rubble, homes and the point |
| `state.py` | `TacticsState`, `Unit`, `Hold`, `TacticsPosition`, `Verb` |
| `events.py` | What a step did: `Placed`, `Moved`, `Built`, `Attacked`, `Captured`, `GameOver`, ... |
| `players.py` | `RandomPlayer`, `GreedyPlayer`, `SitterPlayer` |
| `search.py` | `SearchPlayer`, `Plan`, `Judgment`, `standing`, `material` |
| `code_player.py` | `CodeEvaluator`, `HAND_PICKED`, `value_features`, `move_features`, `look_ahead` |
| `lessons.py` | `Lessons`, `lessons_of`, `positions_of`, `teach`: a search player's games kept to fit to |
| `fit.py` | `fit`, `cases_of`, `FitScore`, `playing`: a code player's constants fit to those games and scored, and those that play |
| `match.py` | `EVAL_SEED`, `small_rules`, `summarize`, `balance` |
| `text.py`, `pictures.py` | Events and summaries as text; boards drawn as hexes |
