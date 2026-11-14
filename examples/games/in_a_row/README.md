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
- Code players: `AlphaZeroPlayer` on an evaluator written as code, which needs no network.
  `LineEvaluator` counts the lines each player keeps open and prefers the middle column, with
  constants picked by hand; `SEARCH_ALONE` is the search knowing only the rules;
  `FittedEvaluator` holds named features (threats by row parity, wins, blocks, lines) with
  constants fit to what a trained agent judged.
- Through `agents`: `UniformPlayer`; `NetPlayer` and `AlphaZeroPlayer` on a `PolicyNet`; and
  `agents.exact.solve`, the value under perfect play of every position of a small board.

## Experiments

| Command | What it does |
|---|---|
| `in_a_row.zero_scratch <minutes> [prefix] [starts] [seed]` | AlphaZero on Connect Four from scratch, in a time box; after each generation, the network alone against uniform and `ThreatPlayer`, the agent with its search against `ThreatPlayer`, and its opening |
| `in_a_row.zero_check <weights> [starts] [simulations]` | The same measures, for saved weights |
| `in_a_row.code_fit <weights> [starts] [prefix] [rounds]` | Fit a code player to the network's agent, in rounds: games, positions judged by its search, constants fit to every lesson so far (each round's games are the last fit's own), and how each fit does |
| `in_a_row.code_report <weights> [shown] [prefix]` | One HTML file on a fit: how each fit does on each round's positions, whether a search steers into its own errors, the labels against a deeper search and exact values, what candidate features would add, and galleries of the widest disagreements and costliest moves |
| `in_a_row.code_ladder <weights> <fit> [starts] [simulations...]` | Code players against the network's agent at equal simulations (16, 64, 256) and at equal time: what each makes of the first columns against perfect play, the time a simulation takes, and ladders |
| `in_a_row.play <fit> [prefix] [port] [weights]` | Play against any player on a page served on this machine (opponents roughly weakest first; with `weights`, the network's agent too), with or without search noise, kept in each record. It says why a search played each move, asks for a note after each game, and keeps both in `{prefix}_games.json`. The cards of `code_report` link to their boards here |

Run each as `uv run python run.py <command>`. There is no `watch`: boards are drawn as text
(`board_text`), and a person sits down with `play`.

## What we found

- **Exact values** (2026-10-28): tic-tac-toe is a draw (solved in 0.16 s), and three in a row
  on 4x3 and 4x4 boards is a first player's win. The number of positions by ply matches OEIS
  A212693.
- **Agent A** (2026-10-28, seed 0, 25 minutes, 65 generations): the network alone beat uniform
  100-0 and `ThreatPlayer` 74-22-4. With its search (96 simulations) it beat `ThreatPlayer`
  96-2-2. It opens in the center column, the only winning first move.
- **Agent B** (2026-10-29, seed 1, 25 minutes, 47 generations): the network alone beat
  `ThreatPlayer` 82-17-1, and with its search 96-4-0. It opens in the center column too.
- **Self-play drifts into draws**: by the end of agent A's run, 190 of 256
  self-play games were drawn. A study on a small board against the exact solver is still open.
- **Agent A fails at the opening** (2026-11-03): its network values every first column
  0.47-0.49, where perfect play gives 0, 0, 1/2, 1, 1/2, 0, 0, and at 1600 simulations its
  search prefers column 1. That is the draw drift. Past the opening it is still far ahead of
  hand-written code at equal simulations.
- **A first code player fit to agent A** (2026-11-04, `code_fit`, 150 starts, 3 rounds, 24,875
  positions; `code_ladder`, 100 starts): 23 constants.
  - Against agent A it scores 37%, 31% and 36% at 16, 64 and 256 simulations (69-121-10,
    53-128-19, 63-119-18), where hand-picked `LineEvaluator` scores 11%, 15% and 8%. At equal
    time, which buys it 2.4x the simulations, 48%, 36% and 40%. Against `ThreatPlayer` it wins
    94-97%, level with or ahead of agent A's 87-95%.
  - Search alone catches the hand-picked lines by 256 simulations (108-92), but not the fit
    (20-179-1): fitted constants keep paying as the search deepens.
  - **The fit finds the parity rule of Connect Four's theory (Allis 1988) unprompted:** a
    threat weighs 0.78 on an odd row and 0.20 on an even one for the first player, and 0.35
    and 0.78 for the second. Features only split threats by row; which rows favor whom comes
    from the agent's judgment.
  - A simulation takes 254 us, against 616 us with the network, 155 us with `LineEvaluator`
    and 112 us with the search alone, on one core.
  - Each round of the fit's own games helped against `LineEvaluator` (at 64 simulations,
    62-34-4, then 81-17-2 and 75-20-5, in 100 games), while its value's correlation with
    agent A's on games kept out fell from 0.81 to 0.49: the features explain less of the
    positions a code player reaches.
- **Why the fit falls behind** (2026-11-04, `code_report`):
  - The drop is the fit's own error. Agent A's labels barely move at 2048 simulations
    (correlation 0.98-0.99), and against exact values from ply 26 they correlate 0.78-0.87 where
    the fit's correlate 0.23-0.62.
  - A search doesn't steer into its own errors more on the positions it chose: picking a move
    adds about +0.02 to +0.03 to its overrating on any round's positions.
  - **It counted threats without their order.** Its widest errors held several threats of a
    player that never cash in, and counting only those with no rival threat below raised the
    value's correlation on games kept out by up to 0.10 when added alone.
  - Moves late in a game aren't the gap: where a wrong move exists, from ply 26, the fit keeps
    the exact value 97-99% of the time, at least as often as agent A. By agent A's own q, the fit's
    moves lose most before ply 21 (0.08 against 0.01 after ply 28), where nothing here gives exact
    values. That q is noisy: every one of the costliest moves shown from ply 29 was a false alarm.
- **With clean threats** (2026-11-04, same runs, two more value features, 25 constants): against
  agent A it scores 44.5%, 39% and 47.5% at 16, 64 and 256 simulations (84-106-10, 72-116-12,
  62-72-66), and 45%, 43% and 43% at equal time, where the first fit scored 37%, 31% and 36%, and
  48%, 36% and 40%. Tested alone on the first fit's lessons, the two features beat it 60-62% head
  to head at every budget.
  - Its value's correlation with exact values from ply 26 rises to 0.80, 0.66 and 0.70 by round,
    from 0.62, 0.23 and 0.48; the agent's own is 0.82. A plain threat on the first player's odd
    rows now weighs little (+0.13), against +1.12 for a clean one: the order carries it.
  - Its moves keep the exact value 98-99% of the time, where a wrong move exists, from ply 26;
    the agent's 97%.
  - The other candidates measured now add under 0.02, so hand-made candidates have run dry here.
    Its widest errors left are late boards with two clean threats, rated about 0.9 and in fact
    lost.
- **A node below the root could stop looking** (2026-11-05): `agents.zero` rescaled values to
  [0, 1] at every node, as `mctx` does. If the first action a node tried looked better than the
  node's own value, rescaling gave it the whole bonus and the others none, however small the
  gap, and it was never left. At 512 simulations, 68% of the fit's nodes below the root with 20
  visits or more had tried one action only, and 36% of agent A's. After 4, 6, 3, a stone in
  column 4 loses by force, yet the fit never looked at the win, and rated it 0.54 for blue after
  1,333 visits. The search now rescales at the root only, which leaves 1% for the fit (agent A:
  11%).
  - The fit at 1024 simulations now beats the old search at 1024 77-22-1, and agent A 26-2-72.
    Both are level at 64 (44% and 54%, 100 games, about +-10 points). So is the search alone at
    any budget up to 1024, MuZero on Colonies at 32 and 128, and Patches below 512.
  - The fit now scores 60% against the old agent A at 1024 simulations each, up from 46%.
  - The other findings above were measured before this fix, and the fit and agent A learned from
    the old search.
  - That hardly hurt the fit's value labels. Judged again on the new search, 2,000 of its positions
    correlate 0.97 with the old values at 256 simulations and 0.96 at 1024, though the move
    changes on 27%. Against exact values from ply 26 (1,000 positions), the labels correlate 0.82
    old, 0.82 new at 256 and 0.85 at 1024 and 4096, where the fit's own value correlates 0.68: its
    features are the limit, not its labels.
- **How far apart the opponents on the page are** (2026-11-05, a round robin of its 11 players
  with noise, 100 games a pair; share of points against all the others): uniform 1.3%, search
  alone at 64 simulations 15%, `ThreatPlayer` 20%, `LineEvaluator` at 64 37%, the fit at 16 and
  64 50% and 62%, agent A at 64 and 256 65% and 66%, the fit at 256 and 1024 70% and 78%, and
  agent A at 1024 86%.
  - Before the fix the top was flat (the fit at 64, 256 and 1024 66%, 67% and 70%, agent A 70%,
    74% and 74%). Now budget shows: the fit at 1024 met the fit at 256 61-34-5, and agent A at
    1024 met agent A at 256 46-3-51.
  - Knowledge still outweighs search: a script with no search beat the search alone at 64, 67-31-2.
  - Noise costs little now: after 4, 6, 3 from the start, with noise the fit picks a safe column
    33 of 40 times at 64 simulations and always at 256 and 1024, where it was 24 of 40, 24 of 40
    and 8 of 20 before the fix. Agent A always does, and so does either without noise.
- **A weakness in `agents.zero`**: with an uneven last round of sequential halving,
  the search could play a lost move over a drawn one. Exact values found it, and it was fixed on
  2026-10-29.

The weights and logs are in `runs/in_a_row/` (not in git).

## Files

| File | What it holds |
|---|---|
| `game.py` | `InARow`: the rules, the lines, and the observation planes |
| `state.py` | `InARowState`, `InARowPosition`, `Placed` |
| `players.py` | `ThreatPlayer` |
| `match.py` | `EVAL_SEED`, `FIRST_MOVES`, and `opening`: what an evaluator makes of the empty board and of each first move |
| `code_player.py` | `LineEvaluator`, `SEARCH_ALONE`, `FittedEvaluator`, their features: evaluators written as code |
| `lessons.py` | `Lessons`, `distinct_positions`, `judge`: positions a teacher's search has judged |
| `fit.py` | `fit`: constants from lessons, scored on games kept out |
| `report.py` | `analyze`, `page`, `CANDIDATES`: a report on a fit, as one HTML file |
| `play.py`, `play.html` | `Table`, `explain`, `rungs`, `serve`: games with a person, why a search moved, and the page |
| `text.py` | `board_text` |
