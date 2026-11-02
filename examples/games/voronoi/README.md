# Voronoi

The Voronoi game on a planet: the third game, and the first off a lattice. It tests `game` where
the other games don't: a `core.MeshSpace` board, cells of unequal areas, any number of players,
and steps that draw nothing.

## The rules

Players take turns placing stones on the land of an icosphere, a fixed number each (`stones`, 3
by default). Each land cell belongs to the player whose stone is fewest moves away over land
(`core.steps` with the sea blocked), and the most land area wins. Areas within `TIE` of each
other are equal. `Voronoi(players=3)` is the same game for three players. The land is drawn
from the seed, and nothing is drawn after it.

## Players

- `Greedy`: places the stone that leaves it furthest ahead of its strongest rival, looking no
  further. With two players, that's a perfect last move, which a test checks against
  `agents.exact`.
- Through `agents`: `UniformPlayer`.

## Experiments

| Command | What it does |
|---|---|
| `voronoi.tournament [starts] [prefix] [players]` | Greedy against uniform in every other seat, then greedy in every seat (which seat does the game favor?); a globe of a game's end |
| `voronoi.watch [start] [prefix] [players...]` | One game, stone by stone, on a world map with each player's land below, in a window or a GIF |

Run each as `uv run python run.py <command>`.

## What we found

Measured on 2026-10-31 with `voronoi.tournament`'s defaults (10 starts, 2 players):

- **Greedy beats uniform 18-2.**
- **The last to place wins:** greedy against itself won 19 of 20 games from the second seat.
  The seat edge is 0.45, far beyond a fair game's noise of 0.23. Placing last is a perfect
  move, so the game as it stands favors the second player heavily.
- **Greedy's first stone is a coin toss:** on an empty board, every cell scores the same, so it
  lands at random. `voronoi.watch` showed one landing at the tip of a peninsula whose only way
  out uniform then happened to block. That's why greedy loses now and then.

## Files

| File | What it holds |
|---|---|
| `game.py` | `Voronoi`, `VoronoiState` (land, stones, the territory as a derived field), `VoronoiPosition`, `Placed`, `board` |
| `players.py` | `Greedy` |
| `match.py` | `EVAL_SEED` |
| `picture.py` | `colors` and `PLAYERS`, for globes and maps |
