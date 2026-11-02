# Settlers

Farmers and herders settling a world of 128x128 hexes, with seasons, droughts, raids and
worn-out soil.

## The model

A step is a month. Each land cell grows food, faster in summer and on fertile soil. Two peoples
harvest it into stores, eat from their stores, have children when fed and die when starving.
When their stores run low they move toward food, carrying their stores with them (a nested
state moved as one). Herders raid farmers they share a cell with. Droughts strike in summer, and
land grazed bare loses fertility, which comes back slowly once left alone. A people too few in a
cell to live on dies out there.

- **Farmers** harvest more on low land and keep their stores longer, but rarely move.
- **Herders** harvest as much in the hills as on the coasts, move readily and win their raids,
  but their stores spoil.

The point is less the history than the run: eight phases over thousands of steps, as a test of
the loop. `Settlement` in `__init__.py` holds the parameters, the state and the phases.

## Run it

```bash
uv run python run.py settlers [years] [prefix]
```

By default it runs 1,000 years (12,000 steps). It prints the totals 20 times along the run and
how fast each stretch ran, and saves a map of the end as `{prefix}_map.png` and a chart of the
history as `{prefix}_history.png` (by default `settlers_*.png`, in the working directory).

## What it shows

Measured on 2026-10-31 with the defaults:

- **The herders spread first and peak early:** about 72,000 by year 200, in 7,900 cells. Then
  they decline as their grazing wears the soil, and mean fertility falls from 0.99 to 0.84.
- **The farmers creep outward and keep growing:** from 500 at year 50 to about 23,500 in 2,700
  cells by year 1,000, while the herders settle near 50,000.
- **Speed:** 452 steps a second on one core, steady along the run: 12,000 steps in 27 s.
