# Population evolution

One founder species on an empty planet of 2,562 cells, adapting to its climates and splitting
into species.

## The model

Each individual carries a heritable trait: the temperature it does best at, one of 32 bins (a
component axis of one field). Births fall off with the distance between that temperature and the
cell's own. Offspring inherit their parent's trait, except a small share born into each
neighboring bin. Crowding raises deaths, felt most from individuals with similar traits. A few
individuals spread to neighboring land every step, and a trait too rare in a cell to live on dies
out there.

The theory it is checked against:

- **mutation against selection:** in one place, traits settle around the best one in a Gaussian
  of known width (Kimura);
- **species:** with competition narrower than the tolerance, traits split into clusters with
  gaps between them, each the main one in its own band of climate;
- **gene flow against local adaptation:** each cell's mean trait follows its temperature, but
  less steeply as more individuals move;
- **a warming climate:** traits lag a moving optimum, and the population collapses above a
  critical speed (Lynch and Lande).

`Evolution` in `__init__.py` holds the parameters, the state and the phases. Most of the checks
are in its tests (`tests/examples/simulations/test_population_evolution.py`).

## Run it

```bash
uv run python run.py population_evolution [steps] [prefix] [warming]
```

By default it runs 6,000 steps without warming. Every 1,000 steps it prints how much of the
continent is occupied, the species (clusters of traits), and how steeply the mean trait follows
the temperature. It saves `{prefix}_map.png` and `{prefix}_history.png` (by default under
`population_evolution_*.png`, in the working directory). With `warming`, the temperature rises by that much a step from
halfway through the run; `1e-4` is about a tenth of the speed at which one place's population
can't keep up, and the cold species die out first.

## What it shows

Measured on 2026-10-31 with the defaults, on a continent of 620 cells:

| Step | Continent occupied | Species | Gradient |
|---|---|---|---|
| 1,000 | 92% | 1 | 0.346 |
| 2,000 | 100% | 3 | 0.797 |
| 3,000 | 100% | 5 | 0.921 |
| 5,000 | 100% | 8 | 0.932 |
| 6,000 | 100% | 8 | 0.930 |

The founders fill the continent as one species, then split into 8 species, one per band of
climate. The mean trait comes to follow the temperature at a slope of about 0.93, just short of
1 because of gene flow. The run takes 10 s (585 steps a second).
