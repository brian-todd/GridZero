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

## As arrays

`arrays.py` has the same phases once more for JAX (`EvolutionArrays(model).phases()`, on
`gridzero.batched`), to run many planets at once on a GPU: other seeds, or other parameters.

```python
import jax, numpy as np
from gridzero import batched
from gridzero.core import icosphere
from examples.simulations.population_evolution import Evolution
from examples.simulations.population_evolution.arrays import EvolutionArrays, spectrum

model, space = Evolution(), icosphere(16)
sim = batched.Simulation(space, EvolutionArrays(model).phases())
starts = [batched.fields(model.start(space, seed)) for seed in range(16)]
planets = jax.tree.map(lambda *xs: np.stack(xs), *starts)
run = jax.jit(jax.vmap(lambda fields: sim.run(fields, batched.root(0), 1000, spectrum)))
ends, spectra = run(planets)  # spectra: (16 planets, 1000 steps, 32 traits)
```

`Evolution` stays the model: the experiment and the tests read it, and the tests hold the
arrays to it step by step (`check_array_model`): every field within 1e-4 of its largest value
over 40 steps, on a globe and on a map, with and without warming. Amounts are float32 there.

Measured on 2026-11-14 (RTX 3080, JAX 0.12.0.1, i9-12900K; 2,562 cells, 200 steps a run), in
steps a second: 85,000 to 99,000 for 4 to 16 planets at once, 64,000 for 64, 30,000 for 1,024;
`Evolution` on `core` does 590 on one core and 3,100 at best on the whole CPU (8 processes).

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
