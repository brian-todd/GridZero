# Predators

Grass, rabbits, foxes and wildfire on a board of 192x128 triangles: a trophic cascade.

## The model

A step is a week. Grass grows back toward a capacity. Rabbits eat it, have young in proportion
to what they eat, die at a steady rate, and hop toward grass and away from foxes. After ten
years foxes arrive: they hunt rabbits, turn half of what they eat into young, and prowl toward
rabbits where their own cell has too few to live on. Lightning starts fires in dry summers, the
wind carries them downwind, and animals caught in a fire die.

A consumer holds its food at the level where its births just replace its deaths. Without foxes,
rabbits hold the grass at their R* (`Predators.rabbits_hold`). With foxes, the foxes hold the
rabbits at theirs (`Predators.foxes_hold`), and the grass grows back to the level those few
rabbits can keep down (`Predators.greener`). Adding a predator makes the world greener. Fires
and movement keep the board from settling, so the means only approach these levels.

The foxes are in the state from the start, with none present until the `foxes arrive` phase
brings a few, once. `Predators` in `__init__.py` holds the parameters, the state and the phases.

## Run it

```bash
uv run python run.py predators [years] [prefix]
```

By default it runs 100 years, with fires and without. It prints the means every 5 years and
against the theory, and saves a map as `{prefix}_map.png` and a history as
`{prefix}_history.png` (by default `predators_*.png`, in the working directory). It takes about 15 s a run.

## What it shows

Measured on 2026-10-31 with the defaults. The theory: rabbits alone hold grass at 0.200 per
unit area; foxes hold rabbits at 0.200, and the grass then grows to 0.500.

| | Grass, rabbits alone | Rabbits, with foxes | Grass, with foxes | Greener |
|---|---|---|---|---|
| Without fire | 0.188 | 0.192 | 0.505 | 2.7x |
| With fire | 0.188 | 0.166 | 0.308 | 1.6x |

Without fire the means land on the theory. Fire keeps knocking the board back, so the cascade
is weaker, and the populations swing from year to year instead of settling.
