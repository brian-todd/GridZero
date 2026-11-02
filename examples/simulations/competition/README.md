# Competition

Two groups competing for one food supply on a 64x64 hex torus, checked against Tilman's R* rule
of ecology.

## The model

Each land cell grows food toward a capacity. Two groups, lowlanders and highlanders, eat it in
proportion to how much there is and how many of them there are, and die at a steady rate. When
food runs short, a cell's food is shared out in proportion to what each group asked for (a
`Claim`). A few people of each group drift to neighboring land every step.

A group alone settles where the food it finds just replaces its deaths: the food density
`R* = mortality / appetite`. Two groups on one food can't both do that, so the one with the
lower R* eats the food down until the other dies out. The run checks this twice:

- **one niche:** every cell favors the same group, which should take every landmass, however
  rare it starts;
- **two niches:** mortality depends on height, so each group has the lower R* on its own half of
  the land. Each should hold its half, with a mixed border where people drift across.

`Competition` in `__init__.py` holds the parameters, the state and the phases.

## Run it

```bash
uv run python run.py competition [prefix]
```

It prints both runs and their checks, and saves a map of the two niches as
`{prefix}_map.png` (by default `competition_map.png`, in the working directory).

## What it shows

Measured on 2026-10-31 with the defaults:

- **One niche:** the lowlanders (R* 0.20) drive the highlanders (R* 0.40) out by step 400.
  The food the winners find when they eat averages 0.2001, matching their R*.
- **Two niches:** 699.5 lowlanders and 712.0 highlanders coexist. The group with the lower R*
  holds 96.8% of the land cells, and 18.7% of land cells hold more than 10% of the other group:
  that's the border.
