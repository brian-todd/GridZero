# Biomes

Grassland and forest on a planet of 2,562 cells, each with its own grazers, and wolves that hunt
in both: apparent competition.

## The model

A step is a week. The planet is grassland in the south and forest in the north, mixed in a band
around the equator, so each biome covers a share of each cell. A biome's plants grow back toward
its share, and its grazers (rabbits on the grassland, deer in the forest) eat them, turn what
they eat into young, die at a steady rate and wander toward plants within their biome. Wolves
arrive after 20 years, hunt the grazers of both biomes and roam toward prey.

Each biome is a nested state, with its grazers nested in it, two levels deep
(`state.forest.grazers.population`). The biomes' phases are written once, for any biome.
`Biomes` in `__init__.py` holds the parameters, the state and the phases.

The theory checked, in three parts:

1. **apart:** without wolves, each biome's grazers hold its plants at their R*;
2. **wolves, no deer:** wolves hold the rabbits at their own R*, and the grass grows back, as in
   `predators`;
3. **wolves and deer** (Holt's apparent competition): where deer feed the wolves, the wolves
   press harder on the rabbits, though rabbits and deer share no food. In the band where the
   biomes mix, rabbits should fall most where the forest covers most of a cell.

## Run it

```bash
uv run python run.py biomes [years]
```

By default it runs 60 years, with deer and without, and prints the means against the theory.
It takes a few seconds and saves no pictures.

## What it shows

Measured on 2026-10-31 with the defaults:

- **Apart:** plants settle at 0.209 on the grassland and 0.310 in the forest, against the
  grazers' R* of 0.208 and 0.309.
- **With wolves:** the theory, in continuous time, holds the grazers at 0.200 and grows the
  grass back to 0.556. Over the last 10 years without deer, rabbits average 0.193 and grass
  0.577; weekly steps shift these by a few percent.
- **Apparent competition:** where the forest covers most of a cell, adding deer cuts the
  rabbits by 40% (0.023 to 0.014 per unit area), though the two never share food. Where the
  forest covers little, the rabbits barely change (+3%).
