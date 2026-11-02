# Migration

Herds following the rains around a planet of 2,562 cells: seven ways of moving amounts, each
measured against the ideal free distribution.

## The model

Rain falls in a band around the planet, and grass grows where it falls. Seven herds share the
planet, one for each way of moving tried here, and none sees the others: each is a world of its
own on the same rain. A herd's animals split their cell's grass equally.

Animals that move to wherever they would eat more stop moving once every occupied cell gives the
same intake (Fretwell and Lucas's ideal free distribution), so a herd's density ends up
proportional to the rain. `distance` measures how far a herd is from that: the share of its
animals that would have to move to reach it, 0 at the ideal and 1 at the opposite. The
experiment runs it twice:

1. **still rain** for 10,000 steps, the band at the equator;
2. **seasons** for five years, the band swinging north and south once a year.

The seven ways are a `Spread` toward food, a `Spread` that leaves only where a neighbor offers
more, a `Flow` down the gradient, a `NeighborAllocation`, a `NeighborChoice` of the best
neighbor, the same choice over a `Ball` of neighbors, and a random walk.

`Migration` in `__init__.py` holds the parameters, the state and the phases.

## Run it

```bash
uv run python run.py migration [prefix]
```

It prints each herd's distance along both runs, and saves globes of the rain and the herds in
the seasons as `{prefix}_globes.png` (by default `migration_globes.png`, in the working directory). It takes about
15 s.

## What it shows

Measured on 2026-10-31. Each figure is a herd's distance from the ideal free distribution:

| Way of moving | Still rain, step 10,000 | Seasons, year 5 |
|---|---|---|
| Spread toward food | 0.122 | 0.343 |
| Spread, leave if better | 0.056 | 0.374 |
| Flow down the gradient | 0.0003 | 0.369 |
| NeighborAllocation | 0.121 | 0.341 |
| NeighborChoice, best | 0.072 | 0.090 |
| same, Ball reach | 0.0009 | 0.328 |
| random walk | 0.405 | 0.413 |

- **Only a method that comes to rest where no neighbor offers more reaches the ideal.** The
  gradient flow and the choice over a ball of neighbors do. A spread that always sends a share
  away turns into diffusion once every cell offers the same, so it settles short of the ideal.
  Greedy choice keeps overshooting. The random walk ignores the grass altogether.
- **In the seasons, the best still-rain methods lag the most.** Only greedy choice keeps up with
  the moving band.
- Every herd's total stays within 1.3e-14 over 15,000 steps: the moves conserve animals.
