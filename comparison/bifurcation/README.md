# bifurcation comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 2b; S1a**: Original author plotting source; requires precomputed external assets as documented.
  - [plot_toysplit_summary.py](plot_toysplit_summary.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.

These two original summary plotters also integrate already trained COATI toy checkpoints. Their core model code is deliberately external; neither the model nor its training is included. Restore the original `GaussToy` / `ToySplit` assets and `src` modules under `external/COATI`. Do not run these expecting a data-free simulation.

For the small-beta supplement use `--betas 0.001 0.01 0.1`; main-panel defaults are `0.1 1 10`.
