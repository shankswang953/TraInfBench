# multi dataset comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **S17; S18**: Modality kinetic energy and joint RNA/ATAC discrepancy from precomputed results.
  - [plot_coati_transport_by_cy_four_datasets.py](plot_coati_transport_by_cy_four_datasets.py)
  - [plot_combined_synchronization_ablation.py](plot_combined_synchronization_ablation.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.
