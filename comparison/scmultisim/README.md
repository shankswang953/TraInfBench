# scmultisim comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 2c; S1b**: Observed and terminal/observed-time lineage distributions, Wasserstein and correct-assignment scores.
  - [plot_synthetic_truth_method_columns_a4.py](plot_synthetic_truth_method_columns_a4.py)
  - [plot_synthetic_coati_cytobridge_reversed_tn_comparison.py](plot_synthetic_coati_cytobridge_reversed_tn_comparison.py)
  - [evaluate_synthetic_predicted_time_sinkhorn.py](evaluate_synthetic_predicted_time_sinkhorn.py)
- **Fig. 2c reference**: Original author plotting source; requires precomputed external assets as documented.
  - [plot_synthetic_overview.py](plot_synthetic_overview.py)
- **S1c; S2a**: Original author plotting source; requires precomputed external assets as documented.
  - [make_ever_branch_jump_table.py](make_ever_branch_jump_table.py)
- **S2b**: Original author plotting source; requires precomputed external assets as documented.
  - [make_combined_distribution_grid.py](make_combined_distribution_grid.py)
- **S2b supporting map grid**: Original author plotting source; requires precomputed external assets as documented.
  - [make_t_architecture_distribution_grid.py](make_t_architecture_distribution_grid.py)
- **S1c; S2a correctness**: Original author plotting source; requires precomputed external assets as documented.
  - [make_observed_time_correctness_table.py](make_observed_time_correctness_table.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.
