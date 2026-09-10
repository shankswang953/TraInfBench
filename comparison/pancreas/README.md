# pancreas comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 5**: Transition matrices, forward/stay/reverse/other fractions, ATAC edge/node error, composition residuals and mass decomposition.
  - [plot_pancreas_coati_celltype_transition_matrix.py](plot_pancreas_coati_celltype_transition_matrix.py)
  - [plot_pancreas_six_method_forward_stay_ambiguous_table.py](plot_pancreas_six_method_forward_stay_ambiguous_table.py)
  - [plot_pancreas_edge_node_atac_summary_table.py](plot_pancreas_edge_node_atac_summary_table.py)
  - [plot_pancreas_six_method_full_composition_residual_heatmap.py](plot_pancreas_six_method_full_composition_residual_heatmap.py)
  - [plot_pancreas_mass_stress_evidence.py](plot_pancreas_mass_stress_evidence.py)
- **S9**: Held-out E15.5 distributions, metrics and compositions.
  - [plot_pancreas_strict_loo_dual_umap.py](plot_pancreas_strict_loo_dual_umap.py)
  - [plot_pancreas_strict_loo_six_metrics_revised.py](plot_pancreas_strict_loo_six_metrics_revised.py)
  - [plot_pancreas_strict_loo_celltype_composition.py](plot_pancreas_strict_loo_celltype_composition.py)
- **S10**: Full-data composition changes and external-method LOO transition matrices.
  - [plot_pancreas_six_method_full_e145_e155_lineage_composition.py](plot_pancreas_six_method_full_e145_e155_lineage_composition.py)
  - [plot_pancreas_four_method_loo_transition_matrices.py](plot_pancreas_four_method_loo_transition_matrices.py)
- **S11 actual panels (caption says S12)**: Relative total mass and cell-type growth.
  - [plot_pancreas_relative_total_mass.py](plot_pancreas_relative_total_mass.py)
  - [plot_pancreas_coati_cytobridge_mean_growth_by_stage.py](plot_pancreas_coati_cytobridge_mean_growth_by_stage.py)
- **S12 actual panels (caption says S11)**: ATAC module-distribution distances and top-two transition targets.
  - [plot_pancreas_atac_module_profile_weighted_summary.py](plot_pancreas_atac_module_profile_weighted_summary.py)
  - [plot_pancreas_full_top_two_targets_appendix.py](plot_pancreas_full_top_two_targets_appendix.py)
- **S19; S20**: Pancreas reference geometry and map AUROC.
  - [plot_pancreas_t_mapping_horizontal_and_umap.py](plot_pancreas_t_mapping_horizontal_and_umap.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.
