# Literal artifact paths and external references

Paths are repository-relative. This inventory covers static strings; dynamically assembled paths and CLI overrides are defined in the source. Inputs and generated outputs are both listed.

### analyze_cytobridge_synthetic.py

- `external/COATI/Synthetic/5scRNA`
- `external/COATI/Synthetic/BalancedSyncSweep`
- `external/COATI/Synthetic/TrainT`
- `results/cytobridge_synthetic_rna10_balanced_n128_i3000/adata.h5ad`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison`
- `results/cytobridge_synthetic_rna10_unbalanced_n128_i3000/adata.h5ad`

### analyze_tigon_synthetic.py

- `data/synthetic_rna10_cytobridge.h5ad`
- `external/COATI/Synthetic/5scRNA`
- `external/COATI/Synthetic/TrainT`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison`
- `results/mioflow_synthetic_rna10_gaga10_n128_i3000_analysis`
- `results/tigon_synthetic_rna10_direct_n128_i3000`
- `results/tigon_synthetic_rna10_direct_n128_i3000_analysis`
- `results/trajectorynet_synthetic_rna10_n128_i3000_analysis`

### analyze_trajectorynet_synthetic.py

- `data/synthetic_rna10_trajectorynet.npz`
- `external/COATI/Synthetic/5scRNA`
- `external/COATI/Synthetic/TrainT`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison`
- `results/trajectorynet_synthetic_rna10_n128_i3000`
- `results/trajectorynet_synthetic_rna10_n128_i3000/checkpt-3000.pth`
- `results/trajectorynet_synthetic_rna10_n128_i3000_analysis`

### cache_cytobridge_synthetic_observed_times.py

- `external/COATI/Synthetic/5scRNA`

### compare_synthetic_methods_coati_protocol.py

- `data/synthetic_rna10_cytobridge.h5ad`
- `external/COATI/Synthetic`
- `results/cytobridge_synthetic_rna10_balanced_d10_e1_n128_i3000/adata.h5ad`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/observed_rna10_umap.npz`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/rna10_umap_model.joblib`
- `results/cytobridge_synthetic_rna10_unbalanced_d10_e1_n128_i3000/adata.h5ad`
- `results/mioflow_synthetic_rna10_gaga10_n128_i3000/model.pt`
- `results/synthetic_rna10_all_method_coati_protocol_comparison`
- `results/tigon_synthetic_rna10_pca_minus2_2_official_n128_i3000`
- `results/trajectorynet_synthetic_rna10_n128_i3000/checkpt-3000.pth`

### evaluate_synthetic_predicted_time_sinkhorn.py

- `external/COATI/Synthetic`
- `results/synthetic_rna10_all_method_coati_protocol_comparison`
- `results/synthetic_rna10_same_space_predicted_time_sinkhorn`

### make_combined_distribution_grid.py

- `external/COATI/Synthetic/UnbalancedSyncSweep`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/observed_rna10_umap.npz`

### make_ever_branch_jump_table.py

- `external/COATI/Synthetic/UnbalancedSyncSweep`

### make_observed_time_correctness_table.py

- `external/COATI/Synthetic/UnbalancedSyncSweep`

### make_t_architecture_distribution_grid.py

- `external/COATI/Synthetic/UnbalancedSyncSweep`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/rna10_umap_model.joblib`

### plot_cytobridge_synthetic_convergence.py

- `results/cytobridge_synthetic_rna10_balanced_d10_e1_n128_i3000_stdout.log`
- `results/cytobridge_synthetic_rna10_d10_e1_convergence`
- `results/cytobridge_synthetic_rna10_unbalanced_d10_e1_n128_i3000_stdout.log`

### plot_synthetic_coati_cytobridge_reversed_tn_comparison.py

- `data/synthetic_rna10_trajectorynet_reversed.npz`
- `external/COATI/Synthetic`
- `results/coati_cytobridge_observed_time_umap_comparison_officialcfg_sum_i3000/comparison_coordinates.npz`
- `results/cytobridge_synthetic_rna10_officialcfg_sum_n128_i3000_observed_times/cytobridge_observed_time_cache.npz`
- `results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns`
- `results/synthetic_rna10_same_space_predicted_time_sinkhorn_cytobridge_officialcfg_sum_i3000/predicted_time_sinkhorn.csv`
- `results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis/metrics.json`
- `results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis/reversed_time_all_initial_trajectories.npz`

### plot_synthetic_overview.py

- `external/COATI/Synthetic`
- `results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/observed_rna10_umap.npz`

### plot_synthetic_truth_method_columns_a4.py

- `data/synthetic_rna10_trajectorynet_reversed.npz`
- `results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/comparison_coordinates.npz`
- `results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/metrics.csv`
- `results/synthetic_rna10_coati_cytobridge_trajectorynet_reversed_method_columns/truth_and_method_columns_a4.png`
