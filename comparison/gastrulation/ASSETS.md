# Literal artifact paths and external references

Paths are repository-relative. This inventory covers static strings; dynamically assembled paths and CLI overrides are defined in the source. Inputs and generated outputs are both listed.

### analyze_gastrulation_full_nmp_mass_decomposition_three_unbalanced.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `data/primal_norm_params.pt`
- `external/COATI/Gastrulation`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad`
- `results/gastrulation_full_nmp_mass_decomposition_three_unbalanced_tigon_upstream_exact_v2_iter20000`
- `results/tigon_gastrulation_full_upstream_public_exact_v2_rngexact_ae10_dopri5pack_n1024_20000_seed1/tigon_checkpoint_iter020000.pt`
- `results/tigon_official_ae_embeddings/gastrulation_ae10_upstream_public_exact_v2`

### analyze_gastrulation_loo_regulatory_edges.py

- `external/COATI`
- `results/gastrulation_loo_latest_external_predictions`
- `results/gastrulation_loo_regulatory_edges_archr_k50_weighted_latest`

### analyze_gastrulation_loo_rostral_gross_cross_lineage.py

- `data/gastrulation_rna_loo_time1_cytobridge.h5ad`
- `results/gastrulation_loo_rostral_gross_cross_lineage`
- `results/gastrulation_loo_shared_t_trajectorynet_base_rna_only_floor`
- `results/gastrulation_loo_time1_coverage_validity`
- `results/gastrulation_loo_time2_shared_t_atac`
- `results/mioflow_gastrulation_loo_time1_pca_gaga10_n1024_20000`
- `results/mioflow_gastrulation_loo_time2_pca_gaga10_n1024_20000`
- `results/trajectorynet_gastrulation_loo_time1_20000`
- `results/trajectorynet_gastrulation_loo_time2_20000`

### analyze_gastrulation_loo_source_lineage_fidelity.py

- `data/gastrulation_rna_loo_time1_cytobridge.h5ad`
- `results/gastrulation_loo_source_lineage_fidelity`
- `results/gastrulation_loo_time1_coverage_validity`
- `results/gastrulation_loo_time2_shared_t_atac`

### analyze_gastrulation_preserved_growth_genes.py

- `data/gastrulation_rna_full.h5ad`
- `results/gastrulation_preserved_growth_genes`

### analyze_gastrulation_strict_loo1_all_source_offlineage.py

- `results/gastrulation_strict_loo1_conservative_cross_germ_leakage`

### analyze_gastrulation_strict_loo1_rostral_endpoint_offlineage.py

- `data/gastrulation_rna_loo_time1_cytobridge.h5ad`
- `results/gastrulation_strict_loo1_benchmark`
- `results/gastrulation_strict_loo1_rostral_endpoint_offlineage`
- `results/trajectorynet_gastrulation_loo_time1_20000`

### analyze_gastrulation_strict_loo2_all_source_offlineage.py

- `data/gastrulation_rna_loo_time2_cytobridge.h5ad`
- `external/COATI/Gastrulation/balancedLOO/trajectory_time2/primary_trajectory_balanced_loo_time2_s0_iter20000.pt`
- `results/gastrulation_strict_loo2_conservative_cross_germ_leakage`
- `results/gastrulation_strict_loo_six_metrics_20k_cy0p3`

### compare_gastrulation_somitic_core_rna_atac_cosine.py

- `external/COATI/Gastrulation/USOT_strict_LOO1/trajectory_iter20000`
- `results/gastrulation_loo_latest_external_predictions_with_trajectorynet_base`
- `results/gastrulation_somitic_gene_module_selection`

### compare_gastrulation_unbalanced_nmp_decomposition.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `data/primal_norm_params.pt`
- `external/COATI/Gastrulation`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad`
- `results/gastrulation_unbalanced_nmp_decomposition`
- `results/tigon_gastrulation_rna_20000/config.json`
- `results/tigon_gastrulation_rna_20000/tigon.pt`

### decompose_gastrulation_nmp_mass_change.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `data/primal_norm_params.pt`
- `external/COATI/Gastrulation`
- `results/gastrulation_nmp_mass_decomposition`

### evaluate_gastrulation_flow_methods_normalized.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `data/gastrulation_rna_primal_norm_params.pt`
- `results/cytobridge_gastrulation_rna_20000/adata.h5ad`
- `results/gastrulation_flow_methods_normalized_interval_eval.csv`
- `results/gastrulation_flow_methods_normalized_rollout_eval.csv`
- `results/gastrulation_normalized_real_data_sanity.csv`
- `results/mioflow_gastrulation_rna_20000/model.pt`
- `wrote: results/gastrulation_flow_methods_normalized_interval_eval.csv`
- `wrote: results/gastrulation_flow_methods_normalized_rollout_eval.csv`

### evaluate_gastrulation_full_cmcc.py

- `FilmSync/Args.py resolves ../data/rna_pca_by_time.npz to the same raw 50D RNA PCA fixture.`
- `data/gastrulation_rna_cytobridge.h5ad`
- `data/gastrulation_rna_trajectorynet.npz`
- `external/COATI/Gastrulation`
- `results/cytobridge_gastrulation_rna_20000/Train/last_model.pth`
- `results/cytobridge_gastrulation_rna_20000/adata.h5ad`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/Train/last_model.pth`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad`
- `results/gastrulation_full_cmcc`
- `results/mioflow_gastrulation_rna_20000/model.pt`
- `results/tigon_gastrulation_rna_20000/config.json`
- `results/tigon_gastrulation_rna_20000/tigon.pt`
- `results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth`
- `results/trajectorynet_gastrulation_rna_20000/checkpt.pth`

### evaluate_gastrulation_full_train_fit_six_metrics.py

- `external/COATI`
- `results/gastrulation_full_train_fit_six_metrics`
- `results/gastrulation_loo_shared_t_trajectorynet_terminal_floor`
- `results/gastrulation_nonloo_all_rna_only_sinkhorn_pareto`
- `results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth`

### evaluate_gastrulation_loo_time1_paired_1nn_atac.py

- `external/COATI`
- `results/cytobridge_gastrulation_loo_time1_20000/Train/last_model.pth`
- `results/cytobridge_gastrulation_loo_time1_20000_unbalanced/Train/last_model.pth`
- `results/gastrulation_loo_time1_paired_1nn_atac`
- `results/mioflow_gastrulation_loo_time1_20000/model.pt`
- `results/tigon_gastrulation_loo_time1_20000/tigon.pt`

### evaluate_gastrulation_loo_time1_shared_t_atac.py

- `Gastrulation/data/TrainT/T_FiLM.pt`
- `Gastrulation/data/TrainT/train_FiLM_MLP.py`
- `external/COATI`
- `results/gastrulation_loo_time1_shared_t_atac`

### evaluate_gastrulation_paired_knn_atac.py

- `Gastrulation/data/atac_lsi_by_time_14D.npz`
- `Gastrulation/data/rna_pca_by_time.npz`
- `data/gastrulation_rna_cytobridge.h5ad`
- `data/gastrulation_rna_trajectorynet.npz`
- `external/COATI`
- `results/cytobridge_gastrulation_rna_20000/Train/last_model.pth`
- `results/cytobridge_gastrulation_rna_20000/adata.h5ad`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/Train/last_model.pth`
- `results/cytobridge_gastrulation_rna_20000_unbalanced/adata.h5ad`
- `results/filmsync_balanced_seed0_20000/checkpoints`
- `results/gastrulation_paired_knn_atac`
- `results/mioflow_gastrulation_rna_20000/model.pt`
- `results/tigon_gastrulation_rna_20000/config.json`
- `results/tigon_gastrulation_rna_20000/tigon.pt`
- `results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth`

### plot_gastrulation_coati_temporal_pair_shuffle_validation.py

- `results/gastrulation_coati_b_objective_lineage_temporal`
- `results/gastrulation_coati_b_temporal_pair_shuffle_validation`

### plot_gastrulation_crispr_t_cross_modal_response_scatter.py

- `results/gastrulation_crispr_t_e85_to_e875`

### plot_gastrulation_crispr_t_rna_terminal_composition.py

- `results/gastrulation_crispr_t_e85_to_e875`

### plot_gastrulation_e85_lineage_growth_programs_no_genes.py

- `results/gastrulation_all_source_stage_growth_gene_programs/figures`
- `results/gastrulation_all_source_stage_growth_gene_programs/module_associations_by_setting.csv`

### plot_gastrulation_full_train_coati_cy_tables.py

- `results/gastrulation_full_train_coati_cy_sweep_revised`

### plot_gastrulation_full_train_six_metrics_revised.py

- `results/gastrulation_full_train_fit_six_metrics`
- `results/gastrulation_full_train_fit_six_metrics_revised`
- `results/gastrulation_nonloo_all_rna_only_sinkhorn_pareto`

### plot_gastrulation_growth_program_auroc_by_stage.py

- `results/gastrulation_genomewide_growth_programs_by_setting/figures`
- `results/gastrulation_genomewide_growth_programs_by_setting/rank_enrichment_by_setting_stage.csv`

### plot_gastrulation_lineage_transition_cosine_compact.py

- `results/gastrulation_lineage_gene_peak_module_comparison`

### plot_gastrulation_loo_atac_umap_selected_methods.py

- `external/COATI/Gastrulation/data`
- `results/gastrulation_loo_time`
- `results/gastrulation_loo_time1_coverage_validity/loo_time1_shared_t_atac_predictions.npz`
- `results/gastrulation_loo_time2_shared_t_atac/loo_time2_shared_t_predictions.npz`

### plot_gastrulation_loo_composition_selected_base.py

- `results/gastrulation_loo_composition_selected_trajectorynet_base`
- `results/gastrulation_loo_shared_t_trajectorynet_base`
- `results/gastrulation_loo_time`

### plot_gastrulation_loo_rna_umap_selected_base.py

- `external/COATI/Gastrulation/data`

### plot_gastrulation_nmp_mass_decomposition_confirmed.py

- `results/gastrulation_nmp_mass_decomposition_interval_restart_pilot`

### plot_gastrulation_observed_nmp_lineage_e85_e875.py

- `external/COATI/Gastrulation/data/gastrulation_rna_processed.h5ad`
- `results/gastrulation_crispr_t_e85_to_e875_gaga10_tigon_ae10`

### plot_gastrulation_rostral_tss_ablation.py

- `external/COATI/Gastrulation`

### plot_gastrulation_rostral_tss_heldout_e85.py

- `results/gastrulation_rostral_tss_heldout_e85`
- `results/mioflow_gastrulation_loo_time2_pca_gaga10_n1024_20000/model.pt`

### plot_gastrulation_strict_loo1_rostral_endpoint_offlineage_table.py

- `results/gastrulation_strict_loo1_rostral_endpoint_offlineage`

### plot_gastrulation_strict_loo_all_source_method_offlineage.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `results/gastrulation_loo_rostral_gross_cross_lineage/trajectorynet_gaussian_rostral_e80.npz`
- `results/gastrulation_loo_rostral_gross_cross_lineage/trajectorynet_gaussian_rostral_e85.npz`
- `results/gastrulation_strict_loo1_benchmark/strict_loo1_e80_benchmark_predictions.npz`
- `results/gastrulation_strict_loo_all_source_method_offlineage`
- `results/tigon_gastrulation_loo_time1_upstream_public_exact_v2_rngexact_ae10_dopri5pack_n1024_20000_seed1`
- `results/tigon_gastrulation_loo_time2_upstream_public_exact_v2_rngexact_ae10_dopri5pack_n1024_20000_seed1`
- `results/tigon_official_ae_embeddings/gastrulation_ae10_upstream_public_exact_v2`

### plot_gastrulation_strict_loo_six_metrics_revised.py

- `results/gastrulation_loo_shared_t_trajectorynet_terminal_floor`
- `results/gastrulation_strict_loo_six_metrics_20k_cy0p3`
- `results/gastrulation_strict_loo_six_metrics_revised_cy0p3`

### plot_gastrulation_strict_loo_sync_ablation_matrix.py

- `results/gastrulation_strict_loo_six_metrics_20k_cy0p3`
- `results/gastrulation_strict_loo_sync_ablation_cy0p3`

### plot_gastrulation_t_celltype_auroc_dotplot.py

- `results/gastrulation_t_celltype_auroc`
- `results/gastrulation_t_celltype_auroc/t_celltype_auroc.csv`

### plot_gastrulation_t_mapping_validation_table.py

- `results/gastrulation_crispr_t_e85_to_e875`

### plot_jsm2026_gastrulation_followup_landscape.py

- `external/COATI/Gastrulation/ResultCompare/ThreeCompare/trajectorynet_native_base_conditioned_composition/caudal_epiblast_seven_methods_trajectorynet_native_base.csv`
- `results/gastrulation_full_nmp_mass_decomposition_three_unbalanced_k1`
- `results/jsm2026_gastrulation_followups_landscape`

### plot_unbalanced_rnaonly_growth_umap_by_time.py

- `data/gastrulation_rna_cytobridge.h5ad`
- `data/gastrulation_rna_primal_norm_params.pt`
- `external/COATI`
- `external/COATI/Gastrulation/UnbalancedRNAOnly/checkpoint/ckpt_s0_e0.1_m100.0_d0.1_iter20000.pth`
- `results/unbalanced_rnaonly_gastrulation_iter20000`

### recompute_gastrulation_trajectorynet_native_base_cohorts.py

- `external/COATI/Gastrulation/ResultCompare/ThreeCompare`
- `results/trajectorynet_gastrulation_rna_20000/checkpt-20000.pth`
