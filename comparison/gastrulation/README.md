# gastrulation comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 3; S3a**: Held-out E8.0/E8.5 RNA and mapped-ATAC distributions and six metrics.
  - [plot_gastrulation_loo_rna_umap_selected_base.py](plot_gastrulation_loo_rna_umap_selected_base.py)
  - [plot_gastrulation_loo_atac_umap_selected_methods.py](plot_gastrulation_loo_atac_umap_selected_methods.py)
  - [plot_gastrulation_strict_loo_six_metrics_revised.py](plot_gastrulation_strict_loo_six_metrics_revised.py)
- **Fig. 4a**: Seven-method caudal-epiblast composition; native-base-conditioned TrajectoryNet, 4x2 layout.
  - [recompute_gastrulation_trajectorynet_native_base_cohorts.py](recompute_gastrulation_trajectorynet_native_base_cohorts.py)
- **Fig. 4b**: NMP retained/incoming/outgoing mass decomposition.
  - [plot_gastrulation_nmp_mass_decomposition_confirmed.py](plot_gastrulation_nmp_mass_decomposition_confirmed.py)
  - [analyze_gastrulation_full_nmp_mass_decomposition_three_unbalanced.py](analyze_gastrulation_full_nmp_mass_decomposition_three_unbalanced.py)
- **Fig. 4c**: WT/T-KO terminal NMP descendants.
  - [plot_gastrulation_crispr_t_rna_terminal_composition.py](plot_gastrulation_crispr_t_rna_terminal_composition.py)
- **Fig. 4d**: RNA/ATAC lineage transition cosine.
  - [plot_gastrulation_lineage_transition_cosine_compact.py](plot_gastrulation_lineage_transition_cosine_compact.py)
  - [compare_gastrulation_somitic_core_rna_atac_cosine.py](compare_gastrulation_somitic_core_rna_atac_cosine.py)
- **Fig. 4e**: Selected temporal RNA/ATAC pairs from existing COATI tables.
  - [plot_gastrulation_coati_objective_rank1_three_pairs.py](plot_gastrulation_coati_objective_rank1_three_pairs.py)
- **Fig. 4f**: S-phase and forward-lineage gene-rank AUROC.
  - [plot_gastrulation_growth_program_auroc_by_stage.py](plot_gastrulation_growth_program_auroc_by_stage.py)
- **S3b; S4a**: Held-out compositions and full-training metric bars (actual PDF panels).
  - [plot_gastrulation_loo_composition_selected_base.py](plot_gastrulation_loo_composition_selected_base.py)
  - [plot_gastrulation_full_train_six_metrics_revised.py](plot_gastrulation_full_train_six_metrics_revised.py)
- **S5; S6**: Synchronization ablation, source taxonomy, lineage leakage and regulatory readouts.
  - [plot_gastrulation_strict_loo_sync_ablation_matrix.py](plot_gastrulation_strict_loo_sync_ablation_matrix.py)
  - [plot_gastrulation_strict_loo_all_source_method_offlineage.py](plot_gastrulation_strict_loo_all_source_method_offlineage.py)
  - [plot_gastrulation_strict_loo1_rostral_endpoint_offlineage_table.py](plot_gastrulation_strict_loo1_rostral_endpoint_offlineage_table.py)
  - [plot_gastrulation_rostral_tss_heldout_e85.py](plot_gastrulation_rostral_tss_heldout_e85.py)
- **S7**: CRISPR reference, cross-modal response, growth programs and paired versus shuffled temporal concordance.
  - [plot_gastrulation_observed_nmp_lineage_e85_e875.py](plot_gastrulation_observed_nmp_lineage_e85_e875.py)
  - [plot_gastrulation_crispr_t_cross_modal_response_scatter.py](plot_gastrulation_crispr_t_cross_modal_response_scatter.py)
  - [plot_gastrulation_e85_lineage_growth_programs_no_genes.py](plot_gastrulation_e85_lineage_growth_programs_no_genes.py)
  - [plot_gastrulation_coati_pair_shuffle_vertical.py](plot_gastrulation_coati_pair_shuffle_vertical.py)
- **S8**: RNA/ATAC fields and stage-specific growth landscapes.
  - [plot_gastrulation_unbalanced_dual_umap_flow.py](plot_gastrulation_unbalanced_dual_umap_flow.py)
  - [plot_gastrulation_unbalanced_growth_distribution.py](plot_gastrulation_unbalanced_growth_distribution.py)
- **S19; S20**: Gastrulation reference geometry and full/LOSO map AUROC.
  - [plot_gastrulation_rna_atac_celltype_reference.py](plot_gastrulation_rna_atac_celltype_reference.py)
  - [plot_gastrulation_t_celltype_auroc_dotplot.py](plot_gastrulation_t_celltype_auroc_dotplot.py)
- **S7b**: Exact panel source identified by plotted content.
  - [plot_gastrulation_t_mapping_validation_table.py](plot_gastrulation_t_mapping_validation_table.py)
- **S4b**: Exact panel source identified by plotted content.
  - [plot_gastrulation_full_train_coati_cy_tables.py](plot_gastrulation_full_train_coati_cy_tables.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.

Full/LOO tables and CRISPR WT/T-KO readouts use different cohorts. Preserve the original cohort and kNN settings. Fig. 4a uses caudal epiblast, not rostral neurectoderm. Full-data AE preprocessing remains transductive where declared.
