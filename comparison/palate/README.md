# palate comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 6a,b**: External H3K27ac/TF peak contrasts and ATAC substate recovery.
  - [plot_palate_external_peak_contrast_table.py](plot_palate_external_peak_contrast_table.py)
  - [evaluate_palate_loo_external_epigenetic_evidence.py](evaluate_palate_loo_external_epigenetic_evidence.py)
  - [plot_palate_atac_substate_f1_rare_no_rna_only.py](plot_palate_atac_substate_f1_rare_no_rna_only.py)
  - [evaluate_palate_loo_atac_regulatory_substates.py](evaluate_palate_loo_atac_regulatory_substates.py)
- **Fig. 6c**: Same-source H3K27ac fate AUROC and separation from cached trajectories.
  - [analyze_palate_full_atac_sync_ablation.py](analyze_palate_full_atac_sync_ablation.py)
- **Fig. 6d; S15b**: Selected and supplementary temporal peak-gene readouts.
  - [plot_palate_peak_gene_max_correlation_supported_pairs.py](plot_palate_peak_gene_max_correlation_supported_pairs.py)
  - [plot_palate_coati_b_supplementary_pairs_2x3.py](plot_palate_coati_b_supplementary_pairs_2x3.py)
- **S13; S14**: Strict held-out E13.5/E14.0 metrics, distributions and compositions.
  - [plot_palate_strict_loo_six_metrics_revised.py](plot_palate_strict_loo_six_metrics_revised.py)
  - [plot_palate_strict_loo_dual_umap.py](plot_palate_strict_loo_dual_umap.py)
  - [plot_palate_loo_celltype_composition.py](plot_palate_loo_celltype_composition.py)
- **S15a**: Full-data external regulatory peak contrast validation.
  - [evaluate_palate_full_external_epigenetic_evidence.py](evaluate_palate_full_external_epigenetic_evidence.py)
- **S19; S21**: Palate reference geometry and map AUROC.
  - [plot_palate_rna_atac_celltype_reference.py](plot_palate_rna_atac_celltype_reference.py)
  - [analyze_palate_t_celltype_auroc.py](analyze_palate_t_celltype_auroc.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.

The external validation evaluates fixed H3K27ac and TF-bound active CRE sets. Download records and peak-set paths are explicit in `evaluate_palate_loo_external_epigenetic_evidence.py`; retain the fixed genomic assembly/coordinate convention. Strict LOO excludes the held-out stage from trajectory and map fitting.
