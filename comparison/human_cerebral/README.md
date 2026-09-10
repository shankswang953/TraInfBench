# human cerebral comparisons

See [public data sources](../../data/README.md) and [external asset placement](../../external/COATI/README.md). Run commands from the repository root. No datasets, model checkpoints or figure binaries are stored here.

## Paper panels

- **Fig. 7**: Source-line W2, growth programs, temporal concordance, mass associations and selected peak-gene lags.
  - [plot_human_cerebral_source_line_w2_combined_10pt.py](plot_human_cerebral_source_line_w2_combined_10pt.py)
  - [plot_human_cerebral_growth_publication_redraw_10pt.py](plot_human_cerebral_growth_publication_redraw_10pt.py)
  - [plot_human_cerebral_publication_redraw_10pt.py](plot_human_cerebral_publication_redraw_10pt.py)
  - [plot_human_cerebral_programme_mass_heatmap_cy0p5.py](plot_human_cerebral_programme_mass_heatmap_cy0p5.py)
  - [plot_human_cerebral_compact_main_lag_v3.py](plot_human_cerebral_compact_main_lag_v3.py)
- **S16**: RNA/ATAC reconstruction against original cells/metacells; reference UMAPs; programme table in growth redraw.
  - [plot_human_cerebral_three_w2_vertical_a4.py](plot_human_cerebral_three_w2_vertical_a4.py)
  - [plot_human_cerebral_original_metacell_umap_compact_10pt.py](plot_human_cerebral_original_metacell_umap_compact_10pt.py)
- **S19; S21**: Organoid reference geometry and map AUROC.
  - [plot_human_cerebral_rna_atac_t_reference.py](plot_human_cerebral_rna_atac_t_reference.py)
  - [analyze_human_cerebral_t_celltype_source_auroc.py](analyze_human_cerebral_t_celltype_source_auroc.py)
- **Fig. 7b**: Exact panel source identified by plotted content.
  - [plot_human_cerebral_three_stage_mass_story_compact.py](plot_human_cerebral_three_stage_mass_story_compact.py)
- **Fig. 7b supporting values**: Exact panel source identified by plotted content.
  - [plot_human_cerebral_three_stage_mass_story.py](plot_human_cerebral_three_stage_mass_story.py)

## Inputs and execution

The plot scripts retain the original result-directory names. Restore the matching `results/...` CSV/NPZ summaries or the `external/COATI/...` manuscript assets. [ASSETS.md](ASSETS.md) lists literal artifact paths found in these scripts; it includes intermediate outputs as well as inputs, so consult `--help` and the relevant loading code for the exact files consumed. Dataset names in historical paths are preserved to avoid confusing incompatible runs.

Use the seven-time D4-D21 archive excluding D16. Original-cell and metacell reference distributions are distinct. Some typography QA requires Arial; the font is resolved through Matplotlib on other systems.


Original ATAC extraction additionally needs R packages `Matrix` and `GenomicRanges`. PDF QA/redrawing needs Poppler (`pdftoppm` on PATH). Arial is needed for exact manuscript typography.
