# Data and frozen inputs (not distributed)

Place prepared benchmark fixtures in this directory, retaining the names expected by the chosen launcher. Put manuscript-side assets in `external/COATI` and generated model outputs in `results`. Do not commit any of them.

| Dataset | Public source | Benchmark representation |
| --- | --- | --- |
| scMultiSim | [scMultiSim](https://github.com/ZhangLabGT/scMultiSim) | RNA PCA10, selected ATAC8 (PC1 and PC3-9), times 0,1,2,3 |
| Mouse gastrulation | [MouseGastrulationData](https://bioconductor.org/packages/MouseGastrulationData/), [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117) | WT subset, RNA PCA50; E7.5,E8.0,E8.5,E8.75 |
| Mouse pancreatic endocrinogenesis | [GSE275562](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275562), [moscot loader](https://moscot.readthedocs.io/en/stable/user/genapi/moscot.datasets.pancreas_multiome.html) | RNA PCA50; E14.5,E15.5,E16.5; legacy directory `moscot` |
| Mouse secondary palate | [GSE218576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218576), [author analysis](https://github.com/fangfang0906/Single_cell_multiome_palate) | CNC-derived mesenchyme, RNA PCA40 / ATAC LSI40; E12.5,E13.5,E14.0,E14.5; legacy directory `MouseBrain` |
| Human cerebral organoids | [RNA E-MTAB-12001](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12001), [ATAC E-MTAB-11998](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11998), [processed data](https://zenodo.org/records/5242913), [author code](https://github.com/quadbiolab/organoid_regulomes) | Computationally paired metacells, corrected RNA PCA30; D4,D7,D9,D11,D12,D18,D21 |

Gastrulation data were originally extracted with `MouseGastrulationData::RAMultiomeData`, followed by the manuscript's filtering and frozen preprocessing. Human RNA and ATAC were measured in separate cells: the pairing is computational, not an experimental paired-cell assay. Dataset-specific external ChIP/CRISPR/peak-set requirements are listed in each comparison folder's asset inventory and analysis script.

TrajectoryNet expects `pca` plus `sample_labels` in `.npz`. CytoBridge/MIOFlow/TIGON adapters expect `.h5ad` with `obsm['X_latent']`, often `obsm['X_pca']`, and an explicit time column (`time_point_processed`, `stage_num`, etc.). Keep physical times and internal ranks separate. Frozen normalization parameters, map checkpoints, reference metadata, cell indices and learned preprocessing are required for exact paper reproduction; re-downloading raw data does not reconstruct these automatically.

No data, including small example cells, are uploaded in this release. Per-dataset [comparison instructions](../comparison/README.md) describe which cached assets a plot needs.
