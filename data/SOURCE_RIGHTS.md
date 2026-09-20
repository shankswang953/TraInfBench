# Source attribution and redistribution review

Reviewed 2026-09-20. This records evidence and remaining questions, not a blanket
permission to redistribute every file. Further filtering, PCA/LSI or format
conversion does not automatically remove conditions attached to the source.

| Dataset | Evidence | Proposed treatment |
| --- | --- | --- |
| scMultiSim | Local `5scRNA/GenHistory.log` records `library(scMultiSim)`, seed 42, 10,000 cells, `GRN_params_100` and `Phyla3`. [Software DESCRIPTION](https://github.com/ZhangLabGT/scMultiSim/blob/bioconductor/DESCRIPTION) lists Artistic-2.0. | Publish our simulated output with generator citation and parameters; confirm bundled GRN/reference-input attribution. Do not infer the output-data license solely from the software license. |
| Gastrulation | [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117), extracted using [MouseGastrulationData](https://bioconductor.org/packages/MouseGastrulationData/) RAMultiomeData, then WT filtering. | Link original accession; name the data creators and describe our derivative. Explicit dataset-specific redistribution terms are not yet established in this review. |
| Pancreas | [GSE275562](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275562), processed annotated multiome source via moscot workflow. | Track both GEO and the precise processed h5mu source; confirm processed-file conditions rather than applying moscot's code license to the data. |
| Palate | [GSE218576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218576); SOFT record retrieved and confirms mouse secondary-palate multiome, four stages, publication PMID 38280850. [Author analysis](https://github.com/fangfang0906/Single_cell_multiome_palate). | Link raw files, describe CNC-derived mesenchyme selection and frozen reductions. No explicit reuse license found in the retrieved SOFT record; confirm source conditions before rehosting processed matrices. |
| Human cerebral organoids | [Zenodo 5242913](https://zenodo.org/records/5242913), API metadata returned `license.id = cc-by-4.0`, `access_right = open`. Also RNA E-MTAB-12001 and ATAC E-MTAB-11998. | Derivatives of files covered by this Zenodo license may be shared under its terms with creator/source attribution, license link and a description of changes. Match local files to the actual source version before claiming every local organoid file is covered. Original raw ArrayExpress files must be considered separately. |

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) allows sharing and adaptation
subject to its conditions, including attribution and indicating changes.
For the licensed organoid source, cite Fleck and colleagues, the source DOI
10.5281/zenodo.5242913, the license, and our time selection, metacell pairing,
correction, dimensionality reduction and normalization. Do not imply source-author
endorsement. Link the associated paper and exact input version in a final release.

[GEO's disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) says files
may generally be downloaded and reproduced unless otherwise stated, but also
explains that NCBI cannot grant unrestricted permission concerning third-party
rights. A missing explicit license is not proof of prohibition; equally, public
access is not enough to assign our own blanket CC BY license to all source files.
When an applicable dataset license already permits redistribution, additional
author permission is not automatically necessary. Otherwise inspect linked source
terms or ask the rights holder; no author has been contacted in this task.

Recommended Zenodo title: **TraInfBench processed benchmark inputs and reference
embeddings**. Credit original data creators in source citations; list our actual
preparation contributors as deposit creators. Describe the deposit as processed
benchmark derivatives, and use related identifiers to link original datasets and
GitHub. Keep differing licenses/conditions explicit per dataset; split deposits
if they cannot be represented accurately under a shared license.
