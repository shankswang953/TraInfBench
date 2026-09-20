# Five-dataset release preparation

The published `processed/` directory contains one `rna.h5ad` and one `atac.h5ad`
per dataset, plus UMAP previews, per-dataset instructions and a SHA-256 manifest.
The small files are included in Git. The original local draft and source-path inventories remain excluded. Source attribution and unresolved source-specific conditions are recorded separately.

Rebuild into a new, empty directory:

```bash
python common/prepare_release_h5ad.py --source-root /path/to/TraInf --output /path/to/new_draft
```

Use the existing scientific environment documented in the repository. The builder
also reads frozen metadata/UMAP assets under the local `data/` and `results/`
directories; it does not download them. Existing output directories are protected.
Source files and results are never modified. Large expression matrices are not
loaded or duplicated merely to extract metadata.

## Format

- `X`, `obsm['X_latent']`: frozen reduced coordinates divided once by the stored scalar.
- `obsm['X_reduced_raw']`: pre-scalar reduced coordinates, **not raw expression/counts**.
- `obsm['X_umap']`: matching observed-cell visualization; origin recorded in `uns`.
- `obs`: cell IDs, physical model time, time index, source time key, row within time,
  available cell-type/lineage labels, and scMultiNODE full/LOO training masks.
- `uns['benchmark']`: source accession, normalization scalar, pairing, preprocessing
  and UMAP provenance. `uns['source_files']` records original file hashes.

Each modality stays separate because its dimensions differ. Corresponding RNA/ATAC
files retain identical row IDs and times, checked after saving. Human organoid rows
are computationally paired metacells, not experimentally paired single cells.
scMultiSim IDs are generated from the simulation's paired time/row order.

The h5ad files are a canonical data exchange format, not replacements for every
historical runner's filename/schema. Some runners expect raw coordinates, others
normalized coordinates, and TrajectoryNet still expects an NPZ. Choose the right
field explicitly; do not divide `X` by the scale again. The masks record scMultiNODE
protocols; other method-specific sampling, mass replication, time reversal and
LOO definitions remain in their adapters. All-data frozen reduction is not an
end-to-end raw-data inductive LOO experiment.

```python
import anndata as ad
rna = ad.read_h5ad('data/processed/palate/rna.h5ad')
x_normalized = rna.obsm['X_latent']
x_raw_pca = rna.obsm['X_reduced_raw']
xy = rna.obsm['X_umap']
train = rna.obs['train_loo1'].to_numpy()
```

## Publication layout

Small reduced h5ad files and UMAP coordinates/previews are included in GitHub. Prefer stable releases rather than frequently replacing
binary data. GitHub warns above 50 MiB per file and blocks files above 100 MiB:
[official limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Zenodo can archive the identical reduced release as well for a DOI.

For Zenodo, distinguish (1) upstream raw sequencing/count data, (2) the author's
processed source data, and (3) our filtered/normalized benchmark derivatives.
Normally link to the existing upstream accession instead of mirroring all raw
files. Deposit our necessary processed count matrices and frozen transformations
with explicit provenance, if permitted. Do not label PCA/LSI matrices as raw data.
No model outputs, training checkpoints, or third-party source code are included.

The local `zenodo_source_inventory.csv` lists large source candidates without
copying them. It is an inventory, not an approved upload manifest. Count-level
h5ad conversion still requires verifying each file's `X`/layers semantics and
source conditions; do not rename a Seurat RDS or MuData file to `.h5ad`.

See [source rights review](SOURCE_RIGHTS.md). No blanket license has been assigned
to the dataset collection; a package's software license does not automatically license its data.
