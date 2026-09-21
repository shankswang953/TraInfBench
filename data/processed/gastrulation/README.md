# gastrulation

已随 TraInfBench 提供的降维模型输入及 UMAP。原始数据来源条件见 [来源说明](../../SOURCE_RIGHTS.md)。

- `rna.h5ad`：35,503 行 × 50 维，21.13 MiB；`rna_umap.png` 为对应预览。
- `atac.h5ad`：35,503 行 × 14 维，7.40 MiB；`atac_umap.png` 为对应预览。

`X` 和 `obsm["X_latent"]` 是归一化后的降维模型输入；`obsm["X_reduced_raw"]` 是未除标量的降维输入（不是原始表达/峰计数）；`obsm["X_umap"]` 为 UMAP 坐标。尺度在 `uns["benchmark"]["scale"]`，不要重复归一化。

RNA UMAP 来源：
```json
{
  "kind": "source_h5ad_embedding",
  "file": "gastrulation_rna_processed.h5ad",
  "key": "X_umap",
  "row_alignment": "stage order + numerical representation equality",
  "note": "Source visualization; may have been fit on a different dimensional representation."
}
```

ATAC UMAP 来源：
```json
{
  "kind": "frozen_benchmark_reducer",
  "file": "atac_umap_model_14D.joblib",
  "sha256": "4f36151caba0cfb938a3639ceb5f15d8fc8c53323dea5705768e1e6a9cf2a6a5",
  "input_space": "raw",
  "max_abs_input_error": 0.0
}
```


LOO 掩码对应 scMultiNODE 已定义协议；其他模型的时间编码、原始/归一化选择和采样仍按各自 README。数值和 ID 已读回核对，哈希见上级 manifest.json。来源与发布计划见 ../../SOURCE_RIGHTS.md 和 ../../RELEASE_PLAN.md。

较大的源 RNA 和 ATAC peak 矩阵：[Zenodo](https://zenodo.org/records/22865714)。
