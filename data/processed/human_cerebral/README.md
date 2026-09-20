# human_cerebral

已随 TraInfBench 提供的降维模型输入及 UMAP。原始数据来源条件见 [来源说明](../../SOURCE_RIGHTS.md)。

- `rna.h5ad`：20,663 行 × 30 维，8.15 MiB；`rna_umap.png` 为对应预览。
- `atac.h5ad`：20,663 行 × 12 维，4.07 MiB；`atac_umap.png` 为对应预览。

`X` 和 `obsm["X_latent"]` 是归一化后的降维模型输入；`obsm["X_reduced_raw"]` 是未除标量的降维输入（不是原始表达/峰计数）；`obsm["X_umap"]` 为 UMAP 坐标。尺度在 `uns["benchmark"]["scale"]`，不要重复归一化。

RNA UMAP 来源：
```json
{
  "kind": "frozen_benchmark_coordinates",
  "file": "umap_coordinates.csv.gz",
  "sha256": "e68535783f10daaec9fc5753986f385f6aceb03d89afd15428d27461424de32f",
  "representation": "Model RNA",
  "row_alignment": "exact paired metacell IDs"
}
```

ATAC UMAP 来源：
```json
{
  "kind": "frozen_benchmark_coordinates",
  "file": "umap_coordinates.csv.gz",
  "sha256": "e68535783f10daaec9fc5753986f385f6aceb03d89afd15428d27461424de32f",
  "representation": "Model ATAC",
  "row_alignment": "exact paired metacell IDs"
}
```

这些是计算配对的 metacells，不是同一细胞的实验配对数据。保留七个时间点，排除 D16。

LOO 掩码对应 scMultiNODE 已定义协议；其他模型的时间编码、原始/归一化选择和采样仍按各自 README。数值和 ID 已读回核对，哈希见上级 manifest.json。来源与发布计划见 ../../SOURCE_RIGHTS.md 和 ../../RELEASE_PLAN.md。

较大的源数据：[Zenodo](https://zenodo.org/records/5242913)。Human cerebral 使用上游记录；其余为本项目按数据集整理的记录。
