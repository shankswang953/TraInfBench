# palate

已随 TraInfBench 提供的降维模型输入及 UMAP。原始数据来源条件见 [来源说明](../../SOURCE_RIGHTS.md)。

- `rna.h5ad`：19,833 行 × 40 维，9.94 MiB；`rna_umap.png` 为对应预览。
- `atac.h5ad`：19,833 行 × 15 维，4.26 MiB；`atac_umap.png` 为对应预览。

`X` 和 `obsm["X_latent"]` 是归一化后的降维模型输入；`obsm["X_reduced_raw"]` 是未除标量的降维输入（不是原始表达/峰计数）；`obsm["X_umap"]` 为 UMAP 坐标。尺度在 `uns["benchmark"]["scale"]`，不要重复归一化。

RNA UMAP 来源：
```json
{
  "kind": "frozen_benchmark_reducer",
  "file": "rna_umap_model.joblib",
  "sha256": "42bce895403bf81e851418f0285b4fd3b70a4840c0b08ffaa91745b3dbf8c4db",
  "input_space": "normalized",
  "max_abs_input_error": 0.0
}
```

ATAC UMAP 来源：
```json
{
  "kind": "frozen_benchmark_reducer",
  "file": "atac_umap_model.joblib",
  "sha256": "29ffbaadf97d0b9bf959c31c4ec7ddfe3c483c81da55efdbded8e8a92c1bb780",
  "input_space": "normalized",
  "max_abs_input_error": 0.0
}
```


LOO 掩码对应 scMultiNODE 已定义协议；其他模型的时间编码、原始/归一化选择和采样仍按各自 README。数值和 ID 已读回核对，哈希见上级 manifest.json。来源与发布计划见 ../../SOURCE_RIGHTS.md 和 ../../RELEASE_PLAN.md。

较大的源数据：[Zenodo](https://zenodo.org/records/22865714)。统一 COATI datasets 记录中的 `palate.zip`。
