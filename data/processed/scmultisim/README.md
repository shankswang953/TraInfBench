# scmultisim

已随 TraInfBench 提供的降维模型输入及 UMAP。原始数据来源条件见 [来源说明](../../SOURCE_RIGHTS.md)。

- `rna.h5ad`：10,000 行 × 10 维，1.56 MiB；`rna_umap.png` 为对应预览。
- `atac.h5ad`：10,000 行 × 8 维，1.33 MiB；`atac_umap.png` 为对应预览。

`X` 和 `obsm["X_latent"]` 是归一化后的降维模型输入；`obsm["X_reduced_raw"]` 是未除标量的降维输入（不是原始表达/峰计数）；`obsm["X_umap"]` 为 UMAP 坐标。尺度在 `uns["benchmark"]["scale"]`，不要重复归一化。

RNA UMAP 来源：
```json
{
  "kind": "frozen_benchmark_coordinate_cache",
  "file": "observed_rna10_umap.npz",
  "sha256": "d5ef08406f95b6405319b7469b8130d464383aa144a506a772b06f38ba99a779",
  "row_alignment": "Original producer concatenates time1..time4; population sequence and dimension checked. Cache has no cell IDs or input matrix; independent per-row identity cannot be verified.",
  "producer": "scripts/fit_project_synthetic_rna10_umap.py",
  "input_space": "raw RNA PCA10"
}
```

ATAC UMAP 来源：
```json
{
  "kind": "new_release_visualization",
  "note": "New visualization of the exact normalized ATAC8 inputs; NOT a manuscript frozen UMAP.",
  "n_neighbors": 30,
  "min_dist": 0.25,
  "random_state": 42,
  "metric": "euclidean",
  "umap_version": "0.5.11"
}
```

ATAC UMAP 是这次新生成的展示坐标，不代表论文冻结 UMAP。RNA 坐标来自现有缓存，验证了类别序列；原缓存没有细胞 ID，不能独立证明每一行身份。模拟细胞 ID 使用时间点与行号生成。

LOO 掩码对应 scMultiNODE 已定义协议；其他模型的时间编码、原始/归一化选择和采样仍按各自 README。数值和 ID 已读回核对，哈希见上级 manifest.json。来源与发布计划见 ../../SOURCE_RIGHTS.md 和 ../../RELEASE_PLAN.md。

scMultiSim 暂不收录在当前 COATI datasets Zenodo 记录中；这里保留小型降维输入及 UMAP。
