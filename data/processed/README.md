# TraInfBench 降维数据

已整理四个真实数据集与 scMultiSim，共 10 个 h5ad，随本仓库提供。

| 数据集 | 细胞/元细胞数 | RNA维度 | ATAC维度 | 两个h5ad合计 MiB |
| --- | ---: | ---: | ---: | ---: |
| gastrulation | 35,503 | 50 | 14 | 28.53 |
| pancreas | 22,604 | 50 | 22 | 20.22 |
| palate | 19,833 | 40 | 15 | 14.20 |
| human_cerebral | 20,663 | 30 | 12 | 12.22 |
| scmultisim | 10,000 | 10 | 8 | 2.89 |

h5ad 总计：78.06 MiB；最大单文件：21.13 MiB。

格式、重建和模型使用限制见 [整理计划](../RELEASE_PLAN.md)，发布条件见 [来源核对](../SOURCE_RIGHTS.md)。

现有 UMAP 坐标不等同于可投影新轨迹的拟合模型。需要投影新点时还需对应的冻结 UMAP 模型及兼容环境；这些 pickle/joblib 不在轻量 h5ad 内。

读取示例：

```python
import anndata as ad
a = ad.read_h5ad("data/processed/palate/rna.h5ad")
x = a.obsm["X_latent"]  # 已归一化的模型输入
xy = a.obsm["X_umap"]
```

这些文件不含原始基因表达或峰计数。不同模型仍须按其说明选择原始降维或归一化坐标；不代表全部历史脚本可直接用此路径运行。
