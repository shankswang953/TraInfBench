# CytoBridge

Install from [zhenyiizhang/CytoBridge](https://github.com/zhenyiizhang/CytoBridge), using the pinned checkout in [external setup](../../external/README.md). The package source belongs in `external/CytoBridge` and is not distributed here.

The runners use CytoBridge's training pipeline. Balanced runs learn velocity; unbalanced runs pretrain growth and then fit velocity plus growth. Inputs are `.h5ad` files with `obsm['X_latent']`, usually also `obsm['X_pca']`, and the indicated time column.

**Local implementation difference:** [benchmark.patch](benchmark.patch) fixes a syntax typo, skips memory-heavy built-in post-fit evaluation, and changes `loss_energy = e1.mean()` to `e1.sum()` because the experiment uses already particle-weighted energy. The final change alters the objective's scaling and is a declared benchmark adaptation, not an unchanged upstream default. The synthetic official-config/sum runner explicitly checks for it.

```bash
python model/CytoBridge/train_cytobridge_20000.py --help
python model/CytoBridge/train_cytobridge_unbalanced.py --help
bash model/CytoBridge/run_cytobridge_synthetic_official_config_sum_3000.sh
bash model/CytoBridge/run_cytobridge_gastrulation_20000.sh
bash model/CytoBridge/run_cytobridge_gastrulation_unbalanced_20000.sh
bash model/CytoBridge/run_cytobridge_palate_20000.sh
bash model/CytoBridge/run_cytobridge_moscot_unbalanced_20000.sh
bash model/CytoBridge/run_cytobridge_human_cerebral_7time_balanced_30000.sh
```

Matching unbalanced/LOO launchers are in this directory. The seven-time organoid suite excludes D16 and uses prepared strict-LOO inputs for its held-out tests. Do not substitute the earlier no-D61 archive.

Saved models are `Train/last_model.pth` and, for unbalanced runs, `Pretrain/last_model.pth`, with `adata.h5ad` and `config.yaml`. `../../comparison/scmultisim/cache_cytobridge_synthetic_observed_times.py` and dataset comparison helpers integrate the fitted fields. Preserve native particle masses in unbalanced evaluation and return to the shared coordinate scale before applying the frozen RNA-to-ATAC map.
