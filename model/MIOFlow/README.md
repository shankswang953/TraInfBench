# MIOFlow

Install from [KrishnaswamyLab/MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow), pinned in [external/versions.json](../../external/versions.json). Apply [benchmark.patch](benchmark.patch) to a local checkout under `external/MIOFlow` before editable installation, as shown in [external setup](../../external/README.md).

This adapter calls upstream `Autoencoder`, `train_gaga_two_phase`, `MIOFlow` and `dataloader_from_pc`; it does not replace MIOFlow's ODE loss. The local patch exposes normalization and frozen mean/std controls plus live loss logging. It is required by the wrapper's constructor arguments.

Current real-data comparisons use **PCA -> StandardScaler -> PHATE-distance GAGA10 -> z-scored GAGA10 -> MIOFlow ODE**. Predictions are inverse-normalized and decoded to the shared PCA space. The former direct-PCA trial launchers are not current benchmark entrypoints.

```bash
python model/MIOFlow/train_mioflow_10000.py --help
bash model/MIOFlow/run_mioflow_synthetic_gaga10_3000.sh
bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh gastrulation full
bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh gastrulation loo1
bash model/MIOFlow/run_mioflow_palate_gaga10_20000.sh full
bash model/MIOFlow/run_mioflow_palate_gaga10_20000.sh loo1
bash model/MIOFlow/run_mioflow_moscot_full_shared_gaga10_n1024_20000.sh
bash model/MIOFlow/run_mioflow_human_cerebral_7time_full_official_gaga10_n256_30000.sh
```

The generic runner also accepts `loo2` where supported; see its dataset/time mapping. The human seven-time LOO runner uses a frozen full-data GAGA representation. This is a fixed-representation (transductive) LOO experiment even though held-out cells are excluded from ODE training. Retain the stored biological-time/rank mapping for irregular time intervals.

Inputs are `.h5ad` common PCA matrices in `obsm['X_latent']` / `obsm['X_pca']`, time labels and the dataset normalization metadata. Outputs include `gaga_model.pt`, `model.pt`, `trajectories.npz` (including decoded `trajectories_pca`), `losses.csv` and `metadata.json`. `analyze_mioflow_synthetic.py` illustrates the shared-space trajectory readout.

The historical filename `train_mioflow_10000.py` is a configurable trainer; the launcher's `--epochs` determines the actual budget.
