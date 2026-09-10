# TIGON

Reference: [yutongo/TIGON](https://github.com/yutongo/TIGON), commit `1ed92cfcc250415fc01b4d344a308b0680cc9635`. Put that checkout in `external/TIGON_upstream_1ed92cf` and install its solver as described in [external setup](../../external/README.md).

Unlike the other three adapters, `run_tigon_moscot_ae.py` is our **vectorized implementation based on public TIGON**. `tigon_official_ae_common.py` and the frozen-embedding runner provide preprocessing, checkpoint selection and decoding. Do not describe it as running unmodified upstream code.

```bash
python model/TIGON/run_tigon_frozen_official_ae_embedding.py --help
python model/TIGON/check_tigon_upstream_public_equivalence.py --help
bash model/TIGON/prepare_tigon_gastrulation_upstream_public_exact_v2.sh
bash model/TIGON/run_tigon_gastrulation_upstream_public_exact_v2.sh full
bash model/TIGON/run_tigon_gastrulation_upstream_public_exact_v2.sh loo_time1
bash model/TIGON/prepare_tigon_moscot_upstream_public_exact_ae5.sh
bash model/TIGON/run_tigon_moscot_upstream_public_exact_ae5_20000.sh
bash model/TIGON/run_tigon_palate_strict_official_ae10_20000.sh
bash model/TIGON/run_tigon_human_cerebral_strict_official_ae10_20000.sh
```

The **current gastrulation v2** uses a frozen full-data log-expression AE10, per-axis [-2,2] scaling, raw KDE-density MSE, exact divergence, TorchDiffEqPack Dopri5, public midpoint action semantics and the public conditional sigma-halving schedule without a floor. It uses 1,024 samples and 20,000 iterations. LOO excludes the stage from trajectory fitting but the AE is transductive. Selection uses observed training stages only, never the held-out stage.

Pancreas uses the dedicated upstream-exact **AE5** runner. Palate and human organoids use their explicitly configured AE10 runners; their representation and bandwidth settings must not be relabelled as gastrulation v2. Synthetic data use `run_tigon_synthetic_pca10_minus2_2_official_3000.sh` (raw PCA10, no AE, per-axis [-2,2]). Earlier sigma-floor/direct-normalized pilots are not interchangeable with these configurations.

Trajectory evaluation restores `tigon.pt` / a selected checkpoint, integrates velocity and log mass, then inverse-scales and decodes through the frozen bridge to the common PCA space. Outputs include configuration, training history, checkpoints, selection metadata and decoded evaluation caches. Native KDE initialization and empirical-cell conditional pushes are distinct analyses.

See [the detailed v2 protocol](PROTOCOL.md) for exact settings and equivalence criteria. The retained upstream license covers the adapted implementation.
