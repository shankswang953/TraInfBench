# TrajectoryNet

Install from [KrishnaswamyLab/TrajectoryNet](https://github.com/KrishnaswamyLab/TrajectoryNet), using the pinned revision and local patch described in [external setup](../../external/README.md). Source belongs in `external/TrajectoryNet`; it is not part of this repository.

We call `TrajectoryNet.main.main` with the upstream parser. Inputs are `.npz` arrays `pca` (cells x dimensions) and `sample_labels` (ordered snapshots). The seed wrapper does not implement a new model. `benchmark.patch` records variable-batch divergence-noise handling, interpolation/regularization handling and loss logging; use the recorded revision rather than an arbitrary newer package.

```bash
python model/TrajectoryNet/run_trajectorynet_seeded.py --help
bash model/TrajectoryNet/run_trajectorynet_synthetic_reversed_3000.sh
bash model/TrajectoryNet/run_trajectorynet_gastrulation_20000.sh
bash model/TrajectoryNet/run_trajectorynet_palate_reversed_20000.sh
bash model/TrajectoryNet/run_trajectorynet_moscot_loo_time1_reversed_20000.sh
bash model/TrajectoryNet/run_trajectorynet_human_cerebral_7time_full_forward_30000.sh
```

LOO launchers live alongside these files. Human organoid full training uses the seven stages D4,D7,D9,D11,D12,D18,D21. Preserve the input preparation file paired with each launcher: forward and reversed rank archives are different protocols.

Trajectory generation and scoring: `analyze_trajectorynet_synthetic_reversed.py`, `../../common/evaluate_terminal_push.py`, and dataset-specific comparison readers restore the checkpoint and apply the recorded clock. Native generation starts from the learned Gaussian base and integrates in the generative direction. Empirical-cell conditional pushes and the cell-type-conditioned native-base composition plot are separate analyses. In LOO, interpolate the held-out biological time between the retained consecutive ranks; never insert the missing cells into training.

Outputs are `results/<run>/checkpt-*.pth`, logs and analysis trajectory caches. Pass `--help` to Python entrypoints for input/output arguments. All displayed UMAPs are projections; distributional scores use the common PCA/ATAC coordinates, not UMAP.
