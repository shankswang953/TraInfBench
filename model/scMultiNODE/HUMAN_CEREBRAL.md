# scMultiNODE: human cerebral, seven-time FULL

This is the **FULL-only** human-cerebral benchmark, using exactly the same
balanced metacells as the corresponding CytoBridge input. The selected source
is `../TraInf/TraInf/humanCerebral/Data/selected_4_7_9_11_12_18_21`:
20,663 metacells per modality across days **4, 7, 9, 11, 12, 18 and 21**.
Day 16 is absent; it is not a held-out target. All LOO scenarios are rejected.
Days 26, 31 and 61 are also outside this frozen seven-time benchmark.

The physical-time coordinate is `(day - 4) / 10`, giving
`[0, 0.3, 0.5, 0.7, 0.8, 1.4, 1.7]`. Times are not rank-compressed to `0..6`.
Inputs and decoded evaluation spaces are frozen normalized **RNA PCA30** and
**ATAC LSI12**, using source normalization scales `63.89521587795941` and
`34.16607592332852`, respectively. Normalization is not refitted. This setup
does not copy rows from the unbalanced CytoBridge mass-prior input, introduce
growth reweighting, or supply paired-row or cell-type-label supervision.

Before training, the adapter requires exact agreement with
`data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_balanced.h5ad`:
normalized RNA coordinates, source IDs, row order and physical times. It also
checks the actual CytoBridge result `adata.h5ad` when available, and validates
ATAC against the paired source normalized arrays. No second normalization is
applied. The reference check is recorded in each run's manifest and fails
closed on a mismatch. These are computationally paired metacells, not a claim
of experimentally measured single-cell pairing.

## Fixed training and export protocol

All numerical settings match the existing scMultiNODE benchmark configuration;
only the dataset and input-space identifiers change:

| Setting | Value |
| --- | --- |
| Shared latent dimension | 10 |
| AE pretraining | 2,000 updates per modality; batch 128 |
| Fusion | 2,000 updates; batch 128 |
| Dynamics | 20,000 updates; generated initial batch 1,024 |
| Native loss reference sampling | Fixed 200 per timepoint, unchanged |
| Learning rates | 0.001 |
| Alignment / dynamics regularization | 0.1 / 0.1 |
| QGW neighbors | 10; other native numerical defaults unchanged |
| Seed / checkpoint interval | 0 / 1,000 dynamics updates |
| Inference batch / output spacing | 512 / 0.025 |

The official-code-based embedding adaptation retains the official architecture
and Euler solver, with signed-output decoders as in the existing benchmark.
Inference advances on the exact observed time grid, then uses the native
piecewise-linear latent interpolant; output spacing is not a smaller solver
step. All **501 day-4 RNA initial metacells** are propagated once in source
order. RNA30 and ATAC12 trajectories are decoded by scMultiNODE's own learned
decoders from the same RNA-anchored latent trajectory. Initial decoded RNA is
an autoencoder reconstruction, not forced to equal the input. Trajectory
weights are uniform; there is no growth/death model or mass-prior adjustment.

There are only 501 initial metacells. For a generated training batch of 1,024,
the unchanged official `forward()` samples these with replacement. This is
not 1,024 distinct original cells; inference exports all 501 unique initials
exactly once. The dense output grid has 69 times, including every observed
physical knot exactly, without float32 near-duplicate query times.

The current CytoBridge setup uses batch **256** and **30,000**
dynamics updates, versus scMultiNODE's **1,024** and **20,000**. This is a
shared-input benchmark, **not equal compute or equal sampling budget**. The
native 200-per-timepoint reference sampling is not advertised as 1,024.

## Checks and launch commands

From the repository root, use the existing environment:

```bash
export PYTHON=/Users/jingfeng/local/Install/conda_env/CytoBridge/bin/python

# Read-only data/config/resource check; inspect memory.passes in the JSON.
QGW_STORAGE=disk bash model/scMultiNODE/run_human_cerebral_benchmark.sh full --check

# Optional bounded engineering smoke, never a formal benchmark.
QGW_STORAGE=disk bash model/scMultiNODE/run_human_cerebral_benchmark.sh full --smoke

# Formal full-data run: start explicitly only after reviewing the checks.
bash model/scMultiNODE/run_human_cerebral_memory_background.sh full
```

Check mode can report resource rejection with a zero exit code; inspect its
report. The smoke reduces training data and updates and uses a separate
`protocol_smoke` output name. It does not validate full-data convergence,
alignment quality, runtime or peak memory.

The memory wrapper selects disk-QGW without changing the numerical alignment
defaults or subsampling formal training cells. The plain
`run_human_cerebral_background.sh full` defaults to official in-memory QGW.
There is no silent storage fallback. Default formal disk output:

```text
results/scmultinode_human_cerebral_7time_d4_d21_no_d16_full_normalized10_n1024_20000_seed0_diskqgw
```

Official storage omits `_diskqgw`; smoke replaces
`normalized10_n1024_20000` with `protocol_smoke`. Launcher logs append
`_launcher.log`. Existing output directories and logs are refused: choose a
new `OUTPUT_DIR` (and optionally `LOG_FILE`) to repeat. There is no overwrite
or automatic resume. The wrappers already use `nohup`; do not append `&`.
A printed PID confirms submission, not successful training.

`HUMAN_CEREBRAL_DATA_DIR` overrides the source directory. `CONFIG` may override
the configuration but must retain the FULL scenario; altered settings need
a distinct output name. The launcher preserves existing cache/thread defaults
and forwards `MEMORY_BUDGET_GIB`, `ROW_BLOCK_SIZE`, `SCRATCH_ROOT`,
`MAX_RSS_GIB` and `MIN_AVAILABLE_GIB` to the same resource guards used by the
other datasets. Disk defaults are row blocks of 256, scratch `/private/tmp`,
maximum sampled RSS 10 GiB and minimum available RAM 6 GiB; preflight requires
16 GiB available by default. These checks do not reserve memory or impose
OS-level hard limits. Inspect resource reports, `manifest.json` and
`qgw_audit.json`; run jobs serially unless combined resources are checked.
See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for disk-QGW limitations.

## Preparation validation

The full-data check passed with maximum absolute RNA-coordinate difference
**0.0** against both the prepared CytoBridge input and its actual trained
`adata.h5ad`. Source IDs, their order, physical times and stored normalized
ATAC blocks also matched exactly. The adapter test suite passed 86 tests,
including 10 human-cerebral protocol tests and an actual official-model
forward check of native replacement sampling for 501 initials / batch 1,024.

A two-update, 64-metacells-per-stage engineering smoke completed in
`results/scmultinode_human_cerebral_7time_d4_d21_no_d16_full_protocol_smoke_seed0_diskqgw`.
It exported RNA `(69, 501, 30)`, ATAC `(69, 501, 12)` and latent `(69, 501, 10)`;
all values were finite. Native-knot agreement was within `6e-8` and checkpoint-only
regeneration reproduced every exported array exactly. Small-test QGW checks
reported no solver warnings. These tests verify plumbing, not biological
alignment quality or convergence. The formal 20,000-update job was not started.
