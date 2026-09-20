# scMultiNODE: palate FULL and LOO

The palate benchmark uses the existing frozen RNA PCA40 and ATAC LSI15
representations under `../TraInf/TraInf/MouseBrain/data`. The directory's name
does not change this dataset's palate identity. Model inputs and both decoded
evaluation spaces are normalized **RNA40 / ATAC15**.

The frozen normalization scalars are RNA `22.22992031699182` and ATAC LSI15
`0.10373610732172349`. CytoBridge trains in **raw RNA PCA40** coordinates;
its shared comparison space divides these by the RNA scalar. scMultiNODE's
normalized RNA inputs therefore match the **CytoBridge evaluation space**,
not its literal raw training coordinates. The benchmark shares source cells
and normalized evaluation spaces while disclosing this numerical difference.

| Scenario | Observed stages | Physical training times | Held out | Training cells per modality |
| --- | --- | --- | --- | ---: |
| `full` | E12.5, E13.5, E14.0, E14.5 | 0, 1, 1.5, 2 | None | 19,833 |
| `loo1` | E12.5, E14.0, E14.5 | 0, 1.5, 2 | E13.5 at time 1 | 14,141 |
| `loo2` | E12.5, E13.5, E14.5 | 0, 1, 2 | E14.0 at time 1.5 | 13,195 |

Original per-stage counts are 2,570, 5,692, 6,638 and 4,933. The runtime input
audit compares observed source cells, row order and normalization scalars
against the CytoBridge source and the actual scenario's `adata.h5ad`, and
verifies ATAC against `X_lsi15`. This is an input identity check, not use of
a separately trained cross-modality map or paired supervision.

The physical coordinate is embryonic day minus 12.5; intervals are not
rank-compressed. Each scenario trains a fresh seed-0 model. In LOO, the
held-out stage is excluded from **both modalities before AE, QGW, fusion and
dynamics training**. All scenarios retain common frozen representations and
normalization scalars. This is training-stage LOO conditional on fixed
representations, not an end-to-end inductive raw-data LOO experiment. No
paired-row or cell-type-label supervision is supplied.

## Fixed settings and inference

Training settings match the existing scMultiNODE pancreas configuration;
dataset, input-space and scenario identifiers select this protocol:

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

The adapter retains the official architecture and Euler solver, with the
same signed-output decoder adaptation as the existing datasets. Formal
inference integrates on each scenario's observed time grid and uses the native
piecewise-linear latent interpolant for queries, including held-out times.
The 0.025 spacing controls saved output density, not additional solver steps.

All **2,570 original E12.5 RNA initial cells** are propagated exactly once in
source order, including in LOO. RNA40 and ATAC15 are produced by scMultiNODE's own
learned decoders from the same RNA-anchored latent trajectory, without a
separately trained Tmap. Initial decoded RNA is an autoencoder reconstruction, not
forced equal to its input. Weights are uniform; no growth/death model or
mass-prior reweighting is added. Matching update or generated-batch counts
does not establish equal compute or equal reference sampling across methods.
The 81 query times produce RNA `(81, 2570, 40)`, ATAC `(81, 2570, 15)` and
shared-latent `(81, 2570, 10)` trajectory arrays in every scenario.

## Check and launch

From the repository root, select one scenario explicitly:

```bash
export PYTHON=/Users/jingfeng/local/Install/conda_env/CytoBridge/bin/python

# Read-only configuration, input and resource checks; inspect memory.passes.
QGW_STORAGE=disk bash model/scMultiNODE/run_palate_benchmark.sh full --check
QGW_STORAGE=disk bash model/scMultiNODE/run_palate_benchmark.sh loo1 --check
QGW_STORAGE=disk bash model/scMultiNODE/run_palate_benchmark.sh loo2 --check

# Optional bounded engineering smoke, with a separate output name.
QGW_STORAGE=disk bash model/scMultiNODE/run_palate_benchmark.sh full --smoke

# Formal run: start only the selected scenario after reviewing its checks.
bash model/scMultiNODE/run_palate_memory_background.sh full
```

Substitute `loo1` or `loo2` in the final command for that scenario. These are
separate commands; start serially unless combined resources have been checked.
The wrappers already use `nohup`; do not append `&`. A printed PID means
submission, not successful training. Check mode may return zero while
reporting resource rejection in JSON. Smokes reduce training data and updates;
they do not validate formal-run convergence, alignment quality or peak memory.

The memory wrapper explicitly selects disk-QGW storage with unchanged
numerical defaults and all observed formal training cells. The plain
`run_palate_background.sh` defaults to official in-memory QGW; there is no
silent fallback. Default disk outputs are:

```text
results/scmultinode_palate_full_normalized10_n1024_20000_seed0_diskqgw
results/scmultinode_palate_loo1_normalized10_n1024_20000_seed0_diskqgw
results/scmultinode_palate_loo2_normalized10_n1024_20000_seed0_diskqgw
```

Official storage omits `_diskqgw`; smoke replaces `normalized10_n1024_20000`
with `protocol_smoke`. Launcher logs append `_launcher.log`. Existing output
directories and logs are refused; there is no overwrite or automatic resume.
Use a new `OUTPUT_DIR` (and optionally `LOG_FILE`) for a deliberate repeat.
`PALATE_DATA_DIR` overrides the source directory. `CONFIG`
can override the configuration but must match the selected scenario; altered
settings need a distinct output name.

The launcher preserves the shared cache/thread defaults and forwards
`MEMORY_BUDGET_GIB`, `ROW_BLOCK_SIZE`, `SCRATCH_ROOT`, `MAX_RSS_GIB` and
`MIN_AVAILABLE_GIB`. Disk defaults are row blocks of 256, scratch `/private/tmp`,
maximum sampled RSS 10 GiB and minimum available RAM 6 GiB; preflight requires
16 GiB available by default. Scratch must have room for both full float64
distance matrices plus headroom. These checks neither reserve resources nor
provide OS-level memory limits. Inspect resource reports, `manifest.json` and
`qgw_audit.json`. Run serially unless combined resources are independently
checked. See
[MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for disk-QGW caveats.

## Preparation validation

All three full-data resource/input preflights passed at preparation time. The
observed RNA coordinates and their normalized counterparts, row IDs and
physical times exactly matched the existing prepared and trained CytoBridge
inputs for each scenario. ATAC LSI15 matched the common reference exactly.
The 96 adapter tests passed, including 10 palate-specific tests covering both
LOO exclusions, poisoned held-out inputs, physical-clock preservation and
native latent interpolation rather than adding held-out integration knots.

Separate two-update, 64-observed-cells-per-stage engineering smokes completed
for full, loo1 and loo2 in
`results/scmultinode_palate_{scenario}_protocol_smoke_seed0_diskqgw`.
All three exported all 2,570 initial cells exactly once, with finite
`(81, 2570, 40)` RNA, `(81, 2570, 15)` ATAC and `(81, 2570, 10)` latent arrays.
Native-knot agreement was within `9e-8`; small-test QGW recorded no warnings.
LOO1 checkpoint-only regeneration reproduced every exported array exactly.
These checks establish pipeline consistency, not biological accuracy,
convergence, full-data runtime or peak memory. Formal 20,000-update runs were
not started during preparation; launch-time resource checks remain active.
