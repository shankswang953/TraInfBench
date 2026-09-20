# scMultiNODE: moscot pancreas full and LOO

This dataset is the pancreas example under `TraInf/TraInf/moscot/data`, not a
second gastrulation dataset. The existing normalized RNA PCA50 and ATAC
PoissonVI22 representations are the model inputs and decoded evaluation spaces.
The ATAC representation is **22D PoissonVI, not gastrulation's 14D LSI**.

## Splits and fixed configuration

| Scenario | Observed stages | Training times | Observed cells per modality |
| --- | --- | --- | ---: |
| `full` | E14.5, E15.5, E16.5 | 0, 1, 2 | 22,604 |
| `loo1` | E14.5, E16.5 | 0, 2 | 12,271 |

There is one internal held-out stage, **E15.5 at time 1**. No `loo2` is configured:
E16.5 is terminal, so holding it out would be a different extrapolation task.
Original per-stage counts are 9,029, 10,333 and 3,242 for each modality. Full and
LOO each train a new model from seed 0. E15.5 is excluded from both modalities
before AE, QGW correspondence, fusion and dynamics training in `loo1`.

The input files are `rna_pca_by_time.npz` and `atac_poissonvi_time_data.npz`, with
`time_0`, `time_1`, `time_2` keys. The adapter uses the existing benchmark's frozen
normalization scalars rather than fitting new LOO-specific scalars. Because
the representations and scalar normalization are frozen from the full atlas,
this is training-stage LOO conditional on fixed representations, not raw-data
inductive LOO. No cell-type labels or paired-row supervision are passed to the
trainer.

| Setting | Both scenarios |
| --- | --- |
| Shared latent dimension | 10 |
| AE pretraining | 2,000 updates per modality; batch 128 |
| Fusion | 2,000 updates; batch 128 |
| Dynamics | 20,000 updates; generated initial batch 1,024 |
| Native reference sampling in losses | Upstream fixed 200 per timepoint, unchanged |
| Learning rates | 0.001 |
| Alignment / dynamics regularization coefficients | 0.1 / 0.1 |
| Architecture / solver | Official hidden widths [50], ReLU, Euler |
| QGW | Native numerical settings; ratio 0.1, neighbors 10, GW, epsilon 0.01 |
| Seed / checkpoints | 0; every 1,000 dynamics updates; final 20,000 predeclared |
| Inference batch / output spacing | 512 / 0.025 |

The official source remains unmodified and pinned to the same revision as the
gastrulation adaptation. Final decoder ReLU is replaced with Identity in the
adapter to support signed input coordinates; both decoders return the original
normalized modality spaces. This is an official-code-based embedding adaptation,
not a literal reproduction of the paper's raw-expression-input experiment.
Matching 20,000 dynamics updates and generated batch 1,024 does not equalize
every method's sample budget or total compute. The hardcoded reference sample
count remains 200 and is not advertised as 1,024.

## Native-time trajectory export

The official Euler implementation takes one step per supplied interval. Formal
inference therefore uses the exact observed training grid, followed by the
native piecewise-linear latent interpolant and both learned decoders. It does
not insert held-out observations or use dense query points as extra solver
steps. For LOO, the training grid is `[0, 2]`, not `[0, 1]`; E15.5 is queried at
the midpoint 1. `trajectory_dt=0.025` affects saved output density only.

All **9,029 original E14.5 RNA cells** are pushed exactly once in source order.
Outputs have shapes RNA `(81, 9029, 50)`, ATAC `(81, 9029, 22)` and shared latent
`(81, 9029, 10)`. Both modalities follow the same RNA-anchored latent trajectory;
the initial decoded RNA is an AE reconstruction, not forced equal to the input.
Weights are uniform, with no growth/death model. Export checks finite values
and agreement with native forward predictions at observed time knots. UMAP and
held-out metrics are not used for checkpoint or hyperparameter selection.

## Checks and background commands

Use the existing environment, from the repository root:

```bash
cd /Users/jingfeng/local/project/TraInfBench
export PYTHON=/Users/jingfeng/local/Install/conda_env/CytoBridge/bin/python

# Read-only resource/data/config checks; inspect memory.passes in the report.
QGW_STORAGE=disk bash model/scMultiNODE/run_pancreas_benchmark.sh full --check
QGW_STORAGE=disk bash model/scMultiNODE/run_pancreas_benchmark.sh loo1 --check
```

Check mode reports resource rejection in JSON; a zero exit code alone does not
mean the memory check passed. A bounded engineering smoke can be requested in
the foreground with `--smoke`; it uses explicitly reduced training data/updates,
is not a full benchmark, and writes a separate `protocol_smoke` output name.

The following commands opt into the same disk-QGW storage adapter used for the
completed gastrulation runs, without changing numerical QGW defaults or
subsampling observed training cells:

```bash
# Full: E14.5, E15.5, E16.5
bash model/scMultiNODE/run_pancreas_memory_background.sh full

# LOO: held-out E15.5. Safest to start after full has finished.
bash model/scMultiNODE/run_pancreas_memory_background.sh loo1
```

The wrapper already uses `nohup`, redirects stdin and writes a launcher log;
do not append another `&`. A printed PID means submission, not successful
training. Existing output directories and logs are refused. Default outputs:

```text
results/scmultinode_pancreas_full_normalized10_n1024_20000_seed0_diskqgw
results/scmultinode_pancreas_loo1_normalized10_n1024_20000_seed0_diskqgw
```

Logs append `_launcher.log`. These pancreas outputs are separate from all
gastrulation results. To repeat, set a new `OUTPUT_DIR`; `LOG_FILE` is optional.
`MOSCOT_DATA_DIR` overrides the default `../TraInf/TraInf/moscot/data` path.
`CONFIG` can override the scenario configuration but must match the scenario;
use a new output name for altered settings. There is no automatic resume or
overwrite. The plain `run_pancreas_background.sh` defaults to official in-memory
QGW; explicitly setting `QGW_STORAGE=disk` also adds `_diskqgw` to its default
output name. It does not silently switch storage mode if preflight fails.

Disk defaults are `ROW_BLOCK_SIZE=256`, `SCRATCH_ROOT=/private/tmp`,
`MAX_RSS_GIB=10`, `MIN_AVAILABLE_GIB=6`, and automatically detected available RAM.
`MEMORY_BUDGET_GIB` can cap that preflight allocation. The scratch parent must
already exist. A single run needs enough scratch for both full float64 distance
matrices plus headroom, and default available-RAM preflight requires 16 GiB.
These are resource policies, not measured peak guarantees. Start jobs serially
unless combined resources have been independently checked; preflight and
sampled guards do not reserve memory or enforce OS-level hard limits.

## Interpretation and audit

Disk storage is a disclosed implementation adaptation, audited against official
storage in bounded gastrulation comparisons; it is not a new alignment loss or
a claim that biological cross-modality alignment is guaranteed. Details are in
[MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md). Poor cell-type correspondence
can occur despite finite losses and visually similar global distributions.
This setup does not change QGW or provide paired supervision in response to the
gastrulation alignment diagnostics.

Inspect each run's `qgw_audit.json`, `manifest.json` and resource reports. Solver
warnings are surfaced and recorded with upstream continuation; invalid coupling
values or materially infeasible marginals stop before fusion. Finite losses,
passed resource checks and successful export do not establish convergence or
biological trajectory accuracy. These must be assessed after training, separately
from code safety checks.

## Preparation validation

The 76 adapter tests, including 10 pancreas-specific tests, passed. Poisoning
the held-out input with NaN/extreme values leaves the LOO training tensors
unchanged. Both real-data resource preflights passed at preparation time;
launch-time checks still apply.

Two-update, 64-cells-per-observed-stage engineering smokes completed for both
splits. They are retained separately in
`results/scmultinode_pancreas_full_protocol_smoke_seed0_diskqgw` and
`results/scmultinode_pancreas_loo1_protocol_smoke_seed0_diskqgw`.
Each exported all 9,029 initial cells in original order at 81 query times,
with finite RNA50, ATAC22 and latent10 values. Agreement with native forward
outputs at training knots had maximum absolute error below `9e-8`.
LOO checkpoint-only trajectory regeneration matched the original export
exactly, including held-out time 1. No QGW solver warnings were recorded in
these small tests. These checks do not establish full-data alignment quality,
convergence, runtime or peak memory. Formal 20,000-update jobs were not started
as part of preparation.
