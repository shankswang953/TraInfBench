# Gastrulation scMultiNODE benchmark protocol

## Fixed comparison settings

| Setting | Full, LOO time1 and LOO time2 |
| --- | --- |
| Input / decoded evaluation space | Existing normalized RNA PCA50 and ATAC LSI14 |
| Joint learned latent | 10 dimensions |
| RNA and ATAC AE pretraining | 2,000 updates each; batch 128 |
| Fusion | 2,000 updates; batch 128 |
| Dynamics | 20,000 updates; generated initial batch 1,024 |
| Native Sinkhorn target sampling | Fixed upstream 200 per timepoint per call, including latent losses |
| Learning rates | 0.001 in all three phases |
| Coefficients | `align_coeff=0.1`, `dyn_reg_coeff=0.1` |
| Architecture and solver | Official hidden widths [50], ReLU and Euler |
| QGW | Official numerical settings; ratio 0.1, neighbors 10, GW, epsilon 0.01; official storage by default, optional disk storage |
| Replicate / checkpoint policy | Seed 0; dynamics checkpoints every 1,000; final 20,000 predeclared |
| Inference | Every original E7.5 RNA cell exactly once; 9,018 paths |

Official source remains clean and pinned to
`c4e444d24849393bc923ffb5373e5cea35476013`. Only public settings are adjusted.
The necessary adapter change remains the final decoder ReLU -> Identity for
signed PCA/LSI coordinates; no other activation, target batch, loss definition,
blur/scaling, optimization rule or numerical solver is replaced. Explicit disk
storage uses a disclosed runtime storage adapter, described in
[MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md); official source stays unmodified.

This is an **official-code-based normalized-embedding adaptation**, not a literal
reproduction of the paper's gene-expression-input experiment. Report AE/fusion
pretraining separately. Matching 20,000 dynamics updates and 1,024 generated
particles does not imply identical sample budgets or compute across methods.
Do not describe the hardcoded reference batch as 1,024.

## Splits and leakage boundaries

| Scenario | Observed stages | Native training times | Cells per modality |
| --- | --- | --- | ---: |
| `full` | E7.5, E8.0, E8.5, E8.75 | 0, 1, 2, 2.5 | 35,503 |
| `loo1` | E7.5, E8.5, E8.75 | 0, 2, 2.5 | 27,707 |
| `loo2` | E7.5, E8.0, E8.75 | 0, 1, 2.5 | 24,033 |

Every split trains a new model from seed 0, without loading a full-model AE,
correspondence, fusion or dynamics checkpoint. The held-out stage is excluded
from **both modalities before all four training components**. Full observed-cell
training is not a fixed small subsample. Original times are retained, not
compressed into integer ranks. No cell-type labels or paired-row supervision
are supplied. All three configurations are otherwise identical.

The benchmark's existing PCA/LSI representations and normalization scalars are
frozen and were constructed using the full atlas. Thus this is strict
**scMultiNODE training-stage LOO conditional on fixed shared representations**,
not end-to-end inductive raw-data LOO. Held-out observations are for evaluation,
never checkpoint/hyperparameter selection in this configuration.

## Trajectory protocol

The official Euler solver takes one numerical step per supplied time interval.
Adding dense query times to the solve changes observed-time predictions. Formal
export therefore solves on each split's actual observed training times, takes
piecewise-linear dense output in the joint latent space, then applies both
learned decoders. This is the native explicit-Euler interpolant, not a finer ODE
solve and not interpolation of decoded observations. A held-out time is only an
interpolation query. The `trajectory_dt=0.025` parameter controls output density,
not the training/inference integration step size.

At every observed knot, each export batch is checked against unmodified official
`model.forward(..., batch_size=None)` (absolute tolerance 1e-5). Coarser/finer
query grids preserve those predictions. The exporter bypasses upstream
`predict()`'s repeated-initial/truncation policy so all 9,018 initial cells are
used exactly once, in source order. Batch 512 controls inference memory only.

Default arrays are time-major: RNA `(101,9018,50)`, ATAC `(101,9018,14)` and
latent `(101,9018,10)`. Both modalities follow the same RNA-anchored latent path;
ATAC is not separately pushed from its own initial cells. Initial decoded RNA
is an AE reconstruction, not forced equal to the observed input. Weights are
uniform: scMultiNODE has no native growth/death model here. Metrics must use the
decoded normalized modality arrays, not the joint latent or UMAP coordinates.

## Memory and execution

### Default official storage

Official QGW constructs full quadratic arrays even with `qgw_sample_ratio=0.1`.
The sparse threshold operation also materializes dense-shaped temporary data.
The runner checks available RAM (including detectable Linux cgroup limits)
before creating training outputs. It never automatically subsamples or rewrites
QGW to fit local memory.

| Scenario | Four dense arrays alone, GiB | Conservative available-RAM gate, GiB |
| --- | ---: | ---: |
| Full | 37.56 | 192 |
| LOO1 | 22.88 | 128 |
| LOO2 | 17.21 | 96 |

The gate is a **planning heuristic, not a measured minimum or peak guarantee**:
five times the four-array baseline, rounded up to 32 GiB. The local machine had
only about 22–23 GiB available during preflight, so the default official-storage
runs were blocked. For that path use a sufficiently provisioned host/allocation,
or explicitly opt into the disk-storage path below. These values refer to available
memory, not advertised physical RAM. Run large jobs serially unless their
combined memory budgets fit. There is no multi-job reservation/queue manager.
On managed servers additionally set `MEMORY_BUDGET_GIB` to the actual available
job allocation; it caps automatic detection and cannot raise it. Keep the same
source revision and compatible environment on the destination host.

From the repository root, audit without training:

```bash
bash model/scMultiNODE/run_gastrulation_benchmark.sh full --check
bash model/scMultiNODE/run_gastrulation_benchmark.sh loo1 --check
bash model/scMultiNODE/run_gastrulation_benchmark.sh loo2 --check
```

Check mode emits JSON; inspect `memory.passes` (a false value is not a nonzero
check-mode exit). On a sufficiently provisioned host, launch the desired case:

```bash
bash model/scMultiNODE/run_gastrulation_background.sh full
# After it finishes, or on a separate allocation:
bash model/scMultiNODE/run_gastrulation_background.sh loo1
bash model/scMultiNODE/run_gastrulation_background.sh loo2
```

The launcher already uses nohup, stdin detachment and a separate launcher log.
It refuses existing output directories/logs. A returned PID means submission,
not successful training: insufficient memory is reported in that job's log.
Defaults: `results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0`;
logs append `_launcher.log`. Set `PYTHON`, `GASTRULATION_DATA_DIR`, `OUTPUT_DIR`
and `LOG_FILE` where needed. A custom `CONFIG` must match the requested scenario;
give changed seeds/settings a new output name. No automatic resume/overwrite.

### Opt-in formal disk storage

`QGW_STORAGE=disk` selects disk-backed float64 distances and sparse coupling
storage. It does **not** select the diagnostic runner or reduce the dataset:
every observed training cell remains included, and held-out stages remain absent
from both modalities before all training components. The public configuration
stays AE2000 / fusion2000 / dynamics20000, generated batch1024, native reference
batch200, latent10 and seed0. QGW numerical defaults and solver-warning behavior
are unchanged. There is no silent fallback from official to disk storage.

Check the desired scenario first, without creating a training run:

```bash
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh full --check
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh loo1 --check
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh loo2 --check
```

Inspect the resource report and `memory.passes`; check mode reports failures in
JSON without a failing exit code. The disk path checks scratch capacity and
available RAM instead of applying the official-storage 192/128/96 GiB gates.
The two full distance files alone need approximately 18.78 GiB for full,
11.44 GiB for LOO1 or 8.61 GiB for LOO2. The scratch check adds 2 GiB headroom;
leave further room for results and other processes. `SCRATCH_ROOT` must already
exist. The RAM check requires the RSS budget plus reserve: 16 GiB available by
default. This is a policy gate, not a measured peak or resource reservation.
Working memory is still needed for KNN, compressed GW and training.

Launch **one desired scenario** with these commands:

```bash
bash model/scMultiNODE/run_gastrulation_memory_background.sh full
# Or either LOO split:
bash model/scMultiNODE/run_gastrulation_memory_background.sh loo1
bash model/scMultiNODE/run_gastrulation_memory_background.sh loo2
```

The wrapper sets `QGW_STORAGE=disk` and delegates to the existing protected
background launcher. It already uses `nohup`; **do not append `&`**. It prints
the PID and log location, and does not overwrite existing outputs or logs.
A submitted PID does not establish successful preflight or training. Defaults:

```text
results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0_diskqgw
results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0_diskqgw_launcher.log
```

The original background launcher remains official-storage by default. To choose
disk storage through it directly, use `QGW_STORAGE=disk` **and set a clearly named
new `OUTPUT_DIR`**. The dedicated wrapper does both automatically.

| Environment override | Disk default | Meaning |
| --- | --- | --- |
| `ROW_BLOCK_SIZE` | `256` | Distance-processing row block, not training minibatch size |
| `SCRATCH_ROOT` | `/private/tmp` | Parent directory for unique temporary distance files |
| `MAX_RSS_GIB` | `10` | Sampled process-RSS stop threshold |
| `MIN_AVAILABLE_GIB` | `6` | Minimum available system RAM reserve |
| `MEMORY_BUDGET_GIB` | automatic detection | Optional allocation cap on preflight available RAM |

`PYTHON`, `GASTRULATION_DATA_DIR`, `CONFIG`, `OUTPUT_DIR` and `LOG_FILE` remain
configurable. Example with explicit defaults and a new repeat name:

```bash
ROW_BLOCK_SIZE=256 SCRATCH_ROOT=/private/tmp MAX_RSS_GIB=10 MIN_AVAILABLE_GIB=6 \
OUTPUT_DIR=results/scmultinode_gastrulation_loo1_normalized10_n1024_20000_seed0_diskqgw_repeat \
bash model/scMultiNODE/run_gastrulation_memory_background.sh loo1
```

For the two LOO runs, serial execution is safest. If overlapping them, first
wait until LOO1 has finished QGW and entered fusion/dynamics, then check available
RAM and disk before starting LOO2. Avoid running both QGW phases together. There
is no scheduler or cross-process resource reservation; per-process sampled
guards are not OS-enforced hard limits and allocation bursts can overshoot.

The full-observed LOO1 pilot measured 2.656 GiB peak sampled RSS and 951.79s
(15.86min), but used only **20 dynamics updates** after AE2000/fusion2000. This
supports that tested memory path, not convergence, a 20k-run runtime estimate,
or guaranteed memory use for full/LOO2. Its inner EMD hit the official iteration
cap. Subsequent runs record and surface solver warnings and coupling diagnostics;
the adapter does not increase the cap, suppress warnings or replace the returned
plan. Finite training losses do not establish QGW alignment convergence.
Inspect `qgw_audit.json` and the manifest's `qgw_quality`: solver warnings mark
`needs_review` while preserving upstream continuation. Nonfinite, negative,
zero-mass or materially infeasible probability plans fail validation before
fusion; that safety check is not a solver replacement or convergence claim.
`resource_live.json` records sampled usage; `resource_guard.json` records a guard
stop. `scratch.json` names the unique temporary scratch directory. Normal cleanup
removes only that temporary data, not results/checkpoints. A watchdog's immediate
exit can leave scratch behind; verify the process is stopped before removing
its exact recorded scratch directory.

## Validation and saved artifacts

Each formal branch passed a separate 64-cells/time/modality test with two updates
per phase; these remain explicitly named engineering smoke outputs under
`results/scmultinode_gastrulation_{full,loo1,loo2}_protocol_smoke_seed0`.
They validate mechanics, not convergence or distribution quality. Each pushes
all 9,018 original initial cells after its tiny training run.

Artifacts include source/normalization hashes, training indices, full config,
upstream revision, memory report, package versions, phase checkpoints,
`model.pt`, compact saved QGW correspondence, normalized trajectories and
finite-value/native-knot equivalence audits. The training loss CSV records the
official already-rounded displayed losses; it does not pretend to recover
unrounded values. Checkpoints are saved after optimizer updates and support
inference, not optimizer-state resume. Nonfinite losses/gradients fail fast;
no gradient clipping or adaptive loss reweighting is introduced.

Re-exporting a benchmark checkpoint through `generate_trajectory.py` preserves
the native-grid protocol. `plot_gastrulation_umap.py` supports all three splits,
uses the same frozen UMAPs and labels the withheld stage; plotting does not train
or select a checkpoint.

```bash
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" -m unittest discover \
  -s model/scMultiNODE -p 'test_*.py' -v
```
