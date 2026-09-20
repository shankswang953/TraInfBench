# Storage-equivalent scMultiNODE QGW adapter

This is an explicitly opt-in storage adaptation, validated with bounded
diagnostics and now available to the formal full/LOO runner. Existing launchers
still default to official storage and its conservative preflight; nothing
switches them automatically. The dedicated disk-storage launcher is
`run_gastrulation_memory_background.sh`. Diagnostic commands below do not start
20,000-step jobs. Original results and official source are untouched.

## What changes

- `_mod_distance`: the identical correlation KNN graph and float64 undirected
  Dijkstra distances, computed in source-row blocks into temporary disk-backed
  arrays. The global disconnected-distance cap and global normalization are
  unchanged. There is no nearest-landmark approximation or float32 distance cast.
- QGW deterministic couplings: CSC-backed column access instead of two dense
  N-by-N coupling arrays. The original official QGW function is reused with
  cloned globals; compressed GW, local EMD, loop order, landmark order, tie
  handling and random-number consumption are retained.
- Correspondence thresholding: apply the same max normalization and <=0.01
  threshold to stored entries and eliminate explicit zeros. Implicit zero values
  remain zero, rather than being inserted into the sparse matrix.
- Distance normalization: operate in blocks without full-matrix copies.
- Initial experiments used flush + MADV_DONTNEED page-cache hints, which did
  not reliably release RSS on this Mac. The current trainer instead uses
  `DiskDistanceArray`: each bounded read/write explicitly opens and closes its
  mapping and returns owned read copies. No mmap view escapes an operation.
  The older persistent-map implementation remains unit-tested as an option;
  copy-on-write mappings are never discarded.

`memory_optimized_train.py` compiles a runtime clone of the pinned official
trainer. Exactly six known storage statements are substituted; a missing or
duplicate statement fails closed. All training iterations, minibatch/RNG calls,
four hardcoded 200-cell losses, coefficients, optimization and Euler solve are
otherwise retained. The signed-decoder adaptation is the same in both compared
models. This is a disclosed storage implementation adaptation, not a claim that
the official trainer ran without any adaptation.

The official dense QGW fallback is not approximated: the storage adapter refuses
the fallback if it would materialize over 512 MiB. Such a refusal is a resource
failure, not a completed fit.

## Meaning of the old RAM thresholds

The existing 192/128/96 GiB formal gates are our conservative 5x planning
heuristic, NOT official requirements or measured RAM peaks. Four dense arrays'
logical byte count is not necessarily simultaneous RSS. The original failed
LOO1 launch was a preflight rejection, not a measured OOM.

The optimized implementation still needs disk scratch for both full float64
distance matrices: approximately 18.78 GiB (full), 11.44 GiB (LOO1), or 8.61 GiB
(LOO2). It also needs working RAM for unchanged KNN and compressed GW. Disk-backed
arrays do not guarantee low RSS on all operating systems; measurements matter.

## Tests already performed

Three real-data, fixed-seed comparisons use 128 cells per observed time/modality,
AE200, fusion20, dynamics10, generated batch1024 and native reference batch200:

| Split | Seed | Largest difference: distances, coupling, model, last gradient, prediction |
| --- | ---: | ---: |
| Full | 1 | 0 for every compared category |
| LOO1 | 0 | 0 for every compared category |
| LOO2 | 2 | 0 for every compared category |

Reports: `results/scmultinode_memory_equivalence_{full,loo1,loo2}_n128_seed{1,0,2}`.
Every comparison reinitializes RNG/model identically before original and optimized
training. This establishes the tested short-run equivalence, not empirical proof
that every possible input or a 20k training run is bitwise identical.

A larger LOO1 pilot used 2,048 cells per observed stage (6,144 per modality),
AE2000, fusion20 and dynamics20 at generated batch1024. It completed in 36.45s
including export, with sampled peak process RSS 2.13 GiB, and exported all 9,018
initial cells. The source pool was explicitly subsampled for this scaling check;
it is not a full-observed-data benchmark. Report:
`results/scmultinode_memory_pilot_loo1_n2048_seed0/manifest.json`.

The first full-observed LOO1 pilot was intentionally interrupted after finishing
the first distance matrix's Dijkstra pass: persistent mmap pages grew to a
sampled 6.30 GiB RSS despite advice. This was NOT an OOM or a training failure.
Its scratch directory was cleaned on normal interruption; the audit remains in
`results/scmultinode_memory_pilot_loo1_allobserved_seed0/stop_reason.json`.

The revised open/close-per-operation strategy was compared again on LOO1
128/time/seed0: distances, coupling, parameters, gradient and predictions again
all had maximum absolute difference 0. Report:
`results/scmultinode_memory_bounded_equivalence_loo1_n128_seed0/manifest.json`.

The revised full-observed LOO1 pilot completed successfully: 27,707 training
cells per modality, AE2000, fusion2000, dynamics20, generated batch1024 and
unchanged native reference batch200. Total wall time was 951.79s (15.86min),
with sampled peak process RSS 2.656 GiB. The two distance-building passes took
222.31s and 224.02s; QGW took 447.51s. Dynamics20 took 9.71s. Thus the major
one-time cost in this pilot was all-cell alignment, not minibatch training.
All 9,018 initial cells were exported, with finite RNA/ATAC predictions shaped
(4, 9018, 50)/(4, 9018, 14). Temporary distance files were removed on normal exit.
Report: `results/scmultinode_memory_bounded_pilot_loo1_allobserved_seed0/manifest.json`.
This is a pipeline/resource test, not a trained 20k model or a distribution-quality
validation. Formal runs must separately opt into `QGW_STORAGE=disk`; the measured
peak and timing do not guarantee resource use or convergence of a 20k run.

The full-observed revised pilot encountered a POT warning that an inner EMD
network-simplex solve hit its default 100,000-iteration cap before optimality.
No solver limit or other numerical parameter was changed. Memory feasibility
and alignment convergence are separate questions; finite subsequent losses do
not establish that this warning is harmless. See the pilot's `solver_warning.json`.
Detailed telemetry was subsequently added for FUTURE runs (not retroactively to
this pilot): all emitted solver warnings, existing GW loss history, final inner
result code, and compressed/raw coupling finite values, mass and marginal errors.
Returned solver plans and parameters are unchanged. GW's nonconvex objective
does not carry a global-optimality guarantee even without an LP warning.

Unit tests cover distance values including disconnected/tied cases, normalized
values, unequal modality sizes, nonuniform masses, QGW and selected alignment
indices, RNG preservation, sparse thresholding, dense fallback refusal,
exclusive scratch files, live mmap/COW safety, and fail-closed trainer cloning.

The formal disk entry point also passed simultaneous LOO1/LOO2 engineering
smokes (64 cells per observed time/modality, two updates per phase). Each saved
AE/fusion/dynamics checkpoints and finite decoded trajectories for all 9,018
initial cells on 101 query times; both exact scratch directories were removed.
Reports: `results/scmultinode_gastrulation_{loo1,loo2}_diskqgw_formal_smoke_seed0`.
This checks entry-point integration, not full-size concurrent resource use.
A fresh LOO1 comparison after integration again found zero maximum absolute
difference in distances, coupling, model, gradients and predictions:
`results/scmultinode_memory_formal_integration_equivalence_loo1_n128_seed0`.
Both full-size LOO preflights passed locally; no 20k job was started by setup.

## Formal full/LOO launch

The formal runner accepts `--qgw-storage official|disk`, default `official`.
The shell launcher forwards `QGW_STORAGE`; the dedicated memory background
wrapper sets it to `disk` and supplies a distinct `_diskqgw` output name.
This formal route retains all observed training cells, AE2000, fusion2000,
dynamics20000, generated batch1024 and native reference batch200. It uses the
same split-specific configs and native-grid exports as the original route.
It does not call `trial_memory_optimized.py` or turn a bounded pilot into a
formal benchmark. Solver settings and warnings remain unchanged; emitted inner
EMD warnings are surfaced and recorded alongside the QGW diagnostics.

```bash
# Read-only formal preflight:
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh loo1 --check
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh loo2 --check

# Formal background training; choose the desired split:
bash model/scMultiNODE/run_gastrulation_memory_background.sh loo1
# Safest: start this after LOO1 completes.
bash model/scMultiNODE/run_gastrulation_memory_background.sh loo2
# The same wrapper also accepts full.
```

No trailing `&` is needed: the launcher already uses `nohup` and prints its PID
and launcher-log path. Default outputs are
`results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0_diskqgw`;
logs append `_launcher.log`. Existing outputs/logs are refused. For repeats,
set a new `OUTPUT_DIR` and optionally `LOG_FILE`.

Defaults are `ROW_BLOCK_SIZE=256`, `SCRATCH_ROOT=/private/tmp`, `MAX_RSS_GIB=10`
and `MIN_AVAILABLE_GIB=6`; these are storage/guard controls, not model settings.
`PYTHON` and `GASTRULATION_DATA_DIR` remain configurable. Formal disk preflight
checks scratch capacity and available memory; a sampled guard remains active
during training. Preflight requires at least the RSS budget plus reserve
(16 GiB by default), and enough free scratch space for both distance files plus
2 GiB. The scratch parent directory must already exist. Guards cannot enforce a
hard OS-level allocation ceiling or reserve resources across jobs. Run the two
LOO jobs serially, or stagger the
second until the first is past QGW and recheck resources before launching it.
Do not overlap both QGW phases without independently adequate RAM and scratch.
Full commands and caveats are in [BENCHMARK.md](BENCHMARK.md).

Formal disk runs write `qgw_audit.json`; the manifest's `qgw_quality` is
`needs_review` when recorded solver warnings/status require attention. Warnings
alone retain upstream continuation. Nonfinite, negative, zero-mass or materially
infeasible probability plans fail validation before fusion. Passing these
checks is not evidence of solver convergence. Resource progress is in
`resource_live.json`, guard stops in `resource_guard.json`, and the exact unique
scratch location in `scratch.json`. A forced watchdog exit may leave scratch
for inspection; do not remove it while its process is still running.

## Bounded diagnostic reproduction

From the repository root, use the shared scientific Python environment. Select
a NEW output directory: existing files are never overwritten.

```bash
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" model/scMultiNODE/trial_memory_optimized.py \
  --mode compare --scenario loo1 --cells-per-time 128 \
  --ae-iters 200 --fusion-iters 20 --iters 10 \
  --max-seconds 180 --max-rss-gib 6 \
  --output-dir results/scmultinode_memory_equivalence_loo1_n128_repeat
```

Full-observed LOO1 diagnostic (still only 20 dynamics updates):

```bash
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" model/scMultiNODE/trial_memory_optimized.py \
  --mode pilot --scenario loo1 --all-observed \
  --ae-iters 2000 --fusion-iters 2000 --iters 20 \
  --max-seconds 1200 --max-rss-gib 10 --min-available-gib 6 --row-block-size 256 \
  --output-dir results/scmultinode_memory_pilot_loo1_allobserved_repeat
```

The process samples its own RSS and system available memory every 0.2s and stops
itself at a resource/time threshold. This is NOT an OS-enforced hard limit;
native allocation bursts can overshoot. Run pilots sequentially. No other user
process is terminated. `resource_live.json` reports progress; a watchdog stop
writes `resource_guard.json` and exits with code75. Guard failure exits76.

Scratch paths are unique `/private/tmp/scmultinode-distance-*` directories, with
their exact path saved to `scratch.json`. Normal exits remove only these scratch
files. A watchdog's immediate exit bypasses Python cleanup; confirm that process
is stopped before removing its exact recorded scratch directory. Keep results,
checkpoints, indices, trajectories and audit manifests for reproducibility.

All pilots export every original initial cell, including cells not used in a
subsampled pilot. Dynamics outputs are native-grid latent interpolants decoded
back to the original normalized RNA/ATAC spaces. No UMAP or held-out-based model
selection is used in these implementation checks.
