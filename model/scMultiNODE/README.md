# scMultiNODE

Benchmark adapters for [official scMultiNODE](https://github.com/rsinghlab/scMultiNODE): training, full/leave-one-stage-out configurations, and RNA/ATAC trajectory export. Third-party source, data and checkpoints are not included.

## Install

Use Python 3.10 for these adapters. Follow the [upstream dependency guide](https://github.com/rsinghlab/scMultiNODE/blob/c4e444d24849393bc923ffb5373e5cea35476013/installation); the benchmark uses the scientific environment described in the root README. The upstream example environment is older and should not be confused with the adapter environment.

From the repository root:

```bash
git clone https://github.com/rsinghlab/scMultiNODE.git external/scMultiNODE
git -C external/scMultiNODE checkout c4e444d24849393bc923ffb5373e5cea35476013
python model/scMultiNODE/benchmark_gastrulation.py --help
```

No editable install is needed: the runners import the local checkout directly. Keep it unmodified; the runners check its revision and working tree. The checkout is ignored by Git. Its code and license remain with the upstream project.

## Datasets and entrypoints

| Dataset | Frozen input coordinates | Supported tasks | Details |
| --- | --- | --- | --- |
| Gastrulation | Normalized RNA PCA50 / ATAC LSI14 | Full, LOO E8.0, LOO E8.5 | [BENCHMARK.md](BENCHMARK.md) |
| Pancreas | Normalized RNA PCA50 / ATAC PoissonVI22 | Full, LOO E15.5 | [PANCREAS.md](PANCREAS.md) |
| Palate | Normalized RNA PCA40 / ATAC LSI15 | Full, LOO E13.5, LOO E14.0 | [PALATE.md](PALATE.md) |
| Human cerebral organoids | Normalized RNA PCA30 / ATAC LSI12 | Full, seven times D4-D21 excluding D16 | [HUMAN_CEREBRAL.md](HUMAN_CEREBRAL.md) |

Set the corresponding data-directory variable to the prepared dataset containing the matrices and frozen normalization files specified in each guide. Run from the repository root. These checks inspect inputs and resource requirements without training:

```bash
export PYTHON=python
export GASTRULATION_DATA_DIR=/path/to/Gastrulation/data
QGW_STORAGE=disk bash model/scMultiNODE/run_gastrulation_benchmark.sh full --check

export MOSCOT_DATA_DIR=/path/to/moscot/data
QGW_STORAGE=disk bash model/scMultiNODE/run_pancreas_benchmark.sh loo1 --check

export PALATE_DATA_DIR=/path/to/MouseBrain/data
QGW_STORAGE=disk bash model/scMultiNODE/run_palate_benchmark.sh loo2 --check

export HUMAN_CEREBRAL_DATA_DIR=/path/to/humanCerebral/Data/selected_4_7_9_11_12_18_21
QGW_STORAGE=disk bash model/scMultiNODE/run_human_cerebral_benchmark.sh full --check
```

Remove `--check` to train. `--smoke` explicitly selects a small engineering run instead of the full budget. Choose a new `OUTPUT_DIR` for repeats; existing outputs are protected. Full jobs use 2,000 AE, 2,000 fusion and 20,000 dynamics updates, with generated batch size 1,024 and unchanged upstream reference batch size 200.

Official QGW storage can require substantial memory. `QGW_STORAGE=disk` opts into the tested storage adapter; its numerical equivalence and resource guards are documented in [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md). This is an implementation adaptation, not unchanged upstream execution.

## What is adapted

The default trainer calls upstream `constructscMultiNODEModel` and `scMultiNODETrain`: modality autoencoders, QGW correspondence, joint fusion and neural ODE training. The benchmark replaces only the **last decoder ReLU with Identity** in both modalities to support signed PCA/LSI/PoissonVI inputs. It retains the encoder, fusion and drift activations and native losses.

LOO removes held-out cells before every scMultiNODE training phase. The supplied PCA/LSI and normalization are frozen all-data representations, so this is not an end-to-end inductive raw-data LOO experiment. Biological clocks are preserved rather than compressed into consecutive ranks. Cell pairing and labels are not training supervision.

## Generate trajectories

Training saves `model.pt`, `manifest.json`, loss records and trajectory archives under `results/`. Re-export from a completed run without retraining:

```bash
python model/scMultiNODE/generate_trajectory.py \
  --run-dir results/YOUR_RUN \
  --output results/YOUR_RUN/trajectories_all_initial.npz
```

By default every original earliest-stage RNA cell is pushed once. The same latent paths are decoded into RNA and ATAC; no COATI mapping is used. Exported weights are uniform because scMultiNODE has no native birth/death model. The initial decoded RNA is a reconstruction, not forced to equal its input.

Formal exports preserve the native observed-time Euler grid and interpolate latent states for dense or held-out queries. Changing output resolution therefore does not change observed-stage endpoints. `--batch-size` controls export memory, not subsampling. The historical `run_gastrulation.py` / smoke launcher is retained as a diagnostic; its dense-query Euler outputs are not interchangeable with formal native-grid exports.

## Checks

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  python -m unittest discover -s model/scMultiNODE -p 'test_*.py'
```

On 2026-09-20, all 96 existing tests passed in the shared Python 3.10 environment. They cover held-out exclusion, frozen normalization, dataset clocks, native-grid export, QGW storage equivalence and resource guards. No full benchmark training was started for publication.
