#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCENARIO=${1:?Usage: run_human_cerebral_benchmark.sh full [--check\|--smoke]}
shift
case "$SCENARIO" in full) ;; *) echo "Unknown scenario: $SCENARIO (expected full; human cerebral LOO is not configured)" >&2; exit 2;; esac
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/local/Install/conda_env/CytoBridge/bin/python" ]]; then
    PYTHON="$HOME/local/Install/conda_env/CytoBridge/bin/python"
  else
    PYTHON=python
  fi
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/private/tmp/trainfbench-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/private/tmp/numba-cache}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
HUMAN_CEREBRAL_DATA_DIR=${HUMAN_CEREBRAL_DATA_DIR:-../TraInf/TraInf/humanCerebral/Data/selected_4_7_9_11_12_18_21}
QGW_STORAGE=${QGW_STORAGE:-official}
case "$QGW_STORAGE" in official|disk) ;; *) echo "Unknown QGW_STORAGE: $QGW_STORAGE (expected official or disk)" >&2; exit 2;; esac
DEFAULT_OUTPUT=results/scmultinode_human_cerebral_7time_d4_d21_no_d16_full_normalized10_n1024_20000_seed0
for option in "$@"; do
  if [[ "$option" == "--smoke" ]]; then
    DEFAULT_OUTPUT=results/scmultinode_human_cerebral_7time_d4_d21_no_d16_full_protocol_smoke_seed0
  fi
done
if [[ "$QGW_STORAGE" == "disk" ]]; then DEFAULT_OUTPUT=${DEFAULT_OUTPUT}_diskqgw; fi
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT}
CONFIG=${CONFIG:-model/scMultiNODE/configs/human_cerebral_full.json}
args=(--qgw-storage "$QGW_STORAGE")
if [[ -n "${MEMORY_BUDGET_GIB:-}" ]]; then args+=(--memory-budget-gib "$MEMORY_BUDGET_GIB"); fi
if [[ -n "${ROW_BLOCK_SIZE:-}" ]]; then args+=(--row-block-size "$ROW_BLOCK_SIZE"); fi
if [[ -n "${SCRATCH_ROOT:-}" ]]; then args+=(--scratch-root "$SCRATCH_ROOT"); fi
if [[ -n "${MAX_RSS_GIB:-}" ]]; then args+=(--max-rss-gib "$MAX_RSS_GIB"); fi
if [[ -n "${MIN_AVAILABLE_GIB:-}" ]]; then args+=(--min-available-gib "$MIN_AVAILABLE_GIB"); fi
exec "$PYTHON" model/scMultiNODE/benchmark_human_cerebral.py \
  --config "$CONFIG" --expected-scenario "$SCENARIO" --data-dir "$HUMAN_CEREBRAL_DATA_DIR" --output-dir "$OUTPUT_DIR" "${args[@]}" "$@"
