#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCENARIO=${1:?Usage: run_gastrulation_benchmark.sh full\|loo1\|loo2 [--check\|--smoke]}
shift
case "$SCENARIO" in full|loo1|loo2) ;; *) echo "Unknown scenario: $SCENARIO" >&2; exit 2;; esac
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/local/Install/conda_env/CytoBridge/bin/python" ]]; then
    PYTHON="$HOME/local/Install/conda_env/CytoBridge/bin/python"
  else
    PYTHON=python
  fi
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
GASTRULATION_DATA_DIR=${GASTRULATION_DATA_DIR:-../TraInf/TraInf/Gastrulation/data}
DEFAULT_OUTPUT=results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0
for option in "$@"; do
  if [[ "$option" == "--smoke" ]]; then
    DEFAULT_OUTPUT=results/scmultinode_gastrulation_${SCENARIO}_protocol_smoke_seed0
  fi
done
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT}
CONFIG=${CONFIG:-model/scMultiNODE/configs/gastrulation_${SCENARIO}.json}
QGW_STORAGE=${QGW_STORAGE:-official}
case "$QGW_STORAGE" in official|disk) ;; *) echo "Unknown QGW_STORAGE: $QGW_STORAGE (expected official or disk)" >&2; exit 2;; esac
args=(--qgw-storage "$QGW_STORAGE")
if [[ -n "${MEMORY_BUDGET_GIB:-}" ]]; then args+=(--memory-budget-gib "$MEMORY_BUDGET_GIB"); fi
if [[ -n "${ROW_BLOCK_SIZE:-}" ]]; then args+=(--row-block-size "$ROW_BLOCK_SIZE"); fi
if [[ -n "${SCRATCH_ROOT:-}" ]]; then args+=(--scratch-root "$SCRATCH_ROOT"); fi
if [[ -n "${MAX_RSS_GIB:-}" ]]; then args+=(--max-rss-gib "$MAX_RSS_GIB"); fi
if [[ -n "${MIN_AVAILABLE_GIB:-}" ]]; then args+=(--min-available-gib "$MIN_AVAILABLE_GIB"); fi
exec "$PYTHON" model/scMultiNODE/benchmark_gastrulation.py \
  --config "$CONFIG" --expected-scenario "$SCENARIO" --data-dir "$GASTRULATION_DATA_DIR" --output-dir "$OUTPUT_DIR" "${args[@]}" "$@"
