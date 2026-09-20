#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/local/Install/conda_env/CytoBridge/bin/python" ]]; then
    PYTHON="$HOME/local/Install/conda_env/CytoBridge/bin/python"
  else
    PYTHON=python
  fi
fi
GASTRULATION_DATA_DIR=${GASTRULATION_DATA_DIR:-../TraInf/TraInf/Gastrulation/data}
OUTPUT_DIR=${OUTPUT_DIR:-results/scmultinode_gastrulation_normalized_smoke_seed0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
"$PYTHON" model/scMultiNODE/run_gastrulation.py \
  --data-dir "$GASTRULATION_DATA_DIR" --output-dir "$OUTPUT_DIR" "$@"
