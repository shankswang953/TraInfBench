#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
CYTOBRIDGE_CACHE_ROOT=${CYTOBRIDGE_CACHE_ROOT:-/tmp/trainfbench-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CYTOBRIDGE_CACHE_ROOT}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CYTOBRIDGE_CACHE_ROOT}/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${CYTOBRIDGE_CACHE_ROOT}/numba}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_balanced.h5ad}
EPOCHS=${EPOCHS:-30000}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
LAMBDA_OT=${LAMBDA_OT:-10.0}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
SAVE_DIR=${SAVE_DIR:-results/cytobridge_human_cerebral_7time_d4_d21_no_d16_balanced_30000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -s "$INPUT_H5AD" ]]; then
  "$PYTHON" model/CytoBridge/prepare_human_cerebral_7time_cytobridge_inputs.py
fi
"$PYTHON" model/CytoBridge/check_human_cerebral_7time_cytobridge_setup.py

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "CytoBridge balanced seven-time full preflight passed"
  echo "  input:          $INPUT_H5AD"
  echo "  output:         $SAVE_DIR"
  echo "  times:          D4,D7,D9,D11,D12,D18,D21 (D16 excluded)"
  echo "  model space:    normalized RNA PCA30"
  echo "  components:     velocity"
  echo "  train epochs:   $EPOCHS"
  echo "  batch size:     $BATCH_SIZE"
  echo "  seed:           $SEED"
  echo "  learning rate:  0.0001"
  echo "  lambda_ot:      $LAMBDA_OT"
  echo "  lambda_energy:  $LAMBDA_ENERGY"
  exit 0
fi

if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  $SAVE_DIR"
    echo "  $LOG_FILE"
    echo "Run with OVERWRITE=1 to replace them."
    exit 1
  fi
  rm -rf "$SAVE_DIR" "$LOG_FILE"
fi
mkdir -p "$(dirname "$LOG_FILE")"

"$PYTHON" model/CytoBridge/train_cytobridge_20000.py \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lambda-ot "$LAMBDA_OT" \
  --lambda-energy "$LAMBDA_ENERGY" \
  --device "$DEVICE" \
  --seed "$SEED" \
  > "$LOG_FILE" 2>&1

echo "CytoBridge balanced seven-time training completed"
echo "  output: $SAVE_DIR"
echo "  log:    $LOG_FILE"
