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
INPUT_H5AD=${INPUT_H5AD:-data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_unbalanced_biological_prior.h5ad}
EPOCHS=${EPOCHS:-30000}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
HIDDEN_DIM=${HIDDEN_DIM:-400}
N_LAYERS=${N_LAYERS:-2}
LR=${LR:-0.00005}
LAMBDA_OT=${LAMBDA_OT:-1.0}
LAMBDA_MASS=${LAMBDA_MASS:-0.01}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
LAMBDA_DENSITY=${LAMBDA_DENSITY:-0.0}
PRETRAIN_LAMBDA_DENSITY=${PRETRAIN_LAMBDA_DENSITY:-0.0}
GLOBAL_MASS=${GLOBAL_MASS:-1}
SAVE_DIR=${SAVE_DIR:-results/cytobridge_human_cerebral_7time_d4_d21_no_d16_unbalanced_biological_prior_30000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -s "$INPUT_H5AD" ]]; then
  "$PYTHON" model/CytoBridge/prepare_human_cerebral_7time_cytobridge_inputs.py
fi
"$PYTHON" model/CytoBridge/check_human_cerebral_7time_cytobridge_setup.py

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "CytoBridge unbalanced seven-time biological-prior full preflight passed"
  echo "  input:          $INPUT_H5AD"
  echo "  output:         $SAVE_DIR"
  echo "  times:          D4,D7,D9,D11,D12,D18,D21 (D16 excluded)"
  echo "  model space:    normalized RNA PCA30"
  echo "  mass prior:     1.00,2.92,4.15,5.36,5.96,22.94,33.99"
  echo "  mass encoding:  official rows_at_t / rows_at_D4"
  echo "  components:     velocity + growth"
  echo "  pretrain/train: $PRETRAIN_EPOCHS/$EPOCHS"
  echo "  batch size:     $BATCH_SIZE"
  echo "  seed:           $SEED"
  echo "  hidden/layers:  $HIDDEN_DIM/$N_LAYERS"
  echo "  learning rate:  $LR"
  echo "  lambda_ot:      $LAMBDA_OT"
  echo "  lambda_mass:    $LAMBDA_MASS"
  echo "  lambda_energy:  $LAMBDA_ENERGY"
  echo "  global_mass:    $GLOBAL_MASS"
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

GLOBAL_MASS_ARGS=(--global-mass)
if [[ "$GLOBAL_MASS" == "0" || "$GLOBAL_MASS" == "false" || "$GLOBAL_MASS" == "False" ]]; then
  GLOBAL_MASS_ARGS=(--no-global-mass)
fi

"$PYTHON" model/CytoBridge/train_cytobridge_unbalanced.py \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --n-layers "$N_LAYERS" \
  --lr "$LR" \
  --lambda-ot "$LAMBDA_OT" \
  --lambda-mass "$LAMBDA_MASS" \
  --lambda-energy "$LAMBDA_ENERGY" \
  --lambda-density "$LAMBDA_DENSITY" \
  --pretrain-lambda-density "$PRETRAIN_LAMBDA_DENSITY" \
  "${GLOBAL_MASS_ARGS[@]}" \
  --device "$DEVICE" \
  --seed "$SEED" \
  > "$LOG_FILE" 2>&1

echo "CytoBridge unbalanced seven-time training completed"
echo "  output: $SAVE_DIR"
echo "  log:    $LOG_FILE"
