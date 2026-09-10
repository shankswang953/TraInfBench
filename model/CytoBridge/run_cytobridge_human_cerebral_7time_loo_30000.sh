#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE=${1:?Usage: bash model/CytoBridge/run_cytobridge_human_cerebral_7time_loo_30000.sh balanced|unbalanced D7|D9|D11|D12|D18}
AGE_INPUT=${2:?Usage: bash model/CytoBridge/run_cytobridge_human_cerebral_7time_loo_30000.sh balanced|unbalanced D7|D9|D11|D12|D18}
AGE=${AGE_INPUT#D}
AGE=${AGE#d}
case "$AGE" in
  7|9|11|12|18) ;;
  *)
    echo "Unsupported held-out age: $AGE_INPUT; use D7, D9, D11, D12, or D18"
    exit 2
    ;;
esac
case "$MODE" in
  balanced|unbalanced) ;;
  *)
    echo "Unknown mode: $MODE; use balanced or unbalanced"
    exit 2
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
CYTOBRIDGE_CACHE_ROOT=${CYTOBRIDGE_CACHE_ROOT:-/tmp/trainfbench-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CYTOBRIDGE_CACHE_ROOT}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CYTOBRIDGE_CACHE_ROOT}/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${CYTOBRIDGE_CACHE_ROOT}/numba}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
PREFIX=human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_strict_loo_D${AGE}
if [[ "$MODE" == "balanced" ]]; then
  INPUT_H5AD=${INPUT_H5AD:-data/${PREFIX}_balanced.h5ad}
  SAVE_DIR=${SAVE_DIR:-results/cytobridge_human_cerebral_7time_d4_d21_no_d16_strict_loo_D${AGE}_balanced_30000}
else
  INPUT_H5AD=${INPUT_H5AD:-data/${PREFIX}_unbalanced_biological_prior.h5ad}
  SAVE_DIR=${SAVE_DIR:-results/cytobridge_human_cerebral_7time_d4_d21_no_d16_strict_loo_D${AGE}_unbalanced_biological_prior_30000}
fi
REFERENCE=${REFERENCE:-data/${PREFIX}_reference.npz}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
EPOCHS=${EPOCHS:-30000}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -s "$INPUT_H5AD" || ! -s "$REFERENCE" ]]; then
  "$PYTHON" model/CytoBridge/prepare_human_cerebral_7time_cytobridge_loo_inputs.py --ages "$AGE"
fi
"$PYTHON" model/CytoBridge/check_human_cerebral_7time_cytobridge_loo_setup.py \
  --age "$AGE" \
  --mode "$MODE"

if [[ "$MODE" == "balanced" ]]; then
  LAMBDA_OT=${LAMBDA_OT:-10.0}
  LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "CytoBridge balanced strict LOO-D${AGE} preflight passed"
    echo "  input:          $INPUT_H5AD"
    echo "  reference:      $REFERENCE"
    echo "  output:         $SAVE_DIR"
    echo "  epochs:         $EPOCHS"
    echo "  batch size:     $BATCH_SIZE"
    echo "  seed:           $SEED"
    echo "  learning rate:  0.0001"
    echo "  lambda_ot:      $LAMBDA_OT"
    echo "  lambda_energy:  $LAMBDA_ENERGY"
    echo "  components:     velocity"
    exit 0
  fi
else
  PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
  HIDDEN_DIM=${HIDDEN_DIM:-400}
  N_LAYERS=${N_LAYERS:-2}
  LR=${LR:-0.00005}
  LAMBDA_OT=${LAMBDA_OT:-1.0}
  LAMBDA_MASS=${LAMBDA_MASS:-0.01}
  LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
  LAMBDA_DENSITY=${LAMBDA_DENSITY:-0.0}
  PRETRAIN_LAMBDA_DENSITY=${PRETRAIN_LAMBDA_DENSITY:-0.0}
  GLOBAL_MASS=${GLOBAL_MASS:-1}
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "CytoBridge unbalanced strict LOO-D${AGE} preflight passed"
    echo "  input:          $INPUT_H5AD"
    echo "  reference:      $REFERENCE"
    echo "  output:         $SAVE_DIR"
    echo "  pretrain/train: $PRETRAIN_EPOCHS/$EPOCHS"
    echo "  batch size:     $BATCH_SIZE"
    echo "  seed:           $SEED"
    echo "  hidden/layers:  $HIDDEN_DIM/$N_LAYERS"
    echo "  learning rate:  $LR"
    echo "  lambda_ot:      $LAMBDA_OT"
    echo "  lambda_mass:    $LAMBDA_MASS"
    echo "  lambda_energy:  $LAMBDA_ENERGY"
    echo "  global_mass:    $GLOBAL_MASS"
    echo "  components:     velocity + growth"
    exit 0
  fi
fi

if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  $SAVE_DIR"
    echo "  $LOG_FILE"
    echo "Run with OVERWRITE=1 to replace them."
    exit 1
  fi
fi
mkdir -p "$(dirname "$LOG_FILE")"

OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
  rm -f "$LOG_FILE"
fi

if [[ "$MODE" == "balanced" ]]; then
  "$PYTHON" model/CytoBridge/train_cytobridge_20000.py \
    --input-h5ad "$INPUT_H5AD" \
    --output-dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lambda-ot "$LAMBDA_OT" \
    --lambda-energy "$LAMBDA_ENERGY" \
    --device "$DEVICE" \
    --seed "$SEED" \
    "${OVERWRITE_ARGS[@]}" \
    > "$LOG_FILE" 2>&1
else
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
    "${OVERWRITE_ARGS[@]}" \
    > "$LOG_FILE" 2>&1
fi

echo "CytoBridge $MODE strict LOO-D${AGE} training completed"
echo "  output: $SAVE_DIR"
echo "  log:    $LOG_FILE"
