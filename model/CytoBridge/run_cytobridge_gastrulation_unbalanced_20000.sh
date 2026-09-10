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

EPOCHS=${EPOCHS:-20000}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-1024}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
HIDDEN_DIM=${HIDDEN_DIM:-400}
N_LAYERS=${N_LAYERS:-2}
LR=${LR:-0.0001}
LAMBDA_OT=${LAMBDA_OT:-10.0}
LAMBDA_MASS=${LAMBDA_MASS:-10.0}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
LAMBDA_DENSITY=${LAMBDA_DENSITY:-0.0}
PRETRAIN_LAMBDA_DENSITY=${PRETRAIN_LAMBDA_DENSITY:-0.0}
DENSITY_TOP_K=${DENSITY_TOP_K:-5}
DENSITY_HINGE_VALUE=${DENSITY_HINGE_VALUE:-0.01}
GLOBAL_MASS=${GLOBAL_MASS:-1}
DATASET_NAME=${DATASET_NAME:-gastrulation}
INPUT_H5AD=${INPUT_H5AD:-data/gastrulation_rna_cytobridge.h5ad}
SAVE_DIR=${SAVE_DIR:-results/cytobridge_gastrulation_rna_20000_unbalanced}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}

if [[ -e "${SAVE_DIR}" || -e "${LOG_FILE}" ]]; then
  if [[ "${OVERWRITE}" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  ${SAVE_DIR}"
    echo "  ${LOG_FILE}"
    echo "Run with OVERWRITE=1 to replace them."
    exit 1
  fi
  rm -rf "${SAVE_DIR}" "${LOG_FILE}"
fi

mkdir -p "$(dirname "${LOG_FILE}")"

echo "Running unbalanced CytoBridge ${DATASET_NAME}"
echo "Input:           ${INPUT_H5AD}"
echo "Output:          ${SAVE_DIR}"
echo "Log:             ${LOG_FILE}"
echo "Pretrain epochs: ${PRETRAIN_EPOCHS}"
echo "Train epochs:    ${EPOCHS}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Device:          ${DEVICE}"
echo "Seed:            ${SEED}"
echo "Hidden dim:      ${HIDDEN_DIM}"
echo "Layers:          ${N_LAYERS}"
echo "Learning rate:   ${LR}"
echo "lambda_ot:       ${LAMBDA_OT}"
echo "lambda_mass:     ${LAMBDA_MASS}"
echo "lambda_energy:   ${LAMBDA_ENERGY}"
echo "lambda_density:  ${LAMBDA_DENSITY}"
echo "pretrain density:${PRETRAIN_LAMBDA_DENSITY}"
echo "global_mass:     ${GLOBAL_MASS}"

GLOBAL_MASS_ARGS=(--global-mass)
if [[ "${GLOBAL_MASS}" == "0" || "${GLOBAL_MASS}" == "false" || "${GLOBAL_MASS}" == "False" ]]; then
  GLOBAL_MASS_ARGS=(--no-global-mass)
fi

"${PYTHON}" model/CytoBridge/train_cytobridge_unbalanced.py \
  --input-h5ad "${INPUT_H5AD}" \
  --output-dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --n-layers "${N_LAYERS}" \
  --lr "${LR}" \
  --lambda-ot "${LAMBDA_OT}" \
  --lambda-mass "${LAMBDA_MASS}" \
  --lambda-energy "${LAMBDA_ENERGY}" \
  --lambda-density "${LAMBDA_DENSITY}" \
  --pretrain-lambda-density "${PRETRAIN_LAMBDA_DENSITY}" \
  --density-top-k "${DENSITY_TOP_K}" \
  --density-hinge-value "${DENSITY_HINGE_VALUE}" \
  "${GLOBAL_MASS_ARGS[@]}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --overwrite \
  > "${LOG_FILE}" 2>&1

echo "Done."
