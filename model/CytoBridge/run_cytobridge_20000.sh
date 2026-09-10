#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=TRUE
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib-cache

PYTHON=${PYTHON:-python}

EPOCHS=${EPOCHS:-50000}
BATCH_SIZE=${BATCH_SIZE:-1024}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
LAMBDA_OT=${LAMBDA_OT:-10.0}
LAMBDA_MASS=${LAMBDA_MASS:-10.0}
LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
INPUT_H5AD=${INPUT_H5AD:-data/moscot_rna_cytobridge.h5ad}
SAVE_DIR=${SAVE_DIR:-results/cytobridge_moscot_rna_50000}
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

echo "Running CytoBridge for ${EPOCHS} epochs with batch size ${BATCH_SIZE}"
echo "Input:  ${INPUT_H5AD}"
echo "Output: ${SAVE_DIR}"
echo "Log:    ${LOG_FILE}"
echo "Device: ${DEVICE}; seed: ${SEED}"
echo "Loss:   OT=${LAMBDA_OT}; mass=${LAMBDA_MASS}; energy=${LAMBDA_ENERGY}"

"${PYTHON}" model/CytoBridge/train_cytobridge_20000.py \
  --input-h5ad "${INPUT_H5AD}" \
  --output-dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lambda-ot "${LAMBDA_OT}" \
  --lambda-mass "${LAMBDA_MASS}" \
  --lambda-energy "${LAMBDA_ENERGY}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1

echo "Done."
