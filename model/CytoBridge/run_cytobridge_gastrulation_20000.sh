#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-external/COATI_WORKSPACE/.cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-external/COATI_WORKSPACE/.cache/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-external/COATI_WORKSPACE/.cache/numba}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}

EPOCHS=${EPOCHS:-20000}
BATCH_SIZE=${BATCH_SIZE:-1024}
DATASET_NAME=${DATASET_NAME:-gastrulation}
INPUT_H5AD=${INPUT_H5AD:-data/gastrulation_rna_cytobridge.h5ad}
SAVE_DIR=${SAVE_DIR:-results/cytobridge_gastrulation_rna_20000}
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

echo "Running CytoBridge ${DATASET_NAME} for ${EPOCHS} epochs"
echo "Input:  ${INPUT_H5AD}"
echo "Output: ${SAVE_DIR}"
echo "Log:    ${LOG_FILE}"

"${PYTHON}" model/CytoBridge/train_cytobridge_20000.py \
  --input-h5ad "${INPUT_H5AD}" \
  --output-dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cpu \
  > "${LOG_FILE}" 2>&1

echo "Done."
