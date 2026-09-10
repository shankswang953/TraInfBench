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

NITERS=${NITERS:-20000}
BATCH_SIZE=${BATCH_SIZE:-1024}
VAL_FREQ=${VAL_FREQ:-1000}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-10}
MAX_DIM=${MAX_DIM:-50}
DATASET_NAME=${DATASET_NAME:-gastrulation}
DATASET=${DATASET:-data/gastrulation_rna_trajectorynet.npz}
SAVE_DIR=${SAVE_DIR:-results/trajectorynet_gastrulation_rna_20000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PLOT_AFTER=${PLOT_AFTER:-1}

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

echo "Running TrajectoryNet ${DATASET_NAME} for ${NITERS} iterations"
echo "Input:  ${DATASET}"
echo "Output: ${SAVE_DIR}"
echo "Log:    ${LOG_FILE}"

"${PYTHON}" -m TrajectoryNet.main \
  --dataset "${DATASET}" \
  --embedding_name pca \
  --max_dim "${MAX_DIM}" \
  --use_cpu \
  --niters "${NITERS}" \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size "${BATCH_SIZE}" \
  --viz_batch_size "${BATCH_SIZE}" \
  --save "${SAVE_DIR}" \
  --viz_freq 1000000 \
  --val_freq "${VAL_FREQ}" \
  --save_freq "${SAVE_FREQ}" \
  --log_freq "${LOG_FREQ}" \
  --solver rk4 \
  --test_solver rk4 \
  --step_size 0.1 \
  --divergence_fn approximate \
  > "${LOG_FILE}" 2>&1

echo "Training finished."

if [[ "${PLOT_AFTER}" == "1" ]]; then
  "${PYTHON}" common/plot_trajectorynet_loss.py --result-dir "${SAVE_DIR}"
fi

echo "Done."
