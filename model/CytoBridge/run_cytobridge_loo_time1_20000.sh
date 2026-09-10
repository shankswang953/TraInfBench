#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
DATASET_NAME=${DATASET_NAME:-"moscot balanced LOO time1"}
DATASET=${DATASET:-data/moscot_rna_loo_time1_trajectorynet.npz}
INPUT_H5AD=${INPUT_H5AD:-data/moscot_rna_loo_time1_cytobridge.h5ad}
LOO_REFERENCE=${LOO_REFERENCE:-data/moscot_rna_loo_time1_reference.npz}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-0}

if [[ ! -s "${DATASET}" || ! -s "${INPUT_H5AD}" || ! -s "${LOO_REFERENCE}" ]]; then
  PREPARE_ARGS=()
  if [[ "${PREPARE_OVERWRITE}" == "1" ]]; then
    PREPARE_ARGS=(--overwrite)
  fi
  "${PYTHON}" common/prepare_loo_time1_inputs.py \
    --trajectorynet-output "${DATASET}" \
    --h5ad-output "${INPUT_H5AD}" \
    --reference-output "${LOO_REFERENCE}" \
    "${PREPARE_ARGS[@]}"
fi

export INPUT_H5AD
export DATASET_NAME
export EPOCHS=${EPOCHS:-20000}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export DEVICE=${DEVICE:-cpu}
export SEED=${SEED:-42}
export LAMBDA_OT=${LAMBDA_OT:-10.0}
export LAMBDA_MASS=${LAMBDA_MASS:-10.0}
export LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_moscot_loo_time1_20000_balanced}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec model/CytoBridge/run_cytobridge_20000.sh
