#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
SOURCE_H5AD=${SOURCE_H5AD:-data/moscot_rna_cytobridge.h5ad}
DATASET=${DATASET:-data/moscot_rna_loo_time1_trajectorynet.npz}
INPUT_H5AD=${INPUT_H5AD:-data/moscot_rna_loo_time1_cytobridge.h5ad}
LOO_REFERENCE=${LOO_REFERENCE:-data/moscot_rna_loo_time1_reference.npz}
HELDOUT_TIME=${HELDOUT_TIME:-1.0}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-0}

if [[ ! -s "${DATASET}" || ! -s "${INPUT_H5AD}" || ! -s "${LOO_REFERENCE}" ]]; then
  PREPARE_ARGS=()
  if [[ "${PREPARE_OVERWRITE}" == "1" ]]; then
    PREPARE_ARGS=(--overwrite)
  fi
  "${PYTHON}" common/prepare_loo_time1_inputs.py \
    --input-h5ad "${SOURCE_H5AD}" \
    --trajectorynet-output "${DATASET}" \
    --h5ad-output "${INPUT_H5AD}" \
    --reference-output "${LOO_REFERENCE}" \
    --heldout-time "${HELDOUT_TIME}" \
    "${PREPARE_ARGS[@]}"
fi

# Keep every training option identical to the moscot full-data run.
export PYTHON INPUT_H5AD
export DATASET_NAME=${DATASET_NAME:-"moscot LOO time1"}
export EPOCHS=${EPOCHS:-20000}
export PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export DEVICE=${DEVICE:-cpu}
export SEED=${SEED:-42}
export LAMBDA_OT=${LAMBDA_OT:-10.0}
export LAMBDA_MASS=${LAMBDA_MASS:-10.0}
export LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
export GLOBAL_MASS=${GLOBAL_MASS:-1}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_moscot_loo_time1_20000_unbalanced}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec bash model/CytoBridge/run_cytobridge_gastrulation_unbalanced_20000.sh "$@"
