#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
SOURCE_H5AD=${SOURCE_H5AD:-data/gastrulation_rna_cytobridge.h5ad}
DATASET=${DATASET:-data/gastrulation_rna_loo_time2_trajectorynet.npz}
INPUT_H5AD=${INPUT_H5AD:-data/gastrulation_rna_loo_time2_cytobridge.h5ad}
LOO_REFERENCE=${LOO_REFERENCE:-data/gastrulation_rna_loo_time2_reference.npz}
HELDOUT_TIME=${HELDOUT_TIME:-2.0}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-0}
TRAJECTORYNET_TIME_ENCODING=${TRAJECTORYNET_TIME_ENCODING:-rank}

if [[ ! -s "${DATASET}" || ! -s "${INPUT_H5AD}" || ! -s "${LOO_REFERENCE}" ]]; then
  PREPARE_ARGS=()
  if [[ "${PREPARE_OVERWRITE}" == "1" ]]; then
    PREPARE_ARGS=(--overwrite)
  fi
  "${PYTHON}" common/prepare_loo_inputs.py \
    --input-h5ad "${SOURCE_H5AD}" \
    --trajectorynet-output "${DATASET}" \
    --h5ad-output "${INPUT_H5AD}" \
    --reference-output "${LOO_REFERENCE}" \
    --heldout-time "${HELDOUT_TIME}" \
    --trajectorynet-time-encoding "${TRAJECTORYNET_TIME_ENCODING}" \
    "${PREPARE_ARGS[@]}"
fi

export INPUT_H5AD
export EPOCHS=${EPOCHS:-20000}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_gastrulation_loo_time2_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec model/CytoBridge/run_cytobridge_20000.sh
