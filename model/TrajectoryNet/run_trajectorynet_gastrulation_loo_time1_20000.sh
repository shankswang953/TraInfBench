#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-data/gastrulation_rna_cytobridge.h5ad}
DATASET=${DATASET:-data/gastrulation_rna_loo_time1_trajectorynet.npz}
LOO_H5AD=${LOO_H5AD:-data/gastrulation_rna_loo_time1_cytobridge.h5ad}
LOO_REFERENCE=${LOO_REFERENCE:-data/gastrulation_rna_loo_time1_reference.npz}
HELDOUT_TIME=${HELDOUT_TIME:-1.0}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-0}

if [[ ! -s "${DATASET}" || ! -s "${LOO_H5AD}" || ! -s "${LOO_REFERENCE}" ]]; then
  PREPARE_ARGS=()
  if [[ "${PREPARE_OVERWRITE}" == "1" ]]; then
    PREPARE_ARGS=(--overwrite)
  fi
  "${PYTHON}" common/prepare_loo_time1_inputs.py \
    --input-h5ad "${INPUT_H5AD}" \
    --trajectorynet-output "${DATASET}" \
    --h5ad-output "${LOO_H5AD}" \
    --reference-output "${LOO_REFERENCE}" \
    --heldout-time "${HELDOUT_TIME}" \
    "${PREPARE_ARGS[@]}"
fi

export DATASET
export NITERS=${NITERS:-20000}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export SAVE_DIR=${SAVE_DIR:-results/trajectorynet_gastrulation_loo_time1_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec model/TrajectoryNet/run_trajectorynet_20000.sh
