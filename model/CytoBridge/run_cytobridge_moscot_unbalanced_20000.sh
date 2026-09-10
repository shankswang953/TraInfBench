#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Match the unbalanced CytoBridge configuration used for Gastrulation and Palate.
export DATASET_NAME=${DATASET_NAME:-moscot}
export INPUT_H5AD=${INPUT_H5AD:-data/moscot_rna_cytobridge.h5ad}
export EPOCHS=${EPOCHS:-20000}
export PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export DEVICE=${DEVICE:-cpu}
export SEED=${SEED:-42}
export LAMBDA_OT=${LAMBDA_OT:-10.0}
export LAMBDA_MASS=${LAMBDA_MASS:-10.0}
export LAMBDA_ENERGY=${LAMBDA_ENERGY:-0.01}
export GLOBAL_MASS=${GLOBAL_MASS:-1}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_moscot_rna_20000_unbalanced}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec bash model/CytoBridge/run_cytobridge_gastrulation_unbalanced_20000.sh "$@"
