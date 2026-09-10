#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export DATASET_NAME=${DATASET_NAME:-"palate LOO time2"}
export INPUT_H5AD=${INPUT_H5AD:-data/palate_rna_loo_time2_cytobridge.h5ad}
export EPOCHS=${EPOCHS:-20000}
export BATCH_SIZE=${BATCH_SIZE:-1024}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_palate_loo_time2_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec bash model/CytoBridge/run_cytobridge_gastrulation_20000.sh "$@"
