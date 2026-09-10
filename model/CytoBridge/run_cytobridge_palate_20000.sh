#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export DATASET_NAME=${DATASET_NAME:-palate}
export INPUT_H5AD=${INPUT_H5AD:-data/palate_rna_cytobridge.h5ad}
export EPOCHS=${EPOCHS:-20000}
export SAVE_DIR=${SAVE_DIR:-results/cytobridge_palate_rna_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec bash model/CytoBridge/run_cytobridge_gastrulation_20000.sh "$@"
