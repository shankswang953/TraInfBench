#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export DATASET_NAME=${DATASET_NAME:-"palate LOO time1"}
export DATASET=${DATASET:-data/palate_rna_loo_time1_trajectorynet.npz}
export MAX_DIM=${MAX_DIM:-40}
export NITERS=${NITERS:-20000}
export SAVE_DIR=${SAVE_DIR:-results/trajectorynet_palate_loo_time1_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}

exec bash model/TrajectoryNet/run_trajectorynet_gastrulation_20000.sh "$@"
