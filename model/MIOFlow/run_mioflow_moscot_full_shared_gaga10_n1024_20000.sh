#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mioflow-moscot-shared-gaga-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-8}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
export INPUT_H5AD=${INPUT_H5AD:-data/moscot_rna_cytobridge.h5ad}
export SAVE_DIR=${SAVE_DIR:-results/mioflow_moscot_full_shared_gaga10_n1024_20000}
export LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
export EPOCHS=${EPOCHS:-20000}
export SAMPLE_SIZE=${SAMPLE_SIZE:-1024}
export N_TRAJECTORIES=${N_TRAJECTORIES:-1000}
export N_BINS=${N_BINS:-101}
export SEED=${SEED:-42}
export DEVICE=${DEVICE:-cpu}
export OVERWRITE=${OVERWRITE:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

export GAGA_CHECKPOINT=
export EMBEDDING_NORMALIZATION_CHECKPOINT=
export EMBEDDING_NORMALIZATION=zscore
export TIME_AXIS=rank
export GAGA_LATENT_DIM=10
export GAGA_HIDDEN_DIMS=${GAGA_HIDDEN_DIMS:-128,64}
export GAGA_MAX_CELLS=${GAGA_MAX_CELLS:-5000}
export GAGA_BATCH_SIZE=${GAGA_BATCH_SIZE:-1024}
export GAGA_ENCODER_EPOCHS=${GAGA_ENCODER_EPOCHS:-100}
export GAGA_DECODER_EPOCHS=${GAGA_DECODER_EPOCHS:-100}
export GAGA_LEARNING_RATE=${GAGA_LEARNING_RATE:-0.001}
export GAGA_DISTANCE_COMPONENTS=${GAGA_DISTANCE_COMPONENTS:-10}
export GAGA_PHATE_KNN=${GAGA_PHATE_KNN:-5}
export GAGA_PHATE_LANDMARKS=${GAGA_PHATE_LANDMARKS:-2000}

echo "MIOFlow pancreas Full: fit the shared GAGA10 representation once"
echo "  Full input: $INPUT_H5AD"
echo "  GAGA input scaler: fitted on every Full-data cell"
echo "  GAGA distance fit: time-stratified sample of $GAGA_MAX_CELLS Full-data cells"
echo "  GAGA latent normalization: Full-data feature-wise zscore"
echo "  MIOFlow epochs: $EPOCHS"
echo "  output: $SAVE_DIR"

"$PYTHON" model/MIOFlow/check_mioflow_moscot_shared_full_gaga10.py \
  --full-h5ad "$INPUT_H5AD"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c \
    "import inspect; from mioflow import MIOFlow, Autoencoder, train_gaga_two_phase; print('[OK] MIOFlow:', inspect.getfile(MIOFlow)); print('[OK] GAGA:', Autoencoder.__name__, train_gaga_two_phase.__name__)"
  echo "Preflight passed; Full training was not started."
  exit 0
fi

exec bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh moscot full
