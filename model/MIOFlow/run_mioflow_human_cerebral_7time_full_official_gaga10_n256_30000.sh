#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mioflow-human-cerebral-7time-official-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-8}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-data/human_cerebral_7time_d4_d21_no_d16_rna_mioflow.h5ad}
NORM_PARAMS=${NORM_PARAMS:-data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_primal_norm_params.pt}
EPOCHS=${EPOCHS:-30000}
SAMPLE_SIZE=${SAMPLE_SIZE:-256}
N_TRAJECTORIES=${N_TRAJECTORIES:-256}
N_BINS=${N_BINS:-121}
SEED=${SEED:-42}
GAGA_LATENT_DIM=${GAGA_LATENT_DIM:-10}
GAGA_HIDDEN_DIMS=${GAGA_HIDDEN_DIMS:-128,64}
GAGA_MAX_CELLS=${GAGA_MAX_CELLS:-5000}
GAGA_BATCH_SIZE=${GAGA_BATCH_SIZE:-1024}
GAGA_ENCODER_EPOCHS=${GAGA_ENCODER_EPOCHS:-100}
GAGA_DECODER_EPOCHS=${GAGA_DECODER_EPOCHS:-100}
GAGA_LEARNING_RATE=${GAGA_LEARNING_RATE:-0.001}
GAGA_DISTANCE_COMPONENTS=${GAGA_DISTANCE_COMPONENTS:-10}
GAGA_PHATE_KNN=${GAGA_PHATE_KNN:-5}
GAGA_PHATE_LANDMARKS=${GAGA_PHATE_LANDMARKS:-2000}
DEVICE=${DEVICE:-cpu}
SAVE_DIR=${SAVE_DIR:-results/mioflow_human_cerebral_7time_d4_d21_no_d16_full_official_gaga10_n256_${EPOCHS}}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ ! -s "$INPUT_H5AD" || ! -s "$NORM_PARAMS" ]]; then
  echo "Prepared MIOFlow input or seven-time normalization is missing."
  echo "Run model/TrajectoryNet/prepare_human_cerebral_7time_trajectorynet_mioflow_inputs.py first."
  exit 1
fi
if [[ "$GAGA_LATENT_DIM" != "10" ]]; then
  echo "This controlled benchmark requires GAGA_LATENT_DIM=10."
  exit 2
fi
"$PYTHON" model/TrajectoryNet/check_human_cerebral_7time_trajectorynet_mioflow_setup.py

echo "MIOFlow official GAGA10 seven-time full configuration"
echo "  observed stages: D4,D7,D9,D11,D12,D18,D21 (D16 excluded)"
echo "  source/model bridge: raw PCA30 / seven-time scale -> normalized PCA30"
echo "  GAGA: StandardScaler + PHATE distances + 10D two-phase autoencoder"
echo "  GAGA epochs: encoder=$GAGA_ENCODER_EPOCHS decoder=$GAGA_DECODER_EPOCHS"
echo "  MIOFlow model space: feature-wise z-scored GAGA10"
echo "  MIOFlow time axis: official consecutive ranks 0..6"
echo "  physical-time metadata: 0,0.3,0.5,0.7,0.8,1.4,1.7"
echo "  rank-grid bins: $N_BINS (20 subdivisions per adjacent rank)"
echo "  MIOFlow epochs: $EPOCHS"
echo "  sample size: $SAMPLE_SIZE"
echo "  generated trajectories: $N_TRAJECTORIES"
echo "  seed: $SEED"
echo "  output: $SAVE_DIR"
echo "  log: $LOG_FILE"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c "import inspect, mioflow; from mioflow import MIOFlow, Autoencoder, train_gaga_two_phase; print('[OK] mioflow', getattr(mioflow, '__version__', '0.1.14')); print('[OK] MIOFlow:', inspect.getfile(MIOFlow)); print('[OK] official GAGA API:', Autoencoder.__name__, train_gaga_two_phase.__name__)"
  echo "Preflight passed; GAGA and MIOFlow training were not started."
  exit 0
fi

if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  $SAVE_DIR"
    echo "  $LOG_FILE"
    exit 1
  fi
  rm -rf "$SAVE_DIR"
  rm -f "$LOG_FILE"
fi

"$PYTHON" model/MIOFlow/train_mioflow_10000.py \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$SAVE_DIR" \
  --latent-key X_latent \
  --require-shared-pca \
  --input-space moscot-normalized \
  --norm-params "$NORM_PARAMS" \
  --time-key time_point_processed \
  --time-axis rank \
  --epochs "$EPOCHS" \
  --sample-size "$SAMPLE_SIZE" \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --lambda-ot 1.0 \
  --lambda-energy 0.01 \
  --energy-time-steps 10 \
  --n-trajectories "$N_TRAJECTORIES" \
  --n-bins "$N_BINS" \
  --loss-log-every 10 \
  --seed "$SEED" \
  --embedding-normalization zscore \
  --use-gaga \
  --gaga-latent-dim "$GAGA_LATENT_DIM" \
  --gaga-hidden-dims "$GAGA_HIDDEN_DIMS" \
  --gaga-max-cells "$GAGA_MAX_CELLS" \
  --gaga-batch-size "$GAGA_BATCH_SIZE" \
  --gaga-encoder-epochs "$GAGA_ENCODER_EPOCHS" \
  --gaga-decoder-epochs "$GAGA_DECODER_EPOCHS" \
  --gaga-learning-rate "$GAGA_LEARNING_RATE" \
  --gaga-distance-components "$GAGA_DISTANCE_COMPONENTS" \
  --gaga-phate-knn "$GAGA_PHATE_KNN" \
  --gaga-phate-landmarks "$GAGA_PHATE_LANDMARKS" \
  --device "$DEVICE" \
  --overwrite \
  > "$LOG_FILE" 2>&1

"$PYTHON" comparison/human_cerebral/annotate_mioflow_human_cerebral_7time_time_axis.py \
  --result-dir "$SAVE_DIR" \
  --overwrite
echo "Training finished: $SAVE_DIR"
