#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET=${1:?Usage: bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh DATASET TASK}
TASK=${2:?Usage: bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh DATASET TASK}
case "$DATASET" in
  moscot|gastrulation|palate|human_cerebral_no_d61) ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 2
    ;;
esac
case "$TASK" in
  full|loo_time1|loo_time2|loo_day7) ;;
  *)
    echo "Unknown task: $TASK"
    exit 2
    ;;
esac
if [[ "$DATASET" == "moscot" && "$TASK" == "loo_time2" ]]; then
  echo "moscot has only three time points; configured tasks are full and loo_time1."
  exit 2
fi
if [[ "$DATASET" == "human_cerebral_no_d61" && "$TASK" != "full" && "$TASK" != "loo_day7" ]]; then
  echo "human_cerebral_no_d61 currently supports full and loo_day7."
  exit 2
fi
if [[ "$DATASET" != "human_cerebral_no_d61" && "$TASK" == "loo_day7" ]]; then
  echo "loo_day7 is configured only for human_cerebral_no_d61."
  exit 2
fi

case "$TASK" in
  full)
    DEFAULT_INPUT_H5AD="data/${DATASET}_rna_cytobridge.h5ad"
    ;;
  loo_time1)
    DEFAULT_INPUT_H5AD="data/${DATASET}_rna_loo_time1_cytobridge.h5ad"
    ;;
  loo_time2)
    DEFAULT_INPUT_H5AD="data/${DATASET}_rna_loo_time2_cytobridge.h5ad"
    ;;
  loo_day7)
    DEFAULT_INPUT_H5AD="data/${DATASET}_rna_loo_day7_cytobridge.h5ad"
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trainfbench-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-8}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-$DEFAULT_INPUT_H5AD}
LATENT_KEY=${LATENT_KEY:-X_latent}
EPOCHS=${EPOCHS:-20000}
if [[ "$DATASET" == "human_cerebral_no_d61" ]]; then
  DEFAULT_SAMPLE_SIZE=256
else
  DEFAULT_SAMPLE_SIZE=1024
fi
SAMPLE_SIZE=${SAMPLE_SIZE:-$DEFAULT_SAMPLE_SIZE}
LOSS_LOG_EVERY=${LOSS_LOG_EVERY:-10}
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
GAGA_CHECKPOINT=${GAGA_CHECKPOINT:-}
EMBEDDING_NORMALIZATION=${EMBEDDING_NORMALIZATION:-identity}
EMBEDDING_NORMALIZATION_CHECKPOINT=${EMBEDDING_NORMALIZATION_CHECKPOINT:-}
DEVICE=${DEVICE:-cpu}
if [[ "$DATASET" == "human_cerebral_no_d61" ]]; then
  DEFAULT_N_BINS=109
  DEFAULT_TIME_AXIS=numeric
else
  DEFAULT_N_BINS=101
  DEFAULT_TIME_AXIS=rank
fi
N_BINS=${N_BINS:-$DEFAULT_N_BINS}
TIME_AXIS=${TIME_AXIS:-$DEFAULT_TIME_AXIS}
N_TRAJECTORIES=${N_TRAJECTORIES:-1000}
SAVE_DIR=${SAVE_DIR:-results/mioflow_${DATASET}_${TASK}_pca_gaga10_n${SAMPLE_SIZE}_20000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -s "$INPUT_H5AD" ]]; then
  echo "Input H5AD is missing or empty: $INPUT_H5AD"
  exit 1
fi
if [[ "$GAGA_LATENT_DIM" != "10" ]]; then
  echo "This controlled benchmark requires GAGA_LATENT_DIM=10."
  exit 2
fi
GAGA_CHECKPOINT_ARGS=()
if [[ -n "$GAGA_CHECKPOINT" ]]; then
  if [[ ! -s "$GAGA_CHECKPOINT" ]]; then
    echo "Frozen GAGA checkpoint is missing or empty: $GAGA_CHECKPOINT"
    exit 1
  fi
  GAGA_CHECKPOINT_ARGS=(--gaga-checkpoint "$GAGA_CHECKPOINT")
fi
EMBEDDING_NORMALIZATION_CHECKPOINT_ARGS=()
if [[ -n "$EMBEDDING_NORMALIZATION_CHECKPOINT" ]]; then
  if [[ ! -s "$EMBEDDING_NORMALIZATION_CHECKPOINT" ]]; then
    echo "Frozen embedding-normalization checkpoint is missing or empty: $EMBEDDING_NORMALIZATION_CHECKPOINT"
    exit 1
  fi
  if [[ "$EMBEDDING_NORMALIZATION" != "zscore" ]]; then
    echo "Frozen embedding normalization requires EMBEDDING_NORMALIZATION=zscore."
    exit 2
  fi
  EMBEDDING_NORMALIZATION_CHECKPOINT_ARGS=(
    --embedding-normalization-checkpoint "$EMBEDDING_NORMALIZATION_CHECKPOINT"
  )
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

echo "Running MIOFlow ${DATASET}/${TASK}: shared PCA -> GAGA10 -> MIOFlow"
echo "Input:       $INPUT_H5AD"
echo "PCA key:     $LATENT_KEY"
echo "Epochs:      $EPOCHS"
echo "Sample size: $SAMPLE_SIZE"
echo "Time axis:   $TIME_AXIS"
echo "Embedding normalization: $EMBEDDING_NORMALIZATION"
echo "Frozen embedding normalization: ${EMBEDDING_NORMALIZATION_CHECKPOINT:-fit in this job}"
echo "Frozen GAGA checkpoint: ${GAGA_CHECKPOINT:-fit in this job}"
echo "Trajectories: $N_TRAJECTORIES"
echo "Seed:         $SEED"
echo "Output:      $SAVE_DIR"
echo "Log:         $LOG_FILE"

"$PYTHON" model/MIOFlow/train_mioflow_10000.py \
  --input-h5ad "$INPUT_H5AD" \
  --output-dir "$SAVE_DIR" \
  --latent-key "$LATENT_KEY" \
  --require-shared-pca \
  --input-space obsm \
  --time-key time_point_processed \
  --time-axis "$TIME_AXIS" \
  --epochs "$EPOCHS" \
  --sample-size "$SAMPLE_SIZE" \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --lambda-ot 1.0 \
  --lambda-energy 0.01 \
  --energy-time-steps 10 \
  --n-trajectories "$N_TRAJECTORIES" \
  --n-bins "$N_BINS" \
  --loss-log-every "$LOSS_LOG_EVERY" \
  --seed "$SEED" \
  --embedding-normalization "$EMBEDDING_NORMALIZATION" \
  "${EMBEDDING_NORMALIZATION_CHECKPOINT_ARGS[@]}" \
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
  "${GAGA_CHECKPOINT_ARGS[@]}" \
  --device "$DEVICE" \
  --overwrite \
  > "$LOG_FILE" 2>&1

echo "Done: $SAVE_DIR"
echo "Decoded shared-PCA trajectories: $SAVE_DIR/trajectories.npz::trajectories_pca"
