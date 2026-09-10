#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mioflow-synthetic-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-8}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-data/synthetic_rna10_cytobridge.h5ad}
RNA_NORM=${RNA_NORM:-external/COATI/Synthetic/5scRNA/primal_norm_params_rna10_w2.pt}
EPOCHS=${EPOCHS:-3000}
SAMPLE_SIZE=${SAMPLE_SIZE:-128}
SEED=${SEED:-0}
GAGA_LATENT_DIM=${GAGA_LATENT_DIM:-10}
GAGA_MAX_CELLS=${GAGA_MAX_CELLS:-5000}
GAGA_BATCH_SIZE=${GAGA_BATCH_SIZE:-1024}
GAGA_ENCODER_EPOCHS=${GAGA_ENCODER_EPOCHS:-100}
GAGA_DECODER_EPOCHS=${GAGA_DECODER_EPOCHS:-100}
SAVE_DIR=${SAVE_DIR:-results/mioflow_synthetic_rna10_gaga10_n128_i3000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
ANALYSIS_DIR=${ANALYSIS_DIR:-results/mioflow_synthetic_rna10_gaga10_n128_i3000_analysis}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -s "$INPUT_H5AD" ]]; then
  echo "Input H5AD is missing or empty: $INPUT_H5AD"
  exit 1
fi
if [[ "$GAGA_LATENT_DIM" != "10" ]]; then
  echo "This synthetic benchmark is configured for GAGA_LATENT_DIM=10."
  exit 2
fi

if [[ -s "$SAVE_DIR/model.pt" && -s "$SAVE_DIR/gaga_model.pt" && -s "$LOG_FILE" && "$OVERWRITE" != "1" ]]; then
  echo "Reusing completed MIOFlow GAGA10 run: $SAVE_DIR"
else
  if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
    if [[ "$OVERWRITE" != "1" ]]; then
      echo "Refusing to overwrite partial MIOFlow output: $SAVE_DIR or $LOG_FILE"
      exit 1
    fi
    rm -rf "$SAVE_DIR" "$LOG_FILE"
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "Running MIOFlow synthetic RNA10: GAGA10 -> ODE -> decoder"
  echo "epochs=$EPOCHS, num_samples=$SAMPLE_SIZE, seed=$SEED"
  PYTHONUNBUFFERED=1 "$PYTHON" model/MIOFlow/train_mioflow_10000.py \
    --input-h5ad "$INPUT_H5AD" \
    --output-dir "$SAVE_DIR" \
    --latent-key X_pca_raw \
    --input-space moscot-normalized \
    --norm-params "$RNA_NORM" \
    --time-key time_point_processed \
    --time-axis rank \
    --epochs "$EPOCHS" \
    --sample-size "$SAMPLE_SIZE" \
    --hidden-dim 64 \
    --learning-rate 0.001 \
    --lambda-ot 1.0 \
    --lambda-energy 0.01 \
    --energy-time-steps 10 \
    --n-trajectories "$SAMPLE_SIZE" \
    --n-bins 61 \
    --loss-log-every 25 \
    --embedding-normalization identity \
    --use-gaga \
    --gaga-latent-dim "$GAGA_LATENT_DIM" \
    --gaga-hidden-dims 128,64 \
    --gaga-max-cells "$GAGA_MAX_CELLS" \
    --gaga-batch-size "$GAGA_BATCH_SIZE" \
    --gaga-encoder-epochs "$GAGA_ENCODER_EPOCHS" \
    --gaga-decoder-epochs "$GAGA_DECODER_EPOCHS" \
    --gaga-learning-rate 0.001 \
    --gaga-distance-components 10 \
    --gaga-phate-knn 5 \
    --gaga-phate-landmarks 2000 \
    --device cpu \
    --seed "$SEED" \
    --overwrite \
    > "$LOG_FILE" 2>&1
fi

if [[ -e "$ANALYSIS_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "Analysis already exists: $ANALYSIS_DIR"
else
  ANALYSIS_ARGS=()
  if [[ "$OVERWRITE" == "1" ]]; then
    ANALYSIS_ARGS=(--overwrite)
  fi
  "$PYTHON" model/MIOFlow/analyze_mioflow_synthetic.py \
    --input-h5ad "$INPUT_H5AD" \
    --model "$SAVE_DIR/model.pt" \
    --result-dir "$SAVE_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    --iterations "$EPOCHS" \
    --num-samples "$SAMPLE_SIZE" \
    "${ANALYSIS_ARGS[@]}"
fi

echo "MIOFlow GAGA10 synthetic run complete."
