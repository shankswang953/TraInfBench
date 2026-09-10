#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

AGE_INPUT=${1:?Usage: bash model/MIOFlow/run_mioflow_human_cerebral_7time_loo_full_gaga10_n256_30000.sh D7|D9|D11|D12|D18}
AGE=${AGE_INPUT#D}
AGE=${AGE#d}
case "$AGE" in
  7|9|11|12|18) ;;
  *)
    echo "Unsupported held-out age: $AGE_INPUT; use D7, D9, D11, D12, or D18"
    exit 2
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mioflow-human-cerebral-7time-loo-full-gaga-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-8}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
INPUT_H5AD=${INPUT_H5AD:-data/human_cerebral_7time_d4_d21_no_d16_rna_mioflow_strict_loo_D${AGE}.h5ad}
NORM_PARAMS=${NORM_PARAMS:-data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_primal_norm_params.pt}
FULL_RESULT_DIR=${FULL_RESULT_DIR:-results/mioflow_human_cerebral_7time_d4_d21_no_d16_full_official_gaga10_n256_30000}
GAGA_CHECKPOINT=${GAGA_CHECKPOINT:-${FULL_RESULT_DIR}/gaga_model.pt}
EMBEDDING_NORMALIZATION_CHECKPOINT=${EMBEDDING_NORMALIZATION_CHECKPOINT:-${FULL_RESULT_DIR}/model.pt}
EPOCHS=${EPOCHS:-30000}
SAMPLE_SIZE=${SAMPLE_SIZE:-256}
N_TRAJECTORIES=${N_TRAJECTORIES:-256}
N_BINS=${N_BINS:-101}
SEED=${SEED:-42}
GAGA_LATENT_DIM=${GAGA_LATENT_DIM:-10}
GAGA_HIDDEN_DIMS=${GAGA_HIDDEN_DIMS:-128,64}
DEVICE=${DEVICE:-cpu}
SAVE_DIR=${SAVE_DIR:-results/mioflow_human_cerebral_7time_d4_d21_no_d16_strict_loo_D${AGE}_fixed_full_gaga10_n256_${EPOCHS}}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ ! -s "$INPUT_H5AD" ]]; then
  "$PYTHON" model/TrajectoryNet/prepare_human_cerebral_7time_trajectorynet_mioflow_loo_inputs.py
fi
for path in "$INPUT_H5AD" "$NORM_PARAMS" "$GAGA_CHECKPOINT" "$EMBEDDING_NORMALIZATION_CHECKPOINT"; do
  if [[ ! -s "$path" ]]; then
    echo "Required LOO/Full-space dependency is missing or empty: $path"
    exit 1
  fi
done
if [[ "$GAGA_LATENT_DIM" != "10" ]]; then
  echo "This controlled benchmark requires GAGA_LATENT_DIM=10."
  exit 2
fi
"$PYTHON" model/TrajectoryNet/check_human_cerebral_7time_trajectorynet_mioflow_loo_setup.py \
  --age "$AGE" \
  --method mioflow \
  --full-gaga-checkpoint "$GAGA_CHECKPOINT" \
  --full-mioflow-checkpoint "$EMBEDDING_NORMALIZATION_CHECKPOINT"

echo "MIOFlow seven-time strict LOO-D${AGE} fixed-Full-GAGA10 configuration"
echo "  Full stages: D4,D7,D9,D11,D12,D18,D21"
echo "  held out only from vector-field training: D${AGE}"
echo "  source space: exact Full raw PCA30 subset / unchanged seven-time RNA scale"
echo "  frozen encoder, decoder, and GAGA input scaler: $GAGA_CHECKPOINT"
echo "  frozen Full GAGA10 feature-wise z-score: $EMBEDDING_NORMALIZATION_CHECKPOINT"
echo "  MIOFlow model space: exactly the Full z-scored GAGA10 space"
echo "  retained time axis: official consecutive ranks 0..5"
echo "  rank-grid bins: $N_BINS across five retained-rank intervals"
echo "  epochs: $EPOCHS"
echo "  sample size: $SAMPLE_SIZE"
echo "  generated trajectories: $N_TRAJECTORIES"
echo "  seed: $SEED"
echo "  output: $SAVE_DIR"
echo "  log: $LOG_FILE"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c "import mioflow; from mioflow import MIOFlow, Autoencoder; print('[OK] mioflow', getattr(mioflow, '__version__', 'unknown')); print('[OK]', MIOFlow.__name__, Autoencoder.__name__)"
  echo "Preflight passed; training was not started."
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
  --embedding-normalization-checkpoint "$EMBEDDING_NORMALIZATION_CHECKPOINT" \
  --use-gaga \
  --gaga-latent-dim "$GAGA_LATENT_DIM" \
  --gaga-hidden-dims "$GAGA_HIDDEN_DIMS" \
  --gaga-checkpoint "$GAGA_CHECKPOINT" \
  --device "$DEVICE" \
  --overwrite \
  > "$LOG_FILE" 2>&1

"$PYTHON" comparison/human_cerebral/annotate_mioflow_human_cerebral_7time_time_axis.py \
  --result-dir "$SAVE_DIR" \
  --heldout-age "$AGE" \
  --overwrite
echo "Training finished: $SAVE_DIR"
