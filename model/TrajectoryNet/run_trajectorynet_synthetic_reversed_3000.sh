#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trajectorynet-synthetic-reversed-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
NITERS=${NITERS:-3000}
BATCH_SIZE=${BATCH_SIZE:-128}
VAL_FREQ=${VAL_FREQ:-500}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-25}
SEED=${SEED:-0}
DATASET=${DATASET:-data/synthetic_rna10_trajectorynet_reversed.npz}
SAVE_DIR=${SAVE_DIR:-results/trajectorynet_synthetic_rna10_reversed_n128_i3000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
ANALYSIS_DIR=${ANALYSIS_DIR:-results/trajectorynet_synthetic_rna10_reversed_n128_i3000_analysis}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -e "$DATASET" ]]; then
  "$PYTHON" model/TrajectoryNet/prepare_synthetic_trajectorynet_reversed_input.py \
    --output-npz "$DATASET"
fi

if [[ -e "$SAVE_DIR/checkpt-3000.pth" && -e "$LOG_FILE" && "$OVERWRITE" != "1" ]]; then
  echo "Reusing completed reversed-time TrajectoryNet run: $SAVE_DIR"
else
  if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
    if [[ "$OVERWRITE" != "1" ]]; then
      echo "Refusing to overwrite partial reversed-time output: $SAVE_DIR or $LOG_FILE"
      exit 1
    fi
    rm -rf "$SAVE_DIR" "$LOG_FILE"
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "Running reversed-time TrajectoryNet synthetic RNA10: iterations=$NITERS, num_samples=$BATCH_SIZE, seed=$SEED"
  PYTHONUNBUFFERED=1 "$PYTHON" model/TrajectoryNet/run_trajectorynet_seeded.py \
    --dataset "$DATASET" \
    --embedding_name pca \
    --max_dim 10 \
    --use_cpu \
    --seed "$SEED" \
    --niters "$NITERS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$BATCH_SIZE" \
    --viz_batch_size "$BATCH_SIZE" \
    --save "$SAVE_DIR" \
    --viz_freq 1000000 \
    --val_freq "$VAL_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --log_freq "$LOG_FREQ" \
    --solver rk4 \
    --test_solver rk4 \
    --step_size 0.1 \
    --divergence_fn approximate \
    > "$LOG_FILE" 2>&1
fi

"$PYTHON" common/plot_trajectorynet_loss.py --result-dir "$SAVE_DIR"

if [[ -e "$ANALYSIS_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "Analysis already exists: $ANALYSIS_DIR"
else
  ANALYSIS_ARGS=()
  if [[ "$OVERWRITE" == "1" ]]; then
    ANALYSIS_ARGS=(--overwrite)
  fi
  "$PYTHON" model/TrajectoryNet/analyze_trajectorynet_synthetic_reversed.py \
    --dataset "$DATASET" \
    --checkpoint "$SAVE_DIR/checkpt-3000.pth" \
    --result-dir "$SAVE_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    "${ANALYSIS_ARGS[@]}"
fi

echo "Reversed-time TrajectoryNet synthetic run complete."
