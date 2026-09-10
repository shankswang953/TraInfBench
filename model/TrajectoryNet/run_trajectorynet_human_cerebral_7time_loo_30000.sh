#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

AGE_INPUT=${1:?Usage: bash model/TrajectoryNet/run_trajectorynet_human_cerebral_7time_loo_30000.sh D7|D9|D11|D12|D18}
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
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trajectorynet-human-cerebral-7time-loo-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
NITERS=${NITERS:-30000}
BATCH_SIZE=${BATCH_SIZE:-256}
VAL_FREQ=${VAL_FREQ:-1000}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-10}
SEED=${SEED:-0}
DATASET=${DATASET:-data/human_cerebral_7time_d4_d21_no_d16_rna_trajectorynet_strict_loo_D${AGE}_forward.npz}
SAVE_DIR=${SAVE_DIR:-results/trajectorynet_human_cerebral_7time_d4_d21_no_d16_strict_loo_D${AGE}_forward_${NITERS}}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PLOT_AFTER=${PLOT_AFTER:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ ! -s "$DATASET" ]]; then
  "$PYTHON" model/TrajectoryNet/prepare_human_cerebral_7time_trajectorynet_mioflow_loo_inputs.py
fi
"$PYTHON" model/TrajectoryNet/check_human_cerebral_7time_trajectorynet_mioflow_loo_setup.py \
  --age "$AGE" \
  --method trajectorynet

echo "TrajectoryNet seven-time strict LOO-D${AGE} normal-direction configuration"
echo "  Full stages: D4,D7,D9,D11,D12,D18,D21"
echo "  held out from training: D${AGE}"
echo "  model space: exact subset of Full seven-time normalized RNA PCA30"
echo "  retained time axis: six consecutive forward ranks 0..5"
echo "  internal time scale: 0.5 per retained-rank interval"
echo "  input: $DATASET"
echo "  iterations: $NITERS"
echo "  batch size: $BATCH_SIZE"
echo "  seed: $SEED"
echo "  output: $SAVE_DIR"
echo "  log: $LOG_FILE"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c "from TrajectoryNet.main import main; from TrajectoryNet.parse import parser; print('[OK] TrajectoryNet imports')"
  echo "Preflight passed; training was not started."
  exit 0
fi

if [[ -e "$SAVE_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  $SAVE_DIR"
    echo "  $LOG_FILE"
    echo "Set OVERWRITE=1 only after inspecting the existing run."
    exit 1
  fi
  rm -rf "$SAVE_DIR"
  rm -f "$LOG_FILE"
fi
mkdir -p "$(dirname "$LOG_FILE")"

PYTHONUNBUFFERED=1 "$PYTHON" model/TrajectoryNet/run_trajectorynet_seeded.py \
  --dataset "$DATASET" \
  --embedding_name pca \
  --max_dim 30 \
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
  --time_scale 0.5 \
  --solver rk4 \
  --test_solver rk4 \
  --step_size 0.1 \
  --divergence_fn approximate \
  > "$LOG_FILE" 2>&1

if [[ "$PLOT_AFTER" == "1" ]]; then
  "$PYTHON" common/plot_trajectorynet_loss.py --result-dir "$SAVE_DIR"
fi
echo "Training finished: $SAVE_DIR"
