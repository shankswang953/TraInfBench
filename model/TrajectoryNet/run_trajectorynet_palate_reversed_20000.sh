#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

USAGE="Usage: bash model/TrajectoryNet/run_trajectorynet_palate_reversed_20000.sh [full|loo1|loo2]"
TASK_INPUT=${1:?$USAGE}
case "$TASK_INPUT" in
  full)
    TASK=full
    DATASET=data/palate_rna_trajectorynet_reversed.npz
    SAVE_DIR_DEFAULT=results/trajectorynet_palate_full_reversed_20000
    ORDER="Gaussian -> E14.5 -> E14.0 -> E13.5 -> E12.5"
    ;;
  loo1|loo_time1)
    TASK=loo_time1
    DATASET=data/palate_rna_loo_time1_trajectorynet_reversed.npz
    SAVE_DIR_DEFAULT=results/trajectorynet_palate_loo_time1_reversed_20000
    ORDER="Gaussian -> E14.5 -> E14.0 -> E12.5 (E13.5 held out)"
    ;;
  loo2|loo_time2)
    TASK=loo_time2
    DATASET=data/palate_rna_loo_time2_trajectorynet_reversed.npz
    SAVE_DIR_DEFAULT=results/trajectorynet_palate_loo_time2_reversed_20000
    ORDER="Gaussian -> E14.5 -> E13.5 -> E12.5 (E14.0 held out)"
    ;;
  *)
    echo "Unknown palate task: $TASK_INPUT (expected full, loo1, or loo2)"
    exit 2
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trajectorynet-palate-reversed-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
NITERS=${NITERS:-20000}
BATCH_SIZE=${BATCH_SIZE:-1024}
VAL_FREQ=${VAL_FREQ:-1000}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-10}
SEED=${SEED:-0}
SAVE_DIR=${SAVE_DIR:-$SAVE_DIR_DEFAULT}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PLOT_AFTER=${PLOT_AFTER:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ ! -s "$DATASET" ]]; then
  "$PYTHON" model/TrajectoryNet/prepare_palate_trajectorynet_reversed_input.py "$TASK" \
    --output-npz "$DATASET"
fi
"$PYTHON" model/TrajectoryNet/prepare_palate_trajectorynet_reversed_input.py "$TASK" \
  --output-npz "$DATASET" \
  --check-only

echo "TrajectoryNet Palate reversed-rank configuration"
echo "  task: $TASK"
echo "  reverse=True sampling: $ORDER"
echo "  reverse=False evaluation: observed E12.5 -> later biological stages"
echo "  input: $DATASET"
echo "  iterations: $NITERS"
echo "  batch size: $BATCH_SIZE"
echo "  seed: $SEED"
echo "  output: $SAVE_DIR"
echo "  log: $LOG_FILE"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c \
    "import inspect; from TrajectoryNet.main import main; from TrajectoryNet.dataset import CustomData; print('[OK] TrajectoryNet main:', inspect.getfile(main)); print('[OK] CustomData:', inspect.getfile(CustomData))"
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
mkdir -p "$(dirname "$LOG_FILE")"

PYTHONUNBUFFERED=1 "$PYTHON" model/TrajectoryNet/run_trajectorynet_seeded.py \
  --dataset "$DATASET" \
  --embedding_name pca \
  --max_dim 40 \
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

if [[ "$PLOT_AFTER" == "1" ]]; then
  "$PYTHON" common/plot_trajectorynet_loss.py --result-dir "$SAVE_DIR"
fi
echo "Training finished: $SAVE_DIR"
