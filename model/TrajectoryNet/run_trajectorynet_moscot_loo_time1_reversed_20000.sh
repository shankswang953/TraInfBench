#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trajectorynet-moscot-reversed-cache}
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
DATASET=${DATASET:-data/moscot_rna_loo_time1_trajectorynet_reversed.npz}
SAVE_DIR=${SAVE_DIR:-results/trajectorynet_moscot_loo_time1_reversed_${NITERS}}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
OVERWRITE=${OVERWRITE:-0}
PLOT_AFTER=${PLOT_AFTER:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ ! -s "$DATASET" ]]; then
  "$PYTHON" model/TrajectoryNet/prepare_trajectorynet_moscot_loo_time1_reversed_input.py \
    --output-npz "$DATASET"
fi
"$PYTHON" model/TrajectoryNet/prepare_trajectorynet_moscot_loo_time1_reversed_input.py \
  --output-npz "$DATASET" \
  --check-only

echo "TrajectoryNet reversed-rank pancreas strict-LOO configuration"
echo "  retained stages: E14.5(rank 1), E16.5(rank 0)"
echo "  held out: E15.5(reversed rank 0.5)"
echo "  Gaussian generative order: E16.5 -> E14.5"
echo "  native density/inverse order: exact E14.5 -> E16.5"
echo "  training iterations: $NITERS"
echo "  training batch size: $BATCH_SIZE (matches COATI num_samples)"
echo "  exported initial cells: 9029 (all E14.5 cells; no sampling)"
echo "  seed: $SEED"
echo "  input: $DATASET"
echo "  output: $SAVE_DIR"
echo "  log: $LOG_FILE"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" -c \
    "import inspect; from TrajectoryNet.main import main; from TrajectoryNet.dataset import CustomData; print('[OK] TrajectoryNet main:', inspect.getfile(main)); print('[OK] CustomData:', inspect.getfile(CustomData))"
  echo "Preflight passed; 20k training was not started."
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
  --max_dim 50 \
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
