#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET=${1:?Usage: bash model/TIGON/run_tigon_official_ae10_dopri5_exact_n1024_20000.sh DATASET TASK}
TASK=${2:?Usage: bash model/TIGON/run_tigon_official_ae10_dopri5_exact_n1024_20000.sh DATASET TASK}
case "$DATASET" in
  moscot|palate|gastrulation) ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 2
    ;;
esac
if [[ "$DATASET" == "moscot" && "$TASK" == "loo_time2" ]]; then
  echo "moscot has only three time points; configured tasks are full and loo_time1."
  exit 2
fi
case "$TASK" in
  full|loo_time1|loo_time2) ;;
  *)
    echo "Unknown task: $TASK"
    exit 2
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trainfbench-cache}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

PYTHON=${PYTHON:-python}
CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/${DATASET}_ae10}
RESULT_DIR=${RESULT_DIR:-results/tigon_${DATASET}_${TASK}_official_ae10_dopri5_exact_n1024_20000}
LOG_FILE=${LOG_FILE:-${RESULT_DIR}_stdout.log}
DEVICE=${DEVICE:-cpu}
TIGON_ITERS=${TIGON_ITERS:-20000}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
TIGON_SEED=${TIGON_SEED:-1}
ACTION_STATE_MODE=${ACTION_STATE_MODE:-upstream_public_exact}
DENSITY_ODE_BACKEND=${DENSITY_ODE_BACKEND:-torchdiffeqpack}
OFFICIAL_DENSITY_SIGMA_SCHEDULE=${OFFICIAL_DENSITY_SIGMA_SCHEDULE:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}

if [[ ! -e "$CACHE_DIR/embedding_ready.json" ]]; then
  echo "Frozen full-data AE cache is not ready:"
  echo "  $CACHE_DIR"
  echo "Run: bash model/TIGON/prepare_tigon_official_ae10_background.sh $DATASET"
  exit 1
fi
if [[ "$OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "OVERWRITE=1 and RESUME=1 cannot be used together"
  exit 1
fi
if [[ -e "$RESULT_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$RESULT_DIR"
    rm -f "$LOG_FILE"
  elif [[ "$RESUME" != "1" ]]; then
    echo "Refusing to overwrite existing output:"
    echo "  $RESULT_DIR"
    echo "  $LOG_FILE"
    exit 1
  fi
fi

resume_arg=()
if [[ "$RESUME" == "1" ]]; then
  resume_arg=(--resume)
fi
sigma_schedule_arg=()
if [[ "$OFFICIAL_DENSITY_SIGMA_SCHEDULE" == "1" ]]; then
  sigma_schedule_arg=(--official-density-sigma-schedule)
fi

"$PYTHON" model/TIGON/run_tigon_frozen_official_ae_embedding.py \
  --dataset "$DATASET" \
  --task "$TASK" \
  --cache-dir "$CACHE_DIR" \
  --outdir "$RESULT_DIR" \
  --tigon-iters "$TIGON_ITERS" \
  --num-samples "$NUM_SAMPLES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --tigon-seed "$TIGON_SEED" \
  --action-state-mode "$ACTION_STATE_MODE" \
  --density-ode-backend "$DENSITY_ODE_BACKEND" \
  --device "$DEVICE" \
  "${sigma_schedule_arg[@]}" \
  "${resume_arg[@]}" > "$LOG_FILE" 2>&1

echo "Done: $RESULT_DIR"
echo "Log: $LOG_FILE"
