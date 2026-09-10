#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

USAGE="Usage: bash model/TIGON/run_tigon_palate_strict_official_ae10_20000.sh [full|loo1|loo2]"
TASK_INPUT=${1:?$USAGE}
case "$TASK_INPUT" in
  full)
    TASK=full
    ;;
  loo1|loo_time1)
    TASK=loo_time1
    ;;
  loo2|loo_time2)
    TASK=loo_time2
    ;;
  *)
    echo "Unknown palate task: $TASK_INPUT (expected full, loo1, or loo2)"
    exit 2
    ;;
esac

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trainfbench-cache}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

PYTHON=${PYTHON:-python}
CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/palate_${TASK}_strict_ae10}
AE_LOG=${AE_LOG:-${CACHE_DIR}_stdout.log}
RESULT_DIR=${RESULT_DIR:-results/tigon_palate_${TASK}_strict_official_ae10_dopri5_exact_n1024_20000}
TIGON_LOG=${TIGON_LOG:-${RESULT_DIR}_stdout.log}
DEVICE=${DEVICE:-cpu}
AE_SEED=${AE_SEED:-4232}
AE_MAX_EPOCHS=${AE_MAX_EPOCHS:-500}
TIGON_ITERS=${TIGON_ITERS:-20000}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
TIGON_SEED=${TIGON_SEED:-1}
ACTION_STATE_MODE=${ACTION_STATE_MODE:-upstream_public_exact}
DENSITY_ODE_BACKEND=${DENSITY_ODE_BACKEND:-torchdiffeqpack}
OFFICIAL_DENSITY_SIGMA_SCHEDULE=${OFFICIAL_DENSITY_SIGMA_SCHEDULE:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}

echo "Palate TIGON strict ${TASK} configuration"
echo "  AE input: 3000-HVG log-expression from retained stages"
echo "  AE: 3000-300-10-300-3000, ReLU, batch norm, dropout=0.2"
echo "  AE seed/max epochs: ${AE_SEED}/${AE_MAX_EPOCHS}"
echo "  latent scaling: retained-cell per-axis min-max to [-2,2]"
echo "  TIGON iterations/samples: ${TIGON_ITERS}/${NUM_SAMPLES}"
echo "  TIGON action/ODE: ${ACTION_STATE_MODE}/${DENSITY_ODE_BACKEND}"
echo "  official sigma schedule: ${OFFICIAL_DENSITY_SIGMA_SCHEDULE}"
echo "  cache: ${CACHE_DIR}"
echo "  result: ${RESULT_DIR}"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
    --dataset palate \
    --task "$TASK" \
    --cache-dir "$CACHE_DIR" \
    --ae-seed "$AE_SEED" \
    --ae-max-epochs "$AE_MAX_EPOCHS" \
    --device "$DEVICE" \
    --validate-only
  "$PYTHON" -c \
    "import torch, torchdiffeq, anndata; from scripts.tigon_official_ae_common import OfficialAutoEncoder; print('[OK] TIGON official AE10 and ODE dependencies')"
  echo "Preflight passed; AE and TIGON training were not started."
  exit 0
fi

if [[ ! -e "$CACHE_DIR/embedding_ready.json" ]]; then
  if [[ -e "$CACHE_DIR" || -e "$AE_LOG" ]]; then
    echo "Refusing a partial or unverified AE cache:"
    echo "  $CACHE_DIR"
    echo "  $AE_LOG"
    echo "Use a new CACHE_DIR, or set OVERWRITE=1 for an intentional restart."
    if [[ "$OVERWRITE" != "1" ]]; then
      exit 1
    fi
  fi
  overwrite_arg=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_arg=(--overwrite)
  fi
  "$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
    --dataset palate \
    --task "$TASK" \
    --cache-dir "$CACHE_DIR" \
    --ae-seed "$AE_SEED" \
    --ae-max-epochs "$AE_MAX_EPOCHS" \
    --device "$DEVICE" \
    "${overwrite_arg[@]}" > "$AE_LOG" 2>&1
  echo "AE ready: $CACHE_DIR"
  echo "AE log: $AE_LOG"
else
  echo "Reusing verified AE cache: $CACHE_DIR"
fi

env \
  CACHE_DIR="$CACHE_DIR" \
  RESULT_DIR="$RESULT_DIR" \
  LOG_FILE="$TIGON_LOG" \
  DEVICE="$DEVICE" \
  TIGON_ITERS="$TIGON_ITERS" \
  NUM_SAMPLES="$NUM_SAMPLES" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  TIGON_SEED="$TIGON_SEED" \
  ACTION_STATE_MODE="$ACTION_STATE_MODE" \
  DENSITY_ODE_BACKEND="$DENSITY_ODE_BACKEND" \
  OFFICIAL_DENSITY_SIGMA_SCHEDULE="$OFFICIAL_DENSITY_SIGMA_SCHEDULE" \
  OVERWRITE="$OVERWRITE" \
  RESUME="$RESUME" \
  bash model/TIGON/run_tigon_official_ae10_dopri5_exact_n1024_20000.sh \
  palate "$TASK"
