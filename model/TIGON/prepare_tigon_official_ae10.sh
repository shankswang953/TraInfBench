#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET=${1:?Usage: bash model/TIGON/prepare_tigon_official_ae10.sh moscot|palate|gastrulation}
case "$DATASET" in
  moscot|palate|gastrulation) ;;
  *)
    echo "Unknown dataset: $DATASET"
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
LOG_FILE=${LOG_FILE:-${CACHE_DIR}_stdout.log}
DEVICE=${DEVICE:-cpu}
AE_SEED=${AE_SEED:-4232}
AE_MAX_EPOCHS=${AE_MAX_EPOCHS:-500}
OVERWRITE=${OVERWRITE:-0}

if [[ -e "$CACHE_DIR" || -e "$LOG_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite an existing frozen AE cache or log:"
    echo "  $CACHE_DIR"
    echo "  $LOG_FILE"
    exit 1
  fi
  rm -rf "$CACHE_DIR"
  rm -f "$LOG_FILE"
fi

overwrite_arg=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_arg=(--overwrite)
fi

"$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
  --dataset "$DATASET" \
  --cache-dir "$CACHE_DIR" \
  --ae-seed "$AE_SEED" \
  --ae-max-epochs "$AE_MAX_EPOCHS" \
  --device "$DEVICE" \
  "${overwrite_arg[@]}" > "$LOG_FILE" 2>&1

echo "Frozen full-data AE ready: $CACHE_DIR"
echo "Log: $LOG_FILE"
