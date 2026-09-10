#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET=${1:?Usage: bash model/TIGON/prepare_tigon_official_ae10_background.sh moscot|palate|gastrulation}
case "$DATASET" in
  moscot|palate|gastrulation) ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 2
    ;;
esac

JOB_DIR=${JOB_DIR:-results/tigon_official_ae_embeddings/background_jobs/${DATASET}}
PID_FILE=${PID_FILE:-${JOB_DIR}/pid}
LAUNCHER_LOG=${LAUNCHER_LOG:-${JOB_DIR}/launcher.log}
OVERWRITE=${OVERWRITE:-0}

if [[ -e "$PID_FILE" || -e "$LAUNCHER_LOG" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite existing background job files:"
    echo "  $PID_FILE"
    echo "  $LAUNCHER_LOG"
    exit 1
  fi
  rm -f "$PID_FILE" "$LAUNCHER_LOG"
fi

mkdir -p "$JOB_DIR"
nohup env \
  OVERWRITE="$OVERWRITE" \
  DEVICE="${DEVICE:-cpu}" \
  bash model/TIGON/prepare_tigon_official_ae10.sh "$DATASET" \
  > "$LAUNCHER_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "Started frozen full-data TIGON AE preparation: dataset=$DATASET pid=$pid"
echo "Launcher log: $LAUNCHER_LOG"
echo "Training log: results/tigon_official_ae_embeddings/${DATASET}_ae10_stdout.log"
