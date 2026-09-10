#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/trainfbench-cache}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

PYTHON=${PYTHON:-python}
CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/moscot_ae5_upstream_public_exact_v2}
TIGON_ITERS=${TIGON_ITERS:-20000}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
TIGON_SEED=${TIGON_SEED:-1}
DEVICE=${DEVICE:-cpu}
EVAL_SAMPLES=${EVAL_SAMPLES:-1024}
EVALUATION_SEED=${EVALUATION_SEED:-20260728}
SELECTION_SINKHORN_BLUR=${SELECTION_SINKHORN_BLUR:-0.05}
POSTPROCESS=${POSTPROCESS:-1}
POSTPROCESS_OVERWRITE=${POSTPROCESS_OVERWRITE:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ "$TIGON_ITERS" != "20000" || "$NUM_SAMPLES" != "1024" || "$CHECKPOINT_EVERY" != "500" ]]; then
  if [[ "${ALLOW_SMOKE_SIZED_RUN:-0}" != "1" ]]; then
    echo "Publication run requires TIGON_ITERS=20000, NUM_SAMPLES=1024, CHECKPOINT_EVERY=500."
    echo "Set ALLOW_SMOKE_SIZED_RUN=1 only for a diagnostic run."
    exit 2
  fi
fi

RESULT_DIR=${RESULT_DIR:-results/tigon_moscot_full_upstream_public_exact_v2_rngexact_ae5_dopri5pack_n${NUM_SAMPLES}_${TIGON_ITERS}_seed${TIGON_SEED}}
LOG_FILE=${LOG_FILE:-${RESULT_DIR}_stdout.log}

if [[ ! -e "$CACHE_DIR/embedding_ready.json" ]]; then
  echo "Frozen moscot AE5 cache is not ready: $CACHE_DIR"
  echo "Run: bash model/TIGON/prepare_tigon_moscot_upstream_public_exact_ae5.sh"
  exit 1
fi
"$PYTHON" model/TIGON/check_tigon_moscot_upstream_public_exact_ae5.py \
  --cache-dir "$CACHE_DIR"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  PREFLIGHT_DIR=${PREFLIGHT_DIR:-results/tigon_moscot_full_upstream_public_exact_ae5_preflight}
  "$PYTHON" model/TIGON/run_tigon_frozen_official_ae_embedding.py \
    --dataset moscot \
    --task full \
    --cache-dir "$CACHE_DIR" \
    --outdir "$PREFLIGHT_DIR" \
    --tigon-iters "$TIGON_ITERS" \
    --num-samples "$NUM_SAMPLES" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --tigon-seed "$TIGON_SEED" \
    --action-state-mode upstream_public_exact \
    --density-ode-backend torchdiffeqpack \
    --latent-source scaled_minus2_2 \
    --latent-standardization none \
    --official-density-sigma-schedule \
    --device "$DEVICE" \
    --audit-only
  exit 0
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

train_command=(
  "$PYTHON" model/TIGON/run_tigon_frozen_official_ae_embedding.py
  --dataset moscot
  --task full
  --cache-dir "$CACHE_DIR"
  --outdir "$RESULT_DIR"
  --tigon-iters "$TIGON_ITERS"
  --num-samples "$NUM_SAMPLES"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --tigon-seed "$TIGON_SEED"
  --action-state-mode upstream_public_exact
  --density-ode-backend torchdiffeqpack
  --latent-source scaled_minus2_2
  --latent-standardization none
  --official-density-sigma-schedule
  --device "$DEVICE"
  "${resume_arg[@]}"
)
if [[ "$RESUME" == "1" ]]; then
  "${train_command[@]}" >> "$LOG_FILE" 2>&1
else
  "${train_command[@]}" > "$LOG_FILE" 2>&1
fi

if [[ "$POSTPROCESS" == "1" ]]; then
  overwrite_arg=()
  if [[ "$POSTPROCESS_OVERWRITE" == "1" ]]; then
    overwrite_arg=(--overwrite)
  fi
  "$PYTHON" model/TIGON/select_tigon_observed_checkpoint.py \
    --dataset moscot \
    --task full \
    --result-dir "$RESULT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --eval-samples "$EVAL_SAMPLES" \
    --sinkhorn-blur "$SELECTION_SINKHORN_BLUR" \
    --evaluation-seed "$EVALUATION_SEED" \
    --selection-initialization native_cov0.02 \
    --require-protocol gastrulation_tigon_upstream_public_exact_v2 \
    --device "$DEVICE" \
    "${overwrite_arg[@]}" >> "$LOG_FILE" 2>&1
  "$PYTHON" model/TIGON/evaluate_tigon_selected_checkpoint.py \
    --dataset moscot \
    --task full \
    --result-dir "$RESULT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --eval-samples "$EVAL_SAMPLES" \
    --sinkhorn-blur "$SELECTION_SINKHORN_BLUR" \
    --evaluation-seed "$EVALUATION_SEED" \
    --device "$DEVICE" \
    "${overwrite_arg[@]}" >> "$LOG_FILE" 2>&1
fi

echo "Done: $RESULT_DIR"
echo "Log: $LOG_FILE"
