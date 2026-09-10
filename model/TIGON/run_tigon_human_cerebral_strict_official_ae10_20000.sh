#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

USAGE="Usage: bash model/TIGON/run_tigon_human_cerebral_strict_official_ae10_20000.sh [full|loo_d7]"
TASK_INPUT=${1:?$USAGE}
case "$TASK_INPUT" in
  full)
    TASK=full
    ;;
  loo_d7|loo_day7)
    TASK=loo_day7
    ;;
  *)
    echo "Unknown human-cerebral task: $TASK_INPUT (expected full or loo_d7)"
    exit 2
    ;;
esac

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
if [[ "$TASK" == "full" ]]; then
  CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/human_cerebral_no_d61_full_ae10}
else
  CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/human_cerebral_no_d61_loo_day7_strict_ae10}
fi
AE_LOG=${AE_LOG:-${CACHE_DIR}_stdout.log}
TIGON_ITERS=${TIGON_ITERS:-20000}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
TIGON_SEED=${TIGON_SEED:-1}
AE_SEED=${AE_SEED:-4232}
AE_MAX_EPOCHS=${AE_MAX_EPOCHS:-500}
DEVICE=${DEVICE:-cpu}
RESULT_DIR=${RESULT_DIR:-results/tigon_human_cerebral_no_d61_${TASK}_strict_official_ae10_dopri5pack_exact_n${NUM_SAMPLES}_${TIGON_ITERS}_seed${TIGON_SEED}}
TIGON_LOG=${TIGON_LOG:-${RESULT_DIR}_stdout.log}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME=${RESUME:-0}

echo "humanCerebral TIGON official AE10 configuration: ${TASK}"
echo "  AE input: original humanBrainRNA 3000-HVG expression"
echo "  AE: 3000-300-10-300-3000, ReLU, batch norm, dropout=0.2"
echo "  AE fit scope: ${TASK} retained cells only"
echo "  latent scaling: fit-scope per-axis min-max to [-2,2]"
echo "  TIGON: public exact action/density, TorchDiffEqPack Dopri5"
echo "  TIGON iterations/samples: ${TIGON_ITERS}/${NUM_SAMPLES}"
echo "  cache: ${CACHE_DIR}"
echo "  result: ${RESULT_DIR}"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  "$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
    --dataset human_cerebral_no_d61 \
    --task "$TASK" \
    --cache-dir "$CACHE_DIR" \
    --ae-seed "$AE_SEED" \
    --ae-max-epochs "$AE_MAX_EPOCHS" \
    --device "$DEVICE" \
    --validate-only
  "$PYTHON" -c \
    "from scripts.tigon_official_ae_common import OfficialAutoEncoder, load_tigon_module; load_tigon_module()._load_upstream_odesolve(); print('[OK] official AE10 and vendored TorchDiffEqPack')"
  echo "Preflight passed; AE and TIGON training were not started."
  exit 0
fi

if [[ "$OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
  echo "OVERWRITE=1 and RESUME=1 cannot be used together"
  exit 1
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
    --dataset human_cerebral_no_d61 \
    --task "$TASK" \
    --cache-dir "$CACHE_DIR" \
    --ae-seed "$AE_SEED" \
    --ae-max-epochs "$AE_MAX_EPOCHS" \
    --device "$DEVICE" \
    "${overwrite_arg[@]}" > "$AE_LOG" 2>&1
  echo "AE ready: $CACHE_DIR"
  echo "AE log: $AE_LOG"
else
  echo "Reusing verified task-specific AE cache: $CACHE_DIR"
fi

if [[ -e "$RESULT_DIR" || -e "$TIGON_LOG" ]]; then
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$RESULT_DIR"
    rm -f "$TIGON_LOG"
  elif [[ "$RESUME" != "1" ]]; then
    echo "Refusing to overwrite existing TIGON output:"
    echo "  $RESULT_DIR"
    echo "  $TIGON_LOG"
    exit 1
  fi
fi

resume_arg=()
if [[ "$RESUME" == "1" ]]; then
  resume_arg=(--resume)
fi
train_command=(
  "$PYTHON" model/TIGON/run_tigon_frozen_official_ae_embedding.py
  --dataset human_cerebral_no_d61
  --task "$TASK"
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
  "${train_command[@]}" >> "$TIGON_LOG" 2>&1
else
  "${train_command[@]}" > "$TIGON_LOG" 2>&1
fi

echo "Done: $RESULT_DIR"
echo "Log: $TIGON_LOG"
