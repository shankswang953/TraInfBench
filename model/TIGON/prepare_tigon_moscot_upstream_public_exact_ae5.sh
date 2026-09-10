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
AE_SEED=${AE_SEED:-4232}
AE_MAX_EPOCHS=${AE_MAX_EPOCHS:-500}
DEVICE=${DEVICE:-cpu}
OVERWRITE=${OVERWRITE:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  "$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
    --dataset moscot \
    --task full \
    --ae-latent-dim 5 \
    --ae-seed "$AE_SEED" \
    --ae-max-epochs "$AE_MAX_EPOCHS" \
    --device "$DEVICE" \
    --validate-only
  exit 0
fi

if [[ -e "$CACHE_DIR/embedding_ready.json" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing protected moscot AE5 cache: $CACHE_DIR"
    "$PYTHON" model/TIGON/check_tigon_moscot_upstream_public_exact_ae5.py \
      --cache-dir "$CACHE_DIR"
    exit 0
  fi
elif [[ -d "$CACHE_DIR" ]] && [[ -n "$(find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Incomplete AE5 cache exists; refusing to overwrite: $CACHE_DIR"
    exit 1
  fi
fi

overwrite_arg=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_arg=(--overwrite)
fi

"$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
  --dataset moscot \
  --task full \
  --cache-dir "$CACHE_DIR" \
  --ae-latent-dim 5 \
  --ae-seed "$AE_SEED" \
  --ae-max-epochs "$AE_MAX_EPOCHS" \
  --device "$DEVICE" \
  "${overwrite_arg[@]}"

"$PYTHON" model/TIGON/prepare_tigon_moscot_official_ae_common_pca_bridge.py \
  --cache-dir "$CACHE_DIR" \
  "${overwrite_arg[@]}"

"$PYTHON" model/TIGON/check_tigon_moscot_upstream_public_exact_ae5.py \
  --cache-dir "$CACHE_DIR"

echo "Prepared frozen full-data moscot TIGON AE5 cache: $CACHE_DIR"
