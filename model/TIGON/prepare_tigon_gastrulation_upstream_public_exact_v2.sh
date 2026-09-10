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
CACHE_DIR=${CACHE_DIR:-results/tigon_official_ae_embeddings/gastrulation_ae10_upstream_public_exact_v2}

if [[ -e "$CACHE_DIR/embedding_ready.json" ]]; then
  echo "Reusing protected v2 AE cache: $CACHE_DIR"
elif [[ -d "$CACHE_DIR" ]] && [[ -n "$(find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Incomplete v2 AE cache exists; refusing to overwrite: $CACHE_DIR"
  exit 1
else
  "$PYTHON" model/TIGON/prepare_tigon_official_ae_embedding.py \
    --dataset gastrulation \
    --cache-dir "$CACHE_DIR" \
    --ae-seed 4232 \
    --ae-max-epochs 500 \
    --device cpu
fi

"$PYTHON" model/TIGON/prepare_tigon_gastrulation_official_ae_common_pca_bridge.py \
  --cache-dir "$CACHE_DIR"

echo "Prepared Gastrulation TIGON v2 AE and common-PCA50 bridge: $CACHE_DIR"
