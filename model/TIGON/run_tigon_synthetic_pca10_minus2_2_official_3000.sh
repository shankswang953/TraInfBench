#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/tigon-synthetic-official-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba-cache}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
TIGON_ITERS=${TIGON_ITERS:-3000}
NUM_SAMPLES=${NUM_SAMPLES:-128}
SEED=${SEED:-0}
SAVE_DIR=${SAVE_DIR:-results/tigon_synthetic_rna10_pca_minus2_2_official_n128_i3000}
LOG_FILE=${LOG_FILE:-${SAVE_DIR}_stdout.log}
ANALYSIS_DIR=${ANALYSIS_DIR:-results/tigon_synthetic_rna10_pca_minus2_2_official_n128_i3000_analysis}
OVERWRITE=${OVERWRITE:-0}

if [[ -s "$SAVE_DIR/tigon.pt" && -s "$LOG_FILE" && "$OVERWRITE" != "1" ]]; then
  echo "Reusing completed official TIGON run: $SAVE_DIR"
else
  if [[ ( -e "$SAVE_DIR" || -e "$LOG_FILE" ) && "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite partial TIGON output: $SAVE_DIR or $LOG_FILE"
    exit 1
  fi
  overwrite_arg=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_arg=(--overwrite)
  fi
  echo "Running official TIGON on original RNA PCA10 scaled per axis to [-2,2]"
  echo "iterations=$TIGON_ITERS, num_samples=$NUM_SAMPLES, seed=$SEED, times=0,1,2,3"
  "$PYTHON" model/TIGON/run_tigon_moscot_ae.py \
    --rna-h5ad data/synthetic_rna10_cytobridge.h5ad \
    --outdir "$SAVE_DIR" \
    --embedding-key X_pca_raw \
    --embedding-normalization per-axis-minmax-minus2-2 \
    --time-key time_point_processed \
    --stage-key time_label \
    --celltype-key population \
    --no-ae \
    --latent-dim 10 \
    --tigon-iters "$TIGON_ITERS" \
    --checkpoint-every 1000 \
    --num-samples "$NUM_SAMPLES" \
    --eval-samples 128 \
    --saliency-cells-per-time 128 \
    --tigon-hidden-dim 16 \
    --tigon-hidden-layers 4 \
    --tigon-lr 3e-3 \
    --training-objective tigon-density \
    --ode-solver dopri5 \
    --density-ode-backend torchdiffeqpack \
    --ode-rtol 1e-3 \
    --ode-atol 1e-5 \
    --action-ode-solver midpoint \
    --action-ode-steps 3 \
    --action-state-mode upstream_public_exact \
    --divergence-estimator exact \
    --density-weight 1e4 \
    --density-sigma-initial 1 \
    --official-density-sigma-schedule \
    --density-sigma-anneal-every 100 \
    --density-sigma-anneal-factor 0.5 \
    --density-sigma-anneal-threshold 3e-4 \
    --density-sigma-anneal-stop-before-end 400 \
    --density-sample-sigma 0.02 \
    --density-kde-chunk-size 1024 \
    --density-stabilization official \
    --growth-action-weight 1 \
    --max-grad-norm 0 \
    --ode-steps 8 \
    --device cpu \
    --seed "$SEED" \
    "${overwrite_arg[@]}" > "$LOG_FILE" 2>&1
fi

analysis_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  analysis_args=(--overwrite)
fi
if [[ -e "$ANALYSIS_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "Analysis already exists: $ANALYSIS_DIR"
else
  "$PYTHON" comparison/scmultisim/analyze_tigon_synthetic.py \
    --result-dir "$SAVE_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    "${analysis_args[@]}"
fi

echo "Official TIGON synthetic PCA10 run complete."
