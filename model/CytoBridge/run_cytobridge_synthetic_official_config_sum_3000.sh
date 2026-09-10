#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
CACHE_ROOT=${CACHE_ROOT:-/tmp/trainfbench-cytobridge-synthetic-officialcfg-sum-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba}
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-3000}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-0}
INPUT_H5AD=${INPUT_H5AD:-data/synthetic_rna10_cytobridge.h5ad}
BALANCED_DIR=${BALANCED_DIR:-results/cytobridge_synthetic_rna10_balanced_officialcfg_sum_n128_i3000}
UNBALANCED_DIR=${UNBALANCED_DIR:-results/cytobridge_synthetic_rna10_unbalanced_officialcfg_sum_n128_i3000}
CACHE_FILE=${CACHE_FILE:-results/cytobridge_synthetic_rna10_officialcfg_sum_n128_i3000_observed_times/cytobridge_observed_time_cache.npz}
SINKHORN_DIR=${SINKHORN_DIR:-results/synthetic_rna10_same_space_predicted_time_sinkhorn_cytobridge_officialcfg_sum_i3000}
UMAP_DIR=${UMAP_DIR:-results/coati_cytobridge_observed_time_umap_comparison_officialcfg_sum_i3000}
CONVERGENCE_DIR=${CONVERGENCE_DIR:-results/cytobridge_synthetic_rna10_officialcfg_sum_i3000_convergence}

if [[ ! -e "$INPUT_H5AD" ]]; then
  "$PYTHON" model/CytoBridge/prepare_synthetic_cytobridge_inputs.py --output-h5ad "$INPUT_H5AD"
fi

run_model() {
  local mode=$1
  local output_dir=$2
  local log_file="${output_dir}_stdout.log"
  if [[ -e "$output_dir/adata.h5ad" && -e "$log_file" && "$OVERWRITE" != "1" ]]; then
    echo "Reusing completed ${mode} run: ${output_dir}"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "Refusing to overwrite partial ${mode} output: ${output_dir} or ${log_file}"
    exit 1
  fi
  "$PYTHON" model/CytoBridge/train_cytobridge_synthetic_official_config_sum.py \
    --mode "$mode" \
    --input-h5ad "$INPUT_H5AD" \
    --output-dir "$output_dir" \
    --epochs "$EPOCHS" \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    > "$log_file" 2>&1
}

run_model balanced "$BALANCED_DIR"
run_model unbalanced "$UNBALANCED_DIR"

if [[ ! -e "$CACHE_FILE" ]]; then
  "$PYTHON" comparison/scmultisim/cache_cytobridge_synthetic_observed_times.py \
    --balanced-adata "$BALANCED_DIR/adata.h5ad" \
    --unbalanced-adata "$UNBALANCED_DIR/adata.h5ad" \
    --output "$CACHE_FILE" \
    --device "$DEVICE"
fi

if [[ ! -e "$SINKHORN_DIR/predicted_time_sinkhorn.csv" ]]; then
  "$PYTHON" comparison/scmultisim/evaluate_synthetic_predicted_time_sinkhorn.py \
    --cytobridge-cache "$CACHE_FILE" \
    --output-dir "$SINKHORN_DIR" \
    --device "$DEVICE"
fi

if [[ ! -e "$UMAP_DIR/coati_cytobridge_observed_time_umap.png" ]]; then
  "$PYTHON" common/plot_coati_cytobridge_observed_time_umap.py \
    --cytobridge-cache "$CACHE_FILE" \
    --sinkhorn-csv "$SINKHORN_DIR/predicted_time_sinkhorn.csv" \
    --cytobridge-subtitle "official config, energy = 0.01" \
    --output-dir "$UMAP_DIR"
fi

if [[ ! -e "$CONVERGENCE_DIR/cytobridge_convergence.png" ]]; then
  "$PYTHON" comparison/scmultisim/plot_cytobridge_synthetic_convergence.py \
    --balanced-log "${BALANCED_DIR}_stdout.log" \
    --unbalanced-log "${UNBALANCED_DIR}_stdout.log" \
    --output-dir "$CONVERGENCE_DIR" \
    --overwrite
fi

echo "CytoBridge official-config weighted-sum experiment complete."
