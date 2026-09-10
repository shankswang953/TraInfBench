#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

USAGE="Usage: bash model/MIOFlow/run_mioflow_palate_gaga10_20000.sh [full|loo1|loo2]"
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

# Match the controlled gastrulation configuration: shared RNA PCA -> GAGA10
# -> MIOFlow.  The two LOO inputs omit E13.5 and E14.0, respectively.
export EPOCHS=${EPOCHS:-20000}
export SAMPLE_SIZE=${SAMPLE_SIZE:-1024}
export LATENT_KEY=${LATENT_KEY:-X_latent}
export GAGA_LATENT_DIM=10
export GAGA_HIDDEN_DIMS=${GAGA_HIDDEN_DIMS:-128,64}
export GAGA_MAX_CELLS=${GAGA_MAX_CELLS:-5000}
export GAGA_BATCH_SIZE=${GAGA_BATCH_SIZE:-1024}
export GAGA_ENCODER_EPOCHS=${GAGA_ENCODER_EPOCHS:-100}
export GAGA_DECODER_EPOCHS=${GAGA_DECODER_EPOCHS:-100}
export GAGA_DISTANCE_COMPONENTS=${GAGA_DISTANCE_COMPONENTS:-10}
export GAGA_PHATE_KNN=${GAGA_PHATE_KNN:-5}
export GAGA_PHATE_LANDMARKS=${GAGA_PHATE_LANDMARKS:-2000}
export EMBEDDING_NORMALIZATION=${EMBEDDING_NORMALIZATION:-identity}
export TIME_AXIS=${TIME_AXIS:-rank}
export N_BINS=${N_BINS:-101}
export N_TRAJECTORIES=${N_TRAJECTORIES:-1000}
export SEED=${SEED:-42}

exec bash model/MIOFlow/run_mioflow_pca_gaga10_n1024_20000.sh palate "$TASK"
