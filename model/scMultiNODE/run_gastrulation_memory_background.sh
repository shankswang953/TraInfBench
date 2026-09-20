#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCENARIO=${1:?Usage: run_gastrulation_memory_background.sh full\|loo1\|loo2}
if [[ "$#" -ne 1 ]]; then echo "Expected exactly one scenario argument" >&2; exit 2; fi
case "$SCENARIO" in full|loo1|loo2) ;; *) echo "Unknown scenario: $SCENARIO" >&2; exit 2;; esac
export QGW_STORAGE=disk
export OUTPUT_DIR=${OUTPUT_DIR:-results/scmultinode_gastrulation_${SCENARIO}_normalized10_n1024_20000_seed0_diskqgw}
exec bash model/scMultiNODE/run_gastrulation_background.sh "$SCENARIO"
