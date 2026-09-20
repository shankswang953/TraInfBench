#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCENARIO=${1:?Usage: run_human_cerebral_memory_background.sh full}
if [[ "$#" -ne 1 ]]; then echo "Expected exactly one scenario argument" >&2; exit 2; fi
case "$SCENARIO" in full) ;; *) echo "Unknown scenario: $SCENARIO (expected full; human cerebral LOO is not configured)" >&2; exit 2;; esac
export QGW_STORAGE=disk
export OUTPUT_DIR=${OUTPUT_DIR:-results/scmultinode_human_cerebral_7time_d4_d21_no_d16_full_normalized10_n1024_20000_seed0_diskqgw}
exec bash model/scMultiNODE/run_human_cerebral_background.sh "$SCENARIO"
