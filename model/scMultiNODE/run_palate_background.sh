#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SCENARIO=${1:?Usage: run_palate_background.sh full\|loo1\|loo2}
if [[ "$#" -ne 1 ]]; then echo "Expected exactly one scenario argument" >&2; exit 2; fi
case "$SCENARIO" in full|loo1|loo2) ;; *) echo "Unknown scenario: $SCENARIO (expected full, loo1 or loo2)" >&2; exit 2;; esac
export QGW_STORAGE=${QGW_STORAGE:-official}
case "$QGW_STORAGE" in official|disk) ;; *) echo "Unknown QGW_STORAGE: $QGW_STORAGE (expected official or disk)" >&2; exit 2;; esac
DEFAULT_OUTPUT=results/scmultinode_palate_${SCENARIO}_normalized10_n1024_20000_seed0
if [[ "$QGW_STORAGE" == "disk" ]]; then DEFAULT_OUTPUT=${DEFAULT_OUTPUT}_diskqgw; fi
export OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}_launcher.log}
if [[ -e "$OUTPUT_DIR" || -e "$LOG_FILE" ]]; then
  echo "Refusing existing output or log; select a new OUTPUT_DIR/LOG_FILE." >&2
  exit 1
fi
mkdir -p "$(dirname "$LOG_FILE")"
( set -o noclobber; nohup bash model/scMultiNODE/run_palate_benchmark.sh "$SCENARIO" > "$LOG_FILE" 2>&1 < /dev/null ) &
echo "Submitted $SCENARIO QGW_STORAGE=$QGW_STORAGE PID=$!"
echo "Log: $LOG_FILE"
if [[ "$QGW_STORAGE" == "disk" ]]; then
  echo "Disk-QGW resource checks or the sampled memory guard can stop this job. Check the log and resource reports."
else
  echo "Official QGW memory preflight can stop this job before training. Check the log."
fi
