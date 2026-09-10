#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Explicitly named entry point for the balanced pancreas LOO benchmark.
exec bash model/CytoBridge/run_cytobridge_loo_time1_20000.sh "$@"
