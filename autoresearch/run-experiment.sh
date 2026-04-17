#!/bin/bash
# Run a single BertNup training experiment with a time budget.
# Usage: ./autoresearch/run-experiment.sh <config.yaml> <data_dir>
#
# Time budget: 20 minutes (1200s) or training completes, whichever first.
# Output: parseable metrics to stdout (val_loss, val_acc, val_auc, vram_mb).
#
# Environment variables:
#   BERTNUP_PYTHON    — Python executable (default: auto-detect)
#   AUTORESEARCH_TIMEOUT — Time budget in seconds (default: 1200 = 20min)

set -euo pipefail

CONFIG="${1:-configs/dnabert1.yaml}"
DATA_DIR="${2:-Data/Stratified_K_fold_data/HS_LC/split_0}"
TIMEOUT="${AUTORESEARCH_TIMEOUT:-1200}"

# Resolve Python
if [[ -n "${BERTNUP_PYTHON:-}" ]]; then
    PYTHON="$BERTNUP_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "==> Experiment: config=$CONFIG data=$DATA_DIR timeout=${TIMEOUT}s"
echo "    Python: $PYTHON"
echo

# Run training with timeout. Kill if exceeds time budget.
# KMP_DUPLICATE_LIB_OK=TRUE for macOS OpenMP compat.
export KMP_DUPLICATE_LIB_OK=TRUE

set +e
timeout "$TIMEOUT" $PYTHON -m bertnup.cli train "$DATA_DIR" --config "$CONFIG" 2>&1
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -eq 124 ]]; then
    echo "---"
    echo "status: timeout (exceeded ${TIMEOUT}s)"
elif [[ $EXIT_CODE -ne 0 ]]; then
    echo "---"
    echo "status: crash (exit code $EXIT_CODE)"
fi
