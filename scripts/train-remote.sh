#!/bin/bash
# Run BertNup training on remote GPU server.
# Usage: ./scripts/train-remote.sh [OPTIONS] -- <bertnup args>
#
# Examples:
#   ./scripts/train-remote.sh --sync-data -- train Data/... --model-name armheb/DNA_bert_3
#   ./scripts/train-remote.sh cross_validate HS_LC --config configs/dnabert1.yaml
#   ./scripts/train-remote.sh --no-sync -- train Data/... --epochs 5
#
# Options:
#   --sync-data    Rsync Data/ to remote (first time)
#   --no-sync      Skip code sync (just run)
#   --host HOST    SSH host (default: ssh1.vast.ai)
#   --port PORT    SSH port (default: 10622)
#   --user USER    SSH user (default: root)
#   --dir DIR      Remote work directory (default: /root/BertNup)

set -euo pipefail

# ── Configurable ──────────────────────────────────────────────────────────
SSH_HOST="${BERTNUP_SSH_HOST:-ssh1.vast.ai}"
SSH_PORT="${BERTNUP_SSH_PORT:-10622}"
SSH_USER="${BERTNUP_SSH_USER:-root}"
REMOTE_DIR="${BERTNUP_REMOTE_DIR:-/root/BertNup}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SYNC_CODE=true
SYNC_DATA=false

# ── Parse options (before -- separator) ───────────────────────────────────
BERTNUP_ARGS=()
parsing_opts=true
while [[ $# -gt 0 ]]; do
    if $parsing_opts; then
        case $1 in
            --) parsing_opts=false; shift ;;
            --sync-data) SYNC_DATA=true; shift ;;
            --no-sync)   SYNC_CODE=false; shift ;;
            --host)      SSH_HOST="$2"; shift 2 ;;
            --port)      SSH_PORT="$2"; shift 2 ;;
            --user)      SSH_USER="$2"; shift 2 ;;
            --dir)       REMOTE_DIR="$2"; shift 2 ;;
            *)           BERTNUP_ARGS+=("$1"); shift ;;
        esac
    else
        BERTNUP_ARGS+=("$1"); shift
    fi
done

RSYNC_OPTS=(-avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no")
SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST")

# ── Sync code ─────────────────────────────────────────────────────────────
if $SYNC_CODE; then
    echo "==> Syncing code to $SSH_USER@$SSH_HOST:$REMOTE_DIR ..."
    rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/" 2>&1 | tail -3
fi

# ── Sync data (optional, ~90MB) ──────────────────────────────────────────
if $SYNC_DATA; then
    echo "==> Syncing data..."
    rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/Data/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/Data/" 2>&1 | tail -3
fi

# ── Run training ─────────────────────────────────────────────────────────
echo "==> Running training on remote GPU..."
echo "    Command: bertnup ${BERTNUP_ARGS[*]}"
echo

"${SSH_CMD[@]}" "cd $REMOTE_DIR && KMP_DUPLICATE_LIB_OK=TRUE pip install -e . -q 2>/dev/null; bertnup ${BERTNUP_ARGS[*]}"
