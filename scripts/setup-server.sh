#!/bin/bash
# Setup BertNup training environment on remote GPU server.
# Usage: ./scripts/setup-server.sh [--host HOST] [--port PORT] [--user USER]
#
# Default: vast.ai server (ssh1.vast.ai:10622, user root)
# Installs: PyTorch+CUDA, transformers, pytorch-lightning, and all BertNup deps.
# Assumes code has already been synced (run train-remote.sh first).

set -euo pipefail

# ── Configurable SSH connection ───────────────────────────────────────────
SSH_HOST="${BERTNUP_SSH_HOST:-ssh1.vast.ai}"
SSH_PORT="${BERTNUP_SSH_PORT:-10622}"
SSH_USER="${BERTNUP_SSH_USER:-root}"
REMOTE_DIR="${BERTNUP_REMOTE_DIR:-/root/BertNup}"

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)   SSH_HOST="$2"; shift 2 ;;
        --port)   SSH_PORT="$2"; shift 2 ;;
        --user)   SSH_USER="$2"; shift 2 ;;
        --dir)    REMOTE_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SSH_CMD="ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST"

echo "==> Setting up BertNup on $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "    Remote dir: $REMOTE_DIR"
echo

# ── Check GPU ─────────────────────────────────────────────────────────────
echo "==> Checking GPU..."
$SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
    echo "ERROR: No GPU detected on remote server."
    exit 1
}
echo

# ── Install PyTorch + CUDA ────────────────────────────────────────────────
echo "==> Installing PyTorch with CUDA 12.4..."
$SSH_CMD "pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" \
    2>&1 | tail -1
echo

# ── Verify PyTorch CUDA ──────────────────────────────────────────────────
echo "==> Verifying PyTorch CUDA..."
$SSH_CMD "python3 -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')\""
echo

# ── Install BertNup deps ─────────────────────────────────────────────────
echo "==> Installing BertNup dependencies..."
$SSH_CMD "cd $REMOTE_DIR && pip install --quiet -e ." 2>&1 | tail -1
echo

# ── Verify bertnup CLI ───────────────────────────────────────────────────
echo "==> Verifying bertnup CLI..."
$SSH_CMD "cd $REMOTE_DIR && bertnup --help | head -3"
echo

echo "==> Setup complete. Ready to train."
