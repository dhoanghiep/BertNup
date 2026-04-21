#!/bin/bash
# Setup BertNup training environment on remote GPU server.
# Usage: ./scripts/setup-server.sh [--host HOST] [--port PORT] [--user USER]
#
# Default: vast.ai server (ssh9.vast.ai:21185, user root)
# Installs: BertNup deps (PyTorch assumed pre-installed on vast.ai).
# Assumes code has already been synced (run train-remote.sh first).
#
# Options:
#   --skip-torch   Skip PyTorch install (default if torch already present)
#   --force-torch  Force PyTorch reinstall

set -euo pipefail

# ── Configurable SSH connection ───────────────────────────────────────────
SSH_HOST="${BERTNUP_SSH_HOST:-ssh9.vast.ai}"
SSH_PORT="${BERTNUP_SSH_PORT:-21185}"
SSH_USER="${BERTNUP_SSH_USER:-root}"
REMOTE_DIR="${BERTNUP_REMOTE_DIR:-/root/BertNup}"

# ── Parse args ────────────────────────────────────────────────────────────
SKIP_TORCH=false
FORCE_TORCH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)        SSH_HOST="$2"; shift 2 ;;
        --port)        SSH_PORT="$2"; shift 2 ;;
        --user)        SSH_USER="$2"; shift 2 ;;
        --dir)         REMOTE_DIR="$2"; shift 2 ;;
        --skip-torch)  SKIP_TORCH=true; shift ;;
        --force-torch) FORCE_TORCH=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SSH_CMD="ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o SetEnv=TERM=xterm-256color $SSH_USER@$SSH_HOST"

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

# ── Install PyTorch + CUDA (skip if already present) ─────────────────────
if $FORCE_TORCH; then
    echo "==> Force-installing PyTorch with CUDA..."
    $SSH_CMD "pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" \
        2>&1 | tail -1
elif $SKIP_TORCH; then
    echo "==> Skipping PyTorch install (--skip-torch)"
else
    # Auto-detect: skip if torch is already importable
    HAS_TORCH=$($SSH_CMD "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'MISSING'")
    if [[ "$HAS_TORCH" != "MISSING" ]]; then
        echo "==> PyTorch $HAS_TORCH already installed, skipping."
    else
        echo "==> Installing PyTorch with CUDA..."
        $SSH_CMD "pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" \
            2>&1 | tail -1
    fi
fi
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
