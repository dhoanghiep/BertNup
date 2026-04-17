# Phase 01: Server Setup Script

## Status: Complete

## Context

- Server: `ssh -p 10622 root@ssh1.vast.ai`
- GPU: RTX 5060 Ti, 16GB, driver 570.153.02
- OS: Linux (vast.ai container)
- PyTorch not installed

## Overview

Create `scripts/setup-server.sh` that SSHes into the vast.ai server and installs all dependencies for BertNup training.

## Requirements

1. Install PyTorch with CUDA support (match driver 570.x → CUDA 12.x)
2. Install Python dependencies from pyproject.toml
3. Install bertnup package in editable mode
4. Verify GPU is accessible to PyTorch

## Implementation Steps

1. Create `scripts/setup-server.sh` with:
   - Configurable SSH connection vars (host, port, user)
   - SSH command that runs remote setup:
     - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
     - `pip install pytorch-lightning transformers omegaconf einops scikit-learn pandas numpy matplotlib seaborn`
     - `pip install -e .` (if code already synced)
     - `python -c "import torch; print(torch.cuda.is_available())"` (verify)
   - Colored output, error handling

2. Make executable: `chmod +x scripts/setup-server.sh`

## Related Code Files

- **Create:** `scripts/setup-server.sh`
- **Read:** `pyproject.toml` (for dependency list)

## Success Criteria

- Running `./scripts/setup-server.sh` installs all deps on remote server
- PyTorch detects the RTX 5060 Ti
- `bertnup --help` works on remote server (after code sync)

## Next Steps

- Phase 02 uses the provisioned server for training
