---
name: server-setup-remote-training
status: completed
created: 2026-04-18
blockedBy: []
blocks: []
---

# Server Setup & Remote Training

## Overview

Create a server setup module and CLI command to deploy BertNup training to a vast.ai GPU server (RTX 5060 Ti, 16GB). The module handles: server provisioning (install deps), code/data sync, and remote training execution.

## Server Details

- **Host:** ssh1.vast.ai, port 10622, user root
- **GPU:** NVIDIA RTX 5060 Ti, 16GB VRAM, driver 570.153.02
- **Missing:** PyTorch, transformers, pytorch-lightning, etc.

## Phases

| # | Phase | Status | Description |
|---|-------|--------|-------------|
| 01 | [Server Setup Script](phase-01-server-setup.md) | Complete | Bash script to install deps on remote server |
| 02 | [Remote Training CLI](phase-02-remote-training.md) | Complete | CLI command + script to sync and train remotely |
| 03 | [Autoresearch Workflow](phase-03-autoresearch-workflow.md) | Complete | Adapt karpathy/autoresearch for BertNup classification |

## Architecture

```
scripts/
├── setup-server.sh       # SSH into server, install all deps
└── train-remote.sh       # Rsync code+data, run training via SSH

autoresearch/
├── program.md            # Agent instructions (human edits)
├── run-experiment.sh     # Single experiment runner (fixed time budget)
└── results.tsv           # Experiment log (gitignored)
```

## Key Design Decisions

1. **Bash scripts, not Python module** - Remote orchestration is fundamentally shell work (rsync, ssh).
2. **rsync for sync** - Efficient incremental sync for code (small) and data (90MB).
3. **Config-driven** - Server details in script vars, training config via existing YAML files.
4. **Reuses existing CLI** - Remote training just calls `bertnup train` or `bertnup cross_validate` on the server.
5. **Autoresearch adapted for classification** - Metric is val_loss (lower=better) instead of val_bpb. Agent modifies YAML configs instead of a single train.py. Time budget via wall-clock.

## Related Plans

- `260416-1542-reproduce-paper-experiments` — This plan enables running those experiments on GPU.
