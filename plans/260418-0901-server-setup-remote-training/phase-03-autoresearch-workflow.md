# Phase 03: Autoresearch Workflow for BertNup

## Status: Complete

## Context

Adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for BertNup's nucleosome positioning classification task. The original autoresearch is designed for LLM pretraining (metric: val_bpb, modifiable: train.py). BertNup needs different choices.

### Key Differences from Original Autoresearch

| Aspect | Original (LLM Pretraining) | BertNup (Classification) |
|--------|---------------------------|--------------------------|
| Metric | val_bpb (bits per byte) | val_loss (cross-entropy, lower=better) |
| Modifiable file | `train.py` (single file) | YAML configs + model hyperparams |
| Fixed files | `prepare.py` | `bertnup/` package (models, data, training) |
| Time budget | 5 min wall-clock | 5 min wall-clock (same) |
| Task | Language modeling | Binary classification |
| Output | val_bpb, peak_vram_mb, mfu | val_loss, val_acc, val_auc, peak_vram_mb |

## Overview

Create `autoresearch/` directory with the autonomous experiment loop infrastructure. The agent modifies YAML configs, runs training, and keeps/discards based on val_loss.

## Requirements

1. **Fixed evaluation harness** — `bertnup/training/trainer.py` is read-only for the agent
2. **Modifiable configs** — Agent edits YAML files in `configs/` (hyperparams, model settings)
3. **Fixed time budget** — Each experiment runs for exactly 5 minutes
4. **Metric** — val_loss (lower = better), logged alongside val_acc and val_auc
5. **Results tracking** — TSV file with commit hash, metrics, status, description
6. **Git-based iteration** — Keep advances branch, discard reverts

## Implementation Steps

### 1. Create `autoresearch/program.md`

Agent instructions (the "research program"). Human edits this file to steer research direction.

Contents:
- Setup instructions (run `scripts/setup-server.sh`, sync data)
- In-scope files the agent can read
- Modifiable vs fixed boundaries
- Experiment loop instructions
- Results logging format
- Ideas for what to try (learning rates, pooling, heads, LoRA, etc.)

### 2. Create `autoresearch/run-experiment.sh`

Single experiment runner script:
```bash
#!/bin/bash
# Runs one BertNup training experiment with a fixed time budget.
# Usage: ./autoresearch/run-experiment.sh <config.yaml> <data_dir>
# Output: writes metrics to stdout in parseable format

CONFIG=${1:-configs/dnabert1.yaml}
DATA_DIR=${2:-Data/Stratified_K_fold_data/HS_LC/split_0}
TIME_BUDGET=${AUTORESEARCH_TIME_BUDGET:-300}  # 5 min default

# Run training with timeout
timeout $TIME_BUDGET /Users/danghiep/miniforge3/envs/bertnup/bin/python -m bertnup.cli \
  train "$DATA_DIR" --config "$CONFIG" 2>&1 | tee run.log

# Extract metrics from output
# Expected output: "val_loss: X.XXXX, val_acc: X.XXXX, val_auc: X.XXXX, vram_mb: XXXX"
grep -E "^val_loss:|^val_acc:|^val_auc:|^vram_mb:" run.log || echo "CRASH"
```

### 3. Create `autoresearch/results.tsv`

Header-only TSV (gitignored):
```
commit	val_loss	val_acc	val_auc	vram_gb	status	description
```

### 4. Add time-budget metric output to trainer

**Modify** `bertnup/training/trainer.py` to print parseable metrics at the end of `run_training()`:
```python
# After evaluation
peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
print(f"val_loss: {loss:.6f}")
print(f"val_acc: {acc:.4f}")
print(f"val_auc: {auc:.4f}")
print(f"vram_mb: {peak_vram:.1f}")
```

This requires adding loss tracking to `run_training()` — currently it only prints sn/sp/acc/f1/mcc/auc but not val_loss. We need to capture val_loss from the best model checkpoint or re-evaluate.

### 5. Add `autoresearch/` to `.gitignore`

Ignore `results.tsv` and `run.log`:
```
autoresearch/results.tsv
autoresearch/run.log
```

## Related Code Files

- **Create:** `autoresearch/program.md`, `autoresearch/run-experiment.sh`
- **Modify:** `bertnup/training/trainer.py` (add parseable metric output)
- **Modify:** `.gitignore` (ignore autoresearch artifacts)

## Agent Experiment Ideas (for program.md)

The agent would be instructed to try:
1. **Hyperparameters** — lr (1e-5 to 5e-4), warmup ratio, batch size, epochs
2. **Pooling** — mean vs max vs attention pooling
3. **Classification head** — single vs enhanced head
4. **Layer reinit** — reinitialize last N layers (0-6)
5. **LoRA** — enable/disable, vary rank (4/8/16/32) and alpha
6. **Regularization** — dropout (0.0-0.3), weight decay, gradient clipping
7. **Precision** — fp32 vs 16-mixed vs bf16
8. **Data augmentation** — reverse complement augmentation
9. **Model selection** — DNABERT-1 k={3,4,5,6}, DNABERT-2, NT variants

## Success Criteria

- `autoresearch/run-experiment.sh configs/dnabert1.yaml Data/...` runs a single timed experiment
- Metrics are printed in parseable format (grep-friendly)
- `program.md` provides enough context for an autonomous agent to run experiments
- Results TSV can track keep/discard decisions with git commits

## Dependencies

- Phase 01 (server setup) — need GPU for training
- Phase 02 (remote training) — autoresearch uses the same sync+run infrastructure

## Next Steps

- Run autoresearch loop on vast.ai GPU overnight
- Compare with reproduce-paper-experiments plan results
