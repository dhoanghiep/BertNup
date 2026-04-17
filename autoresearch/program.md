# BertNup Autoresearch

Autonomous hyperparameter and architecture search for nucleosome positioning prediction.

## Setup

1. **Agree on a run tag** based on today's date (e.g. `apr18`). Branch `autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Sync to server**: `./scripts/train-remote.sh --sync-data -- train Data/Stratified_K_fold_data/HS_LC/split_0 --config configs/dnabert1.yaml` to verify everything works.
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row.
5. **Confirm and go**.

## Experimentation

Each experiment runs on a single GPU (RTX 5060 Ti, 16GB). Training runs for up to **20 minutes** wall-clock or until training completes (10 epochs), whichever comes first. Launch via: `./autoresearch/run-experiment.sh configs/<config>.yaml <data_dir>`.

**What you CAN do:**
- Modify YAML config files in `configs/` — hyperparameters, model settings, pooling, head type, etc.
- Create new config files for experiments.

**What you CANNOT do:**
- Modify `bertnup/` package code (models, data, training). These are fixed.
- Install new packages.
- Modify the evaluation harness.

**The goal: get the lowest val_loss.** Everything is fair game: learning rate, warmup, batch size, epochs, pooling, head type, dropout, LoRA, precision, layer reinit. The only constraint is the code runs without crashing.

**Simplicity criterion**: All else equal, simpler is better. A small improvement from deleting complexity is better than equal improvement from adding it.

**The first run**: Always establish the baseline first — run the training with default config as-is.

## Output format

When the script finishes it prints a summary like:

```
---
val_loss:         0.351234
val_acc:          0.8523
val_auc:          0.9234
training_seconds: 180.5
peak_vram_mb:     4096.2
num_params_M:     110.1
```

Extract metrics:
```
grep -E "^val_loss:|^val_acc:|^val_auc:|^peak_vram_mb:" run.log
```

## Logging results

Log to `autoresearch/results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_loss	val_acc	val_auc	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss (e.g. 0.351234) — use 9.999999 for crashes
3. val_acc (e.g. 0.8523)
4. val_auc (e.g. 0.9234)
5. peak vram in GB (e.g. 4.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description

Example:
```
commit	val_loss	val_acc	val_auc	vram_gb	status	description
a1b2c3d	0.351234	0.8523	0.9234	4.0	keep	baseline
b2c3d4e	0.328901	0.8612	0.9312	4.1	keep	increase LR to 5e-5
c3d4e5f	0.380000	0.8300	0.9000	4.0	discard	switch to max pooling
```

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/apr18`).

LOOP FOREVER:

1. Look at git state: current branch/commit
2. Edit a config in `configs/` with an experimental idea
3. git commit
4. Run experiment: `./autoresearch/run-experiment.sh configs/<config>.yaml <data_dir> > run.log 2>&1`
5. Read results: `grep -E "^val_loss:|^peak_vram_mb:" run.log`
6. If grep empty → crashed. Check `tail -n 50 run.log`. Fix or skip.
7. Record in results.tsv (do NOT commit results.tsv)
8. If val_loss improved (lower) → keep the commit
9. If val_loss equal or worse → `git reset --hard` back

**Timeout**: Each experiment runs up to 20 min. If exceeded 25 min, kill and treat as failure.

**Crashes**: Easy fix (typo, missing import) → fix and re-run. Fundamentally broken → skip, log "crash".

**NEVER STOP**: Do NOT pause to ask. The human may be asleep. Run until manually stopped.

## Ideas to try

1. **Learning rate**: 1e-5 to 5e-4 (default 2e-5)
2. **Warmup ratio**: 0.0 to 0.3 (default 0.1)
3. **Batch size**: 16, 32, 64 (default 32)
4. **Pooling**: mean, max, attention (default mean)
5. **Head type**: single, enhanced (default single)
6. **Dropout**: 0.0 to 0.3 (default 0.1)
7. **Layer reinit**: 0 to 6 (default 0)
8. **LoRA**: rank 4/8/16/32, alpha 8/16/32
9. **Precision**: 32, 16-mixed, bf16
10. **Gradient accumulation**: 1, 2, 4
11. **Reverse complement augmentation**: true/false
12. **Weight decay**: 0.0 to 0.1
13. **K-mer size**: 3, 4, 5, 6 (DNABERT-1 only)
14. **Model backbone**: DNABERT-1, DNABERT-2, NT variants
