# Code Review: Autoresearch Scripts & Trainer Metrics

**Reviewer:** code-reviewer
**Date:** 2026-04-18
**Branch:** feat/modernize-training-architecture
**Scope:** Shell scripts, trainer metric output, gitignore

## Scope

- `scripts/setup-server.sh` -- SSH setup for vast.ai GPU server
- `scripts/train-remote.sh` -- Remote training via rsync+ssh
- `autoresearch/run-experiment.sh` -- Single experiment runner with 20-min timeout
- `autoresearch/program.md` -- Agent instructions for autonomous research
- `bertnup/training/trainer.py` -- lines 169-184, parseable metric output
- `.gitignore` -- autoresearch artifact exclusions

**LOC:** ~250 (scripts ~150, trainer diffs ~15, gitignore ~3)
**Focus:** Recent/specific changes for autoresearch workflow

## Scout Findings

- `trainer.predict()` can return `None` entries if a batch raises -- `torch.cat(outputs)` will crash (pre-existing, affects all three call sites)
- `best_model_path` is unconditionally used in `load_from_checkpoint` and `os.remove` without existence check
- `run_training` deletes checkpoint via `os.remove` but never cleans the `checkpoint_dir` parent directory (orphaned empty dirs)

## Overall Assessment

The scripts are well-structured with consistent style. The trainer metric output is functional but has two correctness bugs that will surface in production. Shell scripts have a mix of acceptable security tradeoffs (vast.ai context) and one `eval` injection risk that should be fixed.

---

## Critical Issues

### C1. `training_seconds` label is misleading -- returns last train_loss value, not elapsed time

**File:** `bertnup/training/trainer.py:180`

```python
print(f"training_seconds: {trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item():.1f}")
```

`trainer.callback_metrics['train_loss']` is the **loss value from the last logged training step**, not a duration in seconds. The label `training_seconds` tells consumers this is elapsed wall-clock time. Any downstream consumer (autoresearch results logger, program.md parser) will record a loss value (e.g. 0.35) as if it were seconds.

**Fix:** Use `time.time()` or `trainer.fit_time` (available in PL 2.x via `trainer.callback_metrics` or manual timing):

```python
import time
start = time.time()
# ... trainer.fit(...)
elapsed = time.time() - start
print(f"training_seconds: {elapsed:.1f}")
```

**Impact:** All autoresearch timing data will be wrong. High -- blocks meaningful analysis.

### C2. `eval "$RSYNC_CMD"` with interpolated variables enables argument injection

**File:** `scripts/train-remote.sh:51,56,62`

```bash
RSYNC_CMD="rsync -avz ... -e 'ssh -p $SSH_PORT'"
eval "$RSYNC_CMD" "$LOCAL_DIR/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/"
```

If `SSH_PORT` or `SSH_HOST` contain shell metacharacters (from env vars or `--port`/`--host` args), the `eval` will interpret them. Since these default to `ssh1.vast.ai:10622`, the risk is low for normal usage, but a user passing `--port "10622 -oProxyCommand=malicious"` would get code execution.

**Fix:** Replace `eval` with an array:

```bash
RSYNC_CMD=(rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' -e "ssh -p $SSH_PORT")
"${RSYNC_CMD[@]}" "$LOCAL_DIR/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/" 2>&1 | tail -3
```

**Impact:** Code execution via crafted CLI args. Low probability in current vast.ai-only usage but violates defense-in-depth.

---

## High Priority

### H1. `best_model_score` is `None` when `save_top_k=1` and no validation runs complete

**File:** `bertnup/training/trainer.py:173`

The guard `if checkpoint_callback.best_model_score is not None` is correct, but `best_model_path` on line 164 is used unconditionally. If no validation epoch completes (e.g. very few training steps + high `val_check_interval`), `best_model_path` will be `None` and `load_from_checkpoint(None)` will crash with an unhelpful error.

**Fix:**

```python
if checkpoint_callback.best_model_path is None:
    raise RuntimeError("No checkpoint was saved. Check val_check_interval vs dataset size.")
best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
```

### H2. `os.remove(checkpoint_callback.best_model_path)` -- silent failure if file already gone

**File:** `bertnup/training/trainer.py:185,289`

If `load_from_checkpoint` or `trainer.predict` raises and is later caught upstream, the checkpoint file might not exist at removal time. In the autoresearch loop this could halt the entire experiment runner due to `set -e` propagation.

**Fix:**

```python
if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
    os.remove(checkpoint_callback.best_model_path)
```

### H3. `peak_vram` measured on CPU will always be 0.0

**File:** `bertnup/training/trainer.py:174`

```python
peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
```

This is correct in terms of the guard. However, `torch.cuda.max_memory_allocated()` returns peak for the **current device**. On multi-GPU systems this only reports device 0. Not a bug per se since `devices=1` is set, but worth noting. More importantly, if running on CPU, the metric silently reports 0.0 -- the autoresearch `program.md` says to log this as `vram_gb`, which will be `0.0` and indistinguishable from a crash per the logging spec ("use 0.0 for crashes").

**Recommendation:** When `torch.cuda.is_available()` is False, print `peak_vram_mb: N/A` instead of `0.0` so downstream consumers can distinguish CPU runs from GPU crashes.

### H4. Shell word splitting on `$SSH_CMD` -- unquoted variable expansion

**File:** `scripts/setup-server.sh:36,50,55,60` and `scripts/train-remote.sh:50,70`

```bash
SSH_CMD="ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST"
$SSH_CMD "some command"
```

`$SSH_CMD` is unquoted, relying on word splitting to expand into separate arguments. If `SSH_HOST` contains spaces (unlikely but possible via env var), this breaks. Also, `BERTNUP_ARGS[*]` on line 70 of `train-remote.sh` joins with the first character of IFS (space) which is correct for the remote shell, but arguments containing spaces will be split again on the remote side.

**Fix for local scripts:** Use an array:

```bash
SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_HOST}")
"${SSH_CMD[@]}" "some command"
```

**Fix for remote command (train-remote.sh:70):** The remote-side escaping is inherently fragile. For this use case (vast.ai, controlled inputs) it is acceptable, but document the limitation.

---

## Medium Priority

### M1. `StrictHostKeyChecking=no` disables SSH host verification

**Files:** `scripts/setup-server.sh:28`, `scripts/train-remote.sh:50`

This is acceptable for ephemeral vast.ai instances but should be scoped to a known host key pattern or at minimum documented. If these scripts are reused for non-vast servers, MITM is trivially possible.

**Recommendation:** Add a comment at the `SSH_CMD` definition explaining the tradeoff and that it is vast.ai-specific.

### M2. `run-experiment.sh` exit code handling with `set -e`

**File:** `autoresearch/run-experiment.sh:37`

```bash
timeout "$TIMEOUT" $PYTHON -m bertnup.cli train "$DATA_DIR" --config "$CONFIG" 2>&1
EXIT_CODE=$?
```

With `set -e`, if `timeout` returns non-zero (including 124 for timeout), the script exits immediately and never reaches `EXIT_CODE=$?`. The timeout/crash status reporting on lines 39-45 is dead code.

**Fix:** Disable exit-on-error for the timeout command:

```bash
set +e
timeout "$TIMEOUT" $PYTHON -m bertnup.cli train "$DATA_DIR" --config "$CONFIG" 2>&1
EXIT_CODE=$?
set -e
```

### M3. Variable `$PYTHON` is unquoted in `run-experiment.sh:35`

```bash
timeout "$TIMEOUT" $PYTHON -m bertnup.cli train "$DATA_DIR" --config "$CONFIG" 2>&1
```

If `BERTNUP_PYTHON` contains a space in the path, this will break. Quote it: `"$PYTHON"`.

### M4. `program.md` says "val_loss" but trainer prints "best validation loss" not "final validation loss"

The autoresearch program optimizes for the lowest `val_loss`, and the trainer correctly reports `checkpoint_callback.best_model_score` which is the best (minimum) validation loss across all epochs. This is correct for the use case. However, the variable name `best_val_loss` is printed as `val_loss:` -- this is fine for grep parsing but could confuse human readers of the code who might expect `val_loss` to mean the last epoch's validation loss.

**Recommendation:** Consider printing both `best_val_loss` and `final_val_loss` for diagnostic purposes. Low priority.

### M5. `configs/default.yaml` referenced in `run-experiment.sh` may not exist

**File:** `autoresearch/run-experiment.sh:14`

```bash
CONFIG="${1:-configs/default.yaml}"
```

The default fallback `configs/default.yaml` does not appear to exist in the repo. If run without arguments, the script will pass a nonexistent config to the training pipeline, which will crash with an unhelpful file-not-found error.

**Fix:** Either create `configs/default.yaml` or change the default to an existing config (e.g., `configs/dnabert1.yaml`), or add an explicit check:

```bash
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
```

---

## Low Priority

### L1. `setup-server.sh` uses `pip install --quiet` -- errors are hidden

The `2>&1 | tail -1` means only the last line of stderr is shown. If installation fails, the "Verifying" step will provide a clearer error, but the root cause is lost.

**Recommendation:** Add `|| echo "WARNING: Install step may have failed"` after each pip install.

### L2. `.gitignore` entries are incomplete for autoresearch

Only `autoresearch/results.tsv` and `autoresearch/run.log` are ignored. If the autoresearch agent creates temporary configs or logs, they could be accidentally committed.

**Recommendation:** Consider `autoresearch/run*.log` or `autoresearch/tmp/` if the agent generates temporary files.

### L3. `train-remote.sh` runs `pip install -e .` on every invocation

Line 70: `pip install -e . -q 2>/dev/null` runs before every training command. This is a defensive measure but adds ~5-10 seconds overhead per invocation. Acceptable for the use case.

---

## Positive Observations

- `set -euo pipefail` used consistently across all shell scripts
- Clear, well-structured CLI arg parsing in all scripts
- Environment variable overrides (`BERTNUP_SSH_HOST`, etc.) provide good flexibility
- The `-- ` separator in `train-remote.sh` cleanly separates wrapper args from passthrough args
- Trainer metric output uses `---` delimiter and consistent formatting for easy grep parsing
- `program.md` is thorough and well-structured as agent instructions
- `.gitignore` correctly excludes autoresearch artifacts without over-excluding

---

## Recommended Actions (Priority Order)

1. **Fix `training_seconds` metric** (C1) -- it outputs loss value instead of wall time. Blocks meaningful autoresearch timing analysis.
2. **Fix `set -e` vs `timeout` interaction** (M2) -- the crash/timeout status reporting is dead code.
3. **Replace `eval` with array** (C2) -- eliminates argument injection vector.
4. **Add `best_model_path` None guard** (H1) -- prevents cryptic crash on edge-case configurations.
5. **Quote `$PYTHON`** (M3) and **validate `$CONFIG` existence** (M5) -- defensive fixes.
6. **Soft-fail checkpoint removal** (H2) -- use existence check before `os.remove`.
7. **Document `StrictHostKeyChecking=no` scope** (M1) -- one-line comment.

---

## Metrics

- **Type Coverage:** Python code is typed (annotations present). Shell scripts have no typing.
- **Test Coverage:** No tests for the new metric output or shell scripts. (Shell scripts are manual-only.)
- **Linting Issues:** 0 syntax errors. One dead-code path (M2). One unquoted variable (M3).

## Unresolved Questions

1. Should the autoresearch `program.md` constrain which configs the agent can create, or is any file in `configs/` fair game? The current spec allows unlimited config creation which could pollute the directory.
2. The `run_training` function deletes the best checkpoint file but not the checkpoint directory. Should cleanup be more thorough (e.g., `shutil.rmtree(checkpoint_dir)`) to avoid accumulating empty directories across many autoresearch runs?
3. Is `configs/default.yaml` expected to be created, or should `run-experiment.sh` require an explicit config argument?

**Status:** DONE_WITH_CONCERNS
**Summary:** Two correctness bugs in trainer.py metric output (training_seconds is loss not time; set-e makes timeout status dead code). One eval injection risk in train-remote.sh. Several medium-priority robustness issues in shell scripts.
**Concerns:** The `training_seconds` bug (C1) will corrupt all autoresearch timing data and should be fixed before any experiments run. The `set -e` interaction with `timeout` (M2) means crash/timeout status is never reported.
