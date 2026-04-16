# Code Review: Phases 04-05 -- Evaluation/Tracking + Extended Backbones

**Date:** 2026-04-16
**Scope:** 12 files (3 new models, 5 modified, 2 new modules, 3 new configs)
**LOC delta:** ~500 net new
**Focus:** Class weights, bootstrap CI, cross-species eval, Evo/HyenaDNA/Caduceus backbones, ensemble, benchmark
**Scout findings:** See Edge Cases section

## Overall Assessment

Phase 4 adds meaningful evaluation capabilities (bootstrap CIs, cross-species eval, significance testing, class weights, WandB tracking). Phase 5 extends the model zoo cleanly using the established `BertNupBase` subclass pattern. However, there are **two critical bugs** (one security-related, one correctness), several **trust_remote_code inconsistencies** that will cause runtime failures for Evo/HyenaDNA/Caduceus, and significant **DRY violations** in the model-class dispatch that will break silently when new backbones are added.

---

## Critical Issues

### C1. `run_evaluation()` in trainer.py does not support Evo/HyenaDNA/Caduceus -- KeyError crash

**File:** `/Users/danghiep/dev/BertNup/bertnup/training/trainer.py:296-301`

```python
model_classes = {
    "dnabert1": BertNupV1,
    "dnabert2": BertNupV2,
    "nucleotide_transformer": BertNupNT,
}
model_class = model_classes[mc.type]
```

This hardcoded map only has three entries. If `mc.type` is `"evo"`, `"hyena_dna"`, or `"caduceus"`, the code raises a `KeyError` and crashes. The same function in `evaluation.py` and `benchmark.py` handles extended backbones via try/except imports, but `run_evaluation()` was not updated.

**Impact:** `bertnup evaluate` command is broken for any new backbone model.

**Fix:** Use the same pattern as `ensemble.py`/`benchmark.py`, or better, centralize the model-class registry in `__init__.py`:

```python
# In bertnup/models/__init__.py
def get_model_class(model_type: str) -> type:
    _registry = {
        "dnabert1": BertNupV1,
        "dnabert2": BertNupV2,
        "nucleotide_transformer": BertNupNT,
    }
    if _EVO_AVAILABLE:
        _registry["evo"] = BertNupEvo
    if _SSM_AVAILABLE:
        _registry["hyena_dna"] = BertNupHyenaDNA
        _registry["caduceus"] = BertNupCaduceus
    if model_type not in _registry:
        raise ValueError(f"Unknown model type: {model_type}")
    return _registry[model_type]
```

Then all callers use `get_model_class(mc.type)` instead of maintaining their own dicts. This is a DRY fix that also prevents the registry from drifting.

### C2. `trust_remote_code` not set for Evo/HyenaDNA/Caduceus in evaluation.py, ensemble.py, benchmark.py

**Files:**
- `/Users/danghiep/dev/BertNup/bertnup/training/evaluation.py:43`
- `/Users/danghiep/dev/BertNup/bertnup/models/ensemble.py:93`
- `/Users/danghiep/dev/BertNup/bertnup/training/benchmark.py:57`

All three locations have:
```python
trust_remote = mc.type == "dnabert2"
```

But the model backbones themselves load with `trust_remote_code=True` (see `evo.py:54,59`, `ssm.py:60,121`). The tokenizer also needs `trust_remote_code=True` for these models because they ship custom tokenizers. Without it, `AutoTokenizer.from_pretrained()` will fail with:

```
ValueError: Loading this model requires you to pass trust_remote_code=True
```

The correct logic already exists in `trainer.py:63`:
```python
trust_remote = mc.type in ("dnabert2", "evo", "hyena_dna", "caduceus")
```

**Impact:** Any evaluation, ensemble prediction, or benchmarking with Evo/HyenaDNA/Caduceus models crashes immediately.

**Fix:** Apply the same `trust_remote_code` logic from `trainer.py:63` to all three files. Or extract it into a shared helper:

```python
def _needs_trust_remote(model_type: str) -> bool:
    return model_type in ("dnabert2", "evo", "hyena_dna", "caduceus")
```

---

## High Priority

### H1. `CrossEntropyLoss` re-instantiated every forward pass

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/base.py:83`

```python
loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
```

This creates a new `CrossEntropyLoss` object on every training step. While functionally correct, it is wasteful and, more importantly, if `class_weights` is a registered buffer (on the correct device), the weight tensor is being moved implicitly each time. PyTorch `CrossEntropyLoss` does not register the weight as a parameter/buffer of the module, so device management relies on the weight being passed at construction time.

**Impact:** Minor performance overhead per batch. Device mismatch risk if the buffer and input are on different devices after model moves.

**Fix:** Create the loss function once in `__init__`:
```python
def __init__(self, ...):
    ...
    if class_weights is not None:
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self._loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
    else:
        self.class_weights = None
        self._loss_fn = nn.CrossEntropyLoss()

def _compute_loss_and_probas(self, logits, labels):
    ...
    if labels is not None:
        loss = self._loss_fn(logits, labels)
```

### H2. `run_kfold_cv()` ignores `use_class_weights` config flag

**File:** `/Users/danghiep/dev/BertNup/bertnup/training/trainer.py:207-268`

`run_training()` correctly handles `config.training.use_class_weights` (lines 119-125), computing inverse-frequency weights from the training CSV and setting them on the model. `run_cross_species_eval()` in `evaluation.py` also handles it (lines 94-106). But `run_kfold_cv()` does not apply class weights at all, despite the flag being in the shared config.

**Impact:** Users who enable `--use-class-weights` will get weighted loss for `train` and `evaluate_cross_species` but unweighted loss for `cross_validate`. Silent correctness bug.

**Fix:** Add the same class-weight computation logic from `run_training()` into the k-fold loop, inside the `for no_split in range(...)` block.

### H3. Division by zero in `compute_all_metrics` for degenerate predictions

**File:** `/Users/danghiep/dev/BertNup/bertnup/data/metrics.py:26-27`

```python
sn = tp / (tp + fn)
sp = tn / (tn + fp)
```

If the model predicts all-0 or all-1, one of `tp+fn` or `tn+fp` will be zero, causing `ZeroDivisionError`. The `f1_score` call above handles this with `zero_division=0`, but `sn` and `sp` are unprotected. This is a pre-existing bug (not introduced in Phase 4/5) but is now amplified because `compute_metrics_with_ci` calls `compute_all_metrics` hundreds of times in bootstrap loops, making a degenerate bootstrap sample more likely to trigger it.

**Fix:**
```python
sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
```

### H4. `compute_metrics_with_ci` crashes when all bootstrap samples are single-class

**File:** `/Users/danghiep/dev/BertNup/bertnup/data/metrics.py:64-72`

If the dataset is small and extremely imbalanced, it is possible for every bootstrap sample to contain only one class. In that case, `bootstrap_results[name]` is an empty list, `np.array([])` has length 0, and `np.mean`, `np.percentile` on an empty array returns `nan`. This propagates as `NaN` in the output dict and silently corrupts downstream comparisons.

**Fix:** Add a guard:
```python
if len(vals) == 0:
    result[name] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    continue
```

---

## Medium Priority

### M1. DRY violation: model-class dispatch duplicated 4 times

**Files:**
- `/Users/danghiep/dev/BertNup/bertnup/models/__init__.py` -- `create_model()`
- `/Users/danghiep/dev/BertNup/bertnup/training/trainer.py:296` -- `run_evaluation()`
- `/Users/danghiep/dev/BertNup/bertnup/training/evaluation.py` -- not present (calls `_evaluate_model_on_test`)
- `/Users/danghiep/dev/BertNup/bertnup/models/ensemble.py:68` -- `predict()`
- `/Users/danghiep/dev/BertNup/bertnup/training/benchmark.py:32` -- `_load_model_and_predict()`

Each location independently imports model classes and builds its own dispatch dict. C1 above is a direct consequence of this pattern. The fix in C1 (centralized `get_model_class()`) eliminates all four copies.

### M2. `BertNupHyenaDNA` and `BertNupCaduceus` have identical `forward()` logic

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/ssm.py`

Lines 65-77 and 126-137 are character-for-character identical. The fallback logic for extracting hidden states from non-standard model outputs is the same. Extract to a shared helper or a common base:

```python
def _extract_hidden(output, fallback_to_logits=True):
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    elif isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    elif isinstance(output, (tuple, list)):
        return output[0]
    elif fallback_to_logits:
        return output.logits
    raise ValueError(f"Cannot extract hidden state from {type(output)}")
```

### M3. `_evaluate_model_on_test` hardcodes `accelerator="auto"`, ignores config

**File:** `/Users/danghiep/dev/BertNup/bertnup/training/evaluation.py:54`

```python
trainer = Trainer(accelerator="auto", devices=1, enable_model_summary=False)
```

All other Trainer instantiations use `_get_accelerator(config.device)`, which respects the user's `--device` config. This function ignores it.

**Fix:** Import and use `_get_accelerator` from `trainer.py` (it's already imported at line 81 in the same file for `run_cross_species_eval`).

### M4. `run_cross_species_eval` hardcodes checkpoint directory

**File:** `/Users/danghiep/dev/BertNup/bertnup/training/evaluation.py:108`

```python
checkpoint_callback = ModelCheckpoint(
    dirpath="model_checkpoint/cross_species",
```

This ignores `config.output.checkpoint_dir`. The user could set `--config` with a custom checkpoint dir, and this function would still write to the hardcoded path.

**Fix:** Use `os.path.join(config.output.checkpoint_dir, "cross_species")`.

### M5. AUC computed but never logged in `on_validation_epoch_end`

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/base.py:118-119`

```python
_, _, _, _, _, auc = compute_all_metrics(probas, labels, verbose=0)
self.log("val_loss", loss)
```

The AUC is computed (at non-trivial cost) but only `val_loss` is logged. The `val_auc` metric is never logged, so it is invisible to early stopping, checkpointing (which monitors `val_loss`), and experiment trackers (WandB).

**Fix:** Add `self.log("val_auc", auc)` after the computation.

### M6. Unused `pickle` import in trainer.py

**File:** `/Users/danghiep/dev/BertNup/bertnup/training/trainer.py:8`

`import pickle` is present but never used anywhere in the file.

### M7. Evo model hardcodes `revision="1.1_fix"` without configurability

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/evo.py:54,59`

```python
config = AutoConfig.from_pretrained(
    pretrained_model_name, trust_remote_code=True, revision="1.1_fix"
)
self.backbone = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name, config=config, trust_remote_code=True, revision="1.1_fix"
)
```

If the Evo team releases a new revision or the user wants to use the default branch, this is not configurable. This should either be documented as intentional pinning or made configurable via the model config.

---

## Low Priority

### L1. `BertNupEvo` uses `AutoModelForCausalLM` while others use `AutoModel`

This is intentional (Evo is a causal LM, not an encoder), but worth a docstring note explaining why the class hierarchy diverges here.

### L2. Ensemble `predict()` re-reads CSV for labels instead of reusing dataset labels

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/ensemble.py:123-124`

After iterating through all models and building test sets, it re-reads the CSV with `pd.read_csv(test_path)` to get labels. The dataset already has this data. Minor I/O waste.

### L3. `bootstrap_auc_comparison` p-value is approximate, not documented as such

**File:** `/Users/danghiep/dev/BertNup/bertnup/data/metrics.py:104`

```python
"p_value": float(np.mean(diffs > 0) * 2),  # two-sided
```

This is a direct percentile-based p-value, not a proper permutation test. For a bioinformatics audience, the method should be documented. Additionally, the `* 2` multiplier can produce p-values > 1.0 when `np.mean(diffs > 0)` is > 0.5.

**Fix:** `min(float(np.mean(diffs > 0) * 2), 1.0)` or use `np.mean(np.abs(diffs) >= np.abs(np.mean(diffs)))` for a proper two-sided test.

### L4. `BertNupEvo.forward()` fallback uses `output.logits` as hidden state proxy

**File:** `/Users/danghiep/dev/BertNup/bertnup/models/evo.py:71-72`

The comment says "not ideal" and indeed it is: passing LM logits (vocab-size dimension) through a classifier expecting `hidden_size` dimension will crash with a shape mismatch. If this fallback is ever hit, it should raise an error instead of silently producing wrong output.

---

## Edge Cases Found by Scout

| ID | Scenario | Risk | Severity |
|----|----------|------|----------|
| C1 | `bertnup evaluate` with Evo/HyenaDNA/Caduceus checkpoint | KeyError crash | Critical |
| C2 | Tokenizer loading for Evo/HyenaDNA/Caduceus in evaluation/benchmark/ensemble | ValueError crash | Critical |
| H2 | `cross_validate --use-class-weights` silently ignores flag | Incorrect loss computation | High |
| H3 | Bootstrap sample with all-same predictions in CI computation | ZeroDivisionError | High |
| H4 | Extremely imbalanced dataset with all single-class bootstrap samples | NaN propagation | High |
| M3 | `evaluate_cross_species` with `--device cpu` runs on GPU instead | Ignores user config | Medium |
| M5 | `val_auc` never logged, invisible to WandB/early stopping | Silent metric loss | Medium |
| L3 | `p_value > 1.0` in bootstrap AUC comparison | Invalid statistics | Low |

---

## Positive Observations

1. **Consistent subclass pattern.** All three new backbones follow the exact `BertNupBase` subclass structure, making the codebase predictable.
2. **Graceful optional dependency handling.** `__init__.py` uses try/except imports for Evo/SSM, and `ensemble.py`/`benchmark.py` do the same. Models that lack deps fail clearly with `ImportError`.
3. **Class weights as registered buffer.** Using `register_buffer` ensures the weights move with the model across devices. This is the correct PyTorch pattern.
4. **Bootstrap implementations are sound.** `compute_metrics_with_ci` and `bootstrap_auc_comparison` use proper resampling with single-class skip guards.
5. **Checkpoint cleanup.** Both `run_training` and `run_cross_species_eval` delete the checkpoint file after evaluation, preventing disk accumulation.
6. **Config YAML separation.** Each backbone has its own config file, keeping defaults clean.

---

## Recommended Actions

1. **[Critical]** Fix `run_evaluation()` model registry to include Evo/HyenaDNA/Caduceus. Better: centralize `get_model_class()` in `__init__.py` and use it everywhere.
2. **[Critical]** Fix `trust_remote_code` in `evaluation.py`, `ensemble.py`, `benchmark.py` to include Evo/HyenaDNA/Caduceus. Extract a shared helper.
3. **[High]** Move `CrossEntropyLoss` instantiation to `__init__` to avoid per-step allocation.
4. **[High]** Add class-weight support to `run_kfold_cv()`.
5. **[High]** Guard division-by-zero in `compute_all_metrics` for `sn` and `sp`.
6. **[High]** Guard empty bootstrap results in `compute_metrics_with_ci`.
7. **[Medium]** Centralize model-class dispatch to eliminate 4x duplication (fixes C1 as side effect).
8. **[Medium]** Extract shared hidden-state extraction logic from `ssm.py`.
9. **[Medium]** Use `config.output.checkpoint_dir` in `run_cross_species_eval`.
10. **[Medium]** Log `val_auc` in `on_validation_epoch_end`.
11. **[Low]** Cap p-value at 1.0 in `bootstrap_auc_comparison`.
12. **[Low]** Remove unused `pickle` import.

---

## Metrics

- Type Coverage: Full (all 6 model types handled in factory + config auto-detection)
- Test Coverage: Structural tests passed per test report; no integration test for cross-species eval
- Linting Issues: Not run (no linter configured)
- Backward Compatibility: DNABERT-1/2/NT paths unchanged; new backbones are additive
- DRY Violations: 4 independent model-class dispatch maps (should be 1)

---

## Unresolved Questions

1. Should Evo's `revision="1.1_fix"` be configurable via `ModelConfig`? Pinning to a specific revision is brittle if the model repo changes.
2. Is `BertNupCaduceus`'s hidden state extraction truly identical to `BertNupHyenaDNA`'s, or do they have different output formats in practice? The identical `forward()` suggests the author assumed they do, but Caduceus uses Mamba and may not produce `last_hidden_state` at all.
3. Should the ensemble predictor be registered as a CLI subcommand? Currently it is only usable programmatically.
4. The `run_benchmark` pairwise comparison has O(n^2) bootstrap tests. For large model sets, this is expensive. Should it be capped or parallelized?
