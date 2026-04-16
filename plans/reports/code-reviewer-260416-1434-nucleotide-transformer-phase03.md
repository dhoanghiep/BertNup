# Code Review: Phase 03 -- Nucleotide Transformer v2 Integration

**Date:** 2026-04-16
**Scope:** 7 files (1 new model class, 5 modified, 1 new config)
**LOC delta:** ~30 net new (model class 57 lines, config 7 lines, plus scattered changes)
**Focus:** NT v2 integration correctness, backward compatibility, edge cases

## Overall Assessment

The integration is clean and consistent with existing V1/V2 patterns. The new `BertNupNT` class is a near-structural clone of `BertNupV2` (intentional -- both use `AutoModel` + custom pooling). No regressions in DNABERT-1/2 paths. However, one **critical** config auto-detection gap will cause a runtime shape mismatch for CLI users who omit the config file.

---

## Critical Issues

### C1. `hidden_size` mismatch when NT model is specified via CLI without config YAML

**File:** `bertnup/config.py` lines 103-107

When a user runs:
```bash
bertnup train DATA_DIR --model-name InstaDeepAI/nucleotide-transformer-500m-human-ref
```

The type is correctly auto-detected as `nucleotide_transformer`, but `hidden_size` stays at 768 (the `default.yaml` default). The NT 500m model outputs 1280-dim embeddings; the 2.5b model outputs 2560-dim. The classifier head is built with `Linear(768, 2)`, so the forward pass will crash with a matrix shape mismatch:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1280 and 768x2)
```

Similarly, `fixed_length` stays at 70 (should be 30 for NT). This causes wasted padding and slightly different truncation behavior.

**Fix:** In `load_config()`, after auto-detecting the model type, also adjust model-specific defaults. Suggested approach:

```python
# After line 107: base.model.type = detected
if detected == "nucleotide_transformer":
    if "model.hidden_size" not in (overrides or []):
        base.model.hidden_size = 1280  # default for 500m; user must override for 2.5b
    if "model.fixed_length" not in (overrides or []):
        base.model.fixed_length = 30
```

A more robust alternative: query `AutoConfig.from_pretrained(model_name).hidden_size` at config time, though this requires a network call for uncached models.

**Impact:** Any CLI invocation that uses an NT model name without `--config configs/nucleotide_transformer.yaml` will crash at the first training step.

### C2. `hidden_size=2560` not handled for NT 2.5b variant

**File:** `configs/nucleotide_transformer.yaml`

The config hardcodes `hidden_size: 1280`, which is correct for the 500m model but wrong for the 2.5b variant (`InstaDeepAI/nucleotide-transformer-2.5b-multi-species` outputs 2560-dim). Users who change the model name in the config without updating `hidden_size` will get the same shape mismatch crash.

**Fix:** Add a comment in the config:
```yaml
hidden_size: 1280  # 1280 for 500m models, 2560 for 2.5b models
```

Better: validate `hidden_size` against the loaded model in `BertNupNT.__init__()`:
```python
actual_hidden = self.backbone.config.hidden_size
if hidden_size != actual_hidden:
    import warnings
    warnings.warn(
        f"hidden_size={hidden_size} does not match model's hidden_size={actual_hidden}. "
        f"Overriding to {actual_hidden}."
    )
    self.hparams.hidden_size = actual_hidden
    # Re-create classifier with correct size
    self.classifier = create_head(head_type=head_type, hidden_size=actual_hidden, ...)
```

---

## High Priority

### H1. Attribute vs dict access inconsistency between BertNupNT and BertNupV2

**Files:** `bertnup/models/nucleotide_transformer.py:54` vs `bertnup/models/dnabert2.py:53`

```python
# BertNupNT:
x = self.pooler(output.last_hidden_state, attention_mask)     # attribute access
# BertNupV2:
x = self.pooler(output["last_hidden_state"], attention_mask)  # dict access
```

Both work because HuggingFace `ModelOutput` supports both access patterns. However, the inconsistency makes the codebase harder to maintain and signals copy-paste without normalization.

**Fix:** Pick one style and apply consistently. Attribute access (`output.last_hidden_state`) is preferred per HuggingFace conventions and is more IDE-friendly.

### H2. No validation that `hidden_size` matches the backbone in any model class

**Files:** `bertnup/models/nucleotide_transformer.py`, `bertnup/models/dnabert2.py`

Neither `BertNupNT` nor `BertNupV2` validate that the user-supplied `hidden_size` matches what the backbone actually outputs. `BertNupV1` gets away with this because DNABERT-1 always outputs 768, but both NT and DNABERT-2 have variants with different hidden sizes.

**Fix:** After loading the backbone, assert or warn:
```python
actual = self.backbone.config.hidden_size
assert hidden_size == actual, f"hidden_size={hidden_size} but backbone outputs {actual}"
```

---

## Medium Priority

### M1. NT config YAML missing `learning_rate`, `batch_size`, and `epochs` tuning

**File:** `configs/nucleotide_transformer.yaml`

The config only specifies model-level settings. For a 500m-parameter model with 1280 hidden dim, the default `batch_size_train: 32` may OOM on consumer GPUs. Consider adding:
```yaml
training:
  batch_size_train: 16
  learning_rate: 1e-4
  precision: "16-mixed"
```

### M2. `Dnabert2Dataset` name is misleading when used for NT

**File:** `bertnup/data/datasets.py:102`

```python
elif model_type in ("dnabert2", "nucleotide_transformer"):
```

The class `Dnabert2Dataset` is reused for NT, which is correct functionally (both tokenize raw sequences with a HuggingFace tokenizer + fixed-length padding). But the class name implies DNABERT-2 ownership. Consider renaming to `RawSequenceDataset` or adding a docstring note.

### M3. `fixed_length=30` for 147bp NT sequences is tight

The tokenizer produces 28 tokens for 147bp. With `fixed_length=30`, there are only 2 padding tokens. If any future NT variant uses a different tokenization scheme (e.g., overlapping k-mers), this would truncate. Consider `fixed_length: 32` for a small safety margin.

---

## Low Priority

### L1. Import ordering in `__init__.py`

`BertNupNT` import is after `BertNupV2` (alphabetical order would put it between base and dnabert1). Minor style note.

### L2. Plan file says `trust_remote_code=True` but implementation correctly omits it

The plan at `phase-03-nucleotide-transformer.md:49` says to use `trust_remote_code=True`, but the implementation correctly does not. This is the right decision (NT uses standard ESM architecture, no custom code). The plan file should be updated to reflect the actual implementation.

---

## Edge Cases Found

| ID | Scenario | Risk | Status |
|----|----------|------|--------|
| C1 | CLI with NT model name, no `--config` | Shape mismatch crash | **Open** |
| C2 | NT 2.5b with `hidden_size=1280` from config | Shape mismatch crash | **Open** |
| H2 | Any model variant with wrong `hidden_size` | Silent until first forward pass | **Open** |
| M3 | Token count exceeds `fixed_length=30` | Truncation | Low risk for current NT variants |

---

## Backward Compatibility

- **DNABERT-1 path:** Unchanged. `_detect_model_type()` still returns `"dnabert1"` for `DNA_bert*` names. `create_model()` dispatch unchanged. `create_dataset()` dispatch unchanged.
- **DNABERT-2 path:** Unchanged. `trust_remote_code=True` still only applied when `mc.type == "dnabert2"`.
- **Checkpoint loading:** `run_evaluation()` correctly maps `"nucleotide_transformer"` to `BertNupNT`. No existing checkpoints affected.
- **Config:** `default.yaml` unchanged. NT config is a separate file.

No regressions detected.

---

## Positive Observations

1. **Consistent architecture pattern.** `BertNupNT` follows the exact same subclass structure as `BertNupV2`, making the codebase predictable.
2. **Correct `trust_remote_code` handling.** NT does not need it (verified by loading the model), and the code correctly only sets `trust_remote=True` for `dnabert2`.
3. **Clean factory extension.** Both `create_model()` and `create_dataset()` use the same `elif` pattern, no special-casing.
4. **`save_hyperparameters()` inherited from base.** Checkpoint save/load will work correctly for NT.
5. **No tokenization divergence.** Reusing `Dnabert2Dataset` was the right call -- NT tokenizer uses the same `AutoTokenizer` + raw sequence + fixed-length padding pattern.

---

## Recommended Actions

1. **[Critical]** Fix `load_config()` to auto-adjust `hidden_size` and `fixed_length` when NT type is detected. This is the only blocking issue.
2. **[High]** Add `hidden_size` validation in `BertNupNT.__init__()` as a safety net against misconfiguration.
3. **[High]** Normalize output access style (attribute vs dict) across `BertNupNT` and `BertNupV2`.
4. **[Medium]** Update the NT config YAML with training-specific tuning and a comment about 2.5b `hidden_size`.
5. **[Low]** Update the plan file to note that `trust_remote_code=True` is not needed for NT.

---

## Metrics

- Type Coverage: Full (all model types handled in factory, config, dataset, trainer)
- Test Coverage: Structural tests passed (10/10 in test report); no end-to-end training test
- Linting Issues: Not run (no linter configured in repo)
- Backward Compatibility: 100% -- no regressions in DNABERT-1/2 paths

---

## Unresolved Questions

1. Should `hidden_size` be auto-detected from the model config at load time? This would require a network call but eliminate an entire class of misconfiguration bugs.
2. What is the target GPU memory for NT 500m training? This determines whether `batch_size_train: 32` is realistic or should be reduced in the NT config.
3. Is NT 2.5b (2560 hidden dim) an intended target? If so, the config and docs need explicit guidance on `hidden_size` per variant.
