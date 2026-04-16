# Code Review: Phases 01+02 Implementation

## Scope

- Files: 13 source files across `bertnup/config.py`, `bertnup/models/`, `bertnup/data/`, `bertnup/training/`, `bertnup/cli.py`, `configs/default.yaml`
- LOC: ~680 new/modified
- Focus: Phases 01 (Training Pipeline) + 02 (Architecture Upgrades)
- Scout findings: 2 critical, 3 high, 4 medium

## Overall Assessment

The refactoring is well-structured with clean module boundaries (heads, pooling, augmentation). Config layering and CLI wiring are correct. However, there are two **blocking bugs** that will crash at runtime, plus a backward compatibility break for existing checkpoints.

---

## Critical Issues (Blocking)

### C1. LoRA params crash `create_model()` with TypeError

**File:** `bertnup/models/__init__.py:18-28`, `bertnup/models/dnabert1.py`, `bertnup/models/dnabert2.py`

The factory passes `use_lora`, `lora_rank`, `lora_alpha` in `common_kwargs` to both `BertNupV1()` and `BertNupV2()`. Neither subclass accepts these parameters in their `__init__` signatures. The parent `BertNupBase.__init__` accepts them, but Python's MRO means the subclass signature wins.

```python
# __init__.py:18-28 passes:
common_kwargs = dict(
    ...
    use_lora=model_config.use_lora,     # <-- V1 and V2 don't accept this
    lora_rank=model_config.lora_rank,   # <-- TypeError: unexpected keyword argument
    lora_alpha=model_config.lora_alpha, # <-- TypeError: unexpected keyword argument
)
```

**Repro:** `create_model(ModelConfig(use_lora=True), 100)` raises `TypeError: BertNupV1.__init__() got an unexpected keyword argument 'use_lora'`.

**Fix:** Add `use_lora`, `lora_rank`, `lora_alpha` to both `BertNupV1.__init__` and `BertNupV2.__init__` signatures, and pass them through to `super().__init__()`.

### C2. EnhancedHead BatchNorm1d crashes with batch_size=1 in training

**File:** `bertnup/models/heads.py:38`

`EnhancedHead` uses `nn.BatchNorm1d` which requires batch_size > 1 in training mode. When the last training batch has exactly 1 sample, this raises `ValueError: Expected more than 1 value per channel when training`.

```python
# heads.py:38
self.bn1 = nn.BatchNorm1d(hidden_size)  # Crashes in train() with batch=1
```

**Fix options (pick one):**
1. Replace `BatchNorm1d` with `LayerNorm` (works with any batch size)
2. Replace with `nn.GroupNorm(1, hidden_size)` (also batch-size independent)
3. Add `drop_last=True` to training DataLoader (loses 1 sample per epoch)
4. Use `nn.InstanceNorm1d` instead

Recommendation: Option 1 (`LayerNorm`) is the standard fix in transformer literature.

---

## High Priority

### H1. Backward compatibility break with old checkpoints

**File:** `bertnup/models/base.py:41`, `bertnup/models/dnabert1.py`

Old `BertNup` model (commit `1ed81a3`) stored classifier weights as `dropout1.*` and `linear1.*`. The refactored code stores them as `classifier.dropout.*` and `classifier.linear.*`. Existing checkpoints from the old code cannot be loaded.

| Old state_dict key | New state_dict key |
|---|---|
| `linear1.weight` | `classifier.linear.weight` |
| `linear1.bias` | `classifier.linear.bias` |
| `dnabert.*` | `dnabert.*` (unchanged) |

**Impact:** `run_evaluation()`, `export_bert_weights()`, and `load_from_checkpoint()` all fail with old checkpoints.

**Fix:** Add a checkpoint migration utility or a `strict=False` load with key remapping in `run_evaluation`. Document the breaking change.

### H2. `warmup_ratio` config field is dead -- never used

**File:** `bertnup/config.py:38`, `bertnup/cli.py:38-39`, `bertnup/models/__init__.py:16`

`TrainingConfig.warmup_ratio` (default 0.1) is accepted from CLI and stored in config, but the factory hardcodes `warmup_steps = int(num_training_steps * 0.1)`. If a user sets `--warmup-ratio 0.2`, it has no effect.

**Fix:** Use `config.training.warmup_ratio` in `create_model()`:
```python
warmup_steps = int(num_training_steps * config.training.warmup_ratio)
```
This requires passing the full config (or just warmup_ratio) to `create_model()`.

### H3. Hparams mutation after model creation is fragile

**File:** `bertnup/training/trainer.py:93-96`

```python
model = create_model(config.model, num_training_steps)
model.hparams.learning_rate = config.training.learning_rate      # mutation
model.hparams.weight_decay = config.training.weight_decay        # mutation
model.hparams.lr_scheduler_type = config.training.lr_scheduler_type  # mutation
```

`learning_rate` and `weight_decay` are already passed via `create_model()` through `config.model`, but `lr_scheduler_type` is not. The mutation bypasses Lightning's hyperparameter tracking and will cause checkpoint reload to use stale values.

**Fix:** Pass `lr_scheduler_type` through `create_model()` or the model constructor instead of mutating hparams post-hoc.

---

## Medium Priority

### M1. Dead code in `run_evaluation()`

**File:** `bertnup/training/trainer.py:262`

```python
model_class = BertNupV1 if mc.type == "dnabert1" else BertNupV2  # dead
```

This line is unreachable because `BertNupV1` is not imported yet at line 262. The actual imports happen at lines 268-269, and the logic is duplicated at lines 271-274. Remove the dead line.

### M2. `augment_with_reverse_complement` uses slow `iterrows()`

**File:** `bertnup/data/augmentation.py:24`

`iterrows()` is O(n) with pandas overhead. For large datasets (100k+ sequences), this creates a noticeable bottleneck. Vectorized alternative:

```python
rc_series = df["sequence"].apply(
    lambda s: str(DNASequence(s).reverse_complement())
)
rc_df = pd.DataFrame({"sequence": rc_series, "label": df["label"]})
return pd.concat([df, rc_df], ignore_index=True)
```

### M3. LoRA target modules are BERT-specific -- won't match DNABERT-2

**File:** `bertnup/models/base.py:60`

```python
target_modules=["query", "key", "value", "dense"],
```

These module names match BERT (DNABERT-1) but DNABERT-2 uses a different architecture (BertAlibi with different naming). `get_peft_model` will silently apply LoRA to 0 modules and print a warning, but the model will train without LoRA actually being applied.

**Fix:** Make `target_modules` configurable per model type, or auto-detect based on backbone architecture.

### M4. `_compute_loss_and_probas` creates dummy loss tensor on GPU

**File:** `bertnup/models/base.py:74`

```python
loss = torch.tensor(0.0, device=logits.device)
```

When `labels is None` (inference mode), a GPU tensor is created unnecessarily. Minor perf issue but also means `predict_step` returns a loss tensor. This is the pre-existing behavior, not a regression.

---

## Low Priority

### L1. `num_training_steps` can be 0 with large `gradient_accumulation_steps`

**File:** `bertnup/training/trainer.py:68`

```python
num_training_steps = len(train_loader) * tc.epochs // tc.gradient_accumulation_steps
```

Integer division can produce 0. The LR scheduler will then have 0 total steps, producing a constant LR of 0.0. Add a `max(1, ...)` guard.

### L2. `run_evaluation` always imports both model classes

**File:** `bertnup/training/trainer.py:268-269`

Both `BertNupV1` and `BertNupV2` are imported unconditionally. Only the needed one should be imported.

### L3. `set_seed` forces deterministic CUDA

**File:** `bertnup/seed.py:18-19`

`cudnn.deterministic = True` and `cudnn.benchmark = False` are set unconditionally, even on CPU-only systems. The `torch.cuda` calls may warn or fail silently on CPU-only installs.

---

## Edge Cases Found by Scout

1. **Checkpoint key mismatch** (confirmed C1 above): old `BertNup` checkpoints have `linear1.*` keys; new `BertNupV1` expects `classifier.linear.*`. Verified with state_dict inspection.
2. **LoRA factory crash** (confirmed C2 above): `create_model()` with `use_lora=True` raises `TypeError`.
3. **BatchNorm1d with batch=1** (confirmed C2 above): `EnhancedHead` crashes in training mode.
4. **Last batch single sample**: If training set size is not divisible by batch_size, the last batch has 1 sample and crashes with EnhancedHead.
5. **DNABERT-2 missing dependency**: `einops` is required by DNABERT-2's model code but not listed in requirements.txt/pyproject.toml.

---

## Positive Observations

- Clean module extraction: `heads.py` and `pooling.py` follow single-responsibility principle
- Factory pattern in `create_head()`, `create_pooling()`, `create_dataset()`, `create_model()` is consistent
- Config layering (YAML defaults -> user config -> CLI overrides) is well-implemented
- `augment_with_reverse_complement` correctly preserves labels and sequence length
- Reverse complement handles non-standard bases (`N`) correctly via the complement map
- New config fields have sensible defaults that maintain backward compatibility at the config level
- LoRA implementation correctly defers `peft` import with a clear error message
- `on_validation_epoch_end` properly handles empty outputs
- AUC guard for single-class batches (`if 0 not in labels or 1 not in labels`) is correct

---

## Recommended Actions (Priority Order)

1. **[CRITICAL]** Add `use_lora`, `lora_rank`, `lora_alpha` params to `BertNupV1.__init__` and `BertNupV2.__init__`, pass through to `super().__init__()`.
2. **[CRITICAL]** Replace `BatchNorm1d` with `LayerNorm` in `EnhancedHead` to fix batch_size=1 crash.
3. **[HIGH]** Add checkpoint migration utility or document breaking change for old checkpoints.
4. **[HIGH]** Wire `warmup_ratio` from config through to `create_model()` instead of hardcoding `0.1`.
5. **[HIGH]** Pass `lr_scheduler_type` through model constructor instead of mutating hparams.
6. **[MEDIUM]** Remove dead `model_class` assignment in `run_evaluation()`.
7. **[MEDIUM]** Make LoRA `target_modules` configurable per model type.
8. **[LOW]** Add `max(1, ...)` guard for `num_training_steps` calculation.

---

## Metrics

- Type Coverage: Good (dataclasses, type hints throughout)
- Test Coverage: Not measured (no test suite found)
- Linting Issues: Clean (no syntax errors, imports verified)
- Backward Compat: BROKEN (old checkpoints won't load)

## Unresolved Questions

1. Should `create_model()` accept the full `Config` object instead of just `ModelConfig` + `num_training_steps`? This would simplify the warmup_ratio and lr_scheduler_type wiring.
2. Is there an existing collection of old checkpoints that need migration? If so, a migration script is needed.
3. Should `einops` be added to `pyproject.toml` dependencies for DNABERT-2 support?
4. The `attention.py` module uses `BertForSequenceClassification` directly (not the new models). Is this intentional, or should it be updated to use `BertNupV1`'s attention weights via the new architecture?

**Status:** DONE_WITH_CONCERNS
**Summary:** Two critical runtime bugs (LoRA factory crash, BatchNorm batch=1 crash) and one backward compatibility break found. All confirmed reproducible. Positive architecture overall but needs fixes before merge.
**Concerns:** LoRA is completely broken at the factory level. EnhancedHead will crash on real training data. Old checkpoints are incompatible without migration.
