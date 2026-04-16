# Phase 01: Training Pipeline Modernization

**Priority:** High
**Status:** Pending
**Risk:** Low
**Dependencies:** None

## Overview

Modernize the training pipeline with established best practices that provide immediate, low-risk improvements. These are drop-in changes to existing code with minimal architectural impact.

## Key Insights

- Mixed precision (bf16) gives 2x speedup and 50% memory reduction with negligible accuracy loss
- Cosine LR schedule outperforms linear warmup for fine-tuning
- Gradient accumulation simulates larger batch sizes when memory-constrained
- Reverse complement augmentation doubles effective training data for free

## Requirements

### Functional
- Add mixed precision training support (bf16/fp16)
- Replace linear warmup schedule with cosine schedule with warmup
- Add gradient accumulation support
- Implement reverse complement data augmentation

### Non-Functional
- Backward compatible with existing configs (new features opt-in)
- No regression in existing model performance
- Keep PyTorch Lightning 2.x compatibility

## Related Code Files

### Modify
- `bertnup/models/base.py` — gradient accumulation in training_step
- `bertnup/training/trainer.py` — precision param, scheduler swap, augmentation in dataloaders
- `bertnup/config.py` — add `precision`, `gradient_accumulation_steps`, `lr_scheduler_type`, `augment_rc` fields
- `bertnup/cli.py` — add CLI args for new training options
- `configs/default.yaml` — add new config keys with sensible defaults

### New
- `bertnup/data/augmentation.py` — reverse complement augmentation logic

## Implementation Steps

### 1. Config updates (`config.py`, `configs/default.yaml`)
- Add `TrainingConfig.precision`: str = "32" (options: "32", "16-mixed", "bf16")
- Add `TrainingConfig.gradient_accumulation_steps`: int = 1
- Add `TrainingConfig.lr_scheduler_type`: str = "cosine" (options: "linear", "cosine")
- Add `TrainingConfig.augment_rc`: bool = False (reverse complement augmentation)

### 2. Mixed precision (`trainer.py`)
- Pass `precision=config.training.precision` to `Trainer()`
- Ensure bf16 works with AdamW and CrossEntropyLoss

### 3. Cosine LR schedule (`base.py`)
- In `configure_optimizers()`, check `lr_scheduler_type`
- Use `get_cosine_schedule_with_warmup` when cosine selected
- Keep linear as fallback option

### 4. Gradient accumulation (`trainer.py`, `base.py`)
- Pass `accumulate_grad_batches=config.training.gradient_accumulation_steps` to `Trainer()`
- Adjust `num_training_steps` calculation: `len(train_loader) * epochs // accumulation_steps`

### 5. Reverse complement augmentation (`augmentation.py`, `datasets.py`)
- Create `augment_with_rc(dataframe)` that doubles each sequence with its reverse complement
- Apply in `Dnabert1Dataset` and `Dnabert2Dataset` when `augment_rc=True`
- Use existing `DNASequence.reverse_complement()` from `sequences.py`

### 6. CLI updates (`cli.py`)
- Add `--precision`, `--gradient-accumulation-steps`, `--lr-scheduler`, `--augment-rc` args

## Todo List

- [ ] Add new config fields to `TrainingConfig` and `default.yaml`
- [ ] Implement cosine LR scheduler in `base.py`
- [ ] Add mixed precision to `Trainer()` calls in `trainer.py`
- [ ] Add gradient accumulation to `Trainer()` calls
- [ ] Create `bertnup/data/augmentation.py` with RC augmentation
- [ ] Integrate augmentation into dataset classes
- [ ] Add new CLI arguments
- [ ] Test with existing DNABERT-1 pipeline to verify no regression

## Success Criteria

- [ ] Mixed precision trains without errors on GPU (bf16)
- [ ] Cosine schedule produces smoother convergence curves
- [ ] Gradient accumulation simulates larger batch sizes
- [ ] RC augmentation doubles training data
- [ ] All existing tests/configs still work with default settings
- [ ] Documented performance comparison (speed, memory, metrics)

## Risk Assessment

**Risk: Low.** These are well-established techniques with minimal code changes. All changes are opt-in via config, preserving backward compatibility.

## Next Steps

- Phase 02 (Architecture Upgrades) can proceed in parallel
- Phase 03 (NT v2 backbone) should wait for this phase to complete
