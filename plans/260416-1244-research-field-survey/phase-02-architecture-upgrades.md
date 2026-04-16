# Phase 02: Architecture Upgrades

**Priority:** High
**Status:** Pending
**Risk:** Medium
**Dependencies:** None (can run parallel with Phase 01)

## Overview

Upgrade model architecture with enhanced classification head, attention pooling, and LoRA support for parameter-efficient fine-tuning.

## Key Insights

- Multi-layer classification head with batch norm gives +3-5% accuracy over single linear layer
- Attention pooling learns which positions matter, outperforms mean/max pooling by +2-4%
- LoRA reduces memory 70-80% and prevents catastrophic forgetting — enables training larger models

## Requirements

### Functional
- Enhanced multi-layer classification head with batch norm
- Attention pooling as third pooling option (after mean, max)
- LoRA/QLoRA support via HuggingFace PEFT library
- Config-driven selection of head type and pooling strategy

### Non-Functional
- Backward compatible — existing single-layer head remains default
- LoRA is opt-in via config flag
- No new required dependencies (PEFT optional)

## Related Code Files

### Modify
- `bertnup/models/base.py` — configurable head, LoRA integration point
- `bertnup/models/dnabert2.py` — attention pooling option
- `bertnup/config.py` — `head_type`, `use_lora`, `lora_rank`, `pooling` fields
- `bertnup/cli.py` — new architecture CLI args

### New
- `bertnup/models/heads.py` — `SingleLayerHead` (existing), `EnhancedHead` (new)
- `bertnup/models/pooling.py` — `MeanPooling`, `MaxPooling`, `AttentionPooling`

## Implementation Steps

### 1. Extract classification heads (`heads.py`)
- Move existing `dropout1 + linear1` from `base.py` into `SingleLayerHead`
- Create `EnhancedHead`: `BatchNorm → Dropout → Linear(hidden, hidden//2) → ReLU → BatchNorm → Dropout → Linear(hidden//2, 2)`
- Both inherit from `nn.Module`, same interface

### 2. Extract pooling strategies (`pooling.py`)
- Move mean/max pooling from `dnabert2.py` into standalone classes
- Create `AttentionPooling`: `Linear(hidden, hidden//2) → Tanh → Linear(hidden//2, 1) → Softmax → WeightedSum`

### 3. Update `base.py`
- Make head configurable via `head_type` param ("single" | "enhanced")
- Default to "single" for backward compatibility
- Move head construction to `_build_head()` method

### 4. Update `dnabert2.py`
- Use pooling classes from `pooling.py`
- Add "attention" as third pooling option

### 5. LoRA integration (`base.py`)
- Add `use_lora` and `lora_rank` config fields
- After loading backbone, optionally wrap with `peft.get_peft_model()`
- Target modules: `["query", "key", "value", "dense"]`
- PEFT becomes optional dependency

### 6. Config and CLI updates
- `ModelConfig.head_type`: str = "single"
- `ModelConfig.use_lora`: bool = False
- `ModelConfig.lora_rank`: int = 8
- `ModelConfig.lora_alpha`: int = 32
- CLI args: `--head-type`, `--use-lora`, `--lora-rank`

## Todo List

- [ ] Create `bertnup/models/heads.py` with SingleLayerHead and EnhancedHead
- [ ] Create `bertnup/models/pooling.py` with Mean, Max, Attention pooling
- [ ] Refactor `base.py` to use configurable head
- [ ] Refactor `dnabert2.py` to use pooling module
- [ ] Add LoRA support via PEFT (optional dependency)
- [ ] Update config, CLI, and YAML defaults
- [ ] Test all combinations (head x pooling x lora on/off)

## Success Criteria

- [ ] EnhancedHead trains and converges
- [ ] AttentionPooling produces valid outputs
- [ ] LoRA reduces trainable params by 70%+
- [ ] SingleLayerHead + mean pooling = same as current (backward compat)
- [ ] All combinations tested with dry-run training

## Risk Assessment

**Risk: Medium.** Refactoring base.py affects all models. Mitigate by keeping existing behavior as default. LoRA adds PEFT dependency — make optional.

## Next Steps

- Phase 03 requires this phase's refactored base.py for new backbone
