# Phase 04: Evaluation & Experiment Tracking

**Priority:** Medium
**Status:** Complete
**Risk:** Low
**Dependencies:** Phase 01 (training pipeline)

## Overview

Improve evaluation rigor with class-weighted loss, bootstrap significance testing, cross-species held-out evaluation, and experiment tracking integration.

## Key Insights

- Class imbalance (nucleosome vs. linker) affects sensitivity — weighted loss helps
- Bootstrap testing gives confidence intervals on AUC/MCC comparisons
- WandB/MLflow provides experiment reproducibility and hyperparameter search
- Cross-species held-out evaluation is the key missing benchmark for SOTA comparison

## Requirements

### Functional
- Class-weighted cross-entropy loss (auto-computed from training data)
- Bootstrap significance testing for model comparisons
- Optional WandB/MLflow experiment tracking
- Cross-species evaluation workflow

### Non-Functional
- Weighted loss is opt-in (default = unweighted, backward compatible)
- Experiment tracking is optional (no hard dependency on WandB/MLflow)
- Significance testing runs as post-hoc analysis, not during training

## Related Code Files

### Modify
- `bertnup/models/base.py` — class-weighted loss option
- `bertnup/data/metrics.py` — add bootstrap significance testing function
- `bertnup/training/trainer.py` — WandB logger, cross-species eval workflow
- `bertnup/config.py` — new config fields
- `bertnup/cli.py` — new CLI commands/args

### New
- `bertnup/training/evaluation.py` — cross-species eval, significance testing, comparison reports

## Implementation Steps

### 1. Class-weighted loss (`base.py`)
- Add `class_weights` parameter to `__init__`
- In `_compute_loss_and_probas()`, use `CrossEntropyLoss(weight=class_weights)` when provided
- Add `TrainingConfig.use_class_weights`: bool = False
- Auto-compute weights from training data in `trainer.py` before model creation

### 2. Bootstrap significance testing (`evaluation.py`, `metrics.py`)
- Add `bootstrap_auc_comparison(y_true, y_pred1, y_pred2, n_bootstraps=1000)`
- Add `compute_metrics_with_ci(probas, labels, n_bootstraps=1000)` returning mean ± std
- Output results as structured dict for logging

### 3. Experiment tracking (`trainer.py`)
- Add optional `WandbLogger` when `experiment.tracking` = "wandb"
- Log hyperparameters, metrics, and model checkpoints
- PEFT/transformers dependency only when tracking enabled

### 4. Cross-species evaluation (`evaluation.py`)
- New `run_cross_species_eval()` function
- Train on species A, evaluate on species B (held-out)
- Report per-species and cross-species metrics
- Support pairwise and leave-one-out cross-species schemes

### 5. Config and CLI
- `TrainingConfig.use_class_weights`: bool = False
- Add `evaluate_cross_species` CLI command
- Optional `ExperimentConfig` section for tracking settings

## Todo List

- [x] Implement class-weighted loss in base.py
- [x] Add bootstrap significance testing to metrics.py
- [x] Create evaluation.py with cross-species workflow
- [x] Add optional WandB logging to trainer.py
- [x] Add `evaluate_cross_species` CLI command
- [x] Update configs and documentation

## Success Criteria

- [x] Class-weighted loss trains without errors
- [x] Bootstrap testing produces confidence intervals
- [x] Cross-species eval reports per-species metrics
- [x] WandB logs metrics when enabled, no error when disabled
- [x] Significance testing confirms/rejects improvements between models

## Risk Assessment

**Risk: Low.** All changes are additive and opt-in. No existing behavior is modified.

## Next Steps

- Phase complete - all features implemented successfully
