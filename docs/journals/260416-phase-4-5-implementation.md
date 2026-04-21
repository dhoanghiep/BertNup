# Phase 4+5 Implementation Session: Extended Backbones & Evaluation

**Date**: 2026-04-16 HH:mm
**Severity**: Medium
**Component**: Training architecture modernization
**Status**: Completed

## What Happened

Implemented Phase 4 (Evaluation & Tracking) and Phase 5 (Extended Backbones) on the `feat/modernize-training-architecture` branch. Added class-weighted loss, bootstrap significance testing, cross-species evaluation, WandB logging, and three new backbones: Evo, HyenaDNA, and Caduceus. Implemented ensemble prediction and centralized model dispatch to fix registry drift bugs.

## The Brutal Truth

This session was incredibly frustrating because we wasted hours fixing critical bugs that the code reviewer caught after implementation. The incomplete model registry and outdated trust_remote logic were particularly painful - they should have been caught during implementation, not review. The web search rate limiting during Evo research felt like a major productivity killer, forcing workarounds that delayed progress.

## Technical Details

- **Critical bugs fixed**: Incomplete model registry (causing KeyError for new backbones), outdated trust_remote logic (Evo model loading failed)
- **Silent correctness bugs**: k-fold ignoring class weights (critical for imbalanced data), div-by-zero in sensitivity/specificity, empty bootstrap samples causing crashes
- **Model registry**: Centralized `get_model_class()` in `__init__.py` after finding drift bugs
- **Division-by-zero guards**: Added to sensitivity/specificity metrics in `metrics.py`
- **Evo defaults**: LoRA enabled by default (7B params too large for consumer GPUs without it)

## What We Tried

Attempted to use web search for Evo research but hit rate limits repeatedly. Switched to HuggingFace direct fetch + existing research files as workaround. Implemented optional dependency handling with graceful ImportError for new backbones. All new backbones use `trust_remote_code=True` with proper error handling.

## Root Cause Analysis

The main issues stemmed from insufficient code review during implementation and inadequate testing of the model registry system. The drift in model registry happened because new backbones were added without updating the central dispatch, causing inconsistent behavior across the codebase. The div-by-zero and empty bootstrap bugs were caused by missing edge case validation in statistical computations.

## Lessons Learned

1. **Test edge cases thoroughly**: Always validate statistical computations with boundary conditions (empty arrays, zero divisors)
2. **Implement as you test**: Don't separate implementation from testing - validate each component as you build it
3. **Centralize registries**: Single source of truth for model dispatch prevents drift and inconsistent behavior
4. **Graceful degradation**: Handle missing dependencies properly rather than letting crashes happen
5. **Pre-review your code**: Anticipate what a reviewer might catch and fix it immediately during implementation

## Next Steps

All bugs from code review have been fixed. The model registry is now centralized and properly maintained. Next phase should focus on comprehensive testing of the new backbones and evaluation of ensemble performance on cross-species tasks.