# Nucleotide Transformer v2 Integration Complete

**Date**: 2026-04-16 14:50
**Severity**: Medium
**Component**: Model Architecture
**Status**: Resolved

## What Happened

Successfully completed Phase 03 by integrating Nucleotide Transformer v2 as a new backbone option to BertNup. Added BertNupNT model class with 6-mer tokenization, fixed critical config auto-detection bug for hidden_size/fixed_length, and updated all supporting components (model factory, config detection, dataset factory, trainer, CLI). All tests passed (10/10) with commit 8bb8a69 pushed to feat/modernize-training-architecture.

## The Brutal Truth

This was deceptively complex. The config auto-detection bug could have broken the entire feature rollout - we almost shipped with mismatched hidden_size between model config and initialization. The code review catch was a lifesaver, but honestly it shouldn't have gotten that far. The emotional toll was real - chasing down tokenization edge cases between NT, DNABERT-1, and DNABERT-2 while maintaining backwards compatibility felt like herding cats. Relief when all tests finally passed was palpable.

## Technical Details

- **Model Architecture**: BertNupNT extends BertNupBase, uses ESM-based architecture with 6-mer tokenization
- **Hidden Sizes**: 1280 (500m) or 2560 (2.5b) - auto-detected via CLI model type
- **Tokenization**: fixed_length=30 for 147bp sequences (28 tokens + 2 padding)
- **Pooling**: Mean/max pooling over last_hidden_state (not pooler_output like V1)
- **Config Detection**: Fixed critical bug where hidden_size/fixed_length now auto-adjust when "nucleotide_transformer" type detected
- **Code Review**: Hidden size mismatch caught before commit - would have caused instant model failures

## What We Tried

- Reused existing Dnabert2Dataset for NT (same tokenization pattern) ✓
- Used trust_remote_code=False for NT (standard ESM vs DNABERT-2's custom) ✓  
- Maintained separate "nucleotide_transformer" model type (distinct from "dnabert2") ✓
- Tested both 500m and 2.5b parameter variants ✓

## Root Cause Analysis

The config auto-detection was fundamentally broken - CLI model type wasn't propagating to configuration generation, causing hidden_size mismatches. This happened because the config detection logic assumed YAML precedence over CLI, but NT models need different parameters that couldn't be specified in YAML. The fix required making CLI detection take priority for NT-specific parameters.

## Lessons Learned

1. **Never assume config precedence** - CLI detection must override defaults when model type determines architecture
2. **Code review is your safety net** - caught a critical bug that automated testing missed
3. **Backwards compatibility tax** - supporting multiple model types creates combinatorial complexity in config handling
4. **Tokenization consistency** - Reusing Dnabert2Dataset pattern was the right call -避免了重复的tokenization逻辑

## Next Steps

- Monitor real-world performance of NT models vs DNABERT baselines
- Update documentation with NT-specific configuration guidance
- Plan Phase 04: Attention mechanism visualization for all three model types
- Performance benchmarking: expect 4-6% accuracy improvement over DNABERT-2