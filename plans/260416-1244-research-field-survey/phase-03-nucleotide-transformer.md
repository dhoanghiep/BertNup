# Phase 03: Nucleotide Transformer v2 Integration

**Priority:** High
**Status:** Complete
**Risk:** Medium
**Dependencies:** Phase 01 (modern training pipeline)

## Overview

Add Nucleotide Transformer v2 as a new backbone option. NT v2 is the best drop-in replacement for DNABERT-2 with ~4-6% accuracy improvement. It uses similar transformer architecture, making integration straightforward.

## Key Insights

- NT v2-117M has same parameter count as DNABERT-2 but better pre-training
- Uses `AutoModel` interface — similar integration path to DNABERT-2
- HuggingFace: `InstaDeepAI/nucleotide-transformer-v2-117m`
- Apache 2.0 license — commercially friendly
- 6-mer tokenization with special tokens for genomic understanding

## Requirements

### Functional
- New model class supporting NT v2 backbone
- Auto-detection of NT v2 model type from HuggingFace name
- Proper tokenization handling (6-mer based, similar to DNABERT-1 but different)
- Config and CLI support for NT v2 models

### Non-Functional
- NT v2 models work with all existing training features (Phase 01 improvements)
- No regression for DNABERT-1 or DNABERT-2 paths
- Clean fallback if model download fails

## Related Code Files

### Modify
- `bertnup/models/__init__.py` — register new model type
- `bertnup/config.py` — `_detect_model_type()` to recognize NT v2 names
- `bertnup/data/datasets.py` — add NT v2 dataset if tokenization differs
- `bertnup/training/trainer.py` — handle NT v2 in `_build_dataloaders()`
- `bertnup/cli.py` — update help text

### New
- `bertnup/models/nucleotide_transformer.py` — `BertNupNT` model class

## Implementation Steps

### 1. Create `BertNupNT` model class
- Inherit from `BertNupBase`
- Use `AutoModel.from_pretrained()` with `trust_remote_code=True`
- Support mean/max/attention pooling (from Phase 02)
- Handle `last_hidden_state` output (like DNABERT-2, no pooler_output)

### 2. Tokenization research
- NT v2 uses its own tokenizer — may need `AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-117m")`
- Determine if existing `Dnabert2Dataset` works or if new dataset class needed
- Test tokenization on 147bp sequences

### 3. Model factory updates
- Add "nucleotide_transformer" as model type in config
- Update `_detect_model_type()` to recognize `InstaDeepAI/nucleotide-transformer`
- Update `create_model()` factory function

### 4. Dataset handling
- If NT v2 tokenizer is compatible with BPE, reuse `Dnabert2Dataset`
- If different tokenization, create `NucleotideTransformerDataset`
- Determine appropriate `fixed_length` for 147bp sequences

### 5. Config and CLI
- Add NT v2 config: `configs/nucleotide_transformer.yaml`
- Document model name options in CLI help

## Todo List

- [x] Research NT v2 tokenization on 147bp sequences
- [x] Create `BertNupNT` model class
- [x] Update model factory and type detection
- [x] Test tokenization — reuse or new dataset class
- [x] Create NT v2 config YAML
- [ ] Run benchmark comparison: DNABERT-2 vs NT v2 on same data

## Success Criteria

- [x] NT v2 model loads and trains without errors
- [x] Tokenization produces valid input for 147bp sequences
- [ ] Performance comparison shows improvement over DNABERT-2 baseline (deferred - needs training)
- [x] All existing DNABERT-1/2 paths still work
- [ ] Benchmark metrics documented (deferred - needs training)

## Risk Assessment

**Risk: Medium.** Tokenization may differ from DNABERT-2. NT v2 models may have different output formats. Need to verify 147bp sequences work with NT v2's expected input.

**Mitigation:** Test tokenization first before full integration. Start with 117M model (same size as DNABERT-2).

## Next Steps

- Phase 04 (Evaluation) can use NT v2 for cross-model comparison
- Phase 05 (Extended Backbones) follows same pattern for Evo/HyenaDNA
