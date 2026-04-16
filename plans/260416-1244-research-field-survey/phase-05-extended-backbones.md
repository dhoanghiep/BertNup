# Phase 05: Extended Backbones & Ensemble Methods

**Priority:** Low (Future)
**Status:** Complete
**Risk:** High
**Dependencies:** Phase 02 (architecture), Phase 03 (NT v2 pattern)

## Overview

Add Evo, HyenaDNA, and Caduceus backbone support, plus ensemble prediction combining multiple models. These are higher-effort integrations with different architectures (state space models).

## Key Insights

- Evo excels at cross-species transfer (1000+ species pre-training) — +5-8% improvement
- HyenaDNA provides 2-3x efficiency gains with state space architecture
- Caduceus offers fastest inference with bidirectional Mamba
- Ensemble of DNABERT-1 + DNABERT-2 + NT v2 can yield +3-7% over any single model
- State space models require different tokenization and integration patterns

## Requirements

### Functional
- Evo model integration for cross-species scenarios
- HyenaDNA/Caduceus integration for efficiency-focused use cases
- Ensemble prediction combining outputs from multiple models
- Model comparison benchmarking tool

### Non-Functional
- Each backbone is optional (can install only what's needed)
- Clear documentation of trade-offs between backbones
- Benchmark results documented for decision-making

## Related Code Files

### New
- `bertnup/models/evo.py` — Evo backbone model
- `bertnup/models/ssm.py` — HyenaDNA/Caduceus backbone (state space models)
- `bertnup/models/ensemble.py` — Ensemble prediction combining multiple models
- `bertnup/training/benchmark.py` — Model comparison and benchmarking utilities
- `configs/evo.yaml`, `configs/hyena_dna.yaml`, `configs/caduceus.yaml`

### Modify
- `bertnup/models/__init__.py` — register new model types
- `bertnup/config.py` — detect new model names
- `bertnup/data/datasets.py` — tokenizer adapters for each backbone

## Implementation Steps

### 1. Evo model integration
- Research Evo tokenization and input format
- Create `BertNupEvo` model class (may follow transformer pattern)
- Test on cross-species datasets (train C. elegans → test H. sapiens)
- Document cross-species performance gains

### 2. State space model integration (HyenaDNA/Caduceus)
- These use fundamentally different architectures (not standard transformer)
- May need custom forward pass, different from `BertNupBase`
- Research SSM-specific tokenization and pooling
- Create `BertNupSSM` or separate classes per model

### 3. Ensemble prediction
- Load multiple fine-tuned checkpoints
- Average (or weighted average) prediction probabilities
- Support heterogeneous ensembles (DNABERT-1 + NT v2 + Evo)
- Learnable ensemble weights via held-out validation

### 4. Benchmarking tool
- Compare all backbones on same dataset splits
- Generate comparison tables (AUC, accuracy, MCC, speed, memory)
- Statistical significance testing between models (from Phase 04)
- Output as CSV/Markdown report

## Todo List

- [x] Research Evo tokenization and integration requirements
- [x] Research HyenaDNA/Caduceus architecture specifics
- [x] Implement Evo backbone (highest priority of extended backbones)
- [x] Implement ensemble prediction
- [x] Create benchmarking tool
- [x] Run full comparison across all backbones
- [x] Document results and recommendations

## Success Criteria

- [x] At least one new backbone (Evo recommended) trains and converges
- [x] Ensemble outperforms best single model
- [x] Benchmark report compares all options with metrics
- [x] Cross-species evaluation shows Evo's advantage

## Risk Assessment

**Risk: High.** State space models have fundamentally different architectures. Tokenization may not be compatible with 147bp sequences. Evo is 350M+ params — may need LoRA (Phase 02) for consumer GPUs.

**Mitigation:** Start with Evo (closest to transformer pattern). Use LoRA for larger models. Validate tokenization early.

## Next Steps

- Phase complete - all features implemented successfully
