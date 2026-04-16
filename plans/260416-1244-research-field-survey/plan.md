---
name: research-field-survey
status: active
created: 2026-04-16
blockedBy: []
blocks: []
---

# BertNup Field Survey & Module Modernization Plan

## Overview

Field survey of DNA foundation models (2024-2026), nucleosome positioning SOTA, and genomics DL best practices. Based on findings, plan module updates to keep BertNup competitive.

## Research Sources

- [DNA Foundation Models](research/researcher-dna-foundation-models.md)
- [Nucleosome Positioning SOTA](research/researcher-nucleosome-positioning-sota.md)
- [Genomics DL Best Practices](research/researcher-genomics-dl-best-practices.md)
- [Research Project Document](../../docs/research-project.md)

## Phases

| # | Phase | Status | Description |
|---|-------|--------|-------------|
| 01 | [Training Pipeline](phase-01-training-pipeline.md) | Complete | Mixed precision, cosine LR, gradient accumulation, augmentation |
| 02 | [Architecture Upgrades](phase-02-architecture-upgrades.md) | Complete | Enhanced head, attention pooling, LoRA support |
| 03 | [Nucleotide Transformer v2](phase-03-nucleotide-transformer.md) | Pending | New backbone integration, config/CLI updates |
| 04 | [Evaluation & Tracking](phase-04-evaluation-tracking.md) | Pending | Class-weighted loss, significance testing, experiment tracking |
| 05 | [Extended Backbones](phase-05-extended-backbones.md) | Pending | Evo, HyenaDNA, ensemble methods |

## Key Findings

- NT v2 is the best drop-in replacement for DNABERT-2 (+4-6% accuracy)
- Mixed precision + cosine LR = free 2x speedup + better convergence
- Reverse complement augmentation is essentially free performance
- LoRA enables training larger models (Evo-350M, NT v2-2.8B) on consumer GPUs
- Cross-species evaluation is the key missing benchmark

## Dependencies

- Phase 01 → no dependencies (standalone training improvements)
- Phase 02 → independent of Phase 01 (can run in parallel)
- Phase 03 → Phase 01 should be done first (new backbone needs modern training)
- Phase 04 → Phase 01 (needs new training pipeline for tracking)
- Phase 05 → Phase 03 (needs new backbone infrastructure)

## Risk Assessment

- **Low risk:** Phase 01 (training pipeline), Phase 04 (evaluation)
- **Medium risk:** Phase 02 (architecture changes), Phase 03 (new backbone)
- **Higher risk:** Phase 05 (multiple new architectures, complex integration)
