---
name: reproduce-paper-experiments
status: active
created: 2026-04-16
blockedBy: []
blocks: []
---

# Reproduce BertNup Paper Experiments

## Overview

Reproduce all experimental results from the BertNup paper (`paper/main.tex`). The paper contains 6 experiments across 4 tables/figures, testing 8 foundation models on nucleosome positioning prediction.

## Paper Experiments Summary

| # | Experiment | Paper Ref | Models | Dataset(s) | Method |
|---|-----------|-----------|--------|------------|--------|
| 1 | Foundation model comparison | Table 4 | 8 models | HS-LC | 8:1:1 split |
| 2 | Dataset 1 comparison (3 species) | Table 5 | DNABERT-1-{3,4,5,6} | H.sapiens, C.elegans, D.melanogaster | 10-fold CV |
| 3 | Dataset 2 comparison (8 sub-datasets) | Figure 1 | DNABERT-1-{3,4,5,6} | HS-LC, HS-PM, HS-5U, DM-LC, DM-PM, DM-5U, Y-WG, Y-PM | 10-fold CV |
| 4 | Mean rank across Dataset 2 | Figure 2 | DNABERT-1-{3,4,5,6} + baselines | All Dataset 2 | Derived from Exp 3 |
| 5 | Attention evolution | Figure 3 | DNABERT-1-3 | HS-LC | Random/Pretrained/Fine-tuned |
| 6 | Attention head patterns | Figure 4 | DNABERT-1-3 | HS-LC | 144 heads, 4 categories |

## Phases

| # | Phase | Status | Description |
|---|-------|--------|-------------|
| 01 | [Data Preparation](phase-01-data-preparation.md) | Pending | Prepare all 11 datasets with correct splits |
| 02 | [Code Gaps](phase-02-code-gaps.md) | Pending | Add BERT-base backbone, 8:1:1 split, fixed warmup steps |
| 03 | [Exp 1 - Foundation Models](phase-03-foundation-model-comparison.md) | Pending | Run 8 models on HS-LC, reproduce Table 4 |
| 04 | [Exp 2+3 - 10-fold CV](phase-04-tenfold-crossvalidation.md) | Pending | Run 10-fold CV on all 11 datasets, reproduce Table 5 + Figure 1 |
| 05 | [Exp 4 - Mean Rank](phase-05-mean-rank-analysis.md) | Pending | Compute mean ranks, reproduce Figure 2 |
| 06 | [Exp 5+6 - Attention](phase-06-attention-analysis.md) | Pending | Attention evolution + head pattern analysis, reproduce Figure 3+4 |
| 07 | [Results Compilation](phase-07-results-compilation.md) | Pending | Aggregate results, format tables, generate figures |

## Key Reproduction Parameters

### Experiment 1 (Table 4) - HS-LC 8:1:1 split
- Batch size: 32, LR: 2e-5, epochs: 10
- Warmup: 10% of max steps, linear decay
- Best model on validation → evaluate on test

### Experiments 2+3 (Table 5, Figure 1) - 10-fold CV
- Batch size: 32, LR: 2e-5, epochs: 10
- Warmup: 40 fixed steps, linear decay
- Val check interval: 0.1 (10 times per epoch)
- Early stopping patience: 10 validation steps
- Mean metrics across 10 folds

### Models to Test
1. BERT-base-cased (110M)
2. DNABERT-1-3 (armheb/DNA_bert_3)
3. DNABERT-1-4 (armheb/DNA_bert_4)
4. DNABERT-1-5 (armheb/DNA_bert_5)
5. DNABERT-1-6 (armheb/DNA_bert_6)
6. DNABERT-2 (zhihan1996/DNABERT-2-117M)
7. NT-500M-human-ref (InstaDeepAI/nucleotide-transformer-500m-human-ref)
8. NT-500M-1000G (InstaDeepAI/nucleotide-transformer-500m-1000g)

## Dependencies

- Phase 01 → no dependencies (data prep)
- Phase 02 → no dependencies (can run parallel with Phase 01)
- Phase 03 → Phase 01 + 02 (needs HS-LC 8:1:1 split + BERT-base backbone)
- Phase 04 → Phase 01 + 02 (needs all datasets prepared)
- Phase 05 → Phase 04 (needs 10-fold CV results)
- Phase 06 → Phase 03 (needs fine-tuned DNABERT-1-3 on HS-LC)
- Phase 07 → Phase 03 + 04 + 05 + 06 (all results)

## Risk Assessment

- **Medium risk:** BERT-base-cased may need custom tokenizer/Dataset for DNA sequences
- **Medium risk:** NT-500M models require significant GPU memory (500M params)
- **Low risk:** DNABERT-1 models are well-tested in current codebase
- **Low risk:** 10-fold CV infrastructure already exists
- **Medium risk:** Attention visualization requires matching exact DNABERT-viz methodology
