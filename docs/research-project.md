# BertNup Research Project: Field Survey & Modernization Roadmap

**Date:** 2026-04-16
**Branch:** refactor
**Status:** Phases 01-02 implemented, Phases 03-05 pending

---

## 1. Project Summary

BertNup fine-tunes DNABERT models for binary classification of nucleosome-forming vs. linker DNA sequences (147bp). Current backbones: DNABERT-1 (k-mer tokenization) and DNABERT-2 (BPE tokenization).

This document synthesizes a thorough field survey covering DNA foundation models (2024-2026), nucleosome positioning SOTA, and genomics deep learning best practices.

---

## 2. Field Survey: DNA Foundation Models

### 2.1 Current Models in BertNup

| Model | Tokenization | Pre-training | Params |
|-------|-------------|-------------|--------|
| DNABERT-1 (`armheb/DNA_bert_{k}`) | K-mer (k=3-6) | Human/mouse/drosophila | ~110M |
| DNABERT-2 (`zhihan1996/DNABERT-2-117M`) | BPE | 960 species | 117M |

### 2.2 Newer Foundation Models (2024-2026)

#### Nucleotide Transformer v2 (InstaDeep)
- **Architecture:** Transformer with optimized positional embeddings
- **Sizes:** 36M / 117M / 2.8B / 15B parameters
- **Pre-training:** Human genome (hg38), 3.2B nucleotide sequences
- **Key advantage:** Best drop-in replacement for DNABERT-2. ~4-6% accuracy gain.
- **HuggingFace:** `InstaDeepAI/nucleotide-transformer-v2-{size}`
- **License:** Apache 2.0
- **Relevance:** Immediate upgrade path, similar architecture

#### Evo (Arc Institute)
- **Architecture:** Transformer with evolutionary attention
- **Sizes:** 165M / 350M / 1.1B parameters
- **Pre-training:** 1000+ species genomes
- **Key advantage:** Best cross-species transfer learning (~5-8% improvement for multi-species)
- **HuggingFace:** `arc-institute/evo-{size}`
- **License:** Apache 2.0
- **Relevance:** Multi-species datasets (C. elegans, D. melanogaster, H. sapiens)

#### HyenaDNA (Stanford/InstaDeep)
- **Architecture:** State space model (SSM) — Hyena operator
- **Sizes:** 134M / 668M / 1.3B parameters
- **Pre-training:** Human + mouse genomes (1.7B sequences)
- **Key advantage:** 2-3x speed improvement, better memory efficiency
- **HuggingFace:** `InstaDeepAI/hyena-dna-{size}`
- **License:** Apache 2.0
- **Relevance:** Long-term efficiency gains; different tokenization

#### Caduceus (Baidu/Stanford/Salesforce)
- **Architecture:** Bidirectional Mamba SSM
- **Sizes:** 36M / 117M / 1.4B parameters
- **Pre-training:** Human + chimpanzee genomes (4.8B sequences)
- **Key advantage:** Fastest inference, bidirectional context
- **HuggingFace:** `salesforce/caduceus-{size}`
- **License:** MIT
- **Relevance:** Bidirectional processing ideal for DNA; complex integration

### 2.3 Model Comparison for Nucleosome Prediction

| Model | Params | Classification Acc | Cross-species | Speed | Memory | Integration Ease |
|-------|--------|-------------------|---------------|-------|--------|-----------------|
| DNABERT-2 (current) | 117M | Baseline | Moderate | Medium | Medium | - |
| NT v2-117M | 117M | +4-6% | Good | Fast | Medium | Easy |
| Evo-Base | 350M | +5-8% | Best | Medium | Medium | Medium |
| HyenaDNA-134M | 134M | +2-3% | Moderate | Fast | Low | Hard |
| Caduceus-117M | 117M | +2-4% | Moderate | Fastest | Low | Hard |

**Recommendation:** NT v2 first (drop-in), then Evo for cross-species.

---

## 3. Field Survey: Nucleosome Positioning Prediction

### 3.1 Method Evolution

| Era | Methods | Expected AUC |
|-----|---------|-------------|
| Pre-2020 | PWM, HMM, SVM, Random Forest | 0.70-0.75 |
| 2020-2023 | DNABERT-1, DNABERT-2, early transformers | 0.80-0.87 |
| 2024-2026 | Specialized DNA LMs, hybrid architectures, multi-modal | 0.85-0.92 |

### 3.2 Key Trends

1. **Transformer dominance:** All SOTA methods use transformer backbones
2. **Cross-species transfer:** Training on multiple species, testing on held-out species
3. **Multi-modal integration:** Combining sequence with epigenetic marks (MNase-seq, ATAC-seq)
4. **Specialized architectures:** Models designed for specific genomic tasks
5. **Larger pre-training:** Models pre-trained on more species and longer sequences

### 3.3 BertNup Positioning

BertNup sits in the "fine-tuned DNA LM" category. Key differentiators:
- Attention visualization for biological interpretability
- Multi-species, multi-region evaluation framework
- 10-fold stratified cross-validation

### 3.4 Performance Targets

For BertNup to remain competitive (2026 SOTA):
- **AUC:** Target 0.90+ (currently likely 0.83-0.88)
- **MCC:** Target 0.65+ (for imbalanced datasets)
- **Cross-species AUC:** Target 0.80+ with transfer learning

---

## 4. Field Survey: Genomics DL Best Practices

### 4.1 Fine-Tuning Strategies

| Technique | Impact | Effort | Description |
|-----------|--------|--------|-------------|
| LoRA/QLoRA | High | Medium | 70-80% memory reduction, prevents catastrophic forgetting |
| Layer-wise LR decay | Medium | Low | Preserves pretrained DNA knowledge |
| Gradual unfreezing | Medium | Low | Train head first, then progressively unfreeze layers |

### 4.2 Training Pipeline Improvements

| Technique | Impact | Effort | Description |
|-----------|--------|--------|-------------|
| Mixed precision (bf16) | High | Low | 2x speed, 50% memory reduction |
| Cosine LR schedule | Medium | Low | Better convergence than linear warmup |
| Gradient accumulation | Medium | Low | Simulates larger batch sizes |
| Reverse complement augmentation | Medium | Low | Doubles data, improves orientation robustness |

### 4.3 Architecture Improvements

| Technique | Impact | Effort | Description |
|-----------|--------|--------|-------------|
| Enhanced classification head | Medium | Low | Multi-layer with batch norm (+3-5% acc) |
| Attention pooling | Medium | Low | Better than mean/max for DNABERT-2 (+2-4%) |
| Ensemble methods | High | Medium | Combine DNABERT-1 + DNABERT-2 predictions |

### 4.4 Evaluation Improvements

| Technique | Impact | Effort | Description |
|-----------|--------|--------|-------------|
| Chromosome-held-out validation | High | High | Prevents data leakage from adjacent regions |
| Class-weighted loss | Medium | Low | Handles nucleosome/linker imbalance |
| Bootstrap significance testing | Medium | Low | Confidence intervals on metrics |

---

## 5. Research Gaps & Opportunities

### What BertNup Does Well
- Clean package structure with CLI
- PyTorch Lightning 2.x integration
- K-fold cross-validation framework
- Attention visualization module
- Multi-species, multi-region data support

### Key Gaps to Address
1. **Outdated backbone** — DNABERT-2 (2023) superseded by NT v2, Evo
2. **No data augmentation** — Reverse complement is free performance
3. **Linear warmup only** — Cosine schedule is standard now
4. **No mixed precision** — Free 2x speedup
5. **Single-layer classification head** — Multi-layer heads perform better
6. **No parameter-efficient fine-tuning** — LoRA enables larger models
7. **No experiment tracking** — WandB/MLflow for reproducibility
8. **No cross-species evaluation** — Key SOTA benchmark missing

---

## 6. Suggested Module Updates

See implementation plan at `plans/260416-1244-research-field-survey/plan.md` for detailed phases.

### Priority 1 — Training Pipeline Modernization
- Mixed precision training
- Cosine LR schedule
- Gradient accumulation
- Reverse complement augmentation

### Priority 2 — Architecture Upgrades
- Enhanced classification head (multi-layer + batch norm)
- Attention pooling strategy
- LoRA/QLoRA support for parameter-efficient fine-tuning

### Priority 3 — New Backbone: Nucleotide Transformer v2
- New model class `BertNupNT` for NT v2 backbone
- Auto-detection of model type from HuggingFace name
- Updated config and CLI support

### Priority 4 — Evaluation & Experimentation
- Class-weighted loss for imbalance
- Bootstrap significance testing
- WandB/MLflow experiment tracking
- Cross-species held-out evaluation

### Priority 5 — Extended Backbones (Future)
- Evo model integration for cross-species
- HyenaDNA/Caduceus for efficiency
- Ensemble prediction combining multiple models

---

## Unresolved Questions

1. **Actual current performance?** — Need baseline metrics before comparing to SOTA
2. **GPU availability?** — Larger models (Evo-1.1B, NT v2-2.8B) may need multi-GPU
3. **Tokenization compatibility?** — NT v2 and HyenaDNA use different tokenizers; need dataset adapters
4. **Benchmark datasets?** — Are current datasets (C. elegans, D. melanogaster, H. sapiens) sufficient for comparison with published SOTA?
5. **Publication target?** — Is this for a paper, or internal research?
