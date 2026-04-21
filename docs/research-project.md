# BertNup Research Project: Field Survey & Modernization Roadmap

**Date:** 2026-04-16
**Branch:** feat/modernize-training-architecture
**Status:** Phases 01-03 implemented, Phases 04-05 pending

---

## 0. Paper Summary (BertNup v1 — Published)

**Paper:** "BertNup: A transformer-based model for nucleosome positioning prediction"
**Authors:** Hiep Hoang Dang, Son Thanh Huynh, Binh Thanh Nguyen
**Venue:** Elsevier (cas-dc format)
**Code:** github.com/dhoanghiep/BertNup

### Problem
Binary classification of 147bp DNA sequences as nucleosome-forming vs. linker DNA, using only sequence information (no epigenetic data).

### Datasets
- **Group 1** (Guo et al.): H. sapiens (4,573 seqs), C. elegans (5,175), D. melanogaster (5,750) — small, per-species
- **Group 2** (Liu et al.): H. sapiens LC/PM/5U, D. melanogaster LC/PM/5U, S. cerevisiae WG/PM — large, per-region

### Models Evaluated
8 transformer-based models fine-tuned: BERT-base, DNABERT-1 (k=3,4,5,6), DNABERT-2, Nucleotide Transformer 500M (human-ref, 1000G).

### Key Results
- **DNABERT-1-3 best overall** — highest mean rank across all 8 Group 2 datasets (rank 3.0)
- **Human data SOTA**: ACC 0.8911, MCC 0.7823, AUC 0.9450 (Group 1 H. sapiens) — beats all prior methods (LeNup, CORENup, DeepNup, etc.)
- **Non-human competitive but not SOTA**: C. elegans AUC 0.9588 vs DeepNup 0.9603; D. melanogaster AUC 0.9259 vs CORENup 0.9400
- **Model size doesn't help much**: 500M NT models performed worse than 86M DNABERT-1
- **DNABERT-2 underperforms DNABERT-1**: BPE tokenization may lose nucleosome-specific features

### Attention Analysis
- **Pre-training → fine-tuning shift**: Random attention → focused on first 5 positions → uniform with elevated focus on last 70bp
- **Nucleotide preferences evolve**: No preference → slight T bias → strong A/T preference (consistent with known A/T periodicity in nucleosomes and poly(dA:dT) in linkers)
- **4 attention head patterns identified**: single-point (regulatory sites), multiple-point (periodic motifs), next-word (sequential), clustered (functional domains)

### Paper Limitations & Future Directions (from Discussion)
1. Explore larger models (NT 2.5B)
2. Multi-species training for cross-species generalization
3. Deeper attention head analysis for biological insights

### What's Missing from v1 Paper
- No data augmentation (reverse complement)
- No parameter-efficient fine-tuning (LoRA)
- No mixed precision training
- Only tested DNABERT-1 and DNABERT-2 backbones (no newer models like NT v2, Evo)
- No cross-species transfer learning experiments
- No statistical significance testing
- No experiment tracking

---

## 1. Project Summary

BertNup fine-tunes DNABERT models for binary classification of nucleosome-forming vs. linker DNA sequences (147bp). Current backbones: DNABERT-1 (k-mer tokenization) and DNABERT-2 (BPE tokenization).

This document synthesizes a thorough field survey covering DNA foundation models (2024-2026), nucleosome positioning SOTA, and genomics deep learning best practices.

---

## 2. Field Survey: DNA Foundation Models

### 2.1 Current Models in BertNup

| Model | Tokenization | Pre-training | Params | Implementation Status |
|-------|-------------|-------------|--------|----------------------|
| DNABERT-1 (`armheb/DNA_bert_{k}`) | K-mer (k=3-6) | Human/mouse/drosophila | ~110M | ✅ Complete |
| DNABERT-2 (`zhihan1996/DNABERT-2-117M`) | BPE | 960 species | 117M | ✅ Complete |
| **Nucleotide Transformer v2** | 6-mer | Human genome (hg38) | 117M/2.8B | ✅ **NEW (Phase 03)** |

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

## 6. Research Plan v2 — Next Paper

Based on v1 paper findings, field survey, and current codebase state.

### 6.1 Research Questions

1. Can newer DNA foundation models (NT v2, Evo) surpass DNABERT-1-3 on nucleosome positioning?
2. Does multi-species training improve cross-species generalization?
3. Can attention analysis from newer models reveal novel biological insights?
4. What is the impact of modern training techniques (LoRA, augmentation, cosine LR)?

### 6.2 Proposed Experiments

#### Experiment A — Backbone Comparison (Reproduce v1 + extend)
- **Goal:** Establish baselines with all backbones on same datasets
- **Models:** DNABERT-1-3/4/5/6, DNABERT-2, NT v2-117M, NT v2-500M, Evo-350M
- **Datasets:** Both groups (all 11 datasets)
- **Method:** 10-fold CV, same hyperparams as v1 (lr=2e-5, bs=32, 10 epochs)
- **Expected:** NT v2 and Evo should beat DNABERT-1-3, especially on non-human data
- **Deliverable:** Table comparing all models across all datasets

#### Experiment B — Training Pipeline Ablation
- **Goal:** Quantify impact of each modernization technique
- **Baseline:** DNABERT-1-3 with v1 pipeline
- **Ablations (additive):**
  1. + Mixed precision (bf16)
  2. + Cosine LR schedule
  3. + Reverse complement augmentation
  4. + Enhanced classification head (2-layer + batch norm)
  5. + Attention pooling
  6. + LoRA (r=16) for parameter-efficient fine-tuning
- **Dataset:** HS-LC (largest human dataset)
- **Expected:** Each technique adds 0.5-2% improvement; combined ~5-8%
- **Deliverable:** Ablation table with ACC, MCC, AUC per configuration

#### Experiment C — Cross-Species Transfer Learning
- **Goal:** Test if multi-species training improves generalization
- **Setup:**
  - Train on H. sapiens → test on C. elegans, D. melanogaster, S. cerevisiae
  - Train on all species → test on each held-out species
  - Train with Evo (pre-trained on 1000+ species) → test on each species
- **Expected:** Evo + multi-species training should close the gap on non-human datasets
- **Deliverable:** Transfer learning matrix (train species × test species)

#### Experiment D — Attention Analysis 2.0
- **Goal:** Deeper interpretability with newer models
- **Analyses:**
  1. Reproduce v1 attention head categorization with best new model
  2. Compare attention patterns across backbones (DNABERT-1 vs NT v2 vs Evo)
  3. Identify species-specific vs universal attention patterns
  4. Correlate attention hotspots with known genomic features (TF binding sites, promoter regions)
- **Expected:** Newer models should reveal more structured, biologically meaningful attention patterns
- **Deliverable:** Attention comparison figures + biological interpretation

### 6.3 Implementation Roadmap

| Phase | Task | Status | Priority |
|-------|------|--------|----------|
| 01-02 | Modernize training pipeline (mixed precision, cosine LR, etc.) | ✅ Done | - |
| 03 | Add NT v2 backbone | ✅ Done | High |
| 04 | Add LoRA support | Pending | High |
| 05 | Data augmentation (reverse complement) | Pending | High |
| 06 | Cross-species evaluation framework | Pending | Medium |
| 07 | Evo backbone integration | Pending | Medium |
| 08 | Enhanced attention analysis tools | Pending | Medium |
| 09 | Run Experiment A (backbone comparison) | Pending | High |
| 10 | Run Experiment B (ablation study) | Pending | High |
| 11 | Run Experiment C (cross-species transfer) | Pending | Medium |
| 12 | Run Experiment D (attention analysis) | Pending | Medium |
| 13 | Write paper v2 | Pending | Low |

### 7.4 Target Venues

- **Bioinformatics** (Oxford) — primary target, genomics methods
- **Briefings in Bioinformatics** — if broader scope
- **NAR Genomics and Bioinformatics** — open access alternative
- **PLOS Computational Biology** — if strong biological insights from attention analysis

### 7.5 Key Differentiators from v1 Paper

1. **Newer backbones** (NT v2, Evo) — not just DNABERT
2. **Modern training techniques** (LoRA, augmentation, mixed precision)
3. **Cross-species transfer experiments** — entirely new
4. **Comparative attention analysis** across models
5. **Ablation study** quantifying each technique's contribution

---

## Unresolved Questions

1. **GPU availability?** — Larger models (Evo-1.1B, NT v2-2.8B) may need multi-GPU
2. **Tokenization compatibility?** — NT v2 and Evo use different tokenizers; need dataset adapters
3. **Benchmark datasets?** — Are current datasets sufficient, or should we add newer nucleosome maps?
4. **Publication timeline?** — Target venue affects experiment scope
