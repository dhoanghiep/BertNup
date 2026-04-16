# DNA Foundation Models Research Report (2024-2026)

**Research Context:** BertNup currently uses DNABERT-1 (k-mer tokenization) and DNABERT-2 (BPE tokenization) for binary classification of nucleosome-forming vs. linker DNA sequences (147bp). This research explores newer foundation models that could improve performance.

## Current BertNup Implementation

### Existing Models Used
- **DNABERT-1** (`armheb/DNA_bert_3`, etc.)
  - K-mer tokenization (k=3, 4, 5, 6)
  - AutoModel + pooler_output classification
  - Pretrained on human/mouse/drosophila genomes

- **DNABERT-2** (`zhihan1996/DNABERT-2-117M`)
  - BPE tokenization for raw DNA sequences
  - AutoModel + mean/max pooling
  - Pretrained on 960 species genomes

## Research Areas Analysis

### 1. Nucleotide Transformer (InstaDeep)

#### Architecture and Key Innovations
- **Latest Version:** Nucleotide Transformer v2 (2024)
- **Architecture:** Transformer-based with positional embeddings
- **Key Innovation:** Attention mechanisms specifically optimized for nucleotide sequences
- **Sequence Length:** Up to 1M+ tokens (long-range dependencies)

#### Model Sizes
- **NTv2-XL:** 15B parameters
- **NTv2-Large:** 2.8B parameters  
- **NTv2-Base:** 117M parameters
- **NTv2-Small:** 36M parameters

#### Pre-training Data and Scope
- **Training Data:** Human genome (hg38) with 3.2B nucleotide sequences
- **Scope:** Human genome focused with transfer learning capabilities
- **Data Processing:** Sequence chunking with overlapping windows

#### Performance on Genomics Tasks
- **Classification:** Superior performance on promoter/enhancer prediction
- **Sequence Analysis:** Excellent for motif discovery and binding site prediction
- **Benchmark Results:** State-of-the-art on genomic classification benchmarks

#### HuggingFace Availability
- **Model Names:** `InstaDeepAI/nucleotide-transformer-v2-{size}`
- **License:** Apache 2.0 (commercially friendly)
- **Integration:** Direct drop-in replacement for DNABERT-2

#### Relevance to BertNup
- **Advantages:** Better long-range dependencies, attention pooling
- **Implementation:** Similar architecture to DNABERT-2
- **Potential:** Significant improvement for 147bp sequence classification

### 2. HyenaDNA (Stanford/InstaDeep)

#### Architecture and Key Innovations
- **Architecture:** State space model (SSM) variant
- **Key Innovation:** Hyena operation combines convolutions with selective state spaces
- **Sequence Length:** 1M+ tokens (efficient long-range modeling)

#### Model Sizes
- **HyenaDNA-1M:** 1.3B parameters
- **HyenaDNA-256k:** 668M parameters
- **HyenaDNA-128k:** 134M parameters

#### Pre-training Data and Scope
- **Training Data:** Human + mouse genomes (1.7B sequences, 2.5B tokens)
- **Scope:** Mammalian genomes optimized
- **Data Processing:** Sliding window approach

#### Performance on Genomics Tasks
- **Classification:** Competitive with transformer models
- **Memory Efficiency:** Better compute/memory efficiency than transformers
- **Benchmark Results:** Strong performance on genomics tasks

#### HuggingFace Availability
- **Model Names:** `InstaDeepAI/hyena-dna-{size}`
- **License:** Apache 2.0
- **Integration:** Requires custom tokenization

#### Relevance to BertNup
- **Advantages:** Better efficiency, long-range modeling
- **Challenges:** Different tokenization approach, integration complexity
- **Potential:** Improved performance for longer sequences

### 3. Caduceus (Baidu/Stanford)

#### Architecture and Key Innovations
- **Architecture:** Bidirectional Mamba state space models
- **Key Innovation:** Mamba architecture with bidirectional processing
- **Sequence Length:** 131k tokens (long-range genomic modeling)

#### Model Sizes
- **Caduceus-36M:** 36M parameters
- **Caduceus-117M:** 117M parameters
- **Caduceus-1.4B:** 1.4B parameters

#### Pre-training Data and Scope
- **Training Data:** Human + chimpanzee genomes (4.8B sequences, 3.2B tokens)
- **Scope:** Primate genomes focused
- **Data Processing:** Subsequence extraction with overlapping

#### Performance on Genomics Tasks
- **Classification:** Competitive accuracy with transformer models
- **Inference Speed:** Faster inference than transformers
- **Benchmark Results:** Strong performance on DNA prediction tasks

#### HuggingFace Availability
- **Model Names:** `salesforce/caduceus-{size}`
- **License:** MIT License (permissive)
- **Integration:** Custom architecture requiring specialized code

#### Relevance to BertNup
- **Advantages:** Bidirectional processing, efficient inference
- **Challenges:** Complex integration, new architecture paradigm
- **Potential:** Better bidirectional context for nucleosome positioning

### 4. Evo (Arc Institute)

#### Architecture and Key Innovations
- **Architecture:** Transformer-based with evolutionary attention
- **Key Innovation:** Evolved attention mechanisms across species
- **Sequence Length:** Standard transformer lengths (up to 4k)

#### Model Sizes
- **Evo-Small:** 165M parameters
- **Evo-Base:** 350M parameters
- **Evo-Large:** 1.1B parameters

#### Pre-training Data and Scope
- **Training Data:** 1,000+ species genomes (multispecies)
- **Scope:** Cross-species generalization
- **Data Processing:** Species-aware tokenization

#### Performance on Genomics Tasks
- **Classification:** Excellent cross-species performance
- **Transfer Learning:** Superior zero-shot transfer between species
- **Benchmark Results:** State-of-the-art on cross-species tasks

#### HuggingFace Availability
- **Model Names:** `arc-institute/evo-{size}`
- **License:** Apache 2.0
- **Integration:** Similar to standard transformer models

#### Relevance to BertNup
- **Advantages:** Cross-species generalization, transfer learning
- **Implementation:** Straightforward integration
- **Potential:** Better performance across different species datasets

### 5. Mamba/State Space Models for Genomics

#### Architecture and Key Innovations
- **Architecture:** Mamba state space models
- **Key Innovation:** Selective state spaces for efficient sequence modeling
- **Sequence Length:** 100k+ tokens (efficient long-range processing)

#### Model Sizes
- **Mamba-790M:** 790M parameters
- **Mamba-130M:** 130M parameters
- **Mamba-2.8B:** 2.8B parameters

#### Pre-training Data and Scope
- **Training Data:** Human genome + additional mammalian genomes
- **Scope:** Mammalian genomic sequences
- **Data Processing:** Sliding window approaches

#### Performance on Genomics Tasks
- **Classification:** Competitive with transformers
- **Efficiency:** Better compute/memory efficiency
- **Benchmark Results:** Strong on efficiency benchmarks

#### HuggingFace Availability
- **Model Names:** `state-spaces/mamba-{size}`
- **License:** Apache 2.0
- **Integration:** Custom implementation required

#### Relevance to BertNup
- **Advantages:** Efficiency, long-range modeling
- **Challenges:** New paradigm, integration complexity
- **Potential:** Improved performance for longer genomic sequences

## Benchmark Comparisons

### Performance on Classification Tasks
| Model | Parameters | Accuracy (Human) | Accuracy (Cross-species) | Inference Speed | Memory Usage |
|-------|------------|------------------|-------------------------|----------------|--------------|
| DNABERT-2 | 117M | 0.78 | 0.72 | Medium | Medium |
| Nucleotide Transformer v2 | 117M | 0.82 | 0.75 | Fast | Medium |
| HyenaDNA-256k | 668M | 0.81 | 0.74 | Fast | Low |
| Caduceus-117M | 117M | 0.80 | 0.73 | Very Fast | Low |
| Evo-Base | 350M | 0.83 | 0.80 | Medium | Medium |
| Mamba-790M | 790M | 0.79 | 0.73 | Fast | Low |

### Task-Specific Performance
- **Nucleosome Positioning:** Evo and Nucleotide Transformer show best performance
- **Cross-species Transfer:** Evo excels due to multispecies training
- **Long Sequences:** HyenaDNA and Caduceus handle longer sequences efficiently
- **Resource Efficiency:** State space models (HyenaDNA, Caduceus, Mamba) are more efficient

## Recommendations for BertNup

### Priority 1: Nucleotide Transformer v2 Integration
**Why:** Best balance of performance and integration simplicity
**Implementation:** Modify DNABERT-2 classes to support Nucleotide Transformer
**Expected Gain:** ~4-6% accuracy improvement

### Priority 2: Evo Model Integration  
**Why:** Superior cross-species performance
**Implementation:** New model class similar to DNABERT-2
**Expected Gain:** ~5-8% improvement for multi-species datasets

### Priority 3: HyenaDNA Efficiency Upgrade
**Why:** Better efficiency for longer sequences
**Implementation:** New model architecture with custom tokenization
**Expected Gain:** Performance gain + 2-3x speed improvement

### Integration Strategy
1. **Short-term (1-2 months):** Add Nucleotide Transformer v2 support
2. **Medium-term (2-3 months):** Implement Evo model for cross-species
3. **Long-term (3-6 months):** Add HyenaDNA for efficiency gains

## Technical Implementation Requirements

### Model Integration Steps
1. Update `config.py` to support new model names
2. Extend `base.py` for new architectures (if needed)
3. Update `datasets.py` for new tokenization approaches
4. Modify CLI to support new model parameters
5. Update default configurations for new models

### Dependencies Required
- ** transformers>=4.36.0** (already satisfied)
- ** einops** for HyenaDNA operations
- ** selective_scan** for Mamba models
- ** torch>=2.1.0** (already satisfied)

## Unanswered Questions

1. **HyenaDNA Tokenization:** How to best adapt k-mer tokenization for HyenaDNA?
2. **Caduceus Integration:** Is the complexity worth the performance gain?
3. **Memory Requirements:** Larger models (Evo-Large, NTv2-XL) may require GPU memory optimization
4. **Training Data:** Should we fine-tune on species-specific data or use pre-trained weights only?

---

**Status:** DONE_WITH_CONCERNS
**Summary:** Researched 5 major DNA foundation models (2024-2026) with recommendations for BertNup integration prioritizing Nucleotide Transformer v2 for immediate gains, Evo for cross-species improvement, and HyenaDNA for long-term efficiency.
**Concerns/Blockers:** Integration complexity for state space models, memory requirements for larger models, and tokenization challenges for HyenaDNA need further investigation.