# Phase 06 - Experiments 5+6: Attention Analysis (Figures 3+4)

## Context Links
- Paper: `paper/main.tex` Section 3.4, 3.5, Figures 3+4
- Attention module: `bertnup/models/attention.py`
- Visualization: `bertnup/visualization/attention_viz.py`
- Model: DNABERT-1-3 on HS-LC dataset

## Overview
- Priority: Medium
- Status: Pending
- Analyze attention score evolution (random → pre-trained → fine-tuned) and categorize attention head patterns.

## Experiment 5: Attention Evolution (Figure 3)

### Paper Description
Compares attention scores across 3 training stages:
1. **Random weights** — all trainable weights re-initialized randomly
2. **Pre-trained weights** — DNABERT-1-3 as-is from HuggingFace
3. **Fine-tuned weights** — after fine-tuning on HS-LC 8:1:1 split

### Visualizations (Figure 3)
- **Panel A:** Attention landscape heatmaps for individual sequences (nucleosome vs linker)
- **Panel B:** Average attention scores per position along 147bp sequence
- **Panel C:** Average attention scores per nucleotide (A, C, G, T) for nucleosome vs linker

### Key findings to reproduce
- Random: uniform/dispersed attention, ~0.0068 mean
- Pre-trained: focused on first 5 positions, slight T preference (0.0071)
- Fine-tuned: uniform distribution, elevated attention on last 70bp, A highest (0.0079), T second

### Implementation Steps

#### Step 1: Create random-weight model
```python
# Load DNABERT-1-3 architecture with random init
model = BertForSequenceClassification.from_pretrained(
    "armheb/DNA_bert_3", output_attentions=True
)
# Re-initialize all weights randomly
for module in model.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
```

#### Step 2: Extract attention from pre-trained model
```python
model = BertForSequenceClassification.from_pretrained(
    "armheb/DNA_bert_3", output_attentions=True
)
# Run inference on HS-LC test set
```

#### Step 3: Extract attention from fine-tuned model
- Fine-tune DNABERT-1-3 on HS-LC 8:1:1 (from Phase 03)
- Export weights via `export_bert_weights()`
- Load with `BertNupAttention` and extract attention

#### Step 4: Compute per-position and per-nucleotide scores
- Use `process_attention_score()` from `bertnup/models/attention.py`
- Aggregate across all test sequences, split by nucleosome/linker
- Normalize: sum of attention scores per sequence = 1

#### Step 5: Generate Figure 3
- Panel A: 2×3 heatmaps (nucleosome/linker × random/pretrained/finetuned)
- Panel B: Line plot of average attention per position
- Panel C: Bar chart of average attention per nucleotide

## Experiment 6: Attention Head Patterns (Figure 4)

### Paper Description
Analyzes all 144 attention heads (12 layers × 12 heads) and categorizes into 4 patterns:

1. **Single-point attention** — focus on individual nucleotides (e.g., Layer 4-Head 10)
2. **Multiple-point attention** — periodic focus on multiple positions (e.g., Layer 10-Head 5)
3. **Next-word attention** — focus on immediate neighbor (no nucleotide preference)
4. **Clustered attention** — grouped focus on larger functional units (no nucleotide preference)

### Visualizations (Figure 4)
- **Panel A:** Attention maps of example heads for 2 DNA sequences
- **Panel B:** Average attention scores per nucleotide for each head category

### Implementation Steps

#### Step 1: Extract attention from all 144 heads
- Use fine-tuned DNABERT-1-3 model
- For each of 12 layers × 12 heads, extract attention scores on test set
- Use `BertNupAttention` with specific layer/head parameters

#### Step 2: Visual inspection and categorization
- For each head, visualize attention map on sample sequences
- Manually categorize into 4 patterns (this is qualitative analysis)
- Paper identifies specific example heads for each category

#### Step 3: Compute nucleotide preferences per category
- For each categorized head, compute average attention per nucleotide (A, C, G, T)
- Split by nucleosome/linker label

#### Step 4: Generate Figure 4
- Panel A: 4 example attention heatmaps (one per category)
- Panel B: Nucleotide preference bar charts per category

## Code Changes Required

### Update `bertnup/visualization/attention_viz.py`
- Add function for per-position attention aggregation
- Add function for per-nucleotide attention aggregation
- Add function for random-weight model attention extraction
- Add multi-stage comparison plotting

### Create `scripts/run_attention_analysis.py`
- Orchestrates the full attention analysis pipeline
- Handles all 3 training stages + all 144 heads
- Generates Figures 3 and 4

## Success Criteria
- [ ] Attention extracted from random, pre-trained, and fine-tuned models
- [ ] Per-position plots match Figure 3B pattern (pre-trained: front-loaded, fine-tuned: last-70bp elevated)
- [ ] Per-nucleotide plots match Figure 3C (fine-tuned A=0.0079, T=0.0074 for linker)
- [ ] All 144 heads visualized and categorized into 4 patterns
- [ ] Example heads match paper's specific layer/head references (Layer 4-Head 10, Layer 10-Head 5)
- [ ] Figures 3 and 4 generated in publication quality
