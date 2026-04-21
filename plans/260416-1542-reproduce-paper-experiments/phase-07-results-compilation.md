# Phase 07 - Results Compilation

## Context Links
- Paper: `paper/main.tex` (all result sections)
- Depends on: All previous phases
- Target: Compile all results into paper-ready tables and figures

## Overview
- Priority: Medium
- Status: Pending
- Aggregate all experiment results, format as publication-ready tables and figures.

## Deliverables

### Tables
| Table | Source | Content |
|-------|--------|---------|
| Table 4 | Phase 03 | 8 foundation models on HS-LC, 6 metrics each |
| Table 5 | Phase 04 | 4 BertNup variants × 3 species (Dataset 1), ACC/MCC/AUC |

### Figures
| Figure | Source | Content |
|--------|--------|---------|
| Figure 1 | Phase 04 | Bar charts: 8 sub-datasets × methods, 3 species panels |
| Figure 2 | Phase 05 | Mean rank bar chart across Dataset 2 |
| Figure 3 | Phase 06 | Attention evolution: heatmaps + per-position + per-nucleotide |
| Figure 4 | Phase 06 | Attention head patterns: examples + nucleotide preferences |

## Implementation Steps

### Step 1: Create results aggregation script
`scripts/compile_results.py`:
- Read all CSV result files from `Results/` directory
- Compute mean ± std across folds
- Format as LaTeX tables matching paper style
- Format as CSV for easy inspection

### Step 2: Generate Table 4 (LaTeX)
- 8 rows (models) × 6 columns (metrics)
- Bold best values per column
- Format: 4 decimal places

### Step 3: Generate Table 5 (LaTeX)
- 11 rows (methods including baselines) × 9 columns (3 species × 3 metrics)
- Include literature baseline values (hardcoded)
- Bold best values per species/metric

### Step 4: Generate Figure 1
- 3 subplots (A: Human, B: Fruit fly, C: Yeast)
- Each subplot: grouped bars for methods, 3 metrics
- Color scheme matching paper (blue/orange tones)
- Error bars for CV std

### Step 5: Generate Figure 2
- Single bar chart of mean ranks
- 11 methods on x-axis, mean rank on y-axis
- Lower = better, highlight DNABERT-1-3

### Step 6: Generate Figure 3
- Multi-panel figure (A, B, C) as described in Phase 06
- Publication-quality formatting
- Matching paper's color scheme

### Step 7: Generate Figure 4
- Multi-panel figure (A, B) as described in Phase 06
- Attention heatmaps + nucleotide preference bars

### Step 8: Create comparison report
- Side-by-side: paper values vs reproduced values
- Delta analysis (difference in percentage points)
- Flag any results >2% deviation for investigation

## Output Structure

```
Results/
├── compiled/
│   ├── table4_foundation_comparison.csv
│   ├── table4_foundation_comparison.tex
│   ├── table5_dataset1_comparison.csv
│   ├── table5_dataset1_comparison.tex
│   ├── figure1_dataset2_barcharts.png
│   ├── figure2_mean_ranks.png
│   ├── figure3_attention_evolution.png
│   ├── figure4_attention_patterns.png
│   └── reproduction_comparison.md    # paper vs reproduced delta report
├── armheb_DNA_bert_3/   # raw fold-by-fold results
├── armheb_DNA_bert_4/
├── armheb_DNA_bert_5/
├── armheb_DNA_bert_6/
├── zhihan1996_DNABERT-2-117M/
├── nt500m_human_ref/
├── nt500m_1000g/
└── bert_base_cased/
```

## Success Criteria
- [ ] All 4 tables/figures generated in publication quality
- [ ] LaTeX table format matches paper's cas-dc document class style
- [ ] Figures match paper's layout (panel labels, color scheme)
- [ ] Comparison report shows reproduced values within ±2% of paper
- [ ] All raw results preserved in CSV format for reproducibility
- [ ] Single script can regenerate all outputs from raw results
