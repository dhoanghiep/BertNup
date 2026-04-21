# Phase 05 - Experiment 4: Mean Rank Analysis (Figure 2)

## Context Links
- Paper: `paper/main.tex` Section 3.3, Figure 2
- Depends on: Phase 04 results
- Target: Reproduce Figure 2 (mean rank bar chart)

## Overview
- Priority: Medium
- Status: Pending
- Compute mean performance rank across Dataset 2, reproduce ranking figure.

## Paper Results to Reproduce (Figure 2)
- Mean rank across 8 sub-datasets in Dataset 2
- DNABERT-1-3 has best mean rank = 3.0
- 3 of 4 BertNup models rank higher than all other methods
- DNABERT-1-5 is the exception (lower rank)

## Methodology

### Step 1: Collect baseline results from literature
Paper cites results from these methods on Dataset 2:
- iNuc_PseKNC (SVM)
- Lo Bosco et al. (CNN)
- DLNN (LSTM)
- LeNup (Inception-style)
- CORENup (CNN+LSTM)
- DeepNup (CNN+GRU)
- NP_CBiR (CNN+BiRNN)

These are extracted from source papers — no re-implementation needed. Paper states: "We meticulously extracted evaluation metrics from the source papers corresponding to the respective methods."

### Step 2: Compile all results into unified table
For each of the 8 sub-datasets, compile metrics from:
- Our Phase 04 results (4 DNABERT-1 models)
- Literature results (7 baseline methods)

### Step 3: Compute ranks
- For each sub-dataset, rank all methods by AUC (or ACC as tiebreaker)
- Rank 1 = best, higher = worse
- Compute mean rank across all 8 sub-datasets for each method

### Step 4: Generate Figure 2
- Bar chart: X-axis = methods, Y-axis = mean rank
- Lower mean rank = better
- Highlight DNABERT-1-3 as best

## Implementation

### Results collection script
Create `scripts/compile_mean_ranks.py`:
1. Read Phase 04 CSV results for each model/dataset
2. Compute mean metrics per dataset
3. Load baseline results from hardcoded table (literature values)
4. Rank methods per dataset
5. Compute mean rank
6. Generate bar chart

### Baseline data table
Need to extract baseline results from paper's references. Key sources:
- iNuc_PseKNC: Guo et al. 2014
- Lo Bosco et al. 2017
- DLNN: Gangi et al. 2018
- LeNup: Zhang et al. 2018
- CORENup: Amato et al. 2020
- DeepNup: Zhou et al. 2022
- NP_CBiR: Han et al. 2022

**Note:** The paper's Figure 1 contains the exact values for all methods. We can extract from paper figures or supplementary materials. Alternatively, hardcode the values shown in the paper.

## Success Criteria
- [ ] Mean ranks computed for all 11 methods across 8 sub-datasets
- [ ] DNABERT-1-3 has lowest (best) mean rank ≈ 3.0
- [ ] 3 of 4 BertNup models outrank all baselines
- [ ] Bar chart generated matching Figure 2 style
