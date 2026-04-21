# Phase 04 - Experiments 2+3: 10-fold Cross-Validation (Table 5 + Figure 1)

## Context Links
- Paper: `paper/main.tex` Section 3.2, 3.3 (Table 5, Figure 1)
- Datasets: All 11 datasets (3 from Dataset 1, 8 from Dataset 2)
- Target: Reproduce Table 5 and Figure 1 results

## Overview
- Priority: High
- Status: Pending
- Run 10-fold CV with DNABERT-1-{3,4,5,6} on all datasets, compare with baselines.

## Paper Results to Reproduce

### Table 5 - Dataset 1 (3 species, 10-fold CV)
| Model | H.sapiens ACC/MCC/AUC | C.elegans ACC/MCC/AUC | D.melanogaster ACC/MCC/AUC |
|-------|----------------------|----------------------|---------------------------|
| DNABERT-1-3 | **0.8911** / **0.7823** / **0.9450** | 0.8987 / 0.7980 / 0.9588 | 0.8544 / 0.7091 / 0.9259 |
| DNABERT-1-4 | 0.8889 / 0.7778 / 0.9414 | 0.8881 / 0.7768 / 0.9523 | 0.8363 / 0.6761 / 0.9122 |
| DNABERT-1-5 | 0.8754 / 0.7511 / 0.9355 | 0.8914 / 0.7836 / 0.9550 | 0.8360 / 0.6747 / 0.9138 |
| DNABERT-1-6 | 0.8830 / 0.7661 / 0.9374 | 0.8947 / 0.7899 / 0.9550 | 0.8433 / 0.6878 / 0.9213 |

### Figure 1 - Dataset 2 (8 sub-datasets, 10-fold CV)
Bar charts showing ACC, MCC, AUC for each method on each sub-dataset.

## Hyperparameters (from paper Section 2.3)

```
batch_size = 32
learning_rate = 2e-5
max_epochs = 10
warmup_steps = 40 (fixed)
lr_scheduler = linear warmup + linear decay
val_check_interval = 0.1 (10 validations per epoch)
early_stopping_patience = 10 (validation steps)
model_selection = best validation loss → test
metrics = mean across 10 folds
```

**Critical difference from Experiment 1:** Fixed warmup_steps=40 (not ratio-based). Early stopping with patience=10 validation steps (not epochs).

## Implementation Steps

### Step 1: Verify 10-fold CV infrastructure
- Current `run_kfold_cv()` already implements:
  - 10-fold split ✓
  - EarlyStopping(patience=10) ✓
  - val_check_interval=0.1 ✓
  - CSV result output ✓
- Need to verify: warmup_steps=40 is used (Phase 02 Gap 2)

### Step 2: Run 10-fold CV for Dataset 1 (3 species × 4 k-mer models)

```bash
PYTHON="/Users/danghiep/miniforge3/envs/bertnup/bin/python"
DATASETS=("Hsapiens" "Celegans" "Dmelanogaster")

for DATASET in "${DATASETS[@]}"; do
  for K in 3 4 5 6; do
    echo "=== $DATASET k=$K ==="
    KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli cross_validate $DATASET \
      --model-name armheb/DNA_bert_$K --kmer $K --epochs 10 \
      --batch-size-train 32 --learning-rate 2e-5 --warmup-steps 40 \
      --val-check-interval 0.1 --early-stopping-patience 10
  done
done
```

### Step 3: Run 10-fold CV for Dataset 2 (8 sub-datasets × 4 k-mer models)

```bash
DATASETS2=("HS_LC" "HS_PM" "HS_5U" "DM_LC" "DM_PM" "DM_5U" "Y_WG" "Y_PM")

for DATASET in "${DATASETS2[@]}"; do
  for K in 3 4 5 6; do
    echo "=== $DATASET k=$K ==="
    KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli cross_validate $DATASET \
      --model-name armheb/DNA_bert_$K --kmer $K --epochs 10 \
      --batch-size-train 32 --learning-rate 2e-5 --warmup-steps 40 \
      --val-check-interval 0.1 --early-stopping-patience 10
  done
done
```

### Step 4: Compute mean metrics
- Read CSV results for each dataset/model combination
- Compute mean ± std for all 6 metrics across 10 folds
- Format as Table 5 layout

### Step 5: Create Figure 1 (bar charts)
- Grouped bar chart for each species (A: Human, B: Fruit fly, C: Yeast)
- X-axis: methods, Y-axis: metric values (ACC, MCC, AUC)
- Use matplotlib with paper styling

## Results Output Structure

```
Results/
├── armheb_DNA_bert_3/
│   ├── Hsapiens.csv      # 10 rows (one per fold)
│   ├── Celegans.csv
│   ├── Dmelanogaster.csv
│   ├── HS_LC.csv
│   ├── HS_PM.csv
│   ├── ... (all 11 datasets)
├── armheb_DNA_bert_4/
│   └── ...
├── armheb_DNA_bert_5/
│   └── ...
└── armheb_DNA_bert_6/
    └── ...
```

## Success Criteria
- [ ] All 44 combinations run (11 datasets × 4 models)
- [ ] Mean metrics within ±2% of paper values
- [ ] DNABERT-1-3 best on human data (matching paper)
- [ ] Results formatted as Table 5
- [ ] Bar chart figures generated matching Figure 1 style
