# Phase 03 - Experiment 1: Foundation Model Comparison (Table 4)

## Context Links
- Paper: `paper/main.tex` Section 3.1, Table 4
- Dataset: HS-LC (8:1:1 split)
- Target: Reproduce Table 4 results

## Overview
- Priority: High
- Status: Pending
- Fine-tune 8 foundation models on HS-LC and compare nucleosome prediction performance.

## Paper Results to Reproduce (Table 4)

| Model | Specificity | Sensitivity | Accuracy | F1 | MCC | AUC |
|-------|------------|-------------|----------|----|-----|-----|
| DNABERT-1-3 | **0.8238** | 0.9632 | **0.9092** | **0.9285** | **0.8085** | **0.9498** |
| DNABERT-1-4 | 0.8015 | 0.9748 | 0.9076 | 0.9281 | 0.8068 | 0.9464 |
| DNABERT-1-5 | 0.7933 | 0.9711 | 0.9022 | 0.9240 | 0.7951 | 0.9452 |
| DNABERT-1-6 | 0.8050 | 0.9731 | 0.9079 | 0.9283 | 0.8071 | 0.9487 |
| DNABERT-2 | 0.7711 | 0.9392 | 0.8740 | 0.9012 | 0.7325 | 0.9216 |
| NT-500M-human-ref | 0.7643 | **0.9777** | 0.8950 | 0.9193 | 0.7817 | 0.9376 |
| NT-500M-1000G | 0.7832 | 0.9675 | 0.8960 | 0.9193 | 0.7818 | 0.9385 |
| BERT-base | 0.7591 | 0.9480 | 0.8747 | 0.9026 | 0.7349 | 0.9198 |

## Hyperparameters (from paper Section 2.3)

```
train/val/test = 8:1:1
batch_size = 32
learning_rate = 2e-5
max_epochs = 10
warmup = 10% of max steps
lr_scheduler = linear warmup + linear decay
model_selection = best validation performance
pooling = CLS token (pooler_output for DNABERT-1, CLS for others)
```

## Implementation Steps

### Step 1: Prepare HS-LC 8:1:1 split
- Use Phase 01/02 single_split functionality
- Random state: 42 (default, paper doesn't specify)
- Save to `Data/Stratified_K_fold_data/HS_LC_single_split/`

### Step 2: Run all 8 models
Execute training for each model with paper hyperparameters:

```bash
DATADIR="Data/Stratified_K_fold_data/HS_LC_single_split"
PYTHON="/Users/danghiep/miniforge3/envs/bertnup/bin/python"

# DNABERT-1 variants (k=3,4,5,6)
for K in 3 4 5 6; do
  KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli train $DATADIR \
    --model-name armheb/DNA_bert_$K --kmer $K --epochs 10 \
    --batch-size-train 32 --learning-rate 2e-5 --warmup-ratio 0.1
done

# DNABERT-2
KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli train $DATADIR \
  --config configs/dnabert2.yaml --epochs 10 \
  --batch-size-train 32 --learning-rate 2e-5 --warmup-ratio 0.1

# NT-500M-human-ref
KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli train $DATADIR \
  --config configs/nt500m-human-ref.yaml --epochs 10 \
  --batch-size-train 32 --learning-rate 2e-5 --warmup-ratio 0.1

# NT-500M-1000G
KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli train $DATADIR \
  --config configs/nt500m-1000g.yaml --epochs 10 \
  --batch-size-train 32 --learning-rate 2e-5 --warmup-ratio 0.1

# BERT-base
KMP_DUPLICATE_LIB_OK=TRUE $PYTHON -m bertnup.cli train $DATADIR \
  --config configs/bert-baseline.yaml --epochs 10 \
  --batch-size-train 32 --learning-rate 2e-5 --warmup-ratio 0.1
```

### Step 3: Collect results
- Parse output from each training run
- Format as Table 4 (matching paper layout)
- Save to `Results/table4_foundation_model_comparison.csv`

## Important Notes
- **Pooling:** DNABERT-1 uses `pooler_output` (BERT's built-in pooler on CLS). Other models use CLS token embedding directly. Verify current implementations match.
- **Randomness:** Paper doesn't specify random seed for Experiment 1. Results may vary slightly. Consider running 3-5 seeds and reporting mean.
- **GPU memory:** NT-500M models are large. May need `batch_size_train=16` or gradient accumulation on limited GPU.

## Success Criteria
- [ ] All 8 models train successfully on HS-LC 8:1:1 split
- [ ] Results are within reasonable range of paper values (±2% for AUC)
- [ ] DNABERT-1-3 is best or near-best performer (matching paper finding)
- [ ] BERT-base has lowest performance (matching paper finding)
- [ ] Results saved in reproducible format (CSV)
