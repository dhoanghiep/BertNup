# Phase 02 - Code Gaps

## Context Links
- Paper: `paper/main.tex` Section 2.3 (Model fine-tuning details)
- Models: `bertnup/models/`
- Training: `bertnup/training/trainer.py`

## Overview
- Priority: High
- Status: Pending
- Fill code gaps needed to reproduce all paper experiments.

## Gaps Identified

### Gap 1: BERT-base-cased Backbone
**Paper context:** Tests BERT-base-cased (110M) as a baseline on HS-LC. Uses standard BPE tokenizer.

**What's needed:** A simple BERT model class that wraps `AutoModel` from `bert-base-cased` with CLS token pooling. DNA sequences are tokenized character-by-character (single character tokenizer).

**Implementation:**
- Create `bertnup/models/bert_baseline.py` with `BertBaseline(BertNupBase)`
- Uses `AutoModel.from_pretrained("bert-base-cased")`
- Pooling: CLS token (index 0) → classifier head
- Tokenizer: `AutoTokenizer.from_pretrained("bert-base-cased")`
- DNA sequence treated as raw text, each character becomes a token
- Register in `bertnup/models/__init__.py` as type `bert_baseline`

**Key detail:** The paper says BERT uses "Single character tokenizer" (Table 3). This means DNA chars A/C/G/T are tokenized individually by BERT's tokenizer. Need to verify this works with BERT's WordPiece tokenizer — may need to add spaces between characters.

### Gap 2: Fixed Warmup Steps for 10-fold CV
**Paper context:** "linear warmup of 40 steps with a linear decay optimizer" for 10-fold CV experiments.

**Current state:** `warmup_ratio=0.1` in default config, which computes warmup_steps dynamically.

**What's needed:** Support for setting exact `warmup_steps` value in config/training. For Experiment 1, use `warmup_ratio=0.1`. For Experiments 2+3, use `warmup_steps=40`.

**Implementation:**
- Add `warmup_steps` parameter to training config (default: null)
- In `BertNupBase.configure_optimizers()`, prefer `warmup_steps` if set, else compute from ratio
- Add config override: `--warmup-steps 40`

### Gap 3: HS-LC 8:1:1 Split for Experiment 1
**Paper context:** "partitioned into training, testing, and validation sets in an 8:1:1 ratio"

**What's needed:** See Phase 01 Step 2.

### Gap 4: NT-500M Model Configs
**Paper context:** Tests two NT variants: `nucleotide-transformer-500m-human-ref` and `nucleotide-transformer-500m-1000g`.

**Current state:** `configs/nucleotide_transformer.yaml` exists but may use a different model variant.

**Implementation:**
- Create `configs/nt500m-human-ref.yaml`
- Create `configs/nt500m-1000g.yaml`
- Both use `type: nucleotide_transformer`
- Model names: `InstaDeepAI/nucleotide-transformer-500m-human-ref` and `InstaDeepAI/nucleotide-transformer-500m-1000g`

### Gap 5: DNABERT-1 k={3,4,5,6} Configs
**Paper context:** Tests all 4 k-mer variants.

**Current state:** Only `configs/dnabert1.yaml` with k=3.

**Implementation:**
- Create `configs/dnabert1-k4.yaml`, `configs/dnabert1-k5.yaml`, `configs/dnabert1-k6.yaml`
- Each with correct `armheb/DNA_bert_{k}` model name and `kmer: {k}`

### Gap 6: BERT-base Config
- Create `configs/bert-baseline.yaml`
- Model: `bert-base-cased`, type: `bert_baseline`

## Implementation Steps

1. Create `bertnup/models/bert_baseline.py` — BERT-base-cased backbone
2. Register in `bertnup/models/__init__.py`
3. Add `warmup_steps` support to training config and base model
4. Create all model configs (BERT, DNABERT-1-{4,5,6}, NT-500M x2)
5. Add `single_split` function to `bertnup/data/preparation.py`
6. Add `split_single` CLI subcommand

## Files to Modify
- `bertnup/models/bert_baseline.py` (NEW)
- `bertnup/models/__init__.py` (register bert_baseline)
- `bertnup/models/base.py` (warmup_steps support)
- `bertnup/config.py` (warmup_steps field)
- `bertnup/data/preparation.py` (single_split function)
- `bertnup/cli.py` (split_single command)
- `configs/bert-baseline.yaml` (NEW)
- `configs/dnabert1-k4.yaml` (NEW)
- `configs/dnabert1-k5.yaml` (NEW)
- `configs/dnabert1-k6.yaml` (NEW)
- `configs/nt500m-human-ref.yaml` (NEW)
- `configs/nt500m-1000g.yaml` (NEW)

## Success Criteria
- [ ] `bertnup train` works with `bert-base-cased` model
- [ ] `warmup_steps` config override works correctly
- [ ] All 8 model configs validate and can instantiate models
- [ ] `bertnup split_single` creates correct 8:1:1 split
