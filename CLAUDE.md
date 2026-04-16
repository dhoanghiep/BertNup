# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BertNup is a transformer-based model for nucleosome positioning prediction. It fine-tunes DNABERT for binary classification of nucleosome-forming vs. linker DNA sequences (147bp). Supports DNABERT-1 (k-mer tokenization), DNABERT-2 (raw sequence), and Nucleotide Transformer v2 (6-mer tokenization, ESM-based).

## Package Structure

```
bertnup/
├── cli.py              # CLI entry point (bertnup command)
├── config.py           # YAML config loading via OmegaConf
├── seed.py             # Reproducible seed setting
├── data/
│   ├── sequences.py    # Sequence, DNASequence, KmerSequence
│   ├── metrics.py      # compute_all_metrics()
│   ├── preparation.py  # FASTA parsing, k-fold splitting
│   └── datasets.py     # Dnabert1Dataset, Dnabert2Dataset
├── models/
│   ├── base.py                    # BertNupBase(LightningModule) — shared training loop
│   ├── dnabert1.py                # BertNupV1 — k-mer, pooler_output
│   ├── dnabert2.py                # BertNupV2 — raw seq, mean/max pooling
│   ├── nucleotide_transformer.py  # BertNupNT — NT v2, ESM-based, mean/max pooling
│   ├── heads.py                   # Classification heads (single, enhanced)
│   ├── pooling.py                 # Pooling strategies (mean, max, attention)
│   └── attention.py               # BertNupAttention, export_bert_weights()
├── training/
│   └── trainer.py      # run_training(), run_kfold_cv(), run_evaluation()
└── visualization/
    └── attention_viz.py # Attention score plotting functions
```

## CLI Commands

```bash
bertnup prepare_data [FASTA_PATHS...] [--save-dir DIR] [--n-splits 10]
bertnup train DATA_DIR --model-name NAME [--kmer K] [--epochs 10]
bertnup evaluate CHECKPOINT TEST_CSV --model-name NAME [--kmer K]
bertnup cross_validate DATA_NAME --model-name NAME --kmer K
bertnup visualize_attention CHECKPOINT --model-name NAME [--kmer K]
```

Config overrides in OmegaConf dotlist format: `--learning-rate 1e-4`, `--epochs 5`, etc. YAML configs in `configs/`.

## Key Conventions

- **Model hierarchy:** `BertNupBase` → `BertNupV1` (DNABERT-1, AutoModel+pooler_output) | `BertNupV2` (DNABERT-2, AutoModel+pooling) | `BertNupNT` (Nucleotide Transformer, AutoModel+pooling). `BertNupAttention` is standalone for attention extraction.
- **PyTorch Lightning 2.x:** Uses `on_validation_epoch_end()` (not `validation_epoch_end`), `save_hyperparameters()`, `accelerator='auto'`.
- **K-mer tokenization:** DNABERT-1 converts DNA to overlapping k-mers via `armheb/DNA_bert_{k}` tokenizers. DNABERT-2 tokenizes raw sequences directly. NT v2 uses 6-mer tokenization (ESM-based, `InstaDeepAI/nucleotide-transformer-*`).
- **Model type auto-detection:** `_detect_model_type()` infers type from model name. Type-specific defaults (hidden_size, fixed_length) are applied automatically when type changes from default.
- **Label encoding:** FASTA `n` = nucleosome (1), else = linker (0).
- **Default hyperparameters:** lr=2e-5, epochs=10, batch_size 32/128, warmup_ratio=0.1, val_check_interval=0.1, seed=0.

## Notebooks

Refactored notebooks in `notebooks/` import from the `bertnup` package. Paths use `../` prefix since notebooks are in a subdirectory.

## Data

- `Data/Dataset_1/` — species-level FASTA (C. elegans, D. melanogaster, H. sapiens)
- `Data/Dataset_2/` — region-level FASTA (promoters, 5'UTR, gene bodies)
- Preprocessed splits → `Data/Stratified_K_fold_data/` (gitignored)

## Setup

```bash
conda activate bertnup  # Use the bertnup conda environment
pip install -e .        # Install package with CLI
bertnup --help          # Verify installation
```

**Note:** On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` before running Python to avoid OpenMP duplicate lib crash.

Pre-trained DNABERT and Nucleotide Transformer models are downloaded automatically by HuggingFace transformers.
