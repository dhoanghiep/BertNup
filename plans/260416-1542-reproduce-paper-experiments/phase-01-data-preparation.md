# Phase 01 - Data Preparation

## Context Links
- Paper: `paper/main.tex` Section 2.1 (Datasets)
- Data prep module: `bertnup/data/preparation.py`
- Data directories: `Data/Dataset_1/`, `Data/Dataset_2/`

## Overview
- Priority: High
- Status: Pending
- Prepare all 11 datasets with correct splits for reproduction experiments.

## Datasets

### Dataset 1 (Guo et al.) - 3 species, 147bp
| Species | Nucleosome | Linker | Total | FASTA |
|---------|-----------|--------|-------|-------|
| H. sapiens | 2,273 | 2,300 | 4,573 | `Hsapiens.fas` |
| C. elegans | 2,567 | 2,608 | 5,175 | `Celegans.fas` |
| D. melanogaster | 2,900 | 2,850 | 5,750 | `Dmelanogaster.fas` |

### Dataset 2 (Liu et al.) - 8 sub-datasets, 147bp
| Name | Nucleosome | Linker | Total | FASTA |
|------|-----------|--------|-------|-------|
| HS-LC | 97,209 | 65,563 | 162,772 | `HS_LC.fas` |
| HS-PM | 56,404 | 44,639 | 101,043 | `HS_PM.fas` |
| HS-5U | 11,769 | 4,880 | 16,649 | `HS_5U.fas` |
| DM-LC | 46,054 | 30,458 | 76,512 | `DM_LC.fas` |
| DM-PM | 48,251 | 28,763 | 77,014 | `DM_PM.fas` |
| DM-5U | 4,669 | 2,704 | 7,373 | `DM_5U.fas` |
| Y-WG | 39,661 | 4,824 | 44,485 | `Y_WG.fas` |
| Y-PM | 1,880 | 4,463 | 31,836 | `Y_PM.fas` |

## Implementation Steps

### Step 1: Prepare 10-fold CV splits for all 11 datasets
- Use existing `bertnup prepare_data` CLI
- Target: `Data/Stratified_K_fold_data/{dataset_name}/split_{0-9}/`
- Random state: 1 (matching existing default)
- n_splits: 10

```bash
# Dataset 1
KMP_DUPLICATE_LIB_OK=TRUE /Users/danghiep/miniforge3/envs/bertnup/bin/python -m bertnup.cli prepare_data \
  Data/Dataset_1/Hsapiens.fas Data/Dataset_1/Celegans.fas Data/Dataset_1/Dmelanogaster.fas \
  --save-dir Data/Stratified_K_fold_data --n-splits 10

# Dataset 2
KMP_DUPLICATE_LIB_OK=TRUE /Users/danghiep/miniforge3/envs/bertnup/bin/python -m bertnup.cli prepare_data \
  Data/Dataset_2/HS_LC.fas Data/Dataset_2/HS_PM.fas Data/Dataset_2/HS_5U.fas \
  Data/Dataset_2/DM_LC.fas Data/Dataset_2/DM_PM.fas Data/Dataset_2/DM_5U.fas \
  Data/Dataset_2/Y_WG.fas Data/Dataset_2/Y_PM.fas \
  --save-dir Data/Stratified_K_fold_data --n-splits 10
```

### Step 2: Prepare HS-LC 8:1:1 split for Experiment 1
- Need a separate single train/test/val split (8:1:1 ratio)
- Save to `Data/Stratified_K_fold_data/HS_LC_single_split/`
- This split is used for the foundation model comparison (Table 4)

### Step 3: Verify data integrity
- Check sample counts match paper tables
- Verify all sequences are 147bp
- Verify label distribution (nucleosome vs linker)

## Code Changes Required

### Add 8:1:1 split function to `bertnup/data/preparation.py`
```python
def single_split(
    data_path: str,
    test_size: float = 0.1,
    val_size: float = 0.111,  # 0.1 / 0.9 ≈ 0.111 for 8:1:1 ratio
    random_state: int = 42,
    save_dir: str | None = None,
) -> None:
    """Split a FASTA dataset into train/val/test (8:1:1 ratio)."""
```

### Add `single_split` CLI subcommand to `bertnup/cli.py`
- New subcommand: `bertnup split_single DATA_PATH [--save-dir DIR] [--random-state 42]`

## Success Criteria
- [ ] All 11 datasets have 10-fold CV splits in `Data/Stratified_K_fold_data/`
- [ ] HS-LC has a separate 8:1:1 split
- [ ] Sample counts match paper Tables 1 and 2
- [ ] All sequences verified as 147bp
