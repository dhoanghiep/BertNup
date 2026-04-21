# DNABERT-2 Autoresearch Launch Report

**Date:** 2026-04-20
**Server:** ssh9.vast.ai:38939 (RTX 5070 Ti, 15.5 GB VRAM, CUDA 13.1)

## Summary

Successfully set up and launched DNABERT-2 autoresearch on vast.ai GPU server.

## Steps Completed

1. **Code Sync** - rsync'd BertNup to `/root/BertNup/` (excluded .git, __pycache__, .DS_Store)
2. **Data Verified** - 10-fold CV splits present at `Data/Stratified_K_fold_data/HS_LC/split_0/`
3. **Dependencies Installed** - torch 2.11.0+cu128, transformers 5.5.4, pytorch-lightning 2.6.1
4. **Compatibility Fixes Applied** (see below)
5. **Autoresearch Launched** - 22 experiments, 5 epochs each, 4h timeout

## Compatibility Fixes Required

The RTX 5070 Ti + CUDA 13.1 + Python 3.12 environment required multiple fixes:

### Fix 1: ALiBi Meta Tensor Error
- **Cause:** DNABERT-2's `bert_layers.py` creates ALiBi tensors with `torch.Tensor()` which fails with torch 2.11's prim dispatch
- **Fix:** Patched cached `bert_layers.py` at `/workspace/.hf_home/modules/transformers_modules/zhihan1996/DNABERT_hyphen_2_hyphen_117M/`:
  - `torch.zeros(...)` -> `torch.zeros(..., device="cpu")`
  - `self.rebuild_alibi_tensor(size=config.alibi_starting_size)` -> added `device="cpu"`
  - `torch.Tensor(...)` -> `torch.tensor(..., device=device, dtype=torch.float32)`

### Fix 2: Triton Flash Attention Incompatibility
- **Cause:** `flash_attn_triton.py` uses deprecated `tl.dot(..., trans_b=True)` removed in triton 3.6
- **Fix:** Disabled flash attention import in `bert_layers.py` - model falls back to PyTorch attention
- **Note:** This reduces throughput but doesn't affect accuracy

### Fix 3: Tuple vs Dict Model Output
- **Cause:** Custom MosaicML BertModel returns `(hidden_states, pooled)` tuple, not dict-like `BaseModelOutput`
- **Fix:** Updated `bertnup/models/dnabert2.py` forward method to handle both tuple and dict outputs

### Fix 4: torchvision Conflict
- **Cause:** torchvision 0.20.1+cu128 conflicts with torch 2.11.0 (NMS operator crash)
- **Fix:** Uninstalled torchvision (not needed by BertNup)

## Baseline Results (1 epoch test)

```
val_loss:         0.244981
val_acc:          0.9122
val_auc:          0.9525
peak_vram_mb:     2962.1
num_params_M:     117.1
speed:            5.89 it/s (~9 min/epoch)
```

## Autoresearch Configuration

- **Script:** `/root/BertNup/autoresearch/agent-loop-dnabert2.sh`
- **Config:** `configs/dnabert2.yaml` (num_workers=4, batch_size_test=128)
- **Data:** `Data/Stratified_K_fold_data/HS_LC/split_0`
- **Timeout:** 14400s (4h) per experiment
- **Results:** `autoresearch/results-dnabert2.tsv`
- **Log:** `autoresearch/agent-dnabert2.log`
- **Total experiments:** 22

## Estimated Runtime

- ~45 min per experiment (5 epochs x 9 min)
- 22 experiments x 45 min = ~16.5 hours total
- Within server rental budget

## Files Modified Locally

1. `configs/dnabert2.yaml` - Added num_workers=4, batch_size_test=128
2. `bertnup/models/dnabert2.py` - Handle both tuple and dict model outputs
3. `autoresearch/agent-loop-dnabert2.sh` - New autoresearch script (created)

## Unresolved Questions

1. Patches to cached DNABERT-2 model files are ephemeral - if HF cache is cleared, patches must be reapplied. Should we create a startup patching script?
2. The MosaicML triton flash attention is permanently broken with triton 3.6+. PyTorch fallback attention works but is slower (~6 it/s vs potential 10+ it/s with flash).
3. torch 2.11.0+cu128 works on RTX 5070 Ti despite it being CUDA 13.1 hardware - driver forward-compat handles it, but cu130 build fails with NCCL symbol errors.
