# Phase 02: Remote Training CLI

## Status: Complete

## Context

- Server provisioned via Phase 01
- Code synced via rsync
- Data is ~90MB (Dataset_1 + Dataset_2)

## Overview

Create `scripts/train-remote.sh` that syncs code+data to the server, runs training via SSH, and streams output locally.

## Requirements

1. Sync code (bertnup/, configs/, pyproject.toml) to remote
2. Sync data (Data/) to remote
3. Execute training command on remote server
4. Stream stdout/stderr back to local terminal
5. Support both `train` and `cross_validate` commands
6. Support all existing CLI flags passed through

## Implementation Steps

1. Create `scripts/train-remote.sh` with:
   - Configurable vars: SSH connection, remote workdir
   - `rsync` code to remote:
     ```bash
     rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
       -e "ssh -p $SSH_PORT" \
       $LOCAL_DIR/ $SSH_USER@$SSH_HOST:$REMOTE_DIR/
     ```
   - `rsync` data to remote (only if `--sync-data` flag)
   - SSH command to run training:
     ```bash
     ssh -p $SSH_PORT $SSH_USER@$SSH_HOST \
       "cd $REMOTE_DIR && KMP_DUPLICATE_LIB_OK=TRUE pip install -e . -q && bertnup $CMD $@"
     ```
   - Pass through all CLI args to remote `bertnup` command

2. Usage examples:
   ```bash
   # First time: sync data + train
   ./scripts/train-remote.sh --sync-data train Data/Stratified_K_fold_data/HS_LC/split_0 --model-name armheb/DNA_bert_3

   # Subsequent: code-only sync + cross_validate
   ./scripts/train-remote.sh cross_validate HS_LC --model-name armheb/DNA_bert_3 --config configs/dnabert1.yaml

   # Just run (no sync)
   ./scripts/train-remote.sh --no-sync train Data/... --model-name ...
   ```

3. Make executable: `chmod +x scripts/train-remote.sh`

## Related Code Files

- **Create:** `scripts/train-remote.sh`
- **Uses:** `bertnup/cli.py` (remote execution)

## Success Criteria

- `./scripts/train-remote.sh train <args>` syncs code and starts training on GPU
- Training output streams back to local terminal
- Results/checkpoints are saved on remote (or pulled back with `--pull-results`)

## Next Steps

- Run reproduce-paper-experiments plan on GPU server
