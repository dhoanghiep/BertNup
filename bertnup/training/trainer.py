"""Training, evaluation, and cross-validation workflows."""

from __future__ import annotations

import csv
import gc
import os
import pickle
import time

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from bertnup.config import Config
from bertnup.data.datasets import create_dataset
from bertnup.data.metrics import compute_all_metrics
from bertnup.models import create_model
from bertnup.seed import set_seed


def _get_logger(config: Config):
    """Create optional experiment logger based on config."""
    tracking = getattr(config, "experiment", None)
    if tracking is None:
        return None
    tracker_type = getattr(tracking, "tracker", None)
    if tracker_type == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            raise ImportError("wandb is required for WandB tracking. Install with: pip install wandb")
        return WandbLogger(
            project=getattr(tracking, "project", "bertnup"),
            name=getattr(tracking, "run_name", None),
            save_dir=getattr(tracking, "save_dir", "wandb_logs"),
        )
    return None


def _get_accelerator(device: str) -> str:
    """Map config device string to PL accelerator."""
    if device == "auto":
        return "auto"
    elif device in ("cuda", "gpu"):
        return "gpu"
    return "cpu"


def _build_dataloaders(config: Config, train_path: str, val_path: str, test_path: str):
    """Create train/val/test DataLoaders from CSV paths."""
    from transformers import AutoTokenizer

    mc = config.model
    tc = config.training

    if mc.type == "dnabert1":
        train_set = create_dataset("dnabert1", train_path, kmer=mc.kmer, augment_rc=tc.augment_rc)
        val_set = create_dataset("dnabert1", val_path, kmer=mc.kmer)
        test_set = create_dataset("dnabert1", test_path, kmer=mc.kmer)
    else:
        trust_remote = mc.type in ("dnabert2", "evo", "hyena_dna", "caduceus")
        tokenizer = AutoTokenizer.from_pretrained(mc.name, trust_remote_code=trust_remote)
        train_set = create_dataset(mc.type, train_path, tokenizer=tokenizer, fixed_length=mc.fixed_length, augment_rc=tc.augment_rc)
        val_set = create_dataset(mc.type, val_path, tokenizer=tokenizer, fixed_length=mc.fixed_length)
        test_set = create_dataset(mc.type, test_path, tokenizer=tokenizer, fixed_length=mc.fixed_length)

    train_loader = DataLoader(
        train_set,
        batch_size=tc.batch_size_train,
        shuffle=True,
        num_workers=tc.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=tc.batch_size_test,
        shuffle=False,
        num_workers=tc.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=tc.batch_size_test,
        shuffle=False,
        num_workers=tc.num_workers,
    )

    num_training_steps = len(train_loader) * tc.epochs // tc.gradient_accumulation_steps
    return train_loader, val_loader, test_loader, test_set, num_training_steps


def run_training(config: Config, data_dir: str) -> dict:
    """Fine-tune a model on a single train/val/test split.

    Args:
        config: Full configuration.
        data_dir: Path to split directory containing train.csv, val.csv, test.csv.

    Returns:
        Dict with evaluation metrics on the test set.
    """
    set_seed(config.seed)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_loader, val_loader, test_loader, test_set, num_training_steps = _build_dataloaders(
        config, train_path, val_path, test_path
    )

    model = create_model(config.model, num_training_steps, warmup_ratio=config.training.warmup_ratio, warmup_steps_override=config.training.warmup_steps)
    # Override learning rate from training config
    model.hparams.learning_rate = config.training.learning_rate
    model.hparams.weight_decay = config.training.weight_decay
    model.hparams.lr_scheduler_type = config.training.lr_scheduler_type

    # Apply class weights if enabled
    if config.training.use_class_weights:
        import pandas as pd
        train_df = pd.read_csv(train_path)
        counts = train_df["label"].value_counts().sort_index().values
        total = counts.sum()
        weights = (total / (len(counts) * counts)).tolist()
        model.hparams.class_weights = weights

    checkpoint_dir = os.path.join(config.output.checkpoint_dir, config.model.name.replace("/", "_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    callbacks = [checkpoint_callback]
    if config.training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config.training.early_stopping_patience,
                mode="min",
            )
        )

    experiment_logger = _get_logger(config)

    trainer = Trainer(
        accelerator=_get_accelerator(config.device),
        devices=1,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        gradient_clip_val=config.training.max_grad_norm,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        val_check_interval=config.training.val_check_interval,
        enable_model_summary=False,
        logger=experiment_logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate on test set
    model_class = type(model)
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt is None:
        print("WARNING: No best checkpoint found. Using last model state.")
        best_model = model
    else:
        best_model = model_class.load_from_checkpoint(best_ckpt)
    outputs = trainer.predict(model=best_model, dataloaders=test_loader)
    probas = torch.cat(outputs).numpy()
    labels = np.vstack(list(test_set.data.label))

    sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas, labels, verbose=1)
    print(f"sn: {sn:.4f}, sp: {sp:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, mcc: {mcc:.4f}, auc: {auc:.4f}")

    # Parseable summary for autoresearch (grep-friendly)
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float("nan")
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else -1.0
    num_params = sum(p.numel() for p in best_model.parameters()) / 1e6
    print("---")
    print(f"val_loss:         {best_val_loss:.6f}")
    print(f"val_acc:          {acc:.4f}")
    print(f"val_auc:          {auc:.4f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"num_params_M:     {num_params:.1f}")

    # Keep best checkpoint for later evaluation
    if best_ckpt:
        print(f"best_checkpoint:  {best_ckpt}")

    return {"sn": sn, "sp": sp, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc}


def run_kfold_cv(
    config: Config,
    data_name: str,
    start_fold: int = 0,
    end_fold: int = 10,
) -> None:
    """Run k-fold cross-validation for a dataset.

    Results are saved to {output.result_dir}/{model_name}/{data_name}.csv.

    Args:
        config: Full configuration.
        data_name: Dataset name (e.g., 'HS_LC').
        start_fold: First fold index (inclusive).
        end_fold: Last fold index (exclusive).
    """
    set_seed(config.seed)

    data_dir = os.path.join(config.data.all_data_dir, data_name)
    result_dir = os.path.join(
        config.output.result_dir,
        config.model.name.replace("/", "_"),
    )
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(result_dir, f"{data_name}.csv")
    with open(result_path, "w") as f:
        f.write(",Sensitivity,Specificity,Accuracy,F1_Score,MCC,AUC\n")

    for no_split in range(start_fold, end_fold):
        gc.collect()
        print(f"Fold {no_split}")

        split_dir = os.path.join(data_dir, f"split_{no_split}")
        train_path = os.path.join(split_dir, "train.csv")
        val_path = os.path.join(split_dir, "val.csv")
        test_path = os.path.join(split_dir, "test.csv")

        train_loader, val_loader, test_loader, test_set, num_training_steps = _build_dataloaders(
            config, train_path, val_path, test_path
        )

        model = create_model(config.model, num_training_steps, warmup_ratio=config.training.warmup_ratio, warmup_steps_override=config.training.warmup_steps)
        model.hparams.learning_rate = config.training.learning_rate
        model.hparams.weight_decay = config.training.weight_decay
        model.hparams.lr_scheduler_type = config.training.lr_scheduler_type

        # Apply class weights if enabled
        if config.training.use_class_weights:
            import pandas as pd
            train_df = pd.read_csv(train_path)
            counts = train_df["label"].value_counts().sort_index().values
            total = counts.sum()
            weights = (total / (len(counts) * counts)).tolist()
            model.hparams.class_weights = weights

        checkpoint_dir = os.path.join(
            config.output.checkpoint_dir, f"{data_name}_fold{no_split}"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
        )
        callbacks = [
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ]

        trainer = Trainer(
            accelerator=_get_accelerator(config.device),
            devices=1,
            precision=config.training.precision,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            gradient_clip_val=config.training.max_grad_norm,
            max_epochs=config.training.epochs,
            callbacks=callbacks,
            val_check_interval=config.training.val_check_interval,
            enable_model_summary=False,
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Evaluate on test set
        model_class = type(model)
        best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
        outputs = trainer.predict(model=best_model, dataloaders=test_loader)
        probas = torch.cat(outputs).numpy()
        labels = np.vstack(list(test_set.data.label))

        print(f"Fold {no_split} test: ")
        sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas, labels, verbose=1)
        print(f"sn: {sn:.4f}, sp: {sp:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, mcc: {mcc:.4f}, auc: {auc:.4f}")

        with open(result_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([no_split, sn, sp, acc, f1, mcc, auc])

        os.remove(checkpoint_callback.best_model_path)

    print(f"Results for {data_name} saved to {result_path}")


def run_evaluation(
    config: Config,
    checkpoint_path: str,
    test_path: str,
) -> dict:
    """Evaluate a saved checkpoint on a test set.

    Args:
        config: Full configuration.
        checkpoint_path: Path to .ckpt file.
        test_path: Path to test CSV.

    Returns:
        Dict with evaluation metrics.
    """
    from transformers import AutoTokenizer

    mc = config.model

    from bertnup.models import get_model_class

    model_class = get_model_class(mc.type)

    if mc.type == "dnabert1":
        test_set = create_dataset("dnabert1", test_path, kmer=mc.kmer)
    else:
        trust_remote = mc.type in ("dnabert2", "evo", "hyena_dna", "caduceus")
        tokenizer = AutoTokenizer.from_pretrained(mc.name, trust_remote_code=trust_remote)
        test_set = create_dataset(mc.type, test_path, tokenizer=tokenizer, fixed_length=mc.fixed_length)

    model = model_class.load_from_checkpoint(checkpoint_path)

    test_loader = DataLoader(
        test_set,
        batch_size=config.training.batch_size_test,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    trainer = Trainer(
        accelerator=_get_accelerator(config.device),
        devices=1,
        enable_model_summary=False,
    )
    outputs = trainer.predict(model=model, dataloaders=test_loader)
    probas = torch.cat(outputs).numpy()
    labels = np.vstack(list(test_set.data.label))

    print("Evaluation results:")
    sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas, labels, verbose=1)
    print(f"sn: {sn:.4f}, sp: {sp:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, mcc: {mcc:.4f}, auc: {auc:.4f}")

    return {"sn": sn, "sp": sp, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc}
