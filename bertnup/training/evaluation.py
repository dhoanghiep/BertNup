"""Cross-species evaluation and significance testing workflows."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bertnup.config import Config
from bertnup.data.datasets import create_dataset
from bertnup.data.metrics import bootstrap_auc_comparison, compute_all_metrics, compute_metrics_with_ci
from bertnup.models import create_model
from bertnup.seed import set_seed


def _compute_class_weights(labels_path: str) -> list[float]:
    """Compute inverse-frequency class weights from a CSV with a 'label' column."""
    import pandas as pd

    df = pd.read_csv(labels_path)
    counts = df["label"].value_counts().sort_index().values
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return weights.tolist()


def _evaluate_model_on_test(
    model: torch.nn.Module,
    test_path: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run prediction and return (probas, labels, metrics_dict)."""
    mc = config.model

    if mc.type == "dnabert1":
        test_set = create_dataset("dnabert1", test_path, kmer=mc.kmer)
    else:
        trust_remote = mc.type in ("dnabert2", "evo", "hyena_dna", "caduceus")
        tokenizer = AutoTokenizer.from_pretrained(mc.name, trust_remote_code=trust_remote)
        test_set = create_dataset(mc.type, test_path, tokenizer=tokenizer, fixed_length=mc.fixed_length)

    test_loader = DataLoader(
        test_set,
        batch_size=config.training.batch_size_test,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    trainer = Trainer(accelerator="auto", devices=1, enable_model_summary=False)
    outputs = trainer.predict(model=model, dataloaders=test_loader)
    probas = torch.cat(outputs).numpy()
    labels = np.vstack(list(test_set.data.label))

    sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas, labels)
    metrics = {"sn": sn, "sp": sp, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc}
    return probas, labels, metrics


def run_cross_species_eval(
    config: Config,
    train_species_dir: str,
    test_species_dirs: list[str],
    output_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Train on one species, evaluate on multiple held-out species.

    Args:
        config: Full configuration.
        train_species_dir: Path to training species directory (train.csv, val.csv, test.csv).
        test_species_dirs: Paths to directories for each test species.
        output_path: Optional CSV path for saving results.

    Returns:
        Dict mapping species name to metrics dict.
    """
    from bertnup.training.trainer import _build_dataloaders, _get_accelerator

    set_seed(config.seed)

    # Build training data
    train_path = os.path.join(train_species_dir, "train.csv")
    val_path = os.path.join(train_species_dir, "val.csv")
    test_path = os.path.join(train_species_dir, "test.csv")

    train_loader, val_loader, _, _, num_training_steps = _build_dataloaders(
        config, train_path, val_path, test_path
    )

    # Compute class weights if enabled
    class_weights = None
    if config.training.use_class_weights:
        class_weights = _compute_class_weights(train_path)

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    model = create_model(config.model, num_training_steps, warmup_ratio=config.training.warmup_ratio)
    model.hparams.learning_rate = config.training.learning_rate
    model.hparams.weight_decay = config.training.weight_decay
    model.hparams.lr_scheduler_type = config.training.lr_scheduler_type
    if class_weights is not None:
        model.hparams.class_weights = class_weights

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoint/cross_species",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )
    callbacks = [checkpoint_callback]
    if config.training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=config.training.early_stopping_patience, mode="min")
        )

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

    # Load best model
    model_class = type(model)
    best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Evaluate on training species + all test species
    results = {}
    species_dirs = [("train_species", test_path)] + [
        (os.path.basename(d), os.path.join(d, "test.csv")) for d in test_species_dirs
    ]

    for species_name, sp_test_path in species_dirs:
        if not os.path.exists(sp_test_path):
            print(f"Skipping {species_name}: {sp_test_path} not found")
            continue
        _, _, metrics = _evaluate_model_on_test(best_model, sp_test_path, config)
        results[species_name] = metrics
        print(f"{species_name}: auc={metrics['auc']:.4f}, acc={metrics['acc']:.4f}, mcc={metrics['mcc']:.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["species", "sn", "sp", "acc", "f1", "mcc", "auc"])
            for species, m in results.items():
                writer.writerow([species, m["sn"], m["sp"], m["acc"], m["f1"], m["mcc"], m["auc"]])
        print(f"Cross-species results saved to {output_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_callback.best_model_path):
        os.remove(checkpoint_callback.best_model_path)

    return results


def run_significance_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    model1_name: str = "model1",
    model2_name: str = "model2",
    n_bootstraps: int = 1000,
) -> dict:
    """Compare two models using bootstrap AUC test.

    Returns:
        Dict with comparison results and interpretation.
    """
    result = bootstrap_auc_comparison(y_true, y_pred1, y_pred2, n_bootstraps=n_bootstraps)
    result["model1"] = model1_name
    result["model2"] = model2_name
    result["significant"] = not (result["ci_low"] <= 0 <= result["ci_high"])
    print(
        f"AUC diff ({model1_name} - {model2_name}): {result['auc_diff']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}], p={result['p_value']:.4f}"
    )
    return result
