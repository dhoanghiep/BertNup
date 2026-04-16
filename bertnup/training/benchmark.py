"""Model comparison and benchmarking utilities."""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bertnup.config import Config
from bertnup.data.datasets import create_dataset
from bertnup.data.metrics import bootstrap_auc_comparison, compute_all_metrics, compute_metrics_with_ci
from bertnup.seed import set_seed


def _load_model_and_predict(
    config: Config,
    checkpoint_path: str,
    test_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, int]:
    """Load a model, run prediction, return (probas, labels, metrics, time_seconds, peak_memory_mb)."""
    mc = config.model

    from bertnup.models import get_model_class

    model_class = get_model_class(mc.type)
    model = model_class.load_from_checkpoint(checkpoint_path)

    # Build test dataset
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

    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer = Trainer(accelerator="auto", devices=1, enable_model_summary=False)
    outputs = trainer.predict(model=model, dataloaders=test_loader)
    elapsed = time.time() - start_time

    probas = torch.cat(outputs).numpy()
    import pandas as pd
    df = pd.read_csv(test_path)
    labels = df["label"].values

    sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas, labels)
    metrics = {"sn": sn, "sp": sp, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc}

    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    del model, trainer
    return probas, labels, metrics, elapsed, peak_mem, n_params


def run_benchmark(
    model_configs: list[tuple[Config, str, str]],
    output_path: str | None = None,
    n_bootstraps: int = 1000,
) -> list[dict]:
    """Compare multiple models on the same test set.

    Args:
        model_configs: List of (config, checkpoint_path, test_path) tuples.
        output_path: Optional CSV path for saving results.
        n_bootstraps: Number of bootstrap samples for confidence intervals.

    Returns:
        List of result dicts with metrics, timing, and parameter counts.
    """
    results = []
    all_probas = []
    labels = None

    for i, (config, ckpt_path, test_path) in enumerate(model_configs):
        set_seed(config.seed)
        model_name = config.model.name.replace("/", "_")
        print(f"Benchmarking [{i+1}/{len(model_configs)}]: {model_name}")

        probas, lbls, metrics, elapsed, peak_mem, n_params = _load_model_and_predict(
            config, ckpt_path, test_path
        )
        all_probas.append(probas)
        if labels is None:
            labels = lbls

        # Compute confidence intervals
        ci = compute_metrics_with_ci(probas, lbls, n_bootstraps=n_bootstraps, seed=config.seed)

        result = {
            "model": model_name,
            "type": config.model.type,
            "n_params": n_params,
            "time_seconds": round(elapsed, 2),
            "peak_memory_mb": round(peak_mem, 2),
        }
        # Add point estimates and CI for each metric
        for metric_name, val in metrics.items():
            result[metric_name] = round(val, 4)
            if metric_name in ci:
                result[f"{metric_name}_ci_low"] = round(ci[metric_name]["ci_low"], 4)
                result[f"{metric_name}_ci_high"] = round(ci[metric_name]["ci_high"], 4)

        results.append(result)
        print(f"  auc={metrics['auc']:.4f} [{ci['auc']['ci_low']:.4f}, {ci['auc']['ci_high']:.4f}]")

    # Pairwise significance tests
    if len(results) >= 2:
        print("\nPairwise AUC significance tests:")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                comp = bootstrap_auc_comparison(
                    labels, all_probas[i], all_probas[j], n_bootstraps=n_bootstraps
                )
                sig = "SIGNIFICANT" if not (comp["ci_low"] <= 0 <= comp["ci_high"]) else "not significant"
                print(
                    f"  {results[i]['model']} vs {results[j]['model']}: "
                    f"diff={comp['auc_diff']:.4f} [{comp['ci_low']:.4f}, {comp['ci_high']:.4f}] ({sig})"
                )

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if results:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\nBenchmark results saved to {output_path}")

    return results
