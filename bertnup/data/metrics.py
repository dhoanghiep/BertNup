"""Evaluation metrics for nucleosome positioning prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_all_metrics(
    probas: np.ndarray, labels: np.ndarray, threshold: float = 0.5, verbose: int = 0
) -> tuple[float, float, float, float, float, float]:
    """Compute sensitivity, specificity, accuracy, F1, MCC, and AUC."""
    preds = (probas > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probas)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).flatten()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    if verbose == 1:
        loss = log_loss(labels, probas)
        print(f"loss: {loss:.6f}")
    return sn, sp, acc, f1, mcc, auc


def compute_metrics_with_ci(
    probas: np.ndarray,
    labels: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all metrics with bootstrap confidence intervals.

    Returns:
        Dict mapping metric name to {"mean": float, "std": float, "ci_low": float, "ci_high": float}.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    alpha = (1 - confidence) / 2

    metric_names = ["sn", "sp", "acc", "f1", "mcc", "auc"]
    bootstrap_results = {m: [] for m in metric_names}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        boot_labels = labels[indices]
        boot_probas = probas[indices]
        # Skip if only one class present in bootstrap sample
        if len(np.unique(boot_labels)) < 2:
            continue
        sn, sp, acc, f1, mcc, auc = compute_all_metrics(boot_probas, boot_labels)
        for name, val in zip(metric_names, [sn, sp, acc, f1, mcc, auc]):
            bootstrap_results[name].append(val)

    result = {}
    for name in metric_names:
        vals = np.array(bootstrap_results[name])
        if len(vals) == 0:
            result[name] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        result[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_low": float(np.percentile(vals, 100 * alpha)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha))),
        }
    return result


def bootstrap_auc_comparison(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap test for whether two models have significantly different AUCs.

    Returns:
        Dict with "auc_diff", "p_value", "ci_low", "ci_high" for the AUC difference (model1 - model2).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        boot_true = y_true[indices]
        if len(np.unique(boot_true)) < 2:
            continue
        auc1 = roc_auc_score(boot_true, y_pred1[indices])
        auc2 = roc_auc_score(boot_true, y_pred2[indices])
        diffs.append(auc1 - auc2)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"auc_diff": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    return {
        "auc_diff": float(np.mean(diffs)),
        "p_value": float(np.mean(diffs > 0) * 2),  # two-sided
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
