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
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    if verbose == 1:
        loss = log_loss(labels, probas)
        print(f"loss: {loss:.6f}")
    return sn, sp, acc, f1, mcc, auc
