"""Ensemble prediction combining multiple fine-tuned models."""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bertnup.config import Config
from bertnup.data.datasets import create_dataset
from bertnup.data.metrics import compute_all_metrics
from bertnup.seed import set_seed


class EnsemblePredictor:
    """Combine predictions from multiple fine-tuned checkpoints.

    Supports uniform or weighted averaging of prediction probabilities
    from heterogeneous models (e.g., DNABERT-1 + NT v2 + Evo).
    """

    def __init__(
        self,
        checkpoints: list[str],
        model_configs: list[Config],
        weights: list[float] | None = None,
    ):
        if len(checkpoints) != len(model_configs):
            raise ValueError("checkpoints and model_configs must have the same length")
        self.checkpoints = checkpoints
        self.model_configs = model_configs

        if weights is not None:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(checkpoints)] * len(checkpoints)

    def predict(
        self,
        test_path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Run ensemble prediction on a test set.

        Returns:
            (probas, labels, metrics_dict)
        """
        all_probas = []

        for ckpt_path, cfg in zip(self.checkpoints, self.model_configs):
            mc = cfg.model
            model_type = mc.type

            # Load checkpoint using centralized registry
            from bertnup.models import get_model_class

            model_class = get_model_class(model_type)
            model = model_class.load_from_checkpoint(ckpt_path)

            # Build test dataset
            if model_type == "dnabert1":
                test_set = create_dataset("dnabert1", test_path, kmer=mc.kmer)
            else:
                trust_remote = model_type in ("dnabert2", "evo", "hyena_dna", "caduceus")
                tokenizer = AutoTokenizer.from_pretrained(mc.name, trust_remote_code=trust_remote)
                test_set = create_dataset(model_type, test_path, tokenizer=tokenizer, fixed_length=mc.fixed_length)

            test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

            trainer = Trainer(
                accelerator="auto" if device == "auto" else ("gpu" if device in ("cuda", "gpu") else "cpu"),
                devices=1,
                enable_model_summary=False,
            )
            outputs = trainer.predict(model=model, dataloaders=test_loader)
            probas = torch.cat(outputs).numpy()
            all_probas.append(probas)

            # Free memory
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Weighted average of probabilities
        labels = None
        ensemble_probas = np.zeros_like(all_probas[0])
        for probas, weight in zip(all_probas, self.weights):
            ensemble_probas += weight * probas

        # Get labels from test CSV
        df = pd.read_csv(test_path)
        labels = df["label"].values

        sn, sp, acc, f1, mcc, auc = compute_all_metrics(ensemble_probas, labels)
        metrics = {"sn": sn, "sp": sp, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc}
        return ensemble_probas, labels, metrics
