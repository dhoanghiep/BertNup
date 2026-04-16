"""BertNup V2 model for DNABERT-2 (raw sequence) fine-tuning."""

from __future__ import annotations

import torch
from transformers import AutoModel

from bertnup.models.base import BertNupBase


class BertNupV2(BertNupBase):
    """DNABERT-2 model with pooling and classification head.

    Uses AutoModel to load a pretrained DNABERT-2 backbone and applies
    mean or max pooling over the last hidden state before the classification head.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_training_steps: int = 0,
        dropout: float = 0.1,
        hidden_size: int = 768,
        pooling: str = "mean",
    ):
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            dropout=dropout,
            hidden_size=hidden_size,
        )
        self.dnabert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.pooling = pooling

    def _pool(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool sequence dimension of last_hidden_state."""
        if self.pooling == "max":
            return torch.max(hidden_state, dim=1)[0]
        return torch.mean(hidden_state, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        output = self.dnabert(input_ids, attention_mask=attention_mask)
        x = self._pool(output["last_hidden_state"])
        x = self.dropout1(x)
        logits = self.linear1(x)
        return self._compute_loss_and_probas(logits, labels)
