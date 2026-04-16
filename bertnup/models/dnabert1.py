"""BertNup V1 model for DNABERT-1 (k-mer based) fine-tuning."""

from __future__ import annotations

import torch
from transformers import AutoModel

from bertnup.models.base import BertNupBase


class BertNupV1(BertNupBase):
    """DNABERT-1 model with custom classification head.

    Uses AutoModel to load a pretrained DNABERT-1 backbone (armheb/DNA_bert_{k})
    and adds a dropout + linear classification head on top of the pooler output.
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
        reinit_layers: int = 0,
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
        self.dnabert = AutoModel.from_pretrained(pretrained_model_name)

        if reinit_layers > 0:
            for i in range(reinit_layers):
                self.dnabert.encoder.layer[-(i + 1)].apply(self.dnabert._init_weights)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.dnabert(input_ids, attention_mask=attention_mask)["pooler_output"]
        x = self.dropout1(x)
        logits = self.linear1(x)
        return self._compute_loss_and_probas(logits, labels)
