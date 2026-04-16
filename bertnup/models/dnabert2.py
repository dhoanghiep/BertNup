"""BertNup V2 model for DNABERT-2 (raw sequence) fine-tuning."""

from __future__ import annotations

import torch
from transformers import AutoModel

from bertnup.models.base import BertNupBase
from bertnup.models.pooling import create_pooling


class BertNupV2(BertNupBase):
    """DNABERT-2 model with pooling and classification head.

    Uses AutoModel to load a pretrained DNABERT-2 backbone and applies
    configurable pooling over the last hidden state before the classification head.
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
        head_type: str = "single",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ):
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            dropout=dropout,
            hidden_size=hidden_size,
            head_type=head_type,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.dnabert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.dnabert = self._apply_lora(self.dnabert)
        self.pooler = create_pooling(pooling, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        output = self.dnabert(input_ids, attention_mask=attention_mask)
        x = self.pooler(output["last_hidden_state"], attention_mask)
        logits = self.classifier(x)
        return self._compute_loss_and_probas(logits, labels)
