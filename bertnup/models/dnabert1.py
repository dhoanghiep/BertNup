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
        self.dnabert = AutoModel.from_pretrained(pretrained_model_name)
        self.dnabert = self._apply_lora(self.dnabert)

        if reinit_layers > 0:
            for i in range(reinit_layers):
                self.dnabert.encoder.layer[-(i + 1)].apply(self.dnabert._init_weights)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.dnabert(input_ids, attention_mask=attention_mask)["pooler_output"]
        logits = self.classifier(x)
        return self._compute_loss_and_probas(logits, labels)
