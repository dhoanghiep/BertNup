"""BertNup Evo model using StripedHyena backbone for DNA classification."""

from __future__ import annotations

import torch
from torch import nn

from bertnup.models.base import BertNupBase
from bertnup.models.pooling import create_pooling


class BertNupEvo(BertNupBase):
    """Evo backbone model using StripedHyena architecture.

    Evo is a 7B parameter biological foundation model using byte-level
    single-nucleotide resolution. Uses AutoModelForCausalLM with
    trust_remote_code=True. Requires the `stripedhyena` package.

    Note: Evo is an autoregressive (causal) model, not bidirectional.
    For classification, we use mean pooling over hidden states.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_training_steps: int = 0,
        dropout: float = 0.1,
        hidden_size: int = 4096,
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
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True, revision="1.1_fix"
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            config=config,
            trust_remote_code=True,
            revision="1.1_fix",
        )
        self.backbone = self._apply_lora(self.backbone)
        self.pooler = create_pooling(pooling, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        output = self.backbone(input_ids, attention_mask=attention_mask)
        # Evo (causal LM) stores hidden states in output.hidden_states
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden = output.hidden_states[-1]
        else:
            # Fallback: use logits as proxy (not ideal, but handles edge cases)
            hidden = output.logits
        x = self.pooler(hidden, attention_mask)
        logits = self.classifier(x)
        return self._compute_loss_and_probas(logits, labels)
