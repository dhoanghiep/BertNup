"""State space model backbones: HyenaDNA and Caduceus.

These models use fundamentally different architectures from standard transformers:
- HyenaDNA: Hyena operation (gated convolutions + attention hybrid)
- Caduceus: Bidirectional Mamba state space model

Both require custom packages and trust_remote_code=True.
"""

from __future__ import annotations

import torch
from transformers import AutoModel

from bertnup.models.base import BertNupBase
from bertnup.models.pooling import create_pooling


class BertNupHyenaDNA(BertNupBase):
    """HyenaDNA backbone using state space model architecture.

    HyenaDNA uses the Hyena operator (gated convolutions) for efficient
    long-range sequence modeling. Requires `hyena_dna` or equivalent
    custom kernels to be installed.

    Note: The HyenaDNA architecture may not output standard
    `last_hidden_state`. This class handles both standard and
    custom output formats.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_training_steps: int = 0,
        dropout: float = 0.1,
        hidden_size: int = 256,
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
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.backbone = self._apply_lora(self.backbone)
        self.pooler = create_pooling(pooling, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        output = self.backbone(input_ids, attention_mask=attention_mask)
        # HyenaDNA may return dict or ModelOutput with different key names
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, dict) and "last_hidden_state" in output:
            hidden = output["last_hidden_state"]
        else:
            # Fallback: try the first tensor in the output tuple
            hidden = output[0] if isinstance(output, (tuple, list)) else output.logits
        x = self.pooler(hidden, attention_mask)
        logits = self.classifier(x)
        return self._compute_loss_and_probas(logits, labels)


class BertNupCaduceus(BertNupBase):
    """Caduceus backbone using bidirectional Mamba state space model.

    Caduceus processes DNA sequences bidirectionally using Mamba SSMs,
    providing efficient inference with linear-time complexity.
    Requires `mamba-ssm` and `causal-conv1d` packages.

    Note: Caduceus is a bidirectional model (unlike standard Mamba
    which is unidirectional), making it well-suited for classification
    tasks where both upstream and downstream context matters.
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
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.backbone = self._apply_lora(self.backbone)
        self.pooler = create_pooling(pooling, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        output = self.backbone(input_ids, attention_mask=attention_mask)
        # Caduceus output format varies by implementation
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, dict) and "last_hidden_state" in output:
            hidden = output["last_hidden_state"]
        else:
            hidden = output[0] if isinstance(output, (tuple, list)) else output.logits
        x = self.pooler(hidden, attention_mask)
        logits = self.classifier(x)
        return self._compute_loss_and_probas(logits, labels)
