"""Model factory for creating the appropriate BertNup model."""

from __future__ import annotations

from bertnup.config import ModelConfig
from bertnup.models.base import BertNupBase
from bertnup.models.dnabert1 import BertNupV1
from bertnup.models.dnabert2 import BertNupV2
from bertnup.models.nucleotide_transformer import BertNupNT


def create_model(model_config: ModelConfig, num_training_steps: int, warmup_ratio: float = 0.1) -> BertNupBase:
    """Create a BertNup model based on configuration.

    Auto-detects model type from model_name if not explicitly set.
    """
    warmup_steps = int(num_training_steps * warmup_ratio)

    common_kwargs = dict(
        pretrained_model_name=model_config.name,
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        dropout=model_config.dropout,
        hidden_size=model_config.hidden_size,
        head_type=model_config.head_type,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
    )

    if model_config.type == "dnabert1":
        return BertNupV1(
            **common_kwargs,
            reinit_layers=model_config.reinit_layers,
        )
    elif model_config.type == "dnabert2":
        return BertNupV2(
            **common_kwargs,
            pooling=model_config.pooling,
        )
    elif model_config.type == "nucleotide_transformer":
        return BertNupNT(
            **common_kwargs,
            pooling=model_config.pooling,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
