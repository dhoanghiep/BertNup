"""Model factory for creating the appropriate BertNup model."""

from __future__ import annotations

from bertnup.config import ModelConfig
from bertnup.models.base import BertNupBase
from bertnup.models.dnabert1 import BertNupV1
from bertnup.models.dnabert2 import BertNupV2


def create_model(model_config: ModelConfig, num_training_steps: int) -> BertNupBase:
    """Create a BertNup model based on configuration.

    Auto-detects model type from model_name if not explicitly set.
    """
    warmup_steps = int(num_training_steps * 0.1)  # default 10% warmup

    common_kwargs = dict(
        pretrained_model_name=model_config.name,
        learning_rate=model_config.dropout,  # will be overridden by training config
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        dropout=model_config.dropout,
        hidden_size=model_config.hidden_size,
    )

    if model_config.type == "dnabert1":
        return BertNupV1(
            pretrained_model_name=model_config.name,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            dropout=model_config.dropout,
            hidden_size=model_config.hidden_size,
            reinit_layers=model_config.reinit_layers,
        )
    elif model_config.type == "dnabert2":
        return BertNupV2(
            pretrained_model_name=model_config.name,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            dropout=model_config.dropout,
            hidden_size=model_config.hidden_size,
            pooling=model_config.pooling,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
