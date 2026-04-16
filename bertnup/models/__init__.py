"""Model factory for creating the appropriate BertNup model."""

from __future__ import annotations

from bertnup.config import ModelConfig
from bertnup.models.base import BertNupBase
from bertnup.models.dnabert1 import BertNupV1
from bertnup.models.dnabert2 import BertNupV2
from bertnup.models.nucleotide_transformer import BertNupNT

# Optional extended backbones — import fails gracefully if deps missing
_EVO_AVAILABLE = True
_SSM_AVAILABLE = True
try:
    from bertnup.models.evo import BertNupEvo
except ImportError:
    _EVO_AVAILABLE = False

try:
    from bertnup.models.ssm import BertNupCaduceus, BertNupHyenaDNA
except ImportError:
    _SSM_AVAILABLE = False


def get_model_class(model_type: str) -> type[BertNupBase]:
    """Get the model class for a given model type.

    Centralized dispatch to avoid duplicating model registries across files.
    """
    _registry = {
        "dnabert1": BertNupV1,
        "dnabert2": BertNupV2,
        "nucleotide_transformer": BertNupNT,
    }
    if _EVO_AVAILABLE:
        from bertnup.models.evo import BertNupEvo
        _registry["evo"] = BertNupEvo
    if _SSM_AVAILABLE:
        from bertnup.models.ssm import BertNupCaduceus, BertNupHyenaDNA
        _registry["hyena_dna"] = BertNupHyenaDNA
        _registry["caduceus"] = BertNupCaduceus

    if model_type not in _registry:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_registry.keys())}")
    return _registry[model_type]


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
    elif model_config.type in ("dnabert2", "nucleotide_transformer", "evo", "hyena_dna", "caduceus"):
        model_class = get_model_class(model_config.type)
        return model_class(
            **common_kwargs,
            pooling=model_config.pooling,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
