"""Configuration loading and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class ModelConfig:
    name: str = "armheb/DNA_bert_3"
    type: str = "dnabert1"  # auto-detected from name if not set
    kmer: Optional[int] = None
    fixed_length: int = 70
    pooling: str = "mean"
    reinit_layers: int = 0
    dropout: float = 0.1
    hidden_size: int = 768
    head_type: str = "single"
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 10
    batch_size_train: int = 32
    batch_size_test: int = 128
    num_workers: int = 0
    warmup_ratio: float = 0.1
    val_check_interval: float = 0.1
    max_grad_norm: float = 10.0
    early_stopping_patience: Optional[int] = None
    precision: str = "32"
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    augment_rc: bool = False
    use_class_weights: bool = False


@dataclass
class DataConfig:
    all_data_dir: str = "Data/Stratified_K_fold_data"
    n_splits: int = 10
    random_state: int = 1


@dataclass
class OutputConfig:
    checkpoint_dir: str = "model_checkpoint"
    result_dir: str = "Results"
    plot_dir: str = "plot_data"
    attention_plot_dir: str = "attention_plots"


@dataclass
class ExperimentConfig:
    tracker: Optional[str] = None  # "wandb" or None
    project: str = "bertnup"
    run_name: Optional[str] = None
    save_dir: str = "wandb_logs"


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _detect_model_type(model_name: str) -> str:
    """Auto-detect model type from the HuggingFace model name."""
    if "DNA_bert" in model_name:
        return "dnabert1"
    elif "nucleotide-transformer" in model_name:
        return "nucleotide_transformer"
    elif "DNABERT-2" in model_name:
        return "dnabert2"
    elif "evo-1" in model_name:
        return "evo"
    elif "hyena-dna" in model_name or "hyena_dna" in model_name:
        return "hyena_dna"
    elif "caduceus" in model_name:
        return "caduceus"
    return "dnabert1"  # fallback


# Defaults that differ from default.yaml per model type
# Applied when auto-detection changes the type (user hasn't set type explicitly)
_DEFAULT_YAML_HIDDEN_SIZE = 768
_DEFAULT_YAML_FIXED_LENGTH = 70
_MODEL_TYPE_DEFAULTS = {
    "nucleotide_transformer": {"hidden_size": 1280, "fixed_length": 30},
    "dnabert2": {"hidden_size": 768, "fixed_length": 70},
    "evo": {"hidden_size": 4096, "fixed_length": 160},
    "hyena_dna": {"hidden_size": 256, "fixed_length": 160},
    "caduceus": {"hidden_size": 768, "fixed_length": 160},
}


def load_config(config_path: Optional[str] = None, overrides: Optional[list[str]] = None) -> Config:
    """Load configuration from YAML file with optional CLI overrides.

    Overrides are in OmegaConf dotlist format, e.g.:
        ["model.name=armheb/DNA_bert_3", "training.epochs=5"]
    """
    # Start with defaults
    base = OmegaConf.load(DEFAULTS_PATH)

    # Layer on user config if provided
    if config_path is not None:
        user_cfg = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, user_cfg)

    # Layer on CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override_cfg)

    # Auto-detect model type if not explicitly set
    if base.model.type == "dnabert1" and "DNA_bert" not in base.model.name:
        detected = _detect_model_type(base.model.name)
        if detected != "dnabert1":
            base.model.type = detected
            # Apply type-specific defaults only when values match default.yaml
            # (i.e. user hasn't overridden them via config file or CLI)
            type_defaults = _MODEL_TYPE_DEFAULTS.get(detected, {})
            if type_defaults.get("hidden_size") and base.model.hidden_size == _DEFAULT_YAML_HIDDEN_SIZE:
                base.model.hidden_size = type_defaults["hidden_size"]
            if type_defaults.get("fixed_length") and base.model.fixed_length == _DEFAULT_YAML_FIXED_LENGTH:
                base.model.fixed_length = type_defaults["fixed_length"]

    # Convert to structured Config
    cfg = OmegaConf.to_container(base, resolve=True)
    model_cfg = ModelConfig(**cfg.get("model", {}))
    training_cfg = TrainingConfig(**cfg.get("training", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))
    output_cfg = OutputConfig(**cfg.get("output", {}))
    experiment_cfg = ExperimentConfig(**cfg.get("experiment", {}))

    return Config(
        seed=cfg.get("seed", 0),
        device=cfg.get("device", "auto"),
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        output=output_cfg,
        experiment=experiment_cfg,
    )
