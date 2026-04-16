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
class Config:
    seed: int = 0
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _detect_model_type(model_name: str) -> str:
    """Auto-detect model type from the HuggingFace model name."""
    if "DNA_bert" in model_name:
        return "dnabert1"
    elif "DNABERT-2" in model_name or "nucleotide-transformer" in model_name:
        return "dnabert2"
    return "dnabert1"  # fallback


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

    # Auto-detect model type if not explicitly set differently from default
    if base.model.type == "dnabert1" and "DNA_bert" not in base.model.name:
        detected = _detect_model_type(base.model.name)
        if detected != "dnabert1":
            base.model.type = detected

    # Convert to structured Config
    cfg = OmegaConf.to_container(base, resolve=True)
    model_cfg = ModelConfig(**cfg.get("model", {}))
    training_cfg = TrainingConfig(**cfg.get("training", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))
    output_cfg = OutputConfig(**cfg.get("output", {}))

    return Config(
        seed=cfg.get("seed", 0),
        device=cfg.get("device", "auto"),
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        output=output_cfg,
    )
