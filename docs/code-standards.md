# BertNup Code Standards and Architecture

**Date:** 2026-04-16  
**Version:** 2.0  
**Scope:** Entire codebase  

## Architecture Overview

BertNup follows a modular architecture with clear separation of concerns. The system is organized into several main components:

```
bertnup/
├── cli.py              # CLI entry point (bertnup command)
├── config.py           # YAML config loading via OmegaConf
├── seed.py             # Reproducible seed setting
├── data/
│   ├── sequences.py    # Sequence, DNASequence, KmerSequence
│   ├── metrics.py      # compute_all_metrics()
│   ├── preparation.py  # FASTA parsing, k-fold splitting
│   └── datasets.py     # Dnabert1Dataset, Dnabert2Dataset
├── models/
│   ├── base.py         # BertNupBase(LightningModule) — shared training loop
│   ├── dnabert1.py     # BertNupV1 — k-mer, pooler_output
│   ├── dnabert2.py     # BertNupV2 — raw seq, mean/max pooling
│   ├── nucleotide_transformer.py # BertNupNT — NT v2, mean/max/attention pooling
│   └── attention.py    # BertNupAttention, export_bert_weights()
├── training/
│   └── trainer.py      # run_training(), run_kfold_cv(), run_evaluation()
└── visualization/
    └── attention_viz.py # Attention score plotting functions
```

## Core Architecture Principles

### 1. Model Hierarchy

All models inherit from `BertNupBase`, which provides the common training infrastructure:

```python
# Base class with shared training logic
class BertNupBase(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        # Backbone, classifier, pooling initialization
        # Training setup
        
    def forward(self, sequences, attention_mask):
        # Standard forward pass
        
    def training_step(self, batch, batch_idx):
        # Standard training logic with metrics
        
    def validation_step(self, batch, batch_idx):
        # Validation logic with metrics
        
    def configure_optimizers(self):
        # Optimizer, scheduler setup
```

### 2. Model Variants

Each model variant implements the same interface but uses different tokenization and pooling strategies:

#### BertNupV1 (DNABERT-1)
- **Tokenization**: K-mer tokenization via `armheb/DNA_bert_{k}` tokenizers
- **Output**: Uses `pooler_output` from HuggingFace model
- **Classification**: Direct linear layer on pooler output
- **Hidden Size**: Fixed at 768

#### BertNupV2 (DNABERT-2)
- **Tokenization**: BPE tokenization via HuggingFace `AutoTokenizer`
- **Output**: Uses `last_hidden_state` (no pooler_output)
- **Pooling**: Mean or max pooling over sequence
- **Hidden Size**: Fixed at 768

#### BertNupNT (Nucleotide Transformer v2)
- **Tokenization**: 6-mer tokenization via HuggingFace `AutoTokenizer`
- **Output**: Uses `last_hidden_state`
- **Pooling**: Mean, max, or attention pooling over sequence
- **Hidden Size**: Auto-detected (1280 for 500M models, 2560 for 2.5B models)

## Coding Standards

### 1. File Naming Convention

- **Python files**: Use descriptive names in lowercase with underscores (e.g., `nucleotide_transformer.py`)
- **Config files**: Use lowercase with underscores (e.g., `nucleotide_transformer.yaml`)
- **Test files**: Prefix with `test_` and descriptive names
- Keep individual files under 200 lines for maintainability

### 2. Class and Function Naming

- **Class names**: PascalCase (e.g., `BertNupNT`, `Dnabert2Dataset`)
- **Function names**: snake_case (e.g., `create_model`, `load_config`)
- **Variable names**: snake_case (e.g., `hidden_size`, `attention_mask`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_BATCH_SIZE`)

### 3. Type Hints

All code must include comprehensive type hints:

```python
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    type: str
    hidden_size: int
    fixed_length: int
    pooling: str

def create_model(model_type: str, config: ModelConfig) -> BertNupBase:
    """Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ('dnabert1', 'dnabert2', 'nucleotide_transformer')
        config: Model configuration
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Implementation
```

### 4. Docstring Standards

Use Google-style docstrings with comprehensive documentation:

```python
def compute_all_metrics(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    y_probs: torch.Tensor
) -> Dict[str, float]:
    """Compute comprehensive classification metrics for nucleosome prediction.
    
    Args:
        y_true: True binary labels (0=linker, 1=nucleosome)
        y_pred: Predicted binary labels
        y_probs: Predicted probabilities for positive class
        
    Returns:
        Dictionary containing:
        - accuracy: Overall classification accuracy
        - mcc: Matthews correlation coefficient
        - auc: Area under ROC curve
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - specificity: True negative rate
        
    Note:
        Handles binary classification metrics suitable for potentially
        imbalanced nucleosome datasets.
    """
    # Implementation
```

### 5. Error Handling

Use appropriate exception types and provide meaningful error messages:

```python
def _detect_model_type(model_name: str) -> str:
    """Detect model type from HuggingFace model name.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Detected model type
        
    Raises:
        ValueError: If model type cannot be determined
    """
    if model_name.startswith("armheb/DNA_bert_"):
        return "dnabert1"
    elif model_name.startswith("zhihan1996/DNABERT-2-"):
        return "dnabert2"
    elif model_name.startswith("InstaDeepAI/nucleotide-transformer-"):
        return "nucleotide_transformer"
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
```

## Configuration Standards

### 1. Config Structure

Use OmegaConf for hierarchical configuration management:

```python
@dataclass
class ModelConfig:
    name: str
    type: str
    hidden_size: int
    fixed_length: int
    pooling: str

@dataclass
class TrainingConfig:
    batch_size_train: int
    batch_size_val: int
    learning_rate: float
    epochs: int
    warmup_ratio: float

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    seed: int
```

### 2. Auto-Detection System

Implement robust auto-detection with fallback mechanisms:

```python
def load_config(overrides: Optional[List[str]] = None) -> Config:
    """Load configuration with auto-detection and overrides.
    
    Args:
        overrides: List of dot-separated overrides (e.g., ['model.name=foo'])
        
    Returns:
        Fully resolved configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Load base config
    cfg = OmegaConf.load("configs/default.yaml")
    
    # Apply overrides
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    
    # Auto-detect model type if not specified
    if cfg.model.type == "auto":
        cfg.model.type = _detect_model_type(cfg.model.name)
    
    # Auto-adjust model-specific parameters
    if cfg.model.type == "nucleotide_transformer":
        _adjust_nt_defaults(cfg)
    
    # Validate configuration
    _validate_config(cfg)
    
    return cfg
```

## Data Processing Standards

### 1. Dataset Classes

Implement dataset classes with proper tokenization and batching:

```python
class Dnabert2Dataset(Dataset):
    """Dataset for raw sequence tokenization (DNABERT-2, NT v2).
    
    Handles tokenization, padding, and truncation for fixed-length sequences.
    """
    
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, fixed_length: int):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.fixed_length = fixed_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            sequence,
            max_length=self.fixed_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

### 2. Data Processing Pipeline

Use efficient data loading with proper batching:

```python
def create_dataset(
    model_type: str,
    sequences: List[str],
    labels: List[int],
    tokenizer,
    fixed_length: int
) -> Dataset:
    """Factory function to create appropriate dataset.
    
    Args:
        model_type: Type of model ('dnabert1', 'dnabert2', 'nucleotide_transformer')
        sequences: List of DNA sequences
        labels: List of binary labels
        tokenizer: Model-specific tokenizer
        fixed_length: Sequence length for padding/truncation
        
    Returns:
        Dataset instance
    """
    if model_type == "dnabert1":
        return Dnabert1Dataset(sequences, labels, tokenizer, fixed_length)
    elif model_type in ("dnabert2", "nucleotide_transformer"):
        return Dnabert2Dataset(sequences, labels, tokenizer, fixed_length)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```

## Model Implementation Standards

### 1. Base Class Implementation

```python
class BertNupBase(pl.LightningModule):
    """Base class for all BertNup models with shared training logic."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize components
        self.backbone = None
        self.classifier = None
        self.pooler = None
        
        # Setup
        self._setup_backbone()
        self._setup_classifier()
        self._setup_pooling()
    
    def _setup_backbone(self):
        """Initialize the backbone transformer model."""
        self.backbone = AutoModel.from_pretrained(self.hparams.model.name)
    
    def _setup_classifier(self):
        """Initialize the classification head."""
        self.classifier = create_head(
            hidden_size=self.hparams.model.hidden_size,
            head_type="linear"
        )
    
    def _setup_pooling(self):
        """Initialize the pooling strategy."""
        self.pooler = create_pooler(
            pooling_type=self.hparams.model.pooling,
            hidden_size=self.hparams.model.hidden_size
        )
    
    def forward(self, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper pooling and classification."""
        # Encode sequences
        output = self.backbone(input_ids=sequences, attention_mask=attention_mask)
        
        # Apply pooling
        pooled = self.pooler(output, attention_mask)
        
        # Classify
        return self.classifier(pooled)
```

### 2. Model-Specific Implementations

Each model variant should follow the same pattern but implement appropriate tokenization and pooling:

```python
class BertNupNT(BertNupBase):
    """BertNup model with Nucleotide Transformer v2 backbone."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Validate hidden_size matches backbone
        actual_hidden = self.backbone.config.hidden_size
        if config.hidden_size != actual_hidden:
            warnings.warn(
                f"hidden_size={config.hidden_size} does not match "
                f"model's hidden_size={actual_hidden}. Overriding."
            )
            self.hparams.model.hidden_size = actual_hidden
            # Re-create classifier with correct size
            self.classifier = create_head(
                head_type="linear",
                hidden_size=actual_hidden
            )
    
    def _setup_backbone(self):
        """Initialize NT v2 backbone with proper config."""
        self.backbone = AutoModel.from_pretrained(
            self.hparams.model.name,
            trust_remote_code=False  # NT v2 doesn't need custom code
        )
```

## Training Pipeline Standards

### 1. LightningModule Integration

```python
class BertNupBase(pl.LightningModule):
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with comprehensive metrics."""
        sequences = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        # Forward pass
        logits = self.forward(sequences, attention_mask)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Loss computation
        loss = F.cross_entropy(logits, labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", (preds == labels).float().mean(), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step with all metrics."""
        sequences = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        # Forward pass
        logits = self.forward(sequences, attention_mask)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Compute all metrics
        metrics = compute_all_metrics(labels, preds, probs)
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=(name == "accuracy"))
        
        return metrics
```

### 2. Training Function

```python
def run_training(data_dir: str, config: Config) -> Optional[str]:
    """Run training with modern PyTorch Lightning features."""
    
    # Load and prepare data
    sequences, labels = load_fasta_data(data_dir)
    train_data, val_data = create_kfold_splits(sequences, labels)
    
    # Create tokenizer and model
    tokenizer = create_tokenizer(config.model.name, config.model.type)
    model = create_model(config.model.type, config.model)
    
    # Create dataloaders
    train_dataset = create_dataset(
        model_type=config.model.type,
        sequences=train_data["sequences"],
        labels=train_data["labels"],
        tokenizer=tokenizer,
        fixed_length=config.model.fixed_length
    )
    
    val_dataset = create_dataset(
        model_type=config.model.type,
        sequences=val_data["sequences"],
        labels=val_data["labels"],
        tokenizer=tokenizer,
        fixed_length=config.model.fixed_length
    )
    
    # Setup trainer with modern features
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=config.training.epochs,
        batch_size=config.training.batch_size_train,
        precision="16-mixed",  # Mixed precision
        gradient_clip_val=1.0,  # Gradient clipping
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath="./checkpoints",
                filename="best-checkpoint",
                monitor="val_accuracy",
                mode="max",
                save_top_k=1
            )
        ],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Return best checkpoint path
    return trainer.checkpoint_callback.best_model_path
```

## Testing Standards

### 1. Unit Test Structure

```python
import pytest
from bertnup.models import BertNupNT
from bertnup.config import load_config

class TestBertNupNT:
    """Unit tests for BertNupNT model."""
    
    def test_model_creation(self):
        """Test model creation with valid config."""
        config = load_config([
            "model.name=InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "model.type=nucleotide_transformer"
        ])
        
        model = BertNupNT(config.model)
        assert model is not None
        assert model.backbone is not None
        assert model.classifier is not None
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        config = load_config([
            "model.name=InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "model.type=nucleotide_transformer"
        ])
        
        model = BertNupNT(config.model)
        
        # Create dummy input
        batch_size = 2
        seq_length = 30
        sequences = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        
        # Forward pass
        outputs = model(sequences, attention_mask)
        assert outputs.shape == (batch_size, 2)  # Binary classification
```

### 2. Integration Testing

```python
class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline with dummy data."""
        # Create dummy dataset
        sequences = ["ATCG" * 36] * 100  # 144bp sequences
        labels = [1] * 50 + [0] * 50
        
        config = load_config([
            "model.name=InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "model.type=nucleotide_transformer",
            "training.epochs=2"
        ])
        
        # Test complete pipeline
        checkpoint = run_training_dummy(sequences, labels, config)
        assert checkpoint is not None
```

## Documentation Standards

### 1. File Headers

Each Python file should include a header with metadata:

```python
"""
BertNup - Nucleosome Positioning Prediction

This module contains the BertNupNT model implementation for Nucleotide Transformer v2.

Author: Hiep Dang
Date: 2026-04-16
Version: 2.0
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel
```

### 2. README Updates

Keep `README.md` and documentation files synchronized with code changes:

- Update model support matrix when adding new backbones
- Document new configuration options
- Update CLI help text with new model types
- Include performance benchmarks when available

## Performance Standards

### 1. Memory Efficiency

- Use mixed precision training (bfloat16)
- Implement gradient accumulation for large batch sizes
- Use appropriate fixed_length to minimize padding
- Monitor GPU memory usage during training

### 2. Training Efficiency

- Use cosine learning rate schedules
- Implement early stopping to prevent overfitting
- Use proper data loading with multiple workers
- Monitor training metrics and adjust hyperparameters

### 3. Model Performance

- Expected accuracy improvements: NT v2 should achieve 4-6% improvement over DNABERT-2
- Cross-species generalization: Target AUC 0.80+ with transfer learning
- Attention visualization should reveal biologically meaningful patterns

## Security and Privacy

### 1. Data Security

- Handle FASTA files with proper validation
- Never log sensitive genomic data
- Use secure storage for checkpoints and models

### 2. Model Security

- Validate model inputs before processing
- Handle model download failures gracefully
- Use proper error handling for unknown model types

## Quality Assurance Checklist

Before committing code, ensure:

1. [ ] All functions have comprehensive type hints
2. [ ] All classes and functions have docstrings
3. [ ] Code follows the established naming conventions
4. [ ] All tests pass (unit tests, integration tests)
5. [ ] Documentation is updated to reflect changes
6. [ ] Configuration files are properly validated
7. [ ] Error handling is comprehensive and meaningful
8. [ ] Performance benchmarks are met or improved
9. [ ] Backward compatibility is maintained
10. [ ] Code follows the modular architecture principles

---

*This code standard document provides comprehensive guidelines for maintaining code quality and consistency across the BertNup project. All team members should follow these standards when developing new features or maintaining existing code.*