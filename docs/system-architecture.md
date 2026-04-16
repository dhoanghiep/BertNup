# BertNup System Architecture Documentation

**Date:** 2026-04-16  
**Version:** 2.0 (Phase 03 Complete)  
**Branch:** feat/modernize-training-architecture  

## System Overview

BertNup is a PyTorch Lightning-based system for nucleosome positioning prediction using transformer-based DNA foundation models. The architecture supports multiple model backbones with a unified training pipeline and comprehensive evaluation framework.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           BertNup System                           │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Interface (cli.py)                                            │
│  ├─ bertnup train                                                  │
│  ├─ bertnup evaluate                                              │
│  ├─ bertnup cross_validate                                        │
│  └─ bertnup visualize_attention                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Configuration System (config.py)                                 │
│  ├─ OmegaConf-based YAML loading                                   │
│  ├─ Auto-detection of model types                                 │
│  └─ Parameter auto-adjustment for NT v2                          │
├─────────────────────────────────────────────────────────────────────┤
│  Data Pipeline                                                     │
│  ├─ FASTA parsing & validation (preparation.py)                    │
│  ├─ K-fold stratified splitting (preparation.py)                  │
│  ├─ Tokenization (datasets.py)                                    │
│  │  ├─ Dnabert1Dataset (k-mer tokenization)                        │
│  │  └─ Dnabert2Dataset (raw sequence tokenization)                 │
│  └─ Metrics computation (metrics.py)                              │
├─────────────────────────────────────────────────────────────────────┤
│  Model Architecture                                                │
│  ├─ BertNupBase (base.py)                                          │
│  │  ├─ Training loop (LightningModule)                             │
│  │  ├─ Loss computation & optimization                            │
│  │  └─ Metrics logging                                              │
│  ├─ BertNupV1 (dnabert1.py) - DNABERT-1 backbone                  │
│  ├─ BertNupV2 (dnabert2.py) - DNABERT-2 backbone                  │
│  └─ BertNupNT (nucleotide_transformer.py) - NT v2 backbone         │
├─────────────────────────────────────────────────────────────────────┤
│  Training & Evaluation Pipeline                                  │
│  ├─ PyTorch Lightning Trainer (trainer.py)                        │
│  ├─ Mixed precision training                                      │
│  ├─ Cosine LR scheduling                                          │
│  ├─ Early stopping                                                │
│  └─ Cross-validation framework                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Visualization & Analysis                                         │
│  ├─ Attention visualization (attention_viz.py)                   │
│  ├─ Training metrics plotting                                     │
│  └─ Model performance analysis                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System

#### Architecture
- **File**: `bertnup/config.py`
- **Framework**: OmegaConf
- **Purpose**: Hierarchical configuration management with auto-detection

#### Key Components

```python
@dataclass
class ModelConfig:
    name: str                    # HuggingFace model name
    type: str                    # Auto-detected model type
    hidden_size: int            # Model embedding dimension
    fixed_length: int           # Sequence length for tokenization
    pooling: str               # Pooling strategy

@dataclass
class TrainingConfig:
    batch_size_train: int       # Training batch size
    batch_size_val: int         # Validation batch size
    learning_rate: float       # Learning rate
    epochs: int                # Number of training epochs
    warmup_ratio: float        # Warmup ratio for scheduler

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    seed: int                  # Random seed for reproducibility
```

#### Auto-Detection Logic

```python
def _detect_model_type(model_name: str) -> str:
    """Detect model type from HuggingFace model name."""
    if "armheb/DNA_bert_" in model_name:
        return "dnabert1"
    elif "zhihan1996/DNABERT-2-" in model_name:
        return "dnabert2"
    elif "InstaDeepAI/nucleotide-transformer-" in model_name:
        return "nucleotide_transformer"
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def _adjust_nt_defaults(cfg: Config):
    """Auto-adjust NT v2 specific parameters."""
    if cfg.model.type == "nucleotide_transformer":
        if "model.hidden_size" not in cfg:
            cfg.model.hidden_size = 1280  # Default for 500M models
        if "model.fixed_length" not in cfg:
            cfg.model.fixed_length = 30    # Optimal for 147bp sequences
```

#### Configuration Flow

```
1. Load base config (configs/default.yaml)
2. Apply CLI overrides
3. Auto-detect model type if 'auto'
4. Auto-adjust NT v2 parameters
5. Validate final configuration
6. Return resolved Config object
```

### 2. Data Processing Pipeline

#### Architecture
- **Files**: `bertnup/data/`
- **Core Classes**: `Dnabert1Dataset`, `Dnabert2Dataset`
- **Purpose**: Tokenization, batching, and data augmentation

#### Data Flow

```
FASTA Files → Parse & Validate → Stratified K-Fold Split → Dataset Creation
```

#### Dataset Classes

##### Dnabert1Dataset (K-mer Tokenization)
```python
class Dnabert1Dataset(Dataset):
    """Dataset for DNABERT-1 k-mer tokenization."""
    
    def __init__(self, sequences: List[str], labels: List[int], 
                 kmer_tokenizer, fixed_length: int):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = kmer_tokenizer
        self.fixed_length = fixed_length
    
    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to k-mers
        kmers = self.tokenizer(sequence_to_kmers(sequence))
        
        # Pad/truncate
        kmers = kmers[:self.fixed_length]
        if len(kmers) < self.fixed_length:
            kmers = kmers + [0] * (self.fixed_length - len(kmers))
        
        return {
            "input_ids": torch.tensor(kmers, dtype=torch.long),
            "attention_mask": torch.ones(self.fixed_length),
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

##### Dnabert2Dataset (Raw Sequence Tokenization)
```python
class Dnabert2Dataset(Dataset):
    """Dataset for raw sequence tokenization (DNABERT-2, NT v2)."""
    
    def __init__(self, sequences: List[str], labels: List[int], 
                 tokenizer, fixed_length: int):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.fixed_length = fixed_length
    
    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize with padding/truncation
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

#### Dataset Factory Pattern

```python
def create_dataset(model_type: str, sequences: List[str], 
                  labels: List[int], tokenizer, 
                  fixed_length: int) -> Dataset:
    """Factory function for dataset creation."""
    
    if model_type == "dnabert1":
        return Dnabert1Dataset(sequences, labels, tokenizer, fixed_length)
    elif model_type in ("dnabert2", "nucleotide_transformer"):
        return Dnabert2Dataset(sequences, labels, tokenizer, fixed_length)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```

### 3. Model Architecture

#### Base Class Architecture

```python
class BertNupBase(pl.LightningModule):
    """Base class for all BertNup models with shared training logic."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Initialize components
        self.backbone = None
        self.classifier = None
        self.pooler = None
        
        # Setup
        self._setup_backbone()
        self._setup_classifier()
        self._setup_pooling()
    
    def _setup_backbone(self):
        """Initialize transformer backbone."""
        self.backbone = AutoModel.from_pretrained(self.hparams.model.name)
    
    def _setup_classifier(self):
        """Initialize classification head."""
        self.classifier = create_head(
            hidden_size=self.hparams.model.hidden_size,
            head_type="linear"
        )
    
    def _setup_pooling(self):
        """Initialize pooling strategy."""
        self.pooler = create_pooler(
            pooling_type=self.hparams.model.pooling,
            hidden_size=self.hparams.model.hidden_size
        )
    
    def forward(self, sequences: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        output = self.backbone(input_ids=sequences, 
                              attention_mask=attention_mask)
        pooled = self.pooler(output, attention_mask)
        return self.classifier(pooled)
```

#### Model Variants

##### BertNupV1 (DNABERT-1)
```python
class BertNupV1(BertNupBase):
    """DNABERT-1 model with k-mer tokenization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # DNABERT-1 specific setup
        self.kmer = int(config.name.split("_")[-1])  # Extract k from model name
        self.tokenizer = DNA_bert_tokenizer(self.kmer)
        
        # DNABERT-1 uses pooler_output (768-dim)
        assert self.backbone.config.pooler_output_dim == 768
    
    def forward(self, sequences, attention_mask):
        # DNABERT-1 specific forward pass
        output = self.backbone(input_ids=sequences, 
                              attention_mask=attention_mask)
        # Use pooler_output directly
        pooled = output.pooler_output
        return self.classifier(pooled)
```

##### BertNupV2 (DNABERT-2)
```python
class BertNupV2(BertNupBase):
    """DNABERT-2 model with raw sequence tokenization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # DNABERT-2 setup handled by base class
        
        # DNABERT-2 needs trust_remote_code
        self.backbone = AutoModel.from_pretrained(
            self.hparams.model.name,
            trust_remote_code=True
        )
    
    def forward(self, sequences, attention_mask):
        # DNABERT-2 uses last_hidden_state with pooling
        output = self.backbone(input_ids=sequences, 
                              attention_mask=attention_mask)
        pooled = self.pooler(output, attention_mask)
        return self.classifier(pooled)
```

##### BertNupNT (Nucleotide Transformer v2)
```python
class BertNupNT(BertNupBase):
    """Nucleotide Transformer v2 model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Auto-detect hidden_size
        actual_hidden = self.backbone.config.hidden_size
        if config.hidden_size != actual_hidden:
            warnings.warn(f"Adjusting hidden_size: {config.hidden_size} → {actual_hidden}")
            self.hparams.model.hidden_size = actual_hidden
            self.classifier = create_head(
                head_type="linear",
                hidden_size=actual_hidden
            )
        
        # NT v2 uses standard AutoModel (no trust_remote_code needed)
        self.backbone = AutoModel.from_pretrained(
            self.hparams.model.name,
            trust_remote_code=False
        )
    
    def forward(self, sequences, attention_mask):
        # NT v2 uses last_hidden_state with pooling
        output = self.backbone(input_ids=sequences, 
                              attention_mask=attention_mask)
        pooled = self.pooler(output, attention_mask)
        return self.classifier(pooled)
```

#### Pooling Strategies

```python
def create_pooler(pooling_type: str, hidden_size: int):
    """Factory function for pooling strategies."""
    
    if pooling_type == "mean":
        return MeanPooling(hidden_size)
    elif pooling_type == "max":
        return MaxPooling(hidden_size)
    elif pooling_type == "attention":
        return AttentionPooling(hidden_size)
    elif pooling_type == "cls":
        # Not supported for V2/NT models
        raise ValueError("CLS pooling not supported for this model type")
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")

class MeanPooling(nn.Module):
    """Mean pooling over sequence dimension."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, output, attention_mask):
        # output.last_hidden_state shape: (batch, seq_len, hidden)
        # attention_mask shape: (batch, seq_len)
        
        mask = attention_mask.unsqueeze(-1).float()
        mask = mask.masked_fill(mask == 0, -1e9)
        
        return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable weights."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, output, attention_mask):
        # Compute attention weights
        attn_weights = self.attention(output.last_hidden_state)
        attn_weights = attn_weights.squeeze(-1)
        
        # Mask padding tokens
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and weight by attention
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        pooled = (output.last_hidden_state * attn_weights).sum(dim=1)
        
        return pooled
```

### 4. Training Pipeline

#### PyTorch Lightning Integration

```python
def run_training(data_dir: str, config: Config) -> str:
    """Run training with modern PyTorch Lightning features."""
    
    # Load and prepare data
    sequences, labels = load_fasta_data(data_dir)
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=config.seed
    )
    
    # Create tokenizer and model
    tokenizer = create_tokenizer(config.model.name, config.model.type)
    model = create_model(config.model.type, config.model)
    
    # Create datasets
    train_dataset = create_dataset(
        config.model.type, train_sequences, train_labels, 
        tokenizer, config.model.fixed_length
    )
    val_dataset = create_dataset(
        config.model.type, val_sequences, val_labels, 
        tokenizer, config.model.fixed_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size_val,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=config.training.epochs,
        batch_size=config.training.batch_size_train,
        precision="16-mixed",  # Mixed precision training
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
        enable_model_summary=True,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return trainer.checkpoint_callback.best_model_path
```

#### Cross-Validation Framework

```python
def run_kfold_cv(data_dir: str, config: Config, n_splits: int = 10) -> Dict[str, List[float]]:
    """Run k-fold cross-validation."""
    
    # Load all data
    sequences, labels = load_fasta_data(data_dir)
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Run training for this fold
        fold_metrics = run_single_fold(
            train_sequences, train_labels, val_sequences, val_labels, config
        )
        all_metrics.append(fold_metrics)
    
    # Aggregate results
    aggregated_metrics = aggregate_cv_results(all_metrics)
    return aggregated_metrics
```

### 5. Evaluation System

#### Metrics Computation

```python
def compute_all_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, 
                       y_probs: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(y_probs):
        y_probs = y_probs.cpu().numpy()
    
    # Binary classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Compute specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }
```

#### Evaluation Pipeline

```python
def run_evaluation(checkpoint_path: str, test_csv: str, config: Config) -> Dict[str, float]:
    """Run evaluation on test data."""
    
    # Load trained model
    model = BertNup.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    
    # Load test data
    test_sequences, test_labels = load_test_data(test_csv)
    
    # Create tokenizer and dataset
    tokenizer = create_tokenizer(config.model.name, config.model.type)
    test_dataset = create_dataset(
        config.model.type, test_sequences, test_labels, 
        tokenizer, config.model.fixed_length
    )
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size_val,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            sequences = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            # Forward pass
            logits = model(sequences, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu())
            all_probs.extend(probs[:, 1].cpu())  # Positive class probabilities
            all_labels.extend(labels.cpu())
    
    # Compute metrics
    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)
    y_probs = torch.tensor(all_probs)
    
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    
    return metrics
```

### 6. Attention Visualization

#### Architecture
- **File**: `bertnup/visualization/attention_viz.py`
- **Purpose**: Visualize attention patterns for biological interpretability

#### Key Functions

```python
def plot_attention_heatmap(sequences: List[str], attention_weights: torch.Tensor, 
                          output_path: str, title: str = "Attention Pattern"):
    """Plot attention heatmap for visual analysis."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot attention weights
    im = ax.imshow(attention_weights.cpu().numpy(), cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Sequence Position')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_attention_patterns(model, sequences: List[str], 
                               output_dir: str):
    """Generate comprehensive attention visualization."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sequence in enumerate(sequences):
        # Extract attention weights
        attention_weights = extract_attention_weights(model, sequence)
        
        # Generate visualizations
        plot_attention_heatmap(
            sequence, attention_weights,
            f"{output_dir}/attention_heatmap_{i}.png",
            f"Attention Pattern - Sequence {i+1}"
        )
        
        plot_attention_summary(
            sequence, attention_weights,
            f"{output_dir}/attention_summary_{i}.png",
            f"Attention Summary - Sequence {i+1}"
        )
```

## Configuration Files

### Model Configuration Hierarchy

```
configs/
├── default.yaml              # Base configuration
├── dnabert1.yaml            # DNABERT-1 specific settings
├── dnabert2.yaml            # DNABERT-2 specific settings
└── nucleotide_transformer.yaml # NT v2 specific settings
```

### Configuration Structure

#### Default Configuration
```yaml
# configs/default.yaml
model:
  name: "armheb/DNA_bert_3"  # Default model
  type: "auto"               # Auto-detect
  hidden_size: 768           # Default hidden size
  fixed_length: 70           # Default sequence length
  pooling: "mean"            # Default pooling

training:
  batch_size_train: 32       # Training batch size
  batch_size_val: 64         # Validation batch size
  learning_rate: 2e-5        # Learning rate
  epochs: 10                 # Number of epochs
  warmup_ratio: 0.1         # Warmup ratio
  val_check_interval: 0.1   # Validation frequency

seed: 0                      # Random seed
```

#### Nucleotide Transformer Configuration
```yaml
# configs/nucleotide_transformer.yaml
model:
  name: "InstaDeepAI/nucleotide-transformer-500m-human-ref"
  type: "nucleotide_transformer"
  hidden_size: 1280          # 1280 for 500M models
  fixed_length: 30           # Optimal for 147bp sequences
  pooling: "mean"            # Pooling strategy

training:
  batch_size_train: 32       # Adjust based on GPU memory
  learning_rate: 2e-5        # Standard fine-tuning LR
  precision: "16-mixed"      # Mixed precision
```

## Data Flow Architecture

### Complete Data Pipeline

```
1. Input: FASTA files with 147bp sequences
2. Processing: Parse FASTA → Validate sequences → Encode labels
3. Splitting: Stratified k-fold cross-validation
4. Tokenization: 
   - DNABERT-1: Convert to k-mers → Pad to 70 tokens
   - DNABERT-2/NT v2: Tokenize directly → Pad to 30/70 tokens
5. Training: PyTorch Lightning with modern features
6. Evaluation: Comprehensive metrics computation
7. Visualization: Attention pattern analysis
```

### Memory Management

- **Mixed Precision**: Reduces memory usage by 50%
- **Gradient Accumulation**: Simulates larger batches
- **Pin Memory**: Faster data transfer to GPU
- **Sequence Length Optimization**: Minimizes padding

### Performance Optimizations

- **Cosine Learning Rate**: Better convergence than linear
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Training stability
- **Mixed Precision**: 2x faster training

## Error Handling and Validation

### Configuration Validation

```python
def _validate_config(cfg: Config):
    """Validate configuration parameters."""
    
    # Model validation
    if cfg.model.type not in ["dnabert1", "dnabert2", "nucleotide_transformer"]:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    
    # Training parameter validation
    if cfg.training.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if cfg.training.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    # NT v2 specific validation
    if cfg.model.type == "nucleotide_transformer":
        if cfg.model.hidden_size not in [1280, 2560]:
            raise ValueError("NT v2 hidden_size must be 1280 or 2560")
```

### Data Validation

```python
def validate_fasta_sequences(sequences: List[str]):
    """Validate FASTA sequences for BertNup requirements."""
    
    for i, seq in enumerate(sequences):
        # Check length (should be 147bp)
        if len(seq) != 147:
            raise ValueError(f"Sequence {i} has length {len(seq)}, expected 147")
        
        # Check valid nucleotides
        if not all(c in "ATCG" for c in seq.upper()):
            raise ValueError(f"Sequence {i} contains invalid nucleotides: {seq}")
        
        # Check if all same nucleotide (edge case)
        if len(set(seq.upper())) == 1:
            warnings.warn(f"Sequence {i} consists of identical nucleotides")
```

## System Integration Points

### CLI Integration

```python
# CLI commands connect to core functionality
@click.group()
def cli():
    """BertNup - Nucleosome Positioning Prediction"""
    pass

@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--model-name", required=True, help="HuggingFace model name")
@click.option("--config", help="YAML config file")
def train(data_dir, model_name, config):
    """Train nucleosome prediction model."""
    
    # Load configuration
    if config:
        cfg = load_config(["--config", config])
    else:
        cfg = load_config([f"model.name={model_name}"])
    
    # Run training
    checkpoint = run_training(data_dir, cfg)
    click.echo(f"Training complete. Best checkpoint: {checkpoint}")
```

### Model Factory Pattern

```python
def create_model(model_type: str, config: ModelConfig) -> BertNupBase:
    """Factory function for model creation."""
    
    if model_type == "dnabert1":
        return BertNupV1(config)
    elif model_type == "dnabert2":
        return BertNupV2(config)
    elif model_type == "nucleotide_transformer":
        return BertNupNT(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```

## Performance Benchmarks

### Expected Performance Improvements

| Model | Expected AUC | Training Time | Memory Usage |
|-------|--------------|---------------|--------------|
| DNABERT-2 | 0.83-0.88 | Baseline | Baseline |
| Nucleotide Transformer v2 | 0.87-0.94 | ~10% faster | 50% less with mixed precision |
| Evo (planned) | Expected 0.88-0.95 | Unknown | Expected similar to NT v2 |

### System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (for 500M models)
- **RAM**: 32GB+ for large datasets
- **Storage**: 50GB+ for models and checkpoints
- **Python**: 3.11+
- **PyTorch**: 2.0+
- **CUDA**: 11.7+ for best performance

## Future Architecture Extensions

### Phase 04: Enhanced Evaluation
- **Class-weighted loss**: Handle imbalanced datasets
- **Bootstrap significance testing**: Statistical validation
- **Experiment tracking**: WandB/MLflow integration
- **Cross-species evaluation**: Transfer learning framework

### Phase 05: Extended Backbones
- **Evo model integration**: Cross-species transfer learning
- **HyenaDNA integration**: State space models
- **Caduceus integration**: Bidirectional processing
- **Ensemble prediction**: Multi-model combination

### Phase 06+: Research Features
- **Attention analysis 2.0**: Deeper biological insights
- **Model distillation**: Smaller, faster models
- **Multi-modal integration**: Combine with epigenetic data
- **Active learning**: Intelligent sample selection

---

*This architecture document provides a comprehensive overview of the BertNup system design, implementation patterns, and integration points. The architecture supports modular extension while maintaining backward compatibility and performance optimization.*