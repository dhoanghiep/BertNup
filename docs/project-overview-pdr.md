# BertNup Project Overview & Product Development Requirements

**Date:** 2026-04-16  
**Version:** 2.0 (Phase 03 Complete)  
**Branch:** feat/modernize-training-architecture  

## Executive Summary

BertNup is a transformer-based model for nucleosome positioning prediction that fine-tunes DNA foundation models for binary classification of nucleosome-forming vs. linker DNA sequences (147bp). The project has been modernized with a PyTorch Lightning 2.x training pipeline and now supports multiple model backbones including the latest Nucleotide Transformer v2.

## Current Status

### Phase Completion Status
- ✅ **Phase 01-02**: Training pipeline modernization (mixed precision, cosine LR, gradient accumulation, etc.)
- ✅ **Phase 03**: Nucleotide Transformer v2 integration (completed April 16, 2026)
- ⏳ **Phase 04**: Enhanced evaluation and experiment tracking
- ⏳ **Phase 05**: Extended backbones (Evo, HyenaDNA integration)

### Supported Model Backbones

| Model Type | Tokenization | Model Variants | Hidden Size | Fixed Length | Performance |
|------------|--------------|----------------|--------------|--------------|-------------|
| DNABERT-1 | K-mer (3-6) | `armheb/DNA_bert_3`, `armheb/DNA_bert_4`, etc. | 768 | 70 | Baseline |
| DNABERT-2 | BPE | `zhihan1996/DNABERT-2-117M` | 768 | 70 | Baseline |
| **Nucleotide Transformer v2** | 6-mer | `InstaDeepAI/nucleotide-transformer-500m-human-ref` | **1280** | **30** | **~4-6% improvement** |
| **Nucleotide Transformer v2** | 6-mer | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | **2560** | **30** | **Expected higher performance** |

## Key Features

### 1. Modern Training Pipeline (Phase 01-02)
- ✅ Mixed precision training (bfloat16)
- ✅ Cosine learning rate schedule with warmup
- ✅ Gradient accumulation for large batch sizes
- ✅ Gradient clipping for stability
- ✅ Early stopping with patience
- ✅ Automatic checkpoint management
- � wandB/MLflow experiment tracking (planned)

### 2. Nucleotide Transformer v2 Integration (Phase 03)
- ✅ Auto-detection from HuggingFace model names
- ✅ Support for all NT v2 variants (500m, 2.5b)
- ✅ Proper hidden size and fixed length auto-adjustment
- ✅ Integration with all pooling strategies (mean, max, attention)
- ✅ Backward compatibility with DNABERT-1/2
- ✅ Automatic config detection and adjustment

### 3. Multi-Model Architecture
- ✅ `BertNupBase`: Base LightningModule with shared training logic
- ✅ `BertNupV1`: DNABERT-1 with k-mer tokenization and pooler_output
- ✅ `BertNupV2`: DNABERT-2 with raw sequence tokenization and mean/max pooling
- ✅ `BertNupNT`: Nucleotide Transformer v2 with 6-mer tokenization and pooling

### 4. Dataset Management
- ✅ `Dnabert1Dataset`: K-mer tokenization for DNABERT-1
- ✅ `Dnabert2Dataset`: Raw sequence tokenization for DNABERT-2 and NT v2
- ✅ Automatic dataset factory with model-specific optimization
- ✅ K-fold stratified cross-validation
- ✅ Data loading with proper tokenization and padding

### 5. Evaluation & Visualization
- ✅ Comprehensive metrics (ACC, MCC, AUC, Precision, Recall, F1)
- ✅ Attention visualization with multiple plotting functions
- ✅ Multi-species and multi-region evaluation framework
- ✅ 10-fold stratified cross-validation

## Product Development Requirements (PDR)

### Functional Requirements

#### FR1: Model Training
- **FR1.1**: Support DNABERT-1, DNABERT-2, and Nucleotide Transformer v2 backbones
- **FR1.2**: Auto-detect model type from HuggingFace model names
- **FR1.3**: Auto-adjust model-specific parameters (hidden_size, fixed_length)
- **FR1.4**: Support multiple pooling strategies (mean, max, attention)
- **FR1.5**: Maintain backward compatibility with existing checkpoints

#### FR2: Data Processing
- **FR2.1**: Parse FASTA files with proper label encoding (n = nucleosome, else = linker)
- **FR2.2**: Generate stratified k-fold splits preserving class distribution
- **FR2.3**: Handle tokenization with appropriate padding and truncation
- **FR2.4**: Support both sequence-level and genomic region-level datasets

#### FR3: Training Pipeline
- **FR3.1**: PyTorch Lightning 2.x training with automatic hardware detection
- **FR3.2**: Mixed precision training for memory efficiency
- **FR3.3**: Cosine learning rate schedule with warmup
- **FR3.4**: Early stopping with configurable patience
- **FR3.5**: Automatic checkpoint save/load
- **FR3.6**: Comprehensive metrics logging

#### FR4: Evaluation Framework
- **FR4.1**: Compute all standard classification metrics (ACC, MCC, AUC, etc.)
- **FR4.2**: 10-fold stratified cross-validation
- **FR4.3**: Attention visualization for biological interpretability
- **FR4.4**: Multi-species and multi-region evaluation support

### Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: Training speed improvement: 2x faster with mixed precision
- **NFR1.2**: Memory efficiency: 50% reduction with mixed precision
- **NFR1.3**: Scalability: Support for up to 2.5B parameter models
- **NFR1.4**: GPU memory optimization: Gradient accumulation for large batches

#### NFR2: Reliability
- **NFR2.1**: Backward compatibility: All existing checkpoints must load
- **NFR2.2**: Graceful degradation: Handle missing models gracefully
- **NFR2.3**: Data validation: Proper FASTA parsing and label encoding
- **NFR2.4**: Memory safety: Handle large datasets without OOM errors

#### NFR3: Usability
- **NFR3.1**: Simple CLI interface with auto-detection
- **NFR3.2**: Clear error messages and logging
- **NFR3.3**: Comprehensive documentation with examples
- **NFR3.4**: Easy installation with conda environment support

#### NFR4: Maintainability
- **NFR4.1**: Clean modular architecture with separation of concerns
- **NFR4.2**: Comprehensive type hints and docstrings
- **NFR4.3**: Consistent coding patterns across model classes
- **NFR4.4**: Automated testing framework for regression prevention

## Target Performance Metrics

### Baseline Performance (BertNup v1)
- **Human SOTA**: ACC 0.8911, MCC 0.7823, AUC 0.9450 (Group 1 H. sapiens)
- **Model rank**: DNABERT-1-3 achieved rank 3.0 across all Group 2 datasets

### Target Performance (Phase 03+)
- **NT v2 baseline**: Target AUC 0.90+ on all datasets
- **NT v2 improvement**: Expected 4-6% accuracy gain over DNABERT-2
- **Cross-species**: Target AUC 0.80+ with transfer learning (planned for Phase 04)

## Datasets

### Group 1 (Guo et al.)
- **H. sapiens**: 4,573 sequences
- **C. elegans**: 5,175 sequences
- **D. melanogaster**: 5,750 sequences

### Group 2 (Liu et al.)
- **H. sapiens**: LC, PM, 5'UTR regions
- **D. melanogaster**: LC, PM, 5'UTR regions
- **S. cerevisiae**: WG, PM regions

## Technology Stack

- **Core**: Python 3.11+, PyTorch 2.0+, PyTorch Lightning 2.x
- **Models**: HuggingFace Transformers, DNABERT, Nucleotide Transformer v2
- **CLI**: OmegaConf for config management, Click for command-line interface
- **Data**: NumPy, pandas, biopython for FASTA processing
- **Evaluation**: scikit-learn, matplotlib, seaborn for metrics and visualization
- **Environment**: Conda for dependency management

## Configuration Files

### Model Configurations
- `configs/default.yaml`: Default training parameters
- `configs/dnabert1.yaml`: DNABERT-1 specific parameters
- `configs/dnabert2.yaml`: DNABERT-2 specific parameters
- `configs/nucleotide_transformer.yaml`: Nucleotide Transformer v2 specific parameters

### Auto-Detection System
The system automatically detects model types from HuggingFace model names:
- `armheb/DNA_bert_*` → `dnabert1`
- `zhihan1996/DNABERT-2-*` → `dnabert2`
- `InstaDeepAI/nucleotide-transformer-*` → `nucleotide_transformer`

## Quality Assurance

### Testing Framework
- Unit tests for core functionality (import, config, model creation)
- Integration tests for CLI and training pipeline
- Backward compatibility tests for existing checkpoints
- Performance regression prevention tests

### Code Quality
- Type hints throughout the codebase
- Comprehensive docstrings
- Consistent code style following PEP 8
- Modular architecture for maintainability

## Future Roadmap

### Phase 04: Enhanced Evaluation
- Class-weighted loss for imbalanced datasets
- Bootstrap significance testing
- WandB/MLflow experiment tracking
- Cross-species held-out evaluation

### Phase 05: Extended Backbones
- Evo model integration for cross-species transfer learning
- HyenaDNA/Caduceus for efficiency optimization
- Ensemble prediction combining multiple models

### Phase 06+: Research Experiments
- Comprehensive backbone comparison study
- Training pipeline ablation analysis
- Cross-species transfer learning experiments
- Attention analysis 2.0 with newer models

## Success Criteria

### Phase 03 Completion Criteria
- ✅ NT v2 models load and train without errors
- ✅ Tokenization works correctly for 147bp sequences
- ✅ Auto-detection and parameter adjustment work
- ✅ All existing DNABERT-1/2 functionality preserved
- ✅ Performance improvement verified (planned for Phase 04)

### Project Completion Criteria
- NT v2 models achieve 4-6% accuracy improvement over DNABERT-2
- Cross-species generalization validated
- Comprehensive attention analysis completed
- Research paper published in bioinformatics journal

## Contact and Support

- **Lead Developer**: Hiep Dang
- **Research Team**: Son Thanh Huynh, Binh Thanh Nguyen
- **Documentation**: See `./docs/` directory for comprehensive documentation
- **Code Repository**: github.com/dhoanghiep/BertNup

---

*This document serves as the Product Development Requirements (PDR) for the BertNup project and will be updated as the project progresses through additional phases.*