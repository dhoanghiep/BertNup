"""Command-line interface for BertNup."""

from __future__ import annotations

import argparse
import os
import sys


def _suppress_logging():
    """Suppress verbose logging from transformers and tokenizers."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import transformers
    transformers.utils.logging.set_verbosity_error()
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def _build_config(args) -> "Config":
    """Build a Config from CLI args, optionally layered on a YAML file."""
    from bertnup.config import load_config

    overrides = []
    if args.model_name:
        overrides.append(f"model.name={args.model_name}")
    if hasattr(args, "kmer") and args.kmer is not None:
        overrides.append(f"model.kmer={args.kmer}")
    if hasattr(args, "fixed_length") and args.fixed_length is not None:
        overrides.append(f"model.fixed_length={args.fixed_length}")
    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        overrides.append(f"training.learning_rate={args.learning_rate}")
    if hasattr(args, "epochs") and args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if hasattr(args, "batch_size_train") and args.batch_size_train is not None:
        overrides.append(f"training.batch_size_train={args.batch_size_train}")
    if hasattr(args, "batch_size_test") and args.batch_size_test is not None:
        overrides.append(f"training.batch_size_test={args.batch_size_test}")
    if hasattr(args, "warmup_ratio") and args.warmup_ratio is not None:
        overrides.append(f"training.warmup_ratio={args.warmup_ratio}")
    if hasattr(args, "val_check_interval") and args.val_check_interval is not None:
        overrides.append(f"training.val_check_interval={args.val_check_interval}")
    if hasattr(args, "max_grad_norm") and args.max_grad_norm is not None:
        overrides.append(f"training.max_grad_norm={args.max_grad_norm}")
    if hasattr(args, "dropout") and args.dropout is not None:
        overrides.append(f"model.dropout={args.dropout}")
    if hasattr(args, "reinit_layers") and args.reinit_layers is not None:
        overrides.append(f"model.reinit_layers={args.reinit_layers}")
    if hasattr(args, "pooling") and args.pooling is not None:
        overrides.append(f"model.pooling={args.pooling}")
    if hasattr(args, "head_type") and args.head_type is not None:
        overrides.append(f"model.head_type={args.head_type}")
    if hasattr(args, "use_lora") and args.use_lora:
        overrides.append("model.use_lora=true")
    if hasattr(args, "lora_rank") and args.lora_rank is not None:
        overrides.append(f"model.lora_rank={args.lora_rank}")
    if hasattr(args, "lora_alpha") and args.lora_alpha is not None:
        overrides.append(f"model.lora_alpha={args.lora_alpha}")
    if hasattr(args, "early_stopping_patience") and args.early_stopping_patience is not None:
        overrides.append(f"training.early_stopping_patience={args.early_stopping_patience}")
    if hasattr(args, "precision") and args.precision is not None:
        overrides.append(f"training.precision={args.precision}")
    if hasattr(args, "gradient_accumulation_steps") and args.gradient_accumulation_steps is not None:
        overrides.append(f"training.gradient_accumulation_steps={args.gradient_accumulation_steps}")
    if hasattr(args, "lr_scheduler") and args.lr_scheduler is not None:
        overrides.append(f"training.lr_scheduler_type={args.lr_scheduler}")
    if hasattr(args, "augment_rc") and args.augment_rc:
        overrides.append("training.augment_rc=true")
    if hasattr(args, "seed") and args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if hasattr(args, "device") and args.device is not None:
        overrides.append(f"device={args.device}")

    config_path = getattr(args, "config", None)
    return load_config(config_path=config_path, overrides=overrides if overrides else None)


# ─── Subcommand handlers ────────────────────────────────────────────────


def cmd_prepare_data(args):
    """Preprocess FASTA files into k-fold CSV splits."""
    from bertnup.data.preparation import k_fold_split

    for data_path in args.data_paths:
        k_fold_split(
            data_path=data_path,
            save_dir=args.save_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )


def cmd_train(args):
    """Fine-tune a model on a single train/val/test split."""
    _suppress_logging()
    from bertnup.training.trainer import run_training

    config = _build_config(args)
    run_training(config, data_dir=args.data_dir)


def cmd_evaluate(args):
    """Evaluate a saved checkpoint on a test set."""
    _suppress_logging()
    from bertnup.training.trainer import run_evaluation

    config = _build_config(args)
    run_evaluation(config, checkpoint_path=args.checkpoint, test_path=args.test_csv)


def cmd_cross_validate(args):
    """Run k-fold cross-validation."""
    _suppress_logging()
    from bertnup.training.trainer import run_kfold_cv

    config = _build_config(args)
    if args.result_dir:
        config.output.result_dir = args.result_dir
    if args.all_data_dir:
        config.data.all_data_dir = args.all_data_dir

    run_kfold_cv(
        config,
        data_name=args.data_name,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
    )


def cmd_visualize_attention(args):
    """Extract and visualize attention scores."""
    _suppress_logging()
    import numpy as np
    import torch
    from pytorch_lightning import Trainer
    from transformers import AutoTokenizer

    from bertnup.data.datasets import Dnabert1Dataset
    from bertnup.data.sequences import DNASequence
    from bertnup.models.attention import (
        BertNupAttention,
        export_bert_weights,
        process_attention_score,
    )
    from bertnup.visualization.attention_viz import (
        plot_average_attention_by_position,
        visualize_dataset_attention,
        visualize_sequence_attention,
        visualize_token2token_scores,
    )

    config = _build_config(args)
    kmer = config.model.kmer or 3
    model_name = config.model.name

    # Export weights for attention extraction
    saved_model = export_bert_weights(args.checkpoint, model_name, "pretrained_bert_export")

    # Load test data if provided
    if args.data_path:
        test_set = Dnabert1Dataset(args.data_path, kmer)
        trainer = Trainer(accelerator="auto", devices=1)

        # Single sequence visualization
        if args.sequence_idx is not None:
            seq_idx = args.sequence_idx
            print("Label:", test_set.data.label[seq_idx])
            attn_model = BertNupAttention(saved_model)
            attn_scores = attn_model(
                test_set[seq_idx]["input_ids"].view(1, -1),
                test_set[seq_idx]["attention_mask"].view(1, -1),
            ).detach().numpy().reshape(1, -1)
            visualize_sequence_attention(attn_scores, test_set.data.sequence[seq_idx])

        # Aggregate visualization
        if args.aggregate:
            from torch.utils.data import DataLoader

            attn_model = BertNupAttention(saved_model)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
            attn_outputs = trainer.predict(model=attn_model, dataloaders=test_loader)
            all_attn = torch.cat(attn_outputs, dim=0)

            processed = []
            for score in all_attn:
                processed.append(process_attention_score(score, kmer=kmer))
            final_scores = np.concatenate(processed, axis=0)

            pos_attn = final_scores[test_set.data.label == 1, :]
            neg_attn = final_scores[test_set.data.label == 0, :]

            visualize_dataset_attention(pos_attn, neg_attn)
            plot_average_attention_by_position(pos_attn, neg_attn)

    # Per-head visualization
    if args.layer is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if args.data_path and args.sequence_idx is not None:
            from bertnup.models.attention import format_attention
            from transformers import BertForSequenceClassification

            seq_idx = args.sequence_idx
            bert_model = BertForSequenceClassification.from_pretrained(
                saved_model, local_files_only=True, output_attentions=True
            )
            kmer_seq = DNASequence(test_set.data.sequence[seq_idx]).to_kmer_sequence(kmer)
            inputs = tokenizer.encode(str(kmer_seq), return_tensors="pt")
            attention = bert_model(inputs)[-1]
            output_attentions = torch.stack(attention)
            xticks = ["CLS"] + list(test_set.data.sequence[seq_idx].upper()) + ["SEP"]
            visualize_token2token_scores(
                output_attentions[args.layer].squeeze().detach().cpu().numpy(),
                head=args.head,
                xticks=xticks,
            )


# ─── Argument parsers ────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared across multiple subcommands."""
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "gpu"], help="Device")
    parser.add_argument("--model-name", type=str, default=None, help="HuggingFace model name or path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")


def _add_training_args(parser: argparse.ArgumentParser):
    """Add training-related arguments."""
    parser.add_argument("--kmer", type=int, default=None, help="K-mer size for DNABERT-1")
    parser.add_argument("--fixed-length", type=int, default=None, help="Max token length for DNABERT-2")
    parser.add_argument("--pooling", type=str, default=None, choices=["mean", "max", "attention"], help="Pooling strategy (DNABERT-2)")
    parser.add_argument("--head-type", type=str, default=None, choices=["single", "enhanced"], help="Classification head type")
    parser.add_argument("--use-lora", action="store_true", default=False, help="Enable LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-train", type=int, default=None)
    parser.add_argument("--batch-size-test", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--val-check-interval", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--reinit-layers", type=int, default=None, help="Reinitialize last N BERT layers")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["32", "16-mixed", "bf16"], help="Training precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr-scheduler", type=str, default=None, choices=["linear", "cosine"], help="LR scheduler type")
    parser.add_argument("--augment-rc", action="store_true", default=False, help="Augment with reverse complement")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bertnup",
        description="BertNup: Transformer-based nucleosome positioning prediction",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # prepare_data
    p = subparsers.add_parser("prepare_data", help="Preprocess FASTA files into k-fold CSV splits")
    p.add_argument("data_paths", nargs="+", help="One or more FASTA file paths")
    p.add_argument("--save-dir", type=str, default="Data/Stratified_K_fold_data")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--random-state", type=int, default=1)

    # train
    p = subparsers.add_parser("train", help="Fine-tune a model on a single split")
    p.add_argument("data_dir", help="Path to split directory (containing train.csv, val.csv, test.csv)")
    _add_common_args(p)
    _add_training_args(p)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a test set")
    p.add_argument("checkpoint", help="Path to .ckpt file")
    p.add_argument("test_csv", help="Path to test CSV")
    _add_common_args(p)
    p.add_argument("--kmer", type=int, default=None)
    p.add_argument("--fixed-length", type=int, default=None)
    p.add_argument("--batch-size-test", type=int, default=None)

    # cross_validate
    p = subparsers.add_parser("cross_validate", help="Run k-fold cross-validation")
    p.add_argument("data_name", help="Dataset name (e.g., HS_LC)")
    _add_common_args(p)
    _add_training_args(p)
    p.add_argument("--result-dir", type=str, default=None, help="Output directory for results")
    p.add_argument("--all-data-dir", type=str, default=None, help="Parent directory of split data")
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--end-fold", type=int, default=10)

    # visualize_attention
    p = subparsers.add_parser("visualize_attention", help="Extract and visualize attention scores")
    p.add_argument("checkpoint", help="Path to fine-tuned .ckpt file")
    _add_common_args(p)
    p.add_argument("--kmer", type=int, default=None)
    p.add_argument("--data-path", type=str, default=None, help="Test CSV for visualization")
    p.add_argument("--sequence-idx", type=int, default=None, help="Single sequence to visualize")
    p.add_argument("--layer", type=int, default=None, help="Attention layer")
    p.add_argument("--head", type=int, default=None, help="Attention head (None = all)")
    p.add_argument("--aggregate", action="store_true", help="Visualize aggregate attention across test set")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "prepare_data": cmd_prepare_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "cross_validate": cmd_cross_validate,
        "visualize_attention": cmd_visualize_attention,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
