"""Attention extraction model and utilities for BertNup."""

from __future__ import annotations

import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, BertForSequenceClassification

from bertnup.data.sequences import DNASequence


class BertNupAttention(LightningModule):
    """Extract attention scores from a fine-tuned BertNup model.

    Wraps BertForSequenceClassification with output_attentions=True to
    extract attention weights from specified layers and heads.
    """

    def __init__(
        self,
        saved_model_path: str,
        start_attn_layer: int | None = None,
        end_attn_layer: int | None = None,
        head: int | None = None,
    ):
        super().__init__()
        self.dnabert = BertForSequenceClassification.from_pretrained(
            saved_model_path, local_files_only=True, output_attentions=True
        )
        if start_attn_layer is None:
            self.start_attn_layer = 11
            self.end_attn_layer = 12
        elif end_attn_layer is None:
            self.end_attn_layer = self.start_attn_layer + 1
        else:
            self.start_attn_layer = start_attn_layer
            self.end_attn_layer = end_attn_layer
        self.head = head
        self.max_length = 147

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        attention = self.dnabert(input_ids)[-1]
        attn = format_attention(attention)
        if self.head is None:
            item_attn_score = attn[self.start_attn_layer : self.end_attn_layer, :, 0:1, :].sum(
                dim=(0, 1, 2)
            )
        else:
            item_attn_score = attn[
                self.start_attn_layer : self.end_attn_layer,
                self.head : self.head + 1,
                :,
                :,
            ].sum(dim=(0, 1, 2))
        return item_attn_score

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attn = format_attention(self.dnabert(input_ids)[-1])
        if self.head is None:
            batch_attn_score = attn[
                :, self.start_attn_layer : self.end_attn_layer, :, 0:1, 1 : self.max_length - 1
            ].sum(dim=(1, 2, 3))
        else:
            batch_attn_score = attn[
                :,
                self.start_attn_layer : self.end_attn_layer,
                self.head : self.head + 1,
                :,
                1 : self.max_length - 1,
            ].sum(dim=(1, 2, 3))
        return batch_attn_score


def format_attention(attention: tuple[torch.Tensor]) -> torch.Tensor:
    """Reshape attention tuples to (num_layers, batch_size, num_heads, seq_len, seq_len)."""
    squeezed = [layer_attention.squeeze(0) for layer_attention in attention]
    return torch.stack(squeezed).transpose(0, 1)


def process_attention_score(attention: np.ndarray, kmer: int) -> np.ndarray:
    """Convert k-mer attention scores back to per-base scores."""
    attention_scores = np.array(attention).reshape(np.array(attention).shape[0], 1)
    real_scores = get_real_score(attention_scores, kmer, "mean")
    real_scores = real_scores / np.linalg.norm(real_scores, ord=1)
    return real_scores.reshape(1, real_scores.shape[0])


def get_real_score(attention_scores: np.ndarray, kmer: int, metric: str) -> np.ndarray:
    """Map k-mer attention scores back to base-level resolution."""
    real_scores = np.zeros(len(attention_scores) + kmer - 1)
    if metric == "mean":
        counts = np.zeros_like(real_scores)
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score
        real_scores = real_scores / counts
    return real_scores


def export_bert_weights(
    checkpoint_path: str,
    model_name: str,
    output_dir: str,
) -> str:
    """Load a BertNupV1 checkpoint and export the underlying BERT weights.

    This is needed for attention visualization, which requires
    BertForSequenceClassification with output_attentions=True.

    Returns the output_dir path where weights were saved.
    """
    from bertnup.models.dnabert1 import BertNupV1

    import os
    os.makedirs(output_dir, exist_ok=True)

    model = BertNupV1.load_from_checkpoint(checkpoint_path, pretrained_model_name=model_name)
    model.dnabert.save_pretrained(output_dir)
    return output_dir
