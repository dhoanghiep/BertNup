"""Pooling strategies for transforming sequence representations."""

from __future__ import annotations

import torch
from torch import nn


class MeanPooling(nn.Module):
    """Average pooling over the sequence dimension, respecting attention mask."""

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden_state.mean(dim=1)


class MaxPooling(nn.Module):
    """Max pooling over the sequence dimension, respecting attention mask."""

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden_state = hidden_state.masked_fill(mask == 0, -1e9)
        return hidden_state.max(dim=1)[0]


class AttentionPooling(nn.Module):
    """Learned attention-based pooling over the sequence dimension.

    Computes attention weights via a small feed-forward network,
    then returns the weighted sum of hidden states. This allows the
    model to focus on positions most relevant for classification.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.attention(hidden_state).squeeze(-1)
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(weights, dim=-1)
        return (hidden_state * weights.unsqueeze(-1)).sum(dim=1)


def create_pooling(pooling_type: str = "mean", hidden_size: int = 768) -> nn.Module:
    """Factory function to create pooling modules."""
    if pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "max":
        return MaxPooling()
    elif pooling_type == "attention":
        return AttentionPooling(hidden_size)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
