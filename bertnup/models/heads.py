"""Classification heads for BertNup models."""

from __future__ import annotations

import torch
from torch import nn


class SingleLayerHead(nn.Module):
    """Original single-layer classification head (backward compatible).

    Dropout → Linear(hidden_size, num_classes).
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear(x)


class EnhancedHead(nn.Module):
    """Enhanced multi-layer classification head with layer normalization.

    LayerNorm → Dropout → Linear(hidden, hidden//2) → ReLU
    → LayerNorm → Dropout → Linear(hidden//2, num_classes).

    Uses LayerNorm instead of BatchNorm to handle batch_size=1
    (last training batch can have a single sample).
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // 2
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, mid)
        self.norm2 = nn.LayerNorm(mid)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.linear1(x))
        x = self.norm2(x)
        x = self.dropout2(x)
        return self.linear2(x)


def create_head(head_type: str = "single", **kwargs) -> nn.Module:
    """Factory function to create classification heads."""
    if head_type == "single":
        return SingleLayerHead(**kwargs)
    elif head_type == "enhanced":
        return EnhancedHead(**kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
