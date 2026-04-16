"""Reproducible seed setting for all random sources."""

from __future__ import annotations

import random


def set_seed(seed: int = 0) -> None:
    """Set all seeds for reproducible results."""
    import numpy as np
    import torch
    from pytorch_lightning import seed_everything

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)
