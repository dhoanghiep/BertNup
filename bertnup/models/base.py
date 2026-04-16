"""Base model class with shared training/validation logic for PyTorch Lightning 2.x."""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from transformers import get_linear_schedule_with_warmup

from bertnup.data.metrics import compute_all_metrics
from bertnup.models.heads import create_head


class BertNupBase(LightningModule):
    """Abstract base for BertNup models.

    Subclasses must implement `_build_backbone()` to load the pretrained
    model and `forward()` to define the forward pass through the head.
    Optionally applies LoRA to the backbone for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        num_training_steps: int = 0,
        dropout: float = 0.1,
        hidden_size: int = 768,
        lr_scheduler_type: str = "linear",
        head_type: str = "single",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = create_head(
            head_type=head_type,
            hidden_size=hidden_size,
            num_classes=2,
            dropout=dropout,
        )
        self._validation_outputs: list[dict[str, Any]] = []

    def _apply_lora(self, backbone):
        """Optionally wrap backbone with LoRA adapters via PEFT."""
        if not self.hparams.use_lora:
            return backbone
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")
        config = LoraConfig(
            r=self.hparams.lora_rank,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none",
        )
        return get_peft_model(backbone, config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        raise NotImplementedError("Subclasses must implement forward()")

    def _compute_loss_and_probas(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply softmax and optionally compute cross-entropy loss."""
        probas = torch.softmax(logits, dim=1)[:, 1]
        loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, probas

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        _, probas = self(input_ids=ids, attention_mask=attention_mask, labels=labels)
        return probas

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids=ids, attention_mask=attention_mask, labels=labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, probas = self(input_ids=ids, attention_mask=attention_mask, labels=labels)
        self._validation_outputs.append({"loss": loss, "probas": probas, "labels": labels})

    def on_validation_epoch_end(self) -> None:
        outputs = self._validation_outputs
        if not outputs:
            return
        loss = sum(o["loss"] for o in outputs) / len(outputs)
        probas = torch.hstack([o["probas"] for o in outputs]).cpu().numpy()
        labels = torch.hstack([o["labels"] for o in outputs]).cpu().numpy()
        if 0 not in labels or 1 not in labels:
            auc = 0
        else:
            _, _, _, _, _, auc = compute_all_metrics(probas, labels, verbose=0)
        self.log("val_loss", loss)
        self._validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_type = getattr(self.hparams, "lr_scheduler_type", "linear")
        if scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                self.hparams.warmup_steps,
                self.hparams.num_training_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                self.hparams.warmup_steps,
                self.hparams.num_training_steps,
            )
        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_cfg]
