"""Dataset classes for DNABERT-1 (k-mer) and DNABERT-2 (raw sequence)."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from bertnup.data.sequences import DNASequence


class Dnabert1Dataset(Dataset):
    """Dataset for DNABERT-1 models using k-mer tokenization.

    Converts raw DNA sequences to overlapping k-mers and tokenizes with
    armheb/DNA_bert_{kmer} tokenizers.
    """

    def __init__(self, data_path: str, kmer: int, max_token_length: int | None = None):
        dataframe = pd.read_csv(data_path)
        self.kmer = kmer
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(f"armheb/DNA_bert_{self.kmer}")
        self.sequence_length = len(self.data["sequence"].iloc[0])
        if max_token_length is None:
            self.max_token_length = self.sequence_length - kmer + 1 + 2  # +2 for CLS/SEP

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        kmer_seq = DNASequence(self.data["sequence"].str.upper().iloc[index]).to_kmer_sequence(
            self.kmer
        )
        encoding = self.tokenizer(
            str(kmer_seq),
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
        )
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(self.data["label"].iloc[index])
        return item

    def __len__(self) -> int:
        return self.len


class Dnabert2Dataset(Dataset):
    """Dataset for DNABERT-2 models using raw sequence tokenization.

    Tokenizes uppercase DNA sequences directly using the provided tokenizer
    with fixed-length padding/truncation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        fixed_length: int = 70,
    ):
        dataframe = pd.read_csv(data_path)
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.fixed_length = fixed_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.data["sequence"].iloc[index].upper(),
            padding="max_length",
            truncation=True,
            max_length=self.fixed_length,
        )
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(self.data["label"].iloc[index])
        return item

    def __len__(self) -> int:
        return self.len


def create_dataset(
    model_type: str,
    data_path: str,
    kmer: int | None = None,
    tokenizer: AutoTokenizer | None = None,
    fixed_length: int = 70,
    max_token_length: int | None = None,
) -> Dataset:
    """Factory function to create the appropriate dataset class."""
    if model_type == "dnabert1":
        if kmer is None:
            raise ValueError("kmer is required for dnabert1 model type")
        return Dnabert1Dataset(data_path, kmer, max_token_length)
    elif model_type == "dnabert2":
        if tokenizer is None:
            raise ValueError("tokenizer is required for dnabert2 model type")
        return Dnabert2Dataset(data_path, tokenizer, fixed_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
