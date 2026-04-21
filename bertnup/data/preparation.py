"""Data preparation: FASTA parsing and k-fold splitting."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import KFold, train_test_split


def get_data(path_name: str) -> list[list[str]]:
    """Parse a FASTA file into [sequence, label] pairs."""
    data = []
    for fasta in SeqIO.parse(open(path_name), "fasta"):
        data.append([str(fasta.seq), fasta.id[0]])
    return data


def generate_dataframe(data: list[list[str]]) -> pd.DataFrame:
    """Convert parsed FASTA data to a DataFrame with 'sequence' and 'label' columns."""
    sequence = [item[0] for item in data]
    lab = [1 if item[1] == "n" else 0 for item in data]
    return pd.DataFrame({"sequence": sequence, "label": lab}).drop_duplicates(
        ignore_index=True
    )


def k_fold_split(
    data_path: str,
    random_state: int = 1,
    save_dir: str | None = None,
    n_splits: int = 10,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] | None:
    """Split a FASTA dataset into k folds with train/val/test sets.

    Each fold uses fold[i] as test, fold[i-1] as validation, and the rest as training.
    If save_dir is provided, writes CSVs; otherwise returns DataFrames.
    """
    data_name = data_path.split("/")[-1][:-4]
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    raw_dataframe = generate_dataframe(get_data(data_path))
    folds_indexes = [test_index for _, test_index in kf.split(raw_dataframe, raw_dataframe.label)]

    split_dataframes = []
    for k_fold_number in range(n_splits):
        test_index = folds_indexes[k_fold_number]
        val_index = folds_indexes[k_fold_number - 1]
        train_index = [
            i
            for i in range(raw_dataframe.shape[0])
            if i not in np.hstack((test_index, val_index))
        ]

        train_set = raw_dataframe.loc[train_index].reset_index(drop=True)
        val_set = raw_dataframe.loc[val_index].reset_index(drop=True)
        test_set = raw_dataframe.loc[test_index].reset_index(drop=True)

        if save_dir is not None:
            save_subdir = f"{save_dir}/{data_name}/split_{k_fold_number}/"
            os.makedirs(save_subdir, exist_ok=True)
            train_set.to_csv(save_subdir + "train.csv", index=False)
            val_set.to_csv(save_subdir + "val.csv", index=False)
            test_set.to_csv(save_subdir + "test.csv", index=False)
        else:
            split_dataframes.append((train_set, val_set, test_set))

    if save_dir is not None:
        print(f"Split {data_path} to {n_splits} groups, saved to {save_dir}/{data_name}")
    else:
        return split_dataframes


def single_split(
    data_path: str,
    test_size: float = 0.1,
    val_size: float = 0.111,
    random_state: int = 42,
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Split a FASTA dataset into train/val/test (8:1:1 ratio).

    First splits off test_size as test, then splits val_size of remaining as val.
    Default params give 80/10/10 split.
    If save_dir is provided, writes CSVs; otherwise returns DataFrames.
    """
    data_name = data_path.split("/")[-1][:-4]
    df = generate_dataframe(get_data(data_path))

    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state, stratify=train_val["label"])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if save_dir is not None:
        subdir = f"{save_dir}/{data_name}_single_split/"
        os.makedirs(subdir, exist_ok=True)
        train.to_csv(f"{subdir}train.csv", index=False)
        val.to_csv(f"{subdir}val.csv", index=False)
        test.to_csv(f"{subdir}test.csv", index=False)
        print(f"Split {data_path} to train/val/test ({len(train)}/{len(val)}/{len(test)}), saved to {subdir}")
    else:
        return train, val, test
