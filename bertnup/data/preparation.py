"""Data preparation: FASTA parsing and k-fold splitting."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import KFold


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
