"""DNA sequence augmentation for training data expansion."""

from __future__ import annotations

import pandas as pd

from bertnup.data.sequences import DNASequence


def augment_with_reverse_complement(df: pd.DataFrame) -> pd.DataFrame:
    """Double a DataFrame by appending reverse complement of each sequence.

    For each row, creates a new row with the reverse complement sequence
    and the same label. The reverse complement of a nucleosome-forming
    sequence is also nucleosome-forming (DNA is double-stranded).

    Args:
        df: DataFrame with 'sequence' and 'label' columns.

    Returns:
        DataFrame with original + reverse complement rows.
    """
    rc_rows = []
    for _, row in df.iterrows():
        rc_seq = str(DNASequence(row["sequence"]).reverse_complement())
        rc_rows.append({"sequence": rc_seq, "label": row["label"]})

    rc_df = pd.DataFrame(rc_rows)
    return pd.concat([df, rc_df], ignore_index=True)
