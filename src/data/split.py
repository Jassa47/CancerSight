"""Lesion-based stratified train/val/test split for HAM10000."""

import pandas as pd
from sklearn.model_selection import train_test_split


def get_splits(metadata_path: str, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
    """
    Split HAM10000 at the lesion level to prevent data leakage.
    Returns three DataFrames: train, val, test.
    """
    df = pd.read_csv(metadata_path)

    # One row per lesion for splitting
    lesions = df.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]]

    train_val_lesions, test_lesions = train_test_split(
        lesions, test_size=test_size, stratify=lesions["dx"], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train_lesions, val_lesions = train_test_split(
        train_val_lesions, test_size=relative_val, stratify=train_val_lesions["dx"], random_state=random_state
    )

    train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].reset_index(drop=True)

    return train_df, val_df, test_df
