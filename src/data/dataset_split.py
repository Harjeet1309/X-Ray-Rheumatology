"""Patient-level dataset splitting utilities (PRD Section 5.4).

Ensures that radiographs from the same patient never appear across multiple splits
(train, validation, test) while preserving official RAM-H1200 splits when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def create_patient_level_splits(
    metadata_df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "is_ra",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe at the patient level with stratification on `label_col`."""
    if "split" in metadata_df.columns and set(metadata_df["split"].unique()).issuperset(
        {"train", "val", "test"}
    ):
        train_df = metadata_df[metadata_df["split"] == "train"].copy()
        val_df = metadata_df[metadata_df["split"] == "val"].copy()
        test_df = metadata_df[metadata_df["split"] == "test"].copy()
        verify_zero_patient_leakage(train_df, val_df, test_df, patient_col=patient_col)
        return train_df, val_df, test_df

    rng = np.random.default_rng(seed)
    patient_groups = (
        metadata_df.groupby(patient_col)[label_col].agg(lambda x: int(round(x.mean()))).reset_index()
    )

    train_patients, val_patients, test_patients = [], [], []
    for label_val in sorted(patient_groups[label_col].unique()):
        pids = patient_groups[patient_groups[label_col] == label_val][patient_col].tolist()
        rng.shuffle(pids)
        n = len(pids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train_patients.extend(pids[:n_train])
        val_patients.extend(pids[n_train : n_train + n_val])
        test_patients.extend(pids[n_train + n_val :])

    train_df = metadata_df[metadata_df[patient_col].isin(set(train_patients))].copy()
    val_df = metadata_df[metadata_df[patient_col].isin(set(val_patients))].copy()
    test_df = metadata_df[metadata_df[patient_col].isin(set(test_patients))].copy()

    verify_zero_patient_leakage(train_df, val_df, test_df, patient_col=patient_col)
    return train_df, val_df, test_df


def verify_zero_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> Dict[str, int]:
    """Raise ValueError if any patient appears in more than one split."""
    p_train = set(train_df[patient_col].unique())
    p_val = set(val_df[patient_col].unique())
    p_test = set(test_df[patient_col].unique())

    tv = p_train & p_val
    tt = p_train & p_test
    vt = p_val & p_test
    if tv or tt or vt:
        raise ValueError(
            f"Patient leakage detected across splits! train&val={tv}, train&test={tt}, val&test={vt}"
        )
    return {
        "train_patients": len(p_train),
        "val_patients": len(p_val),
        "test_patients": len(p_test),
    }
