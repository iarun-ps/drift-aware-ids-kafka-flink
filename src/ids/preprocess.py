"""Cleaning rules for CIC-IDS2017 flow features (shared by offline and streaming code)."""
import numpy as np
import pandas as pd

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP",
           "Timestamp", "Label", "row_order", "source_file"]

# constant on Monday (the training day), so unusable as features
CONSTANT_COLS = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# -1 means "no TCP window seen", so keep it
KEEP_NEGATIVE = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ID_COLS and c not in CONSTANT_COLS]


def clean_features(df: pd.DataFrame, medians: pd.Series = None):
    """Return (float32 feature table, medians used for filling)."""
    X = df[feature_columns(df)].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    to_clip = [c for c in X.columns if c not in KEEP_NEGATIVE]
    X[to_clip] = X[to_clip].clip(lower=0)
    if medians is None:
        medians = X.median()
    X = X.fillna(medians)
    return X.astype("float32"), medians
