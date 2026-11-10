"""Cleaning and scaling rules for CIC-IDS2017 flow features (shared by offline and streaming code)."""
import json

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

# dropped on purpose: the raw port number is a shortcut to the attack type
DROP_COLS = ["Destination Port"]

# -1 means "no TCP window seen", so keep it
KEEP_NEGATIVE = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

CLIP_Z = 10.0


def feature_columns(df: pd.DataFrame) -> list:
    skip = set(ID_COLS) | set(CONSTANT_COLS) | set(DROP_COLS)
    return [c for c in df.columns if c not in skip]


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


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """log1p on every feature; the TCP window columns are shifted by +1 first."""
    L = X.astype("float64").copy()
    for c in L.columns:
        L[c] = np.log1p(L[c] + (1.0 if c in KEEP_NEGATIVE else 0.0))
    return L


def fit_spec(X_clean: pd.DataFrame, medians: pd.Series) -> dict:
    """Learn the scaling parameters (call on training data only)."""
    L = log_transform(X_clean)
    return {
        "columns": list(L.columns),
        "medians": [float(medians[c]) for c in L.columns],
        "mean": [float(v) for v in L.mean()],
        "std": [float(v) for v in L.std().clip(lower=1e-6)],
        "clip_z": CLIP_Z,
    }


def transform(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply cleaning + log + standardization using a fitted spec."""
    cols = spec["columns"]
    medians = pd.Series(spec["medians"], index=cols)
    X, _ = clean_features(df, medians=medians)
    X = X[cols]
    L = log_transform(X)
    Z = (L - np.array(spec["mean"])) / np.array(spec["std"])
    return Z.clip(-spec["clip_z"], spec["clip_z"]).astype("float32")


def save_spec(spec: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def load_spec(path) -> dict:
    with open(path) as f:
        return json.load(f)


def time_split(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15)):
    """Split in time order into (train, validation, holdout)."""
    df = df.sort_values(["Timestamp", "row_order"]).reset_index(drop=True)
    n = len(df)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
