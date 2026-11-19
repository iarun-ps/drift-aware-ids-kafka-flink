"""Prequential evaluation: score traffic before updating the model."""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from ids.preprocess import transform


def prequential_batches(
    df: pd.DataFrame,
    spec: dict,
    score_batch: Callable[[np.ndarray], np.ndarray],
    update_batch: Callable[[np.ndarray, np.ndarray], None] | None = None,
    *,
    batch_size: int = 1024,
) -> Iterator[pd.DataFrame]:
    """
    Evaluate chronological batches using test-then-train.

    score_batch(X): returns one anomaly score per flow.
    update_batch(X, scores): optional unsupervised model update.

    Labels are used only for evaluation and are never passed
    to the model or its update function.

    Note: Updates occur after scoring each entire batch.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    required = {"Timestamp", "row_order", "Label"}

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    ordered = df.sort_values(
        ["Timestamp", "row_order"],
        kind="mergesort"
    ).reset_index(drop=True)

    if ordered["Timestamp"].isna().any():
        raise ValueError("Dataset contains missing timestamps")

    for start in range(0, len(ordered), batch_size):
        batch = ordered.iloc[start:start + batch_size]

        X = transform(batch, spec).to_numpy(
            dtype=np.float32,
            copy=False
        )

        # TEST: Generate scores using the current model.
        scores = np.asarray(
            score_batch(X),
            dtype=np.float64
        ).reshape(-1)

        if len(scores) != len(batch):
            raise ValueError("Incorrect number of anomaly scores")

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite anomaly scores detected")

        # TRAIN: Update only after scoring this batch.
        # True attack labels are never provided to the updater.
        if update_batch is not None:
            update_batch(X, scores.copy())

        # Return predictions and labels for evaluation.
        yield pd.DataFrame({
            "Timestamp": batch["Timestamp"].to_numpy(),
            "row_order": batch["row_order"].to_numpy(),
            "score": scores,
            "is_attack": (
                batch["Label"].astype(str).str.strip()
                != "BENIGN"
            ).to_numpy()
        })
