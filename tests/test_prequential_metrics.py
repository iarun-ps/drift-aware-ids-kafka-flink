"""Automated tests for prequential evaluation and metrics."""

import numpy as np
import pandas as pd

from ids.metrics import evaluate_scores
from ids.prequential import prequential_batches


def test_prequential_order_and_updates():
    spec = {
        "columns": ["Flow Duration"],
        "medians": [0.0],
        "mean": [0.0],
        "std": [1.0],
        "clip_z": 10.0,
    }

    df = pd.DataFrame({
        "Flow Duration": [10.0, 20.0, 30.0],
        "Timestamp": pd.to_datetime([
            "2017-07-03 10:02:00",
            "2017-07-03 10:00:00",
            "2017-07-03 10:01:00",
        ]),
        "row_order": [2, 0, 1],
        "Label": ["BENIGN", "Attack", "BENIGN"],
    })

    state = {"version": 0}
    events = []

    def scorer(X):
        events.append("score")
        return np.full(len(X), state["version"])

    def updater(X, scores):
        events.append("update")
        state["version"] += 1

    batches = list(
        prequential_batches(
            df,
            spec,
            scorer,
            updater,
            batch_size=2,
        )
    )

    out = pd.concat(batches, ignore_index=True)

    assert out["Timestamp"].is_monotonic_increasing
    assert out["score"].tolist() == [0.0, 0.0, 1.0]
    assert out["is_attack"].tolist() == [True, False, False]
    assert events == ["score", "update", "score", "update"]


def test_evaluation_metrics():
    results = evaluate_scores(
        [False, False, True, True],
        [0.1, 0.8, 0.9, 0.4],
        threshold=0.5,
    )

    assert results["tp"] == 1
    assert results["tn"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 1

    assert np.isclose(results["precision"], 0.5)
    assert np.isclose(results["recall"], 0.5)
    assert np.isclose(results["f1"], 0.5)
    assert np.isclose(results["auroc"], 0.75)
    assert np.isclose(results["aucpr"], 5 / 6)


def test_benign_only_evaluation():
    results = evaluate_scores(
        [False, False],
        [0.1, 0.2],
        threshold=0.5,
    )

    assert np.isnan(results["auroc"])
    assert np.isnan(results["aucpr"])
    assert np.isnan(results["f1"])
    assert results["false_alarm_rate"] == 0.0
