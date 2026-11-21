"""Common evaluation metrics for intrusion detection experiments."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def evaluate_scores(y_true, scores, threshold):
    """Evaluate anomaly scores against binary attack labels.

    Labels are used only for evaluation, never for training.
    Higher anomaly scores indicate more suspicious traffic.
    """

    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)

    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    if len(y) != len(s):
        raise ValueError("Labels and scores must have equal lengths")

    if not np.isfinite(s).all():
        raise ValueError("Anomaly scores must be finite")

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    predicted = s > threshold

    tp = int(np.sum(predicted & y))
    tn = int(np.sum(~predicted & ~y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))

    n_attacks = tp + fn
    n_benign = tn + fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_attacks if n_attacks else float("nan")

    if n_attacks:
        f1 = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn)
            else 0.0
        )
    else:
        f1 = float("nan")

    false_alarm_rate = (
        fp / n_benign if n_benign else float("nan")
    )

    # AUROC and AUCPR require both classes to be present.
    if n_attacks > 0 and n_benign > 0:
        auroc = float(roc_auc_score(y, s))
        aucpr = float(average_precision_score(y, s))
    else:
        auroc = float("nan")
        aucpr = float("nan")

    return {
        "flows": len(y),
        "attacks": n_attacks,
        "attack_share": n_attacks / len(y),
        "auroc": auroc,
        "aucpr": aucpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_rate": recall,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
