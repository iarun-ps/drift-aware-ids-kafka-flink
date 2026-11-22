"""Online ADWIN drift detection using minute-level anomaly scores."""

import numpy as np
from river import drift


class MedianRatioADWIN:
    def __init__(self, threshold, delta=0.002, clock=8, min_flows=20):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Threshold must be positive and finite")
        if min_flows < 1:
            raise ValueError("min_flows must be positive")

        self.threshold = float(threshold)
        self.min_flows = min_flows
        self.detector = drift.ADWIN(delta=delta, clock=clock)
        self.minutes_processed = 0
        self.drift_count = 0

    def update_minute(self, scores):
        """Process one completed minute of scores; no labels are used."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        if len(scores) < self.min_flows:
            return None

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite anomaly scores")

        ratio = float(
            np.clip(np.median(scores) / self.threshold, 0, 1)
        )

        self.detector.update(ratio)
        self.minutes_processed += 1

        detected = bool(self.detector.drift_detected)

        if detected:
            self.drift_count += 1

        return {
            "median_ratio": ratio,
            "drift_detected": detected,
        }
