"""Coordinate ADWIN detection and deferred adaptation candidates."""

import numpy as np

from ids.adaptation_buffer import AdaptationBuffer
from ids.adwin_monitor import MedianRatioADWIN


class AdaptiveController:
    def __init__(
        self,
        threshold,
        cutoff,
        delta=0.002,
        clock=8,
        min_flows=20,
        max_alert_rate=0.05,
        capacity=4096,
        min_samples=256,
    ):
        if not 0 <= max_alert_rate <= 1:
            raise ValueError("Invalid maximum alert rate")

        self.threshold = float(threshold)
        self.max_alert_rate = float(max_alert_rate)
        self.min_flows = min_flows
        self.pending_drift = False

        self.monitor = MedianRatioADWIN(
            threshold=threshold,
            delta=delta,
            clock=clock,
            min_flows=min_flows,
        )

        self.buffer = AdaptationBuffer(
            cutoff=cutoff,
            capacity=capacity,
            min_samples=min_samples,
        )

    def observe_minute(self, X, scores):
        """Score information is observed after a minute completes."""
        X = np.asarray(X, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        if X.ndim != 2 or len(X) != len(scores):
            raise ValueError("Feature and score dimensions do not match")

        if not np.isfinite(X).all() or not np.isfinite(scores).all():
            raise ValueError("Non-finite values detected")

        if len(scores) < self.min_flows:
            return None

        result = self.monitor.update_minute(scores)
        detected = result["drift_detected"]

        alert_rate = float(np.mean(scores > self.threshold))
        suspicious = alert_rate > self.max_alert_rate

        accepted = 0

        if detected:
            # Start a new request; discard pre-drift candidates.
            self.pending_drift = True
            self.buffer.clear()

        elif suspicious:
            # Keep the request pending, but discard unsafe candidates.
            self.buffer.clear()

        elif self.pending_drift:
            # Collect only from later eligible minutes.
            accepted = self.buffer.add_batch(X, scores)

        candidate_ready = bool(
            self.pending_drift
            and not suspicious
            and not detected
            and self.buffer.ready()
        )

        return {
            "median_ratio": result["median_ratio"],
            "alert_rate": alert_rate,
            "drift_detected": detected,
            "pending_drift": self.pending_drift,
            "suspicious_minute": suspicious,
            "accepted": int(accepted),
            "buffer_size": len(self.buffer.memory),
            "candidate_ready": candidate_ready,
        }

    def mark_adapted(self):
        """Call only after a model update completes successfully."""
        if not self.pending_drift or not self.buffer.ready():
            raise RuntimeError("No ready adaptation request")

        self.buffer.clear()
        self.pending_drift = False
