"""Conservative, label-free candidate buffer for model adaptation."""

from collections import deque

import numpy as np


class AdaptationBuffer:
    def __init__(self, cutoff, capacity=4096, min_samples=256):
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("Invalid cutoff")
        if capacity < 1 or not 1 <= min_samples <= capacity:
            raise ValueError("Invalid buffer configuration")

        self.cutoff = float(cutoff)
        self.capacity = capacity
        self.min_samples = min_samples
        self.memory = deque(maxlen=capacity)
        self.n_features = None
        self.total_accepted = 0

    def add_batch(self, X, scores):
        """Accept low-scoring observations without using labels."""
        X = np.asarray(X, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        if X.ndim != 2 or len(X) != len(scores):
            raise ValueError("Input and score dimensions do not match")

        if not np.isfinite(X).all() or not np.isfinite(scores).all():
            raise ValueError("Non-finite values detected")

        if self.n_features is None:
            self.n_features = X.shape[1]
        elif X.shape[1] != self.n_features:
            raise ValueError("Feature count changed")

        accepted = X[scores <= self.cutoff]

        for row in accepted:
            self.memory.append(row.copy())

        self.total_accepted += len(accepted)
        return len(accepted)

    def ready(self):
        """Whether enough candidate records are available."""
        return len(self.memory) >= self.min_samples

    def snapshot(self):
        """Return selected records without changing the buffer."""
        if not self.ready():
            return None

        return np.stack(self.memory).astype(np.float32)

    def clear(self):
        self.memory.clear()
