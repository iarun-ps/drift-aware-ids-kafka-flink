"""Half-Space Trees baseline for streaming anomaly detection."""

import numpy as np
from river import anomaly


class HalfSpaceTreesBaseline:
    def __init__(
        self,
        n_features=67,
        n_trees=10,
        height=8,
        window_size=250,
        seed=42,
    ):
        self.n_features = n_features
        self.model = anomaly.HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            limits={
                f"f{i}": (-10.0, 10.0)
                for i in range(n_features)
            },
            seed=seed,
        )
        self.updates = 0

    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Invalid input feature dimensions")

        if not np.isfinite(X).all():
            raise ValueError("Input contains non-finite values")

        return X

    def _record(self, row):
        return {
            f"f{i}": float(value)
            for i, value in enumerate(row)
        }

    def initialize(self, X):
        """Learn historical benign training data."""
        X = self._prepare(X)

        for row in X:
            self.model.learn_one(self._record(row))
            self.updates += 1

    def score_batch(self, X):
        """Generate anomaly scores without updating the model."""
        X = self._prepare(X)

        return np.array([
            self.model.score_one(self._record(row))
            for row in X
        ], dtype=np.float64)

    def update_batch(self, X, scores):
        """Update the model after scoring each batch."""
        X = self._prepare(X)

        if len(X) != len(scores):
            raise ValueError("Scores and observations must match")

        for row in X:
            self.model.learn_one(self._record(row))
            self.updates += 1
