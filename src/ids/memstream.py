"""MemStream-style baseline using a frozen encoder and FIFO memory."""

import numpy as np
import torch


class MemStreamBaseline:
    def __init__(
        self,
        autoencoder,
        memory_size=256,
        k_neighbors=3,
        decay=0.9,
        beta=1.0,
    ):
        if memory_size < 1 or not 1 <= k_neighbors <= memory_size:
            raise ValueError("Invalid memory size or neighbor count")
        if not 0 < decay <= 1:
            raise ValueError("decay must be between 0 and 1")
        if not np.isfinite(beta):
            raise ValueError("beta must be finite")

        self.encoder = autoencoder.encoder.eval()
        self.memory_size = memory_size
        self.k = k_neighbors
        self.decay = decay
        self.beta = beta
        self.memory = None
        self.pointer = 0
        self.updates = 0

    @torch.no_grad()
    def encode(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Expected a 2D feature matrix")

        tensor = torch.from_numpy(X)
        return self.encoder(tensor).cpu().numpy()

    def initialize(self, X):
        """Initialize memory using historical benign training data."""
        Z = self.encode(X)

        if len(Z) < self.k:
            raise ValueError("Insufficient initial memory samples")

        self.memory = Z[-self.memory_size:].copy()
        self.pointer = 0
        self.updates = 0

    def score_batch(self, X):
        """Score observations against the current memory."""
        if self.memory is None:
            raise RuntimeError("Memory is not initialized")

        Z = self.encode(X)
        if len(Z) == 0:
            return np.empty(0, dtype=np.float64)

        distances = np.abs(
            Z[:, None, :] - self.memory[None, :, :]
        ).sum(axis=2)

        nearest = np.partition(
            distances,
            self.k - 1,
            axis=1,
        )[:, :self.k]

        nearest.sort(axis=1)

        weights = self.decay ** np.arange(self.k)

        return (
            nearest @ weights / weights.sum()
        ).astype(np.float64)

    def update_batch(self, X, scores):
        """Update FIFO memory only for sufficiently normal observations."""
        if self.memory is None:
            raise RuntimeError("Memory is not initialized")

        Z = self.encode(X)
        scores = np.asarray(scores).reshape(-1)

        if len(Z) != len(scores):
            raise ValueError("Scores and observations must match")

        for z, anomaly_score in zip(Z, scores):
            if anomaly_score < self.beta:
                self.memory[self.pointer] = z
                self.pointer = (
                    self.pointer + 1
                ) % len(self.memory)
                self.updates += 1
