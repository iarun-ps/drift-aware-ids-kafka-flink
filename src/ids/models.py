"""Autoencoder anomaly detector (PyTorch). Score = reconstruction error per flow."""
import numpy as np
import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit(model, X, epochs=10, batch_size=512, lr=1e-3, seed=0):
    """Train on (benign) data X; returns the average loss per epoch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        total = 0.0
        for i in range(0, len(data), batch_size):
            xb = data[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / len(data))
    return history


@torch.no_grad()
def score(model, X, batch_size=8192):
    """Anomaly score per row: mean squared reconstruction error."""
    model.eval()
    X = np.asarray(X)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32)
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out)
