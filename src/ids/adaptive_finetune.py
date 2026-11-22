"""Conservative fine-tuning of a candidate autoencoder."""

import copy

import numpy as np
import torch

from ids.models import score


def fine_tune_candidate(
    model,
    X,
    epochs=1,
    lr=1e-5,
    batch_size=256,
    max_weight_change=0.01,
):
    """Return an updated model copy; never modify the original."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2 or len(X) == 0:
        raise ValueError("Expected a nonempty feature matrix")
    if not np.isfinite(X).all():
        raise ValueError("Training data contains non-finite values")
    if epochs < 1 or batch_size < 1 or lr <= 0 or max_weight_change <= 0:
        raise ValueError("Invalid training configuration")

    candidate = copy.deepcopy(model).cpu()
    n_features = candidate.encoder[0].in_features

    if X.shape[1] != n_features:
        raise ValueError("Incorrect number of features")

    original_weights = [
        p.detach().clone() for p in candidate.parameters()
    ]

    before = float(np.mean(score(candidate, X)))

    data = torch.from_numpy(X)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=lr)

    candidate.train()

    for _ in range(epochs):
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]

            optimizer.zero_grad()
            reconstruction = candidate(batch)
            loss = torch.mean((reconstruction - batch) ** 2)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            loss.backward()
            optimizer.step()

    candidate.eval()
    after = float(np.mean(score(candidate, X)))

    weight_change = max(
        float(torch.max(torch.abs(p.detach() - original)).item())
        for p, original in zip(candidate.parameters(), original_weights)
    )

    if not np.isfinite(after) or weight_change > max_weight_change:
        raise RuntimeError("Candidate failed update safety checks")

    return candidate, {
        "loss_before": before,
        "loss_after": after,
        "max_weight_change": weight_change,
    }
