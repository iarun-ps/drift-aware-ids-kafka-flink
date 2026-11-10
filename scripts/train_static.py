"""Train the static autoencoder on Monday (benign) and set an alert threshold."""
import json

import numpy as np
import torch

from ids.data import REPO_ROOT, load_day
from ids.models import AutoEncoder, fit, score
from ids.preprocess import load_spec, time_split, transform

spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
train, val, hold = time_split(load_day("mon"))
Xtr, Xva, Xho = (transform(d, spec).values for d in (train, val, hold))

model = AutoEncoder(Xtr.shape[1])
history = fit(model, Xtr, epochs=10, seed=0)
for i, h in enumerate(history, 1):
    print(f"epoch {i}: loss {h:.4f}")

thr = float(np.quantile(score(model, Xva), 0.99))
fpr = float((score(model, Xho) > thr).mean())
print("threshold (99th percentile of validation):", thr)
print("false alarm rate on benign holdout:", fpr)

torch.save(model.state_dict(), REPO_ROOT / "models" / "static_ae.pt")
with open(REPO_ROOT / "models" / "static_ae_meta.json", "w") as f:
    json.dump({"threshold": thr, "epochs": 10, "seed": 0,
               "n_features": int(Xtr.shape[1])}, f, indent=2)
print("saved model and metadata")
