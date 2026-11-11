"""Export the static autoencoder to ONNX (input: features, output: score per row)."""
import json

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ids.data import REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform


class Scorer(nn.Module):
    """Wraps the autoencoder so the output is the reconstruction error per row."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return ((self.ae(x) - x) ** 2).mean(dim=1)


meta = json.load(open(REPO_ROOT / "models" / "static_ae_meta.json"))
ae = AutoEncoder(meta["n_features"])
ae.load_state_dict(torch.load(REPO_ROOT / "models" / "static_ae.pt"))
ae.eval()

path = REPO_ROOT / "models" / "static_ae.onnx"
torch.onnx.export(
    Scorer(ae).eval(), (torch.zeros(1, meta["n_features"]),), str(path),
    input_names=["x"], output_names=["score"],
    dynamic_axes={"x": {0: "batch"}, "score": {0: "batch"}},
    opset_version=17, dynamo=False,
)
print("wrote", path, f"({path.stat().st_size / 1024:.1f} KB)")

# parity check against PyTorch on real, preprocessed Monday holdout flows
spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
hold = time_split(load_day("mon"))[2]
X = transform(hold, spec).values[:5000].astype("float32")

sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
onnx_scores = sess.run(None, {"x": X})[0]
torch_scores = score(ae, X)
print("batch of", len(X), "-> max abs diff:", float(np.abs(onnx_scores - torch_scores).max()))
one = sess.run(None, {"x": X[:1]})[0]
print("single row works:", one.shape, "| matches:", bool(np.allclose(one, torch_scores[:1], atol=1e-5)))
