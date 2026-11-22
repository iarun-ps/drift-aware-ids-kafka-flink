"""Export approved adaptive Autoencoders as versioned ONNX models."""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ids.models import score


class Scorer(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return ((self.ae(x) - x) ** 2).mean(dim=1)


def export_approved_model(engine, reference_X, output_dir):
    """Export and verify the currently approved adaptive model."""

    if engine.model_version < 1:
        raise ValueError("No adaptive model has been approved")

    if engine.candidate_model is not None:
        raise RuntimeError("Candidate is awaiting validation")

    X = np.asarray(reference_X, dtype=np.float32)

    if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Invalid reference features")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version = f"adaptive-v{engine.model_version}"

    model_path = output_dir / f"{version}.onnx"
    metadata_path = output_dir / f"{version}.json"
    temporary_path = output_dir / f".{version}.tmp.onnx"

    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Model version already exists: {version}")

    model = engine.active_model.cpu().eval()

    try:
        torch.onnx.export(
            Scorer(model).eval(),
            (torch.zeros(1, X.shape[1]),),
            str(temporary_path),
            input_names=["x"],
            output_names=["score"],
            dynamic_axes={
                "x": {0: "batch"},
                "score": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        session = ort.InferenceSession(
            str(temporary_path),
            providers=["CPUExecutionProvider"],
        )

        onnx_scores = session.run(None, {"x": X})[0]
        torch_scores = score(model, X)

        if onnx_scores.shape != torch_scores.shape:
            raise RuntimeError("ONNX output shape mismatch")

        max_diff = float(
            np.max(np.abs(onnx_scores - torch_scores))
        )

        if not np.isfinite(max_diff) or max_diff > 1e-5:
            raise RuntimeError(
                f"ONNX parity failed: max_diff={max_diff}"
            )

        metadata = {
            "model_version": version,
            "input_name": "x",
            "output_name": "score",
            "input_dtype": "float32",
            "n_features": int(X.shape[1]),
            "threshold": float(engine.controller.threshold),
            "opset": 17,
            "parity_rows": int(len(X)),
            "parity_max_abs_diff": max_diff,
        }

        os.replace(temporary_path, model_path)

        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"PASS: Exported {version}")
    print(f"ONNX parity max difference: {max_diff:.2e}")

    return model_path, metadata_path
