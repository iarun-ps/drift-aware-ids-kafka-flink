"""Check that Arun's ONNX model loads and scores one flow."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "static_ae.onnx"

session = ort.InferenceSession(
    str(MODEL_PATH), providers=["CPUExecutionProvider"]
)

print("Input :", session.get_inputs()[0].name, session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].name, session.get_outputs()[0].shape)

row = np.zeros((1, 67), dtype=np.float32)
score = session.run(["score"], {"x": row})[0]
print("Score for one all-zero row:", score)