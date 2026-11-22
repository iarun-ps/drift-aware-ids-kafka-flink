"""Parity check: ONNX scores must match Arun's expected scores on 100 flows."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "parity_100.csv"
MODEL = ROOT / "models" / "static_ae.onnx"
TOLERANCE = 1e-5

df = pd.read_csv(FIXTURE)
feature_cols = [c for c in df.columns if c not in ("flow_id", "expected_score")]
assert len(feature_cols) == 67, len(feature_cols)

X = df[feature_cols].to_numpy(dtype=np.float32)
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
scores = session.run(["score"], {"x": X})[0]

diff = np.abs(scores - df["expected_score"].to_numpy())
print("Rows checked      :", len(df))
print("Max abs difference:", diff.max())
print("PASS" if diff.max() < TOLERANCE else "FAIL")
