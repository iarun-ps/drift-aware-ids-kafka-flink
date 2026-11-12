# Contracts between Part 1 (models), Part 2 (pipeline) and Part 3 (baselines/report)

## 1. Features and preprocessing (final)
- Spec file: `models/preprocess_spec.json` (67 features, in this exact order: `spec["columns"]`).
- Raw input to the spec: the CIC-IDS2017 flow columns, with names stripped of spaces and
  typos fixed (see `src/ids/data.py`). Identifier columns and `Destination Port` are NOT used.
- Per-flow steps (must match `src/ids/preprocess.py::transform`):
  1. Missing or infinite value -> the feature's median (`spec["medians"]`).
  2. Negative value -> 0, except `Init_Win_bytes_forward` and `Init_Win_bytes_backward`
     (-1 is a valid marker there and is kept).
  3. `log1p(value)`; for the two Init_Win columns use `log1p(value + 1)`.
  4. `(x - mean) / std` using `spec["mean"]` and `spec["std"]`.
  5. Clip to [-10, 10] (`spec["clip_z"]`).

## 2. Model file (final)
- `models/static_ae.onnx`: input `x` float32 `[N, 67]`, output `score` float32 `[N]`.
  The score is the reconstruction error (higher = more anomalous).
- Alert threshold: `models/static_ae_meta.json` -> `threshold` (99th percentile of benign
  validation scores). A flow is an alert if `score > threshold`.
- Parity requirement: Flink scores must match `ids.models.score` to about 1e-5.

## 3. Kafka messages (PROPOSED - Part 2 to confirm or change)
- Topic `flows` (JSON, one flow per message):
  `{"flow_id": str, "produce_ts_ms": int, "label": str, "features": {name: float or null}}`
  Raw feature values; infinite or missing values are sent as null. Flink applies section 1.
  `label` is carried only for evaluation; the model never uses it.
- Topic `alerts` (JSON): `{"flow_id": str, "score": float, "is_alert": bool,
  "produce_ts_ms": int, "alert_ts_ms": int, "model_version": str}`
  Latency of a flow = `alert_ts_ms - produce_ts_ms`.

## 4. Results files (PROPOSED)
- CSV in `results/`, one row per run or segment, with a `git_sha` column (see
  `results/exp1_static_baseline.csv`). Latency runs also record seed, rate and parallelism.
