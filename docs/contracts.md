# Contracts between Part 1 (models), Part 2 (pipeline) and Part 3 (baselines/report)

## 1. Features and preprocessing (final)
- Spec file: `models/preprocess_spec.json` (67 features, in this exact order: `spec["columns"]`).
- Raw input to the spec: the CIC-IDS2017 flow columns, with names stripped of spaces and
  typos fixed (see `src/ids/data.py`). Identifier columns and `Destination Port` are NOT used.
- The Kafka producer applies `src/ids/preprocess.py::transform` exactly once.
- Per-flow preprocessing steps:
  1. Missing or infinite value -> the feature's median (`spec["medians"]`).
  2. Negative value -> 0, except `Init_Win_bytes_forward` and `Init_Win_bytes_backward`
     (-1 is a valid marker there and is kept).
  3. `log1p(value)`; for the two Init_Win columns use `log1p(value + 1)`.
  4. `(x - mean) / std` using `spec["mean"]` and `spec["std"]`.
  5. Clip to [-10, 10] (`spec["clip_z"]`).

## 2. Model file (final)
- `models/static_ae.onnx`: input `x` float32 `[N, 67]`, output `score` float32 `[N]`.
  The score is the reconstruction error (higher = more anomalous).
- Model version: `static-v1` (include in every prediction/alert output).
- Alert threshold: `0.1953052431344986` from `models/static_ae_meta.json`
  (99th percentile of benign validation scores). Alert if `score > threshold`.
- Parity requirement: Flink scores must match `ids.models.score` to about 1e-5.

## 3. Kafka messages (PROPOSED - Part 2 to confirm or change)
- Topic `flows` (JSON, one flow per message):
  `{"flow_id": str, "produce_ts_ms": int, "label": str, "features": {name: float}}`
  The producer sends 67 already-preprocessed feature values from `transform()`.
  Values must be finite and correspond to `spec['columns']`.
  Flink assembles them in that exact order as float32 `[1, 67]` and performs
  ONNX scoring only. Flink must NOT repeat preprocessing.
  `label` is carried only for evaluation; the model never uses it.
- Topic `alerts` (JSON): `{"flow_id": str, "score": float, "is_alert": bool,
  "produce_ts_ms": int, "alert_ts_ms": int, "model_version": str}`
  Latency of a flow = `alert_ts_ms - produce_ts_ms`.

## 4. Results files (PROPOSED)
- CSV in `results/`, one row per run or segment, with a `git_sha` column (see
  `results/exp1_static_baseline.csv`). Latency runs also record seed, rate and parallelism.
