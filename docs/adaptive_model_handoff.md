# Adaptive ONNX Model Handoff

## Available Models

| Version | Model | Metadata |
|---|---|---|
| adaptive-v1 | models/adaptive/adaptive-v1.onnx | models/adaptive/adaptive-v1.json |
| adaptive-v2 | models/adaptive/adaptive-v2.onnx | models/adaptive/adaptive-v2.json |
| adaptive-v3 | models/adaptive/adaptive-v3.onnx | models/adaptive/adaptive-v3.json |

## Model Contract

- Input name: x
- Input shape: [batch, 67]
- Input dtype: float32
- Output name: score
- Output shape: [batch]
- Output meaning: Autoencoder reconstruction MSE
- Detection threshold: 0.1953052431344986
- ONNX opset: 17
- Preprocessing: Use models/preprocess_spec.json
- Feature preprocessing must occur exactly once.

## Approved Updates

- adaptive-v1: Wednesday, 2017-07-05 11:09
- adaptive-v2: Thursday, 2017-07-06 15:45
- adaptive-v3: Friday, 2017-07-07 16:31

## Verification

All models passed PyTorch/ONNX parity on 5,000
preprocessed Monday validation records.

Maximum absolute differences:
- adaptive-v1: 3.5762786865234375e-07
- adaptive-v2: 2.384185791015625e-07
- adaptive-v3: 2.384185791015625e-07

Tolerance: 1e-5.

## Integration Notes

These versions were produced by retrospective chronological
adaptation. They must not be used to score traffic occurring
before their respective approval times during historical evaluation.

The streaming pipeline currently uses static-v1.
Adaptive ONNX export is implemented, but automatic model
reloading and live drift-driven updating in Flink are not yet
implemented.

A safe reload mechanism should validate the replacement
model before activation, preserve the current model on failure,
and record the version used for each prediction.
