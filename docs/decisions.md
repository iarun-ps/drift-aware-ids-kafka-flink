# Design decisions

## D1: Adaptive detector = autoencoder + ADWIN (METER-inspired), not the official METER code
- Checked 2026-11-12. Official code: github.com/zjiaqi725/METER (VLDB 2024, Zhu et al.).
- It pins Python 3.7.6, PyTorch 1.5.1, hypnettorch 0.0.4 and edl-pytorch 0.0.2. Our environment is
  Python 3.10 with current PyTorch/ONNX, and the Flink pipeline needs ONNX export.
- It runs from a single main.py on small benchmark datasets, so adapting it to CIC-IDS2017 segments
  and the streaming pipeline would exceed the half-day limit set in the plan.
- Chosen instead (the fallback in the project description):
  static autoencoder (static detector) + ADWIN on anomaly scores (drift controller)
  + fine-tuning on recent flows after a drift signal (offline update).
- Not implemented: METER's hypernetwork-generated parameter shift and evidential uncertainty.
- Report wording: "METER-inspired"; state the reason above and do not claim a METER reproduction.
