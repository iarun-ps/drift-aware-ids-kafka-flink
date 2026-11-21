"""ADWIN on the per-minute alert rate of the static model (no model updates)."""
import json

import pandas as pd
import torch
from river import drift

from ids.data import DAY_FILES, REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform

DELTAS = [0.002, 1e-4, 1e-6]
MIN_FLOWS = 20          # ignore minutes with too few flows to give a stable rate

spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
meta = json.load(open(REPO_ROOT / "models" / "static_ae_meta.json"))
model = AutoEncoder(meta["n_features"])
model.load_state_dict(torch.load(REPO_ROOT / "models" / "static_ae.pt"))
thr = meta["threshold"]

detectors = {d: drift.ADWIN(delta=d) for d in DELTAS}
rows = []
for name, _ in DAY_FILES:
    df = load_day(name)
    if name == "mon":
        df = time_split(df)[2]
        name = "mon_holdout"
    df = df.sort_values(["Timestamp", "row_order"]).reset_index(drop=True)
    df["alert"] = score(model, transform(df, spec).values) > thr
    df["minute"] = df["Timestamp"].dt.floor("min")
    g = df.groupby("minute")["alert"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= MIN_FLOWS]
    counts = {d: 0 for d in DELTAS}
    for rate in g["mean"]:
        for d, det in detectors.items():
            det.update(float(rate))
            if det.drift_detected:
                counts[d] += 1
    row = {"segment": name, "minutes_used": len(g)}
    row.update({f"delta={d:g}": counts[d] for d in DELTAS})
    rows.append(row)
    del df

out = pd.DataFrame(rows)
path = REPO_ROOT / "results" / "adwin_minute.csv"
out.to_csv(path, index=False)
print(out.to_string(index=False))
print("TOTAL per delta:")
print(out.drop(columns=["segment", "minutes_used"]).sum().to_string())
print("wrote", path)
