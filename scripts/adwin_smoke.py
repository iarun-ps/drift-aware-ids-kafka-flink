"""Smoke test: where does ADWIN fire on the static model's alert stream? (no model updates)"""
import json
import sys

import pandas as pd
import torch
from river import drift

from ids.data import DAY_FILES, REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform

delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002

spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
meta = json.load(open(REPO_ROOT / "models" / "static_ae_meta.json"))
model = AutoEncoder(meta["n_features"])
model.load_state_dict(torch.load(REPO_ROOT / "models" / "static_ae.pt"))
thr = meta["threshold"]

adwin = drift.ADWIN(delta=delta)
rows = []
for name, _ in DAY_FILES:
    df = load_day(name)
    if name == "mon":
        df = time_split(df)[2]
        name = "mon_holdout"
    df = df.sort_values(["Timestamp", "row_order"]).reset_index(drop=True)
    alerts = score(model, transform(df, spec).values) > thr
    ts = df["Timestamp"].to_numpy()
    events = []
    for i, a in enumerate(alerts):
        adwin.update(float(a))
        if adwin.drift_detected:
            events.append(str(ts[i])[:19])
    rows.append({
        "segment": name,
        "flows": len(df),
        "alert_rate": round(float(alerts.mean()), 4),
        "drift_events": len(events),
        "first_event": events[0] if events else "",
        "delta": delta,
    })
    del df

out = pd.DataFrame(rows)
path = REPO_ROOT / "results" / "adwin_smoke.csv"
out.to_csv(path, index=False)
print(out.to_string(index=False))
print("total drift events:", int(out["drift_events"].sum()), "| wrote", path)
