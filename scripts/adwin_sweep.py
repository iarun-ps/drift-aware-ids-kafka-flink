"""Count ADWIN drift events per segment for several delta values (static model, no updates)."""
import json

import pandas as pd
import torch
from river import drift

from ids.data import DAY_FILES, REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform

DELTAS = [0.002, 1e-4, 1e-6, 1e-8, 1e-10]

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
    alerts = score(model, transform(df, spec).values) > thr
    counts = {d: 0 for d in DELTAS}
    for a in alerts:
        x = float(a)
        for d, det in detectors.items():
            det.update(x)
            if det.drift_detected:
                counts[d] += 1
    row = {"segment": name, "flows": len(df)}
    row.update({f"delta={d:g}": counts[d] for d in DELTAS})
    rows.append(row)
    del df

out = pd.DataFrame(rows)
path = REPO_ROOT / "results" / "adwin_sweep.csv"
out.to_csv(path, index=False)
print(out.to_string(index=False))
print("TOTAL per delta:")
print(out.drop(columns=["segment", "flows"]).sum().to_string())
print("wrote", path)
