"""Cache per-minute stats of the static model, then sweep ADWIN on two signals."""
import json

import pandas as pd
import torch
from river import drift

from ids.data import DAY_FILES, REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform

MIN_FLOWS = 20
DELTAS = [0.002, 0.01, 0.05, 0.2]
CLOCK = 8
SHORT = {"mon_holdout": "mon", "fri_pm_portscan": "fri_ps", "fri_pm_ddos": "fri_dd"}

meta = json.load(open(REPO_ROOT / "models" / "static_ae_meta.json"))
thr = meta["threshold"]
cache = REPO_ROOT / "results" / "static_minute_stats.csv"

if cache.exists():
    stats = pd.read_csv(cache)
    print("loaded cache", cache.name)
else:
    spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
    model = AutoEncoder(meta["n_features"])
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "static_ae.pt"))
    parts = []
    for name, _ in DAY_FILES:
        df = load_day(name)
        if name == "mon":
            df = time_split(df)[2]
            name = "mon_holdout"
        df = df.sort_values(["Timestamp", "row_order"]).reset_index(drop=True)
        df["score"] = score(model, transform(df, spec).values)
        df["alert"] = df["score"] > thr
        df["minute"] = df["Timestamp"].dt.floor("min")
        g = df.groupby("minute").agg(flows=("alert", "size"),
                                     alert_rate=("alert", "mean"),
                                     median_score=("score", "median")).reset_index()
        g.insert(0, "segment", name)
        parts.append(g)
        del df
    stats = pd.concat(parts, ignore_index=True)
    stats.to_csv(cache, index=False)
    print("wrote cache", cache.name)

stats = stats[stats["flows"] >= MIN_FLOWS].copy()
stats["median_ratio"] = (stats["median_score"] / thr).clip(0, 1)
segments = list(dict.fromkeys(stats["segment"]))

rows, events = [], []
for sig in ["alert_rate", "median_ratio"]:
    for d in DELTAS:
        det = drift.ADWIN(delta=d, clock=CLOCK)
        counts = {s: 0 for s in segments}
        for seg, minute, val in zip(stats["segment"], stats["minute"], stats[sig]):
            det.update(float(val))
            if det.drift_detected:
                counts[seg] += 1
                events.append({"signal": sig, "delta": d, "segment": seg, "minute": minute})
        row = {"signal": sig, "delta": d}
        row.update({SHORT.get(s, s): c for s, c in counts.items()})
        row["TOTAL"] = sum(counts.values())
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(REPO_ROOT / "results" / "adwin_grid.csv", index=False)
pd.DataFrame(events).to_csv(REPO_ROOT / "results" / "adwin_events.csv", index=False)
print(out.to_string(index=False))
