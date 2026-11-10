"""Experiment 1, first pass: score every segment with the static autoencoder."""
import json
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ids.data import DAY_FILES, REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform

spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
meta = json.load(open(REPO_ROOT / "models" / "static_ae_meta.json"))
model = AutoEncoder(meta["n_features"])
model.load_state_dict(torch.load(REPO_ROOT / "models" / "static_ae.pt"))
thr = meta["threshold"]
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                     text=True, cwd=REPO_ROOT).stdout.strip()

rows = []
for name, _ in DAY_FILES:
    df = load_day(name)
    if name == "mon":
        df = time_split(df)[2]          # benign holdout only (never trained on)
        name = "mon_holdout"
    s = score(model, transform(df, spec).values)
    y = (df["Label"] != "BENIGN").to_numpy()
    n_att = int(y.sum())
    both = 0 < n_att < len(y)
    rows.append({
        "segment": name,
        "flows": len(df),
        "attacks": n_att,
        "attack_share": round(n_att / len(df), 4),
        "auroc": roc_auc_score(y, s) if both else np.nan,
        "aucpr": average_precision_score(y, s) if both else np.nan,
        "detection_rate": float((s[y] > thr).mean()) if n_att else np.nan,
        "false_alarm_rate": float((s[~y] > thr).mean()),
        "git_sha": sha,
    })
    del df

out = pd.DataFrame(rows)
path = REPO_ROOT / "results" / "exp1_static_baseline.csv"
out.to_csv(path, index=False)
print(out.drop(columns="git_sha").round(3).to_string(index=False))
print("wrote", path)
