"""Experiment 1 pilot: compare three models on identical Wednesday flows."""

import json
import subprocess

import numpy as np
import pandas as pd
import torch

from ids.data import REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.memstream import MemStreamBaseline
from ids.half_space_trees import HalfSpaceTreesBaseline
from ids.preprocess import load_spec, time_split, transform
from ids.prequential import prequential_batches
from ids.metrics import evaluate_scores

SEED = 42
BATCH_SIZE = 256
CALIBRATION_FLOWS = 2500
HOLDOUT_FLOWS = 2500
TEST_FLOWS = 5000

spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
meta = json.loads(
    (REPO_ROOT / "models" / "static_ae_meta.json").read_text()
)

ae = AutoEncoder(meta["n_features"])
ae.load_state_dict(torch.load(
    REPO_ROOT / "models" / "static_ae.pt",
    map_location="cpu",
    weights_only=True,
))
ae.eval()

train, val, hold = time_split(load_day("mon"))
wed = load_day("wed").sample(n=TEST_FLOWS, random_state=SEED)

validation_data = val.head(CALIBRATION_FLOWS)
holdout_data = hold.head(HOLDOUT_FLOWS)

mem = MemStreamBaseline(
    ae, memory_size=256, k_neighbors=3, beta=1.0
)
mem.initialize(transform(train.tail(256), spec).values)

hst = HalfSpaceTreesBaseline(
    n_features=len(spec["columns"]),
    n_trees=10,
    height=8,
    window_size=250,
    seed=SEED,
)
hst.initialize(transform(train.tail(1000), spec).values)

models = [
    ("Static AE", lambda X: score(ae, X), None),
    ("MemStream", mem.score_batch, mem.update_batch),
    ("Half-Space Trees", hst.score_batch, hst.update_batch),
]


def evaluate_stream(df, score_fn, update_fn):
    batches = prequential_batches(
        df,
        spec,
        score_fn,
        update_fn,
        batch_size=BATCH_SIZE,
    )
    return pd.concat(list(batches), ignore_index=True)


sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
).stdout.strip()

rows = []

for name, score_fn, update_fn in models:
    print(f"Evaluating {name}...", flush=True)

    validation = evaluate_stream(
        validation_data, score_fn, update_fn
    )
    threshold = float(np.quantile(validation["score"], 0.99))

    holdout = evaluate_stream(
        holdout_data, score_fn, update_fn
    )
    predictions = evaluate_stream(
        wed, score_fn, update_fn
    )

    metrics = evaluate_scores(
        predictions["is_attack"],
        predictions["score"],
        threshold,
    )

    rows.append({
        "model": name,
        "threshold": threshold,
        "validation_fpr": float(
            (validation["score"] > threshold).mean()
        ),
        "holdout_fpr": float(
            (holdout["score"] > threshold).mean()
        ),
        "auroc": metrics["auroc"],
        "aucpr": metrics["aucpr"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "wednesday_fpr": metrics["false_alarm_rate"],
        "test_flows": len(predictions),
        "attacks": int(predictions["is_attack"].sum()),
        "git_sha": sha,
    })

results = pd.DataFrame(rows)

assert len(results) == 3
assert results["test_flows"].eq(TEST_FLOWS).all()
assert results["attacks"].nunique() == 1

output = REPO_ROOT / "results" / "exp1_model_comparison_pilot.csv"
output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)

print("\nMODEL COMPARISON")
print(results.drop(columns="git_sha").round(4).to_string(index=False))
print("\nSaved:", output)
print("PASS: Reproducible pilot comparison completed!")
