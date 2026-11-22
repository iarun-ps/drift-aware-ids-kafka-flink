"""Experiment 4: Compare AE + ADWIN safeguard configurations."""

import argparse
from collections import deque
import json
import subprocess

import numpy as np
import pandas as pd
import torch

from ids.data import REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform
from ids.adaptive_engine import AdaptiveAE
from ids.metrics import evaluate_scores


spec = load_spec(REPO_ROOT / "models/preprocess_spec.json")
meta = json.loads(
    (REPO_ROOT / "models/static_ae_meta.json").read_text()
)

static = AutoEncoder(meta["n_features"])
static.load_state_dict(
    torch.load(
        REPO_ROOT / "models/static_ae.pt",
        map_location="cpu",
        weights_only=True,
    )
)
static.eval()

threshold = float(meta["threshold"])

# All calibration data comes from Monday, before the test period.
_, validation, monday_holdout = time_split(load_day("mon"))

validation_X = transform(validation, spec).to_numpy(
    dtype=np.float32
)

cutoff = float(
    np.quantile(score(static, validation_X), 0.05)
)

reference_X = validation_X[:5000].copy()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    choices=["full", "no_alert_gate", "no_score_filter"],
    required=True,
)
variant = parser.parse_args().variant

if variant == "no_score_filter":
    cutoff = float(np.finfo(np.float32).max)

max_alert_rate = 1.0 if variant == "no_alert_gate" else 0.05

engine = AdaptiveAE(
    static,
    threshold=threshold,
    cutoff=cutoff,
    max_alert_rate=max_alert_rate,
)

print("Threshold:", threshold)
print("Adaptation cutoff:", cutoff, flush=True)

results = []
events = []
candidate_audit = []
candidate_labels = deque(
    maxlen=engine.controller.buffer.capacity
)


def replay(name, df, evaluate=False):
    """Process completed minutes in chronological order."""
    df = (
        df.dropna(subset=["Timestamp"])
        .sort_values(["Timestamp", "row_order"])
        .reset_index(drop=True)
    )

    X = transform(df, spec).to_numpy(dtype=np.float32)

    predictions = np.empty(len(df), dtype=np.float32)
    audit_labels = (
        df["Label"].astype(str).str.strip().ne("BENIGN")
        .to_numpy()
    )
    versions = np.empty(len(df), dtype=np.int32)

    # Group by minute without adding an extra feature column.
    minute_groups = df.groupby(
        df["Timestamp"].dt.floor("min"), sort=True
    ).indices

    print(
        f"Replaying {name}: {len(df)} flows, "
        f"{len(minute_groups)} minutes",
        flush=True,
    )

    for minute, indices in minute_groups.items():
        minute_X = X[indices]

        version_used = engine.model_version
        scores, event = engine.process_minute(minute_X)
        predictions[indices] = scores
        versions[indices] = version_used

        # Read-only audit; labels never enter the model/controller.
        if event and (
            event.get("drift_detected")
            or event.get("suspicious_minute")
        ):
            candidate_labels.clear()
        elif event and event.get("pending_drift"):
            accepted = scores <= engine.controller.buffer.cutoff
            candidate_labels.extend(
                audit_labels[indices][accepted].tolist()
            )

        if event and event.get("drift_detected"):
            events.append({
                "segment": name,
                "minute": minute,
                "event": "drift_detected",
                "model_version": engine.model_version,
            })

        if event and event.get("candidate_proposed"):
            assert len(candidate_labels) == len(
                engine.controller.buffer.memory
            ), "Audit and training buffer disagree"

            n = len(candidate_labels)
            attacks = int(sum(candidate_labels))

            candidate_audit.append({
                "variant": variant,
                "segment": name,
                "minute": minute,
                "training_candidates": n,
                "training_attacks": attacks,
                "contamination_rate": attacks / n,
            })

            report = engine.validate_and_promote(reference_X)
            candidate_labels.clear()

            events.append({
                "segment": name,
                "minute": minute,
                "event": (
                    "candidate_approved"
                    if report["approved"]
                    else "candidate_rejected"
                ),
                "model_version": engine.model_version,
            })

    assert np.isfinite(predictions).all()

    if evaluate:
        labels = (
            df["Label"].astype(str).str.strip().ne("BENIGN")
            .to_numpy()
        )

        baseline_scores = score(static, X)

        periods = [
            ("all", np.ones(len(df), dtype=bool)),
            ("before_update", versions == 0),
            ("after_update", versions > 0),
        ]

        for period, mask in periods:
            if not mask.any():
                continue

            for model_name, model_scores in [
                ("Static AE", baseline_scores),
                ("AE + ADWIN", predictions),
            ]:
                metrics = evaluate_scores(
                    labels[mask],
                    model_scores[mask],
                    threshold,
                )

                results.append({
                    "segment": name,
                    "period": period,
                    "model": model_name,
                    **metrics,
                })

    print(
        f"Completed {name}; active version: "
        f"{engine.model_version}",
        flush=True,
    )


# Warm up the drift monitor using earlier traffic.
replay("mon_holdout", monday_holdout)
replay("tue", load_day("tue"))

# Wednesday is the comparison period.
replay("wed", load_day("wed"), evaluate=True)

# Continue chronological evaluation on Thursday and Friday.
for name in [
    "thu_am",
    "thu_pm",
    "fri_am",
    "fri_pm_portscan",
    "fri_pm_ddos",
]:
    replay(name, load_day(name), evaluate=True)

sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
).stdout.strip()

output = pd.DataFrame(results)
output["git_sha"] = sha

results_path = (
    REPO_ROOT / f"results/exp4_{variant}_pilot.csv"
)
events_path = (
    REPO_ROOT / f"results/exp4_{variant}_events.csv"
)

output.to_csv(results_path, index=False)
pd.DataFrame(
    events,
    columns=["segment", "minute", "event", "model_version"],
).to_csv(events_path, index=False)

print("\nSTATIC VS ADAPTIVE")
print(
    output[
        [
            "period", "model", "flows", "attacks", "auroc",
            "aucpr", "precision", "recall",
            "f1", "false_alarm_rate",
        ]
    ].round(4).to_string(index=False)
)

print("\nDrift events:", sum(
    e["event"] == "drift_detected" for e in events
))
print("Approved updates:", engine.model_version)
print("Saved:", results_path)
print("Saved:", events_path)

audit_path = (
    REPO_ROOT / f"results/exp4_{variant}_candidate_audit.csv"
)
pd.DataFrame(candidate_audit).to_csv(
    audit_path, index=False
)
print("Saved:", audit_path)
print("\nTRAINING CONTAMINATION AUDIT")
print(pd.DataFrame(candidate_audit).to_string(index=False))
print("PASS: Chronological adaptive pilot completed!")
