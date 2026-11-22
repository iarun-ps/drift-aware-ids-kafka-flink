"""Compare Flink's streamed scores with Arun's expected scores."""
import argparse
import json

import numpy as np
import pandas as pd
from kafka import KafkaConsumer

from ids.data import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "parity_100.csv"
META = REPO_ROOT / "models" / "static_ae_meta.json"
TOLERANCE = 1e-5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="parity-out")
    parser.add_argument("--bootstrap", default="localhost:9092")
    args = parser.parse_args()

    df = pd.read_csv(FIXTURE)
    expected = df["expected_score"].to_numpy()
    threshold = json.loads(META.read_text())["threshold"]

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=15000,
    )
    results, duplicates, errors = {}, 0, 0
    for record in consumer:
        msg = json.loads(record.value)
        if msg.get("status") != "ok":
            errors += 1
            continue
        if msg["flow_id"] in results:
            duplicates += 1
        results[msg["flow_id"]] = msg

    ids = [f"parity-{i:03d}" for i in range(len(df))]
    missing = [i for i in ids if i not in results]
    print(f"Expected flows : {len(df)}")
    print(f"Scored ok      : {len(results)}")
    print(f"Missing        : {len(missing)}")
    print(f"Duplicates     : {duplicates}")
    print(f"Error results  : {errors}")
    if missing:
        print("FAIL: some flows are missing")
        return

    streamed = np.array([results[i]["score"] for i in ids])
    diff = np.abs(streamed - expected)
    alerts_streamed = np.array([results[i]["is_alert"] for i in ids])
    alerts_expected = expected > threshold
    print(f"Max abs score difference : {diff.max():.3e}")
    print(f"Alert flags that differ  : {int((alerts_streamed != alerts_expected).sum())}")
    ok = diff.max() < TOLERANCE and not duplicates and not errors
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()