"""Send deliberately broken messages to test Flink's error handling."""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

from ids.data import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "parity_100.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="errors-in")
    parser.add_argument("--bootstrap", default="localhost:9092")
    args = parser.parse_args()

    df = pd.read_csv(FIXTURE)
    cols = [c for c in df.columns if c not in ("flow_id", "expected_score")]
    features = {c: float(df.at[0, c]) for c in cols}

    def message(flow_id, feats=None):
        return {
            "flow_id": flow_id,
            "produce_ts_ms": int(time.time() * 1000),
            "label": "fixture",
            "features": features if feats is None else feats,
        }

    items = list(features.items())
    reordered = dict([items[1], items[0]] + items[2:])
    with_nan = dict(features)
    with_nan[cols[5]] = float("nan")
    with_text = dict(features)
    with_text[cols[5]] = "abc"
    no_id = message("x")
    del no_id["flow_id"]

    cases = [
        ("valid flow", json.dumps(message("case-1"))),
        ("not JSON", "this is not json"),
        ("missing flow_id", json.dumps(no_id)),
        ("only 10 features", json.dumps(message("case-4", dict(items[:10])))),
        ("features in wrong order", json.dumps(message("case-5", reordered))),
        ("NaN value", json.dumps(message("case-6", with_nan))),
        ("text value", json.dumps(message("case-7", with_text))),
        ("features is a list", json.dumps(message("case-8", list(features.values())))),
        ("duplicate of case 1", json.dumps(message("case-1"))),
    ]

    producer = KafkaProducer(bootstrap_servers=args.bootstrap, acks="all", retries=5)
    for i, (name, payload) in enumerate(cases, start=1):
        producer.send(args.topic, key=f"case-{i}".encode(), value=payload.encode())
        print(f"sent {i}: {name}")
    producer.flush()
    print(f"Sent {len(cases)} messages. Expect 2 ok and 7 error results.")


if __name__ == "__main__":
    main()