"""Send Arun's 100 parity-test flows (already prepared features) to Kafka."""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

from ids.data import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "parity_100.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="parity-in")
    parser.add_argument("--bootstrap", default="localhost:9092")
    args = parser.parse_args()

    df = pd.read_csv(FIXTURE)
    feature_cols = [c for c in df.columns if c not in ("flow_id", "expected_score")]
    assert len(feature_cols) == 67, len(feature_cols)

    producer = KafkaProducer(bootstrap_servers=args.bootstrap, acks="all", retries=5)
    for i in range(len(df)):
        flow_id = f"parity-{i:03d}"  # row number, because Flow IDs can repeat
        message = {
            "flow_id": flow_id,
            "produce_ts_ms": int(time.time() * 1000),
            "label": "fixture",
            "features": {c: float(df.at[i, c]) for c in feature_cols},
        }
        producer.send(args.topic, key=flow_id.encode(), value=json.dumps(message).encode())
    producer.flush()
    print(f"Sent {len(df)} fixture flows to topic '{args.topic}'")


if __name__ == "__main__":
    main()