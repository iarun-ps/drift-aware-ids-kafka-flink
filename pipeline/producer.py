"""Replay CIC-IDS2017 flows into Kafka at a controlled rate.

Each flow is prepared once with transform() from ids.preprocess (contract
section 1), so Flink only has to score it. The rate is the OFFERED rate.
"""
import argparse
import json
import time

import numpy as np
from kafka import KafkaProducer

from ids.data import REPO_ROOT, load_day
from ids.preprocess import load_spec, transform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment", default="mon",
                   help="mon, tue, wed, thu_am, thu_pm, fri_am, fri_pm_portscan, fri_pm_ddos")
    p.add_argument("--topic", default="flows")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--rate", type=float, default=100.0,
                   help="offered rate in records per second")
    p.add_argument("--max-rows", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
    columns = spec["columns"]

    # same order every run: by capture time, then original row order
    df = load_day(args.segment)
    df = df.sort_values(["Timestamp", "row_order"])
    df = df.head(args.max_rows).reset_index(drop=True)

    values = transform(df, spec).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("prepared features contain NaN or infinite values")
    flow_ids = [f"{args.segment}-{int(r)}" for r in df["row_order"]]
    labels = df["Label"].astype(str).tolist()

    producer = KafkaProducer(
    bootstrap_servers=args.bootstrap,
    acks="all",
    retries=5,
    linger_ms=20,
    batch_size=65536,
    key_serializer=lambda k: k.encode("utf-8"),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    counts = {"ok": 0, "failed": 0}

    def on_ok(_metadata):
        counts["ok"] += 1

    def on_error(exc):
        counts["failed"] += 1
        print("send failed:", exc)

    print(f"Sending {len(df)} flows from '{args.segment}' to topic "
          f"'{args.topic}' at {args.rate} records/s")
    start = time.monotonic()
    for i in range(len(df)):
        wait = start + i / args.rate - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        message = {
            "flow_id": flow_ids[i],
            "produce_ts_ms": int(time.time() * 1000),
            "label": labels[i],
            "features": dict(zip(columns, map(float, values[i]))),
        }
        producer.send(args.topic, key=flow_ids[i], value=message) \
            .add_callback(on_ok).add_errback(on_error)
    producer.flush()
    elapsed = time.monotonic() - start

    print(f"Done: {counts['ok']} sent, {counts['failed']} failed, "
          f"{elapsed:.1f}s, achieved {counts['ok'] / elapsed:.1f} records/s")


if __name__ == "__main__":
    main()