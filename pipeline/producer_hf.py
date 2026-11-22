"""Stream CIC-IDS2017 traffic from Hugging Face into Kafka."""

import argparse
import json
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from kafka import KafkaProducer

from ids.preprocess import load_spec, transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send Hugging Face CIC-IDS2017 traffic to Kafka."
    )

    parser.add_argument("--topic", default="flows-test")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--label", default=None,
                        help="Optional label filter, e.g. BENIGN or DDoS")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rate <= 0:
        raise ValueError("Rate must be greater than zero.")

    if args.max_rows <= 0:
        raise ValueError("Maximum rows must be greater than zero.")

    # Load the existing model preprocessing specification.
    spec = load_spec("models/preprocess_spec.json")
    columns = spec["columns"]

    # Connect to the Hugging Face dataset.
    print("Connecting to Hugging Face dataset...")

    dataset = load_dataset(
        "bvsam/cic-ids-2017",
        "traffic_labels",
        split="train",
        streaming=True
    )

    # Collect the requested number of records.
    rows = []

    for row in dataset:
        if args.label and row["Label"].strip().upper() != args.label.upper():
            continue

        rows.append(row)

        if len(rows) >= args.max_rows:
            break

    if not rows:
        raise ValueError("No matching records found.")

    df = pd.DataFrame(rows)

    print("Records downloaded:", len(df))
    print("Traffic labels:")
    print(df["Label"].value_counts().to_string())

    # Apply the existing preprocessing.
    print("Applying existing preprocessing...")

    X = transform(df, spec)
    values = X.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinite values.")

    # Connect to Kafka.
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        acks="all",
        retries=5,
        linger_ms=20,
        batch_size=65536,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8")
    )

    counts = {"sent": 0, "failed": 0}

    def on_success(metadata):
        counts["sent"] += 1

    def on_error(error):
        counts["failed"] += 1
        print("Kafka send failed:", error)

    print(f"Sending records to Kafka topic: {args.topic}")

    start = time.monotonic()

    for i in range(len(df)):

        # Control the offered sending rate.
        wait = start + i / args.rate - time.monotonic()

        if wait > 0:
            time.sleep(wait)

        message = {
            "flow_id": f"hf-{i}",
            "produce_ts_ms": int(time.time() * 1000),
            "label": str(df.iloc[i]["Label"]),
            "features": dict(
                zip(columns, map(float, values[i]))
            )
        }

        producer.send(
            args.topic,
            key=message["flow_id"],
            value=message
        ).add_callback(on_success).add_errback(on_error)

    producer.flush()
    elapsed = time.monotonic() - start
    producer.close()

    print("\nKafka sending completed.")
    print("Successfully sent:", counts["sent"])
    print("Failed:", counts["failed"])
    print("Elapsed seconds:", round(elapsed, 2))
    print("Achieved rate:", round(counts["sent"] / elapsed, 2),
          "records/second")


if __name__ == "__main__":
    main()