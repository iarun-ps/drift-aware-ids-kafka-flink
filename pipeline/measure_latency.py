"""Read scored results from Kafka and report latency, throughput and sanity checks."""
import argparse
import json

import numpy as np
from kafka import KafkaConsumer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="scores")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--expected", type=int, default=None, help="how many flows were sent")
    parser.add_argument("--idle-sec", type=int, default=20, help="stop after this many silent seconds")
    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=args.idle_sec * 1000,
    )

    seen = {}
    ok = errors = alerts = 0
    latencies = []
    points = []
    first_produce = None
    last_completion = None

    print(f"Reading '{args.topic}' (stops after {args.idle_sec}s of silence)...")
    for record in consumer:
        msg = json.loads(record.value)
        fid = msg.get("flow_id")
        seen[fid] = seen.get(fid, 0) + 1
        if msg.get("status") != "ok":
            errors += 1
            continue
        ok += 1
        alerts += 1 if msg["is_alert"] else 0
        produce = msg["produce_ts_ms"]
        done = msg["completion_ts_ms"]
        latencies.append(done - produce)
        points.append((produce, done - produce))
        first_produce = produce if first_produce is None else min(first_produce, produce)
        last_completion = done if last_completion is None else max(last_completion, done)

    total = ok + errors
    duplicates = sum(1 for n in seen.values() if n > 1)
    print(f"\nResults read:     {total}")
    print(f"  ok:             {ok}")
    print(f"  error:          {errors}")
    print(f"  duplicates:     {duplicates} flow_ids seen more than once")
    if args.expected is not None:
        print(f"  missing:        {args.expected - len(seen)} of {args.expected} expected")
    if not latencies:
        print("No ok results, nothing to measure.")
        return

    lat = np.array(latencies, dtype=float)
    seconds = (last_completion - first_produce) / 1000.0
    print(f"  alerts:         {alerts} ({100.0 * alerts / ok:.1f}% of ok)")
    print("\nLatency (ms) = completion time - produce time")
    print(f"  median (p50):   {np.percentile(lat, 50):.0f}")
    print(f"  p95:            {np.percentile(lat, 95):.0f}")
    print(f"  p99:            {np.percentile(lat, 99):.0f}")
    print(f"  max:            {lat.max():.0f}")
    print(f"\nThroughput: {len(seen)} unique flows in {seconds:.1f} s = {len(seen) / seconds:.0f} flows/s")
    print("\nLatency by minute of the run (ms)")
    base = min(p for p, _ in points)
    buckets = {}
    for p, l in points:
        buckets.setdefault(int((p - base) // 60000), []).append(l)
    for minute in sorted(buckets):
        arr = np.array(buckets[minute], dtype=float)
        print(f"  minute {minute:2d}: flows={len(arr):6d}  median={np.percentile(arr, 50):8.0f}  max={arr.max():8.0f}")

if __name__ == "__main__":
    main()