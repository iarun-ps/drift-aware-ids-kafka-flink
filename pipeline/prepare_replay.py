"""Prepare a day of flows ONCE and save them as ready-to-send lines.

Each line holds the final JSON message with a placeholder where the send
time goes. replay_producer.py fills in the time at the moment of sending.
"""
import argparse
import json
import time

import numpy as np

from ids.data import REPO_ROOT, load_day
from ids.preprocess import load_spec, transform

PLACEHOLDER = "@@TS@@"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", default="mon")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="default: the whole segment")
    args = parser.parse_args()

    spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
    columns = spec["columns"]

    t0 = time.time()
    df = load_day(args.segment).sort_values(["Timestamp", "row_order"])
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df.reset_index(drop=True)
    values = transform(df, spec).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("prepared features contain NaN or infinite values")
    print(f"Prepared {len(df)} rows in {time.time() - t0:.0f}s, writing lines...")

    # the short text must mean exactly the same float32 number
    step = max(1, len(values) // 2000)
    for v in values[::step].ravel():
        assert np.float32(float(str(v))) == v, "float32 text round trip failed"

    key_prefix = [json.dumps(c) + ": " for c in columns]
    row_orders = df["row_order"].astype(int).tolist()
    labels = df["Label"].astype(str).tolist()

    out_dir = REPO_ROOT / "data" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.segment}.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            flow_id = f"{args.segment}-{row_orders[i]}"
            features = ", ".join(p + str(v) for p, v in zip(key_prefix, values[i]))
            line = (f'{{"flow_id": {json.dumps(flow_id)}, '
                    f'"produce_ts_ms": {PLACEHOLDER}, '
                    f'"label": {json.dumps(labels[i])}, '
                    f'"features": {{{features}}}}}')
            f.write(flow_id + "\t" + line + "\n")
    print(f"Wrote {len(df)} flows to {out_path} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()