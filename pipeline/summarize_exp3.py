"""Summarise Experiment 3: mean and spread over the repeats of each setting."""
import argparse

import pandas as pd

from ids.data import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/exp3")
    args = parser.parse_args()
    root = REPO_ROOT / args.dir

    df = pd.read_csv(root / "summary.csv").drop_duplicates(subset="run_id", keep="last")
    valid = df[df["producer_valid"] == True]  # noqa: E712

    table = valid.groupby(["partitions", "parallelism", "rate_offered"]).agg(
        runs=("run_id", "count"),
        p50_ms=("p50_ms", "mean"),
        p50_sd=("p50_ms", "std"),
        p95_ms=("p95_ms", "mean"),
        p95_sd=("p95_ms", "std"),
        p99_ms=("p99_ms", "mean"),
        throughput_fps=("throughput_fps", "mean"),
        missing=("missing", "sum"),
        errors=("errors", "sum"),
    ).round(1).reset_index()

    print(table.to_string(index=False))
    table.to_csv(root / "aggregate.csv", index=False)
    print(f"\nRuns in summary.csv: {len(df)}, with a valid producer: {len(valid)}")
    print(f"Wrote {root / 'aggregate.csv'}")


if __name__ == "__main__":
    main()