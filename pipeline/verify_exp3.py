"""Independent re-computation of Experiment 3 numbers from the raw per-flow files."""
import argparse

import numpy as np
import pandas as pd

from ids.data import REPO_ROOT

CHECKS = [  # (my value, column in summary.csv, allowed difference from rounding)
    ("ok", "ok", 0), ("errors", "errors", 0), ("duplicates", "duplicates", 0),
    ("missing", "missing", 0), ("alerts", "alerts", 0),
    ("p50", "p50_ms", 0.5), ("p95", "p95_ms", 0.5), ("p99", "p99_ms", 0.5),
    ("max", "max_ms", 0.5), ("throughput", "throughput_fps", 0.05),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/exp3")
    args = parser.parse_args()
    root = REPO_ROOT / args.dir
    summary = pd.read_csv(root / "summary.csv").set_index("run_id")

    out = []
    for run_id, s in summary.iterrows():
        raw = pd.read_csv(root / run_id / "scores.csv")
        ok = raw[raw["status"] == "ok"]
        latency = (ok["completion_ts_ms"] - ok["produce_ts_ms"]).to_numpy(float)
        span = (ok["completion_ts_ms"].max() - ok["produce_ts_ms"].min()) / 1000.0
        unique_ok = ok["flow_id"].nunique()
        mine = {
            "ok": len(ok), "errors": len(raw) - len(ok),
            "duplicates": len(raw) - raw["flow_id"].nunique(),
            "missing": int(s["rows_sent"]) - raw["flow_id"].nunique(),
            "alerts": int(ok["is_alert"].astype(str).str.lower().eq("true").sum()),
            "p50": np.percentile(latency, 50), "p95": np.percentile(latency, 95),
            "p99": np.percentile(latency, 99), "max": latency.max(),
            "throughput": unique_ok / span,
        }
        problems = [f"{name}: mine {mine[name]:.1f} vs summary {s[col]}"
                    for name, col, tol in CHECKS if abs(mine[name] - s[col]) > tol]
        negative = int((latency < 0).sum())
        if negative:
            problems.append(f"{negative} negative latencies")
        if ok["model_version"].nunique() != 1:
            problems.append("more than one model_version")
        if not np.isfinite(ok["score"].astype(float)).all():
            problems.append("non-finite scores")
        out.append({"run_id": run_id, **{k: round(v, 1) for k, v in mine.items()},
                    "min_latency_ms": latency.min(), "negative_latencies": negative,
                    "match": not problems, "problems": "; ".join(problems)})
        print(f"{run_id:22s} {'OK' if not problems else 'MISMATCH'}   "
              f"p50={mine['p50']:.0f} p95={mine['p95']:.0f} p99={mine['p99']:.0f} "
              f"throughput={mine['throughput']:.1f}  {'; '.join(problems)}", flush=True)

    pd.DataFrame(out).to_csv(root / "verification.csv", index=False)
    good = sum(1 for r in out if r["match"])
    print(f"\n{good} of {len(out)} runs match summary.csv. Wrote {root / 'verification.csv'}")


if __name__ == "__main__":
    main()