"""Make the Experiment 3 figures and a metrics summary from the exported results.

Reads  results/exp3/summary.csv and results/exp3_metrics/metrics_per_run.csv
Writes results/exp3_figures/*.png and results/exp3_figures/metrics_summary.csv
Only new files are created; nothing existing is changed.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp3_figures"
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(ROOT / "results" / "exp3" / "summary.csv")
runs = runs[runs.producer_valid.astype(str) == "True"].copy()
runs["setting"] = runs.apply(lambda r: f"{int(r.partitions)} part / par {int(r.parallelism)}", axis=1)
SETTINGS = ["1 part / par 1", "2 part / par 2", "4 part / par 4"]
RATES = [800, 1500]
COLORS = {800: "#2b6cb0", 1500: "#dd6b20"}


def grouped_bar(ax, column, ylabel, title):
    width = 0.38
    for i, rate in enumerate(RATES):
        means, sds = [], []
        for s in SETTINGS:
            v = runs[(runs.setting == s) & (runs.rate_offered == rate)][column]
            means.append(v.mean())
            sds.append(v.std(ddof=1))
        ax.bar(np.arange(3) + (i - 0.5) * width, means, width, yerr=sds, capsize=3,
               label=f"{rate} flows/s offered", color=COLORS[rate])
    ax.set_xticks(range(3))
    ax.set_xticklabels(SETTINGS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


# 1. median latency
fig, ax = plt.subplots(figsize=(7, 4))
grouped_bar(ax, "p50_ms", "median latency (ms)", "Median latency (p50), mean of 3 runs, bars = std")
save(fig, "1_latency_p50.png")

# 2. p95 and p99
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
grouped_bar(axes[0], "p95_ms", "latency (ms)", "p95 latency")
grouped_bar(axes[1], "p99_ms", "latency (ms)", "p99 latency")
save(fig, "2_latency_p95_p99.png")

# 3. throughput
fig, ax = plt.subplots(figsize=(7, 4))
grouped_bar(ax, "throughput_fps", "throughput (flows/s)", "Throughput (equals offered rate in every setting)")
for rate in RATES:
    ax.axhline(rate, color=COLORS[rate], ls="--", lw=0.8)
save(fig, "3_throughput.png")

# 7. throughput vs latency
fig, ax = plt.subplots(figsize=(7, 4.5))
markers = {"1 part / par 1": "o", "2 part / par 2": "s", "4 part / par 4": "^"}
for s in SETTINGS:
    for rate in RATES:
        d = runs[(runs.setting == s) & (runs.rate_offered == rate)]
        ax.scatter(d.throughput_fps, d.p50_ms, marker=markers[s], color=COLORS[rate], s=50,
                   label=f"{s}, {rate}/s")
ax.set_xlabel("throughput (flows/s)")
ax.set_ylabel("median latency (ms)")
ax.set_title("Throughput versus latency (one point per run)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
save(fig, "7_throughput_vs_latency.png")

# metrics: CPU, memory, backpressure, busy, backlog
m = pd.read_csv(ROOT / "results" / "exp3_metrics" / "metrics_per_run.csv")
m["value"] = pd.to_numeric(m["value"], errors="coerce")
rows = []
for run_id, g in m.groupby("run_id"):
    src = g[g.metric == "source_records_in_per_s"].groupby("seconds_from_run_start").value.sum()
    active = src[src > 50]
    if active.empty:
        continue
    lo, hi = active.index.min(), active.index.max()
    g = g[(g.seconds_from_run_start >= lo) & (g.seconds_from_run_start <= hi)]
    pick = lambda name: g[g.metric == name].value
    rows.append({
        "run_id": run_id,
        "cpu_mean": pick("jvm_cpu_load").mean(),
        "cpu_max": pick("jvm_cpu_load").max(),
        "heap_mean_mb": pick("jvm_heap_used_bytes").mean() / 1e6,
        "heap_max_mb": pick("jvm_heap_used_bytes").max() / 1e6,
        "metaspace_max_mb": pick("jvm_metaspace_used_bytes").max() / 1e6,
        "busy_mean_ms_per_s": pick("busy_ms_per_s").mean(),
        "idle_mean_ms_per_s": pick("idle_ms_per_s").mean(),
        "backpressured_mean_ms_per_s": pick("backpressured_ms_per_s").mean(),
        "backpressured_max_ms_per_s": pick("backpressured_ms_per_s").max(),
        "pending_mean": g[g.metric == "pending_records"].groupby("seconds_from_run_start").value.sum().mean(),
        "pending_max": g[g.metric == "pending_records"].groupby("seconds_from_run_start").value.sum().max(),
    })
mr = pd.DataFrame(rows).merge(runs[["run_id", "setting", "rate_offered"]], on="run_id")
mr.to_csv(OUT / "metrics_per_run_summary.csv", index=False)
agg = mr.drop(columns="run_id").groupby(["setting", "rate_offered"]).mean(numeric_only=True).round(3)
agg.to_csv(OUT / "metrics_summary.csv")
print("\nMetrics during the sending window (mean over 3 runs):")
print(agg.to_string())

# 4/5/6. CPU, memory, backpressure
fig, ax = plt.subplots(figsize=(7, 4))
mr["cpu_pct"] = mr.cpu_mean * 100
runs_backup, runs = runs, mr
grouped_bar(ax, "cpu_pct", "JVM CPU load (%)", "Task manager CPU load while sending (mean of 3 runs)")
runs = runs_backup
save(fig, "4_cpu.png")

fig, ax = plt.subplots(figsize=(7, 4))
runs_backup, runs = runs, mr
grouped_bar(ax, "heap_mean_mb", "JVM heap used (MB)", "Task manager heap while sending (mean of 3 runs)")
runs = runs_backup
save(fig, "5_jvm_memory.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
runs_backup, runs = runs, mr
grouped_bar(axes[0], "backpressured_mean_ms_per_s", "ms per second (0-1000)", "Backpressured time")
grouped_bar(axes[1], "busy_mean_ms_per_s", "ms per second (0-1000)", "Busy time")
runs = runs_backup
save(fig, "6_backpressure_busy.png")
print("\nDone. Figures are in", OUT)