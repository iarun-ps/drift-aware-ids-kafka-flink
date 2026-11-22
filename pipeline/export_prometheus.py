"""Export Prometheus metrics for every finished Experiment 3 run (one row per sample, tagged with run_id)."""
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from ids.data import REPO_ROOT

PROM = "http://localhost:9090"
STEP = "5s"
MARGIN_S = 10
METRICS = {
    "busy_ms_per_s":        "flink_taskmanager_job_task_busyTimeMsPerSecond",
    "idle_ms_per_s":        "flink_taskmanager_job_task_idleTimeMsPerSecond",
    "backpressured_ms_per_s": "flink_taskmanager_job_task_backPressuredTimeMsPerSecond",
    "pending_records":      "flink_taskmanager_job_task_operator_pendingRecords",
    "source_records_in_per_s": 'flink_taskmanager_job_task_operator_numRecordsInPerSecond{operator_name=~".*Source.*"}',
    "jvm_cpu_load":         "flink_taskmanager_Status_JVM_CPU_Load",
    "jvm_heap_used_bytes":  "flink_taskmanager_Status_JVM_Memory_Heap_Used",
    "jvm_metaspace_used_bytes": "flink_taskmanager_Status_JVM_Memory_Metaspace_Used",
    "kafka_topic_offset":   'kafka_topic_partition_current_offset{topic=~"flows-p.*|scores-p.*"}',
}
LABELS = ["task_name", "subtask_index", "operator_name", "topic", "partition"]


def query_range(expr, start, end):
    params = urllib.parse.urlencode({"query": expr, "start": start.timestamp(),
                                     "end": end.timestamp(), "step": STEP})
    with urllib.request.urlopen(f"{PROM}/api/v1/query_range?{params}", timeout=60) as r:
        return json.load(r)["data"]["result"]


def main():
    out_dir = Path(REPO_ROOT) / "results" / "exp3_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for meta_path in sorted((Path(REPO_ROOT) / "results" / "exp3").glob("*/meta.json")):
        meta = json.load(open(meta_path))
        start = datetime.fromisoformat(meta["start_utc"])
        end = start + timedelta(seconds=int(meta["duration_s"]) + MARGIN_S)
        runs.append((meta, start, end))
    print(f"{len(runs)} finished runs found")

    rows_total = 0
    with open(out_dir / "metrics_per_run.csv", "w", newline="") as f, \
         open(out_dir / "run_windows.csv", "w", newline="") as w:
        writer = csv.writer(f)
        writer.writerow(["run_id", "metric", "time_utc", "seconds_from_run_start", "value"] + LABELS)
        win = csv.writer(w)
        win.writerow(["run_id", "partitions", "parallelism", "rate_offered", "start_utc",
                      "end_utc_with_margin", "input_topic", "output_topic", "job_id"])
        for meta, start, end in runs:
            win.writerow([meta["run_id"], meta["partitions"], meta["parallelism"], meta["rate_offered"],
                          start.isoformat(), end.isoformat(), meta["input_topic"],
                          meta["output_topic"], meta["job_id"]])
            for name, expr in METRICS.items():
                for series in query_range(expr, start, end):
                    labels = series["metric"]
                    if name == "kafka_topic_offset" and not (
                            labels.get("topic") in (meta["input_topic"], meta["output_topic"])):
                        continue  # keep only this run's own topics
                    for ts, value in series["values"]:
                        t = datetime.utcfromtimestamp(float(ts))
                        writer.writerow([meta["run_id"], name, t.isoformat(timespec="seconds"),
                                         round(float(ts) - start.timestamp(), 1), value]
                                        + [labels.get(k, "") for k in LABELS])
                        rows_total += 1
            print(f"  {meta['run_id']} done")
    print(f"\nWrote {rows_total} rows to {out_dir}")


if __name__ == "__main__":
    main()