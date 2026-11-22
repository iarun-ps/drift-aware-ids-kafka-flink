"""Experiment 3 runner: Kafka partitions x Flink parallelism x offered rate.

For every run: fresh topics, fresh Flink job, replay with replay_producer.py,
raw results saved to CSV, meta.json and one line in summary.csv.
Finished runs are skipped, so the script can be stopped and started again.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
from kafka import KafkaConsumer

from ids.data import REPO_ROOT

FLINK = "http://localhost:8081"
SUMMARY_FIELDS = [
    "run_id", "git_sha", "model_version", "partitions", "parallelism",
    "rate_offered", "workers", "rows_sent", "ok", "errors", "duplicates",
    "missing", "alerts", "p50_ms", "p95_ms", "p99_ms", "max_ms",
    "throughput_fps", "producer_flush_wait_s", "producer_valid", "start_utc",
    "duration_s",
]


def log(message):
    print(f"  {message}", flush=True)


def sh(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def rest(path, method="GET"):
    request = urllib.request.Request(FLINK + path, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
    return json.loads(body) if body else {}


def cancel_all_jobs():
    for job in rest("/jobs/overview")["jobs"]:
        if job["state"] in ("RUNNING", "RESTARTING", "CREATED"):
            rest(f"/jobs/{job['jid']}?mode=cancel", method="PATCH")
    time.sleep(3)


def wait_for_taskmanager(slots_needed, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            managers = rest("/taskmanagers")["taskmanagers"]
            if len(managers) == 1 and managers[0]["freeSlots"] >= slots_needed:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Flink task manager did not become ready")


def wait_job_running(job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = rest(f"/jobs/{job_id}")
        if info["state"] in ("FAILED", "CANCELED", "FINISHED"):
            raise RuntimeError(f"job ended early: {info['state']}")
        if info["state"] == "RUNNING" and all(v["status"] == "RUNNING" for v in info["vertices"]):
            return
        time.sleep(2)
    raise RuntimeError("job did not reach RUNNING")


def collect(topic, expected, idle_sec=30):
    consumer = KafkaConsumer(
        topic, bootstrap_servers="localhost:9092", auto_offset_reset="earliest",
        enable_auto_commit=False, consumer_timeout_ms=idle_sec * 1000)
    rows, seen = [], set()
    for record in consumer:
        message = json.loads(record.value)
        rows.append(message)
        seen.add(message.get("flow_id"))
        if len(seen) >= expected:
            break
    consumer.close()
    return rows


def git_sha():
    sha = sh(["git", "rev-parse", "--short", "HEAD"]).strip()
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"]).strip()
    return sha + ("-dirty" if dirty else "")


def run_one(args, out_root, run_id, partitions, parallelism, rate):
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{run_id}-{int(time.time())}"
    in_topic, out_topic = f"flows-{suffix}", f"scores-{suffix}"
    started = datetime.now(timezone.utc)
    t0 = time.time()

    log("restarting the job manager and the task manager (clean start)")
    sh(["docker", "restart", "ids-jobmanager", "ids-taskmanager"])
    time.sleep(15)
    wait_for_taskmanager(parallelism)

    log(f"creating topics with {partitions} partition(s)")
    for topic in (in_topic, out_topic):
        sh(["docker", "exec", "ids-kafka", "/opt/kafka/bin/kafka-topics.sh",
            "--bootstrap-server", "localhost:19092", "--create", "--topic", topic,
            "--partitions", str(partitions), "--replication-factor", "1"])

    log(f"starting Flink job with parallelism {parallelism}")
    output = sh(["docker", "exec", "ids-jobmanager", "/opt/flink/bin/flink", "run", "-d",
                 "-p", str(parallelism), "-py", "/opt/ids/pipeline/flink_score.py",
                 "--input", in_topic, "--output", out_topic,
                 "--group", f"ids-{suffix}", "--unbounded"])
    match = re.search(r"JobID ([0-9a-f]{32})", output)
    if not match:
        raise RuntimeError("could not find the JobID in:\n" + output)
    job_id = match.group(1)
    wait_job_running(job_id)
    time.sleep(5)

    workers = max(1, math.ceil(rate / 500))
    target_rows = int(rate * args.duration)
    log(f"sending {target_rows} flows at {rate}/s with {workers} worker(s)")
    produced = subprocess.run(
        [sys.executable, "pipeline/replay_producer.py", "--segment", args.segment,
         "--topic", in_topic, "--rate", str(rate), "--workers", str(workers),
         "--max-rows", str(target_rows)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    producer_log = produced.stdout + produced.stderr
    (run_dir / "producer.log").write_text(producer_log)
    done = re.search(r"Done: (\d+) sent, (\d+) failed", producer_log)
    flush = re.search(r"final flush wait: ([0-9.]+)s", producer_log)
    rows_sent = int(done.group(1)) if done else 0
    flush_wait = float(flush.group(1)) if flush else float("nan")
    producer_valid = bool(done) and "WARNING" not in producer_log

    log("collecting results")
    rows = collect(out_topic, expected=max(rows_sent, 1))
    rest(f"/jobs/{job_id}?mode=cancel", method="PATCH")

    with open(run_dir / "scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flow_id", "status", "score", "is_alert", "produce_ts_ms",
                         "completion_ts_ms", "model_version", "reason"])
        for m in rows:
            writer.writerow([m.get("flow_id"), m.get("status"), m.get("score"),
                             m.get("is_alert"), m.get("produce_ts_ms"),
                             m.get("completion_ts_ms"), m.get("model_version"),
                             m.get("reason")])

    ok_rows = [m for m in rows if m.get("status") == "ok"]
    ids = [m.get("flow_id") for m in rows]
    unique = len(set(ids))
    latency = np.array([m["completion_ts_ms"] - m["produce_ts_ms"] for m in ok_rows], dtype=float)
    if len(ok_rows):
        span = (max(m["completion_ts_ms"] for m in ok_rows)
                - min(m["produce_ts_ms"] for m in ok_rows)) / 1000.0
        throughput = len(set(m["flow_id"] for m in ok_rows)) / span
        p50, p95, p99 = (float(np.percentile(latency, q)) for q in (50, 95, 99))
        worst = float(latency.max())
    else:
        throughput = p50 = p95 = p99 = worst = float("nan")

    summary = {
        "run_id": run_id, "git_sha": git_sha(),
        "model_version": ok_rows[0]["model_version"] if ok_rows else "",
        "partitions": partitions, "parallelism": parallelism,
        "rate_offered": rate, "workers": workers, "rows_sent": rows_sent,
        "ok": len(ok_rows), "errors": len(rows) - len(ok_rows),
        "duplicates": len(ids) - unique, "missing": rows_sent - unique,
        "alerts": sum(1 for m in ok_rows if m.get("is_alert")),
        "p50_ms": round(p50), "p95_ms": round(p95), "p99_ms": round(p99),
        "max_ms": round(worst), "throughput_fps": round(throughput, 1),
        "producer_flush_wait_s": flush_wait, "producer_valid": producer_valid,
        "start_utc": started.isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0),
    }
    summary_path = out_root / "summary.csv"
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    meta = dict(summary)
    meta.update({
        "segment": args.segment, "send_seconds_target": args.duration,
        "flink": "1.20.5", "kafka": "3.9.1", "input_topic": in_topic,
        "output_topic": out_topic, "job_id": job_id,
        "environment": "single-node Docker on a Mac (4 CPUs, 8 GiB for Docker)",
    })
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))  # marks the run as done
    log(f"p50={summary['p50_ms']} ms  p95={summary['p95_ms']} ms  "
        f"throughput={summary['throughput_fps']}/s  missing={summary['missing']}  "
        f"errors={summary['errors']}  producer_valid={producer_valid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="1:1,2:2,4:4",
                        help="partitions:parallelism pairs, comma separated")
    parser.add_argument("--rates", default="800,1500")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=int, default=60, help="seconds of sending per run")
    parser.add_argument("--segment", default="mon")
    parser.add_argument("--out", default="results/exp3")
    args = parser.parse_args()

    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]
    rates = [int(r) for r in args.rates.split(",")]
    plan = [(rep, p, k, r) for rep in range(1, args.repeats + 1)
            for (p, k) in configs for r in rates]
    print(f"{len(plan)} runs planned, results go to {out_root}", flush=True)

    for n, (rep, p, k, r) in enumerate(plan, start=1):
        run_id = f"p{p}-par{k}-r{r}-rep{rep}"
        if (out_root / run_id / "meta.json").exists():
            print(f"[{n}/{len(plan)}] {run_id}: already done, skipping", flush=True)
            continue
        print(f"[{n}/{len(plan)}] {run_id}", flush=True)
        try:
            run_one(args, out_root, run_id, p, k, r)
        except Exception as exc:
            print(f"  RUN FAILED: {exc}", flush=True)
            (out_root / run_id).mkdir(parents=True, exist_ok=True)
            (out_root / run_id / "error.txt").write_text(str(exc))
            try:
                cancel_all_jobs()
            except Exception:
                pass
    print("All runs finished.", flush=True)


if __name__ == "__main__":
    main()