"""Live test of safe model reload: switch models while flows stream through Flink."""
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from ids.data import REPO_ROOT
from run_experiment import (cancel_all_jobs, collect, log, rest, sh,
                            wait_for_taskmanager, wait_job_running)

POINTER = Path(REPO_ROOT) / "models" / "active_model.json"
RATE, SECONDS = 300, 90
SWITCHES = [(20, "adaptive-v1"), (40, "adaptive-v2"), (60, "adaptive-v9")]  # v9 does not exist


def switcher(t_start):
    for at, version in SWITCHES:
        time.sleep(max(0, t_start + at - time.time()))
        POINTER.write_text('{"model_version": "%s"}' % version)
        log(f"t={at}s  pointer set to {version}")


def main():
    POINTER.unlink(missing_ok=True)
    suffix = str(int(time.time()))
    in_topic, out_topic = f"flows-reload-{suffix}", f"scores-reload-{suffix}"
    try:
        log("restarting Flink (clean start)")
        sh(["docker", "restart", "ids-jobmanager", "ids-taskmanager"])
        time.sleep(15)
        wait_for_taskmanager(1)
        for topic in (in_topic, out_topic):
            sh(["docker", "exec", "ids-kafka", "/opt/kafka/bin/kafka-topics.sh",
                "--bootstrap-server", "localhost:19092", "--create", "--topic", topic,
                "--partitions", "1", "--replication-factor", "1"])
        out = sh(["docker", "exec", "ids-jobmanager", "/opt/flink/bin/flink", "run", "-d",
               "-p", "1", "-pyfs", "/opt/ids/pipeline/model_holder.py",
               "-py", "/opt/ids/pipeline/flink_score.py",
                  "--input", in_topic, "--output", out_topic,
                  "--group", f"ids-{suffix}", "--unbounded"])
        job_id = re.search(r"JobID ([0-9a-f]{32})", out).group(1)
        wait_job_running(job_id)
        time.sleep(5)

        rows_target = RATE * SECONDS
        t_start = time.time()
        threading.Thread(target=switcher, args=(t_start,), daemon=True).start()
        log(f"sending {rows_target} flows at {RATE}/s")
        produced = subprocess.run(
            [sys.executable, "pipeline/replay_producer.py", "--segment", "mon",
             "--topic", in_topic, "--rate", str(RATE), "--workers", "1",
             "--max-rows", str(rows_target)],
            capture_output=True, text=True, cwd=REPO_ROOT)
        done = re.search(r"Done: (\d+) sent, (\d+) failed", produced.stdout + produced.stderr)
        sent = int(done.group(1))

        rows = collect(out_topic, expected=sent)
        
    finally:
        POINTER.unlink(missing_ok=True)
        try:
            cancel_all_jobs()
        except Exception:
            pass

    ids = {m.get("flow_id") for m in rows}
    errors = [m for m in rows if m.get("status") != "ok"]
    print("\n=== RESULT ===")
    print(f"flows sent: {sent}   unique flows scored: {len(ids)}   missing: {sent - len(ids)}   errors: {len(errors)}")
    ok = sorted((m for m in rows if m.get("status") == "ok"), key=lambda m: m["completion_ts_ms"])
    t0 = ok[0]["completion_ts_ms"]
    by_version = Counter(m["model_version"] for m in ok)
    for version, count in by_version.items():
        times = [(m["completion_ts_ms"] - t0) / 1000 for m in ok if m["model_version"] == version]
        print(f"{version:12s} {count:6d} flows   from {min(times):5.1f}s to {max(times):5.1f}s")
    print("every record has model_version:", all(m.get("model_version") for m in rows))


if __name__ == "__main__":
    main()