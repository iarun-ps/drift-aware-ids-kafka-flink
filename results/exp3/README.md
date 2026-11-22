# Experiment 3: Kafka partitions x Flink parallelism

Single-node Docker (Kafka 3.9.1, Flink 1.20.5, 1 task manager with 4 slots) on a Mac with
4 CPUs and 8 GiB for Docker. Static ONNX autoencoder (`static-v1`) scores CIC-IDS2017
Monday flows (benign traffic only). This is not a distributed or cloud deployment, and the
results must not be described as validated cluster scalability.

## How the results were produced

    python pipeline/run_experiment.py        # 18 runs, resumable (finished runs are skipped)
    python pipeline/summarize_exp3.py        # writes aggregate.csv
    python pipeline/verify_exp3.py           # independent re-computation, writes verification.csv

`run_experiment.py` was started several times. Runs that already had a `meta.json` were skipped,
so it was used as a resumable job, not as one continuous batch. All runs happened on
2026-11-22 between 07:31 and 09:03 UTC (one run takes about 110 to 125 s).

## Design
- Settings (partitions:parallelism): 1:1, 2:2, 4:4
- Offered rates: 800 and 1500 flows/s, about 60 s of sending per run (48,000 and 90,000 flows)
- 3 repeats per setting, 18 runs in total. The original plan asked for 5 repeats on selected
  settings; 3 repeats on a smaller grid were used because of time and laptop limits.
  The 3x3 grid was reduced to the matched settings 1:1, 2:2, 4:4 (no mixed settings such as
  4 partitions with parallelism 1).
- Each run uses fresh Kafka topics (with the run's number of partitions) and a fresh Flink job.
- Flink containers are restarted before every run (Metaspace and stale slot state otherwise build up).
- No warm-up period is trimmed: every flow of a run is included. The first minute of a run can
  show a small start-up effect (fresh Flink worker), so analysis may choose to trim it and say so.

## Reproducibility: Git versions and the "-dirty" runs
`summary.csv` records the Git version of the code that was checked out for each run.

| Runs | git_sha | Restart before each run |
| --- | --- | --- |
| 12 runs: 1:1 and 2:2, repeats 1 to 3 | `338ba3d` (clean) | task manager only |
| 6 runs: 4:4, repeats 1 to 3 | `338ba3d-dirty` | job manager and task manager |

"Dirty" means the working tree had a change that was not yet committed. Here the only change
was an edit to `pipeline/run_experiment.py` (the restart of both containers, 7 lines). It was
added because earlier 4:4 attempts failed with "could not acquire the minimum required resources"
(Flink did not get 4 slots). Those failed attempts produced no results. The edited script is
committed in `c7a0d7d`. The difference between `338ba3d` and `c7a0d7d` in `pipeline/` is only that
edit and the new `summarize_exp3.py`. The measurement code (`replay_producer.py`, Flink job,
model, timestamps) is identical for all 18 runs. The restart method does not change what is
measured, but 4:4 runs started from a freshly restarted job manager as well.

## Measurement rules
- Latency (ms) = `completion_ts_ms` - `produce_ts_ms`, for flows with status `ok`.
  - `produce_ts_ms` is stamped by the replay producer (on the Mac host) just before the
    flow is handed to the Kafka client.
  - `completion_ts_ms` is stamped inside the Flink Python scoring step (in the Docker VM)
    after scoring, before the write to the output topic. It does not include the sink write.
  - The two timestamps come from two clocks (Mac and Docker VM), assumed to agree.
    `verification.csv` reports the minimum latency per run and the number of negative
    latencies, which would indicate clock problems.
- Throughput = unique flows scored / ((last completion - first produce) / 1000).
- Percentiles (p50, p95, p99, max) are computed per run from the raw per-flow latencies
  (linear interpolation, `numpy.percentile`). `aggregate.csv` then takes the mean (and the
  sample standard deviation, n-1) of the 3 per-run values for each setting. It does not
  average percentiles to make an overall percentile.
- A run counts only if the producer kept up: `producer_valid` is true when the producer
  reported no warning (its delivery queue was empty at the end, final flush wait under 1 s;
  largest observed 0.19 s). All 18 runs are valid.
- Offered rates stay below the producer limit (about 1000 flows/s per producer process,
  about 2000 in total on this Mac). `replay_producer.py` uses 1 worker per 500 flows/s
  (2 workers at 800/s, 3 at 1500/s).

## Files
- `summary.csv`: one line per run (run_id, git_sha, model_version, partitions, parallelism,
  rate_offered, workers, rows_sent, ok, errors, duplicates, missing, alerts, p50/p95/p99/max
  latency in ms, throughput_fps, producer_flush_wait_s, producer_valid, start_utc, duration_s)
- `aggregate.csv`: mean and sample standard deviation over the 3 repeats per setting
- `verification.csv`: independent re-computation from the raw files (written by `verify_exp3.py`)
- `<run_id>/meta.json`: everything in the run's summary line plus topics, Flink job id, versions
- `<run_id>/producer.log`: producer output of the run. Not in Git; it is in `exp3_raw.zip`.
- `<run_id>/scores.csv`: RAW per-flow results. Not in Git because of size (about 95 MB);
  shared as `exp3_raw.zip` (about 22 MB).

## Raw per-flow file (`scores.csv`)
One row per message read from the run's output topic, in the order they were read
(not sorted by time or by flow_id; with several partitions the order mixes partitions).

| Column | Meaning |
| --- | --- |
| `flow_id` | `<segment>-<row_order>`, for example `mon-12345`; unique per flow within a run |
| `status` | `ok` or `error` (all 18 runs have only `ok`) |
| `score` | autoencoder reconstruction error (static-v1) |
| `is_alert` | `True` when score > threshold 0.1953052431344986 |
| `produce_ts_ms` | send time, milliseconds since the Unix epoch (Mac clock) |
| `completion_ts_ms` | scoring completion time, milliseconds since the Unix epoch (Docker VM clock) |
| `model_version` | `static-v1` |
| `reason` | empty for `ok` rows; the error reason for `error` rows |

The flows are the first 48,000 or 90,000 Monday flows, ordered by capture time then original
row order. Their original 2017 capture times are not part of the file and are not latency starts.

## Findings
- Throughput equals the offered rate in every setting (about 790 and 1465 to 1480 flows/s);
  no flows lost, no duplicates, no errors. Flink and Kafka were not saturated, so more
  partitions and workers gave no throughput gain.
- The load generator was the limiting component (about 2000 flows/s in total), so the point
  where Flink saturates was not measured.
- Median latency rises with parallelism (800/s: 336, 581, 1149 ms for 1, 2, 4 workers;
  1500/s: 241, 384, 632 ms) and falls with the offered rate. Likely cause, not tested: PyFlink
  groups records into bundles before the Python scoring step, and each worker's bundle fills
  more slowly when the flows are split across more workers.
- Repeats agree closely (standard deviation of the median 2 to 49 ms; the 1:1 run at 1500/s
  has the largest spread in p95).

## Not measured / limits
- CPU, memory, Kafka consumer lag and Flink backpressure were not saved per run. They were
  shown live in Grafana and are in Prometheus for the time of the runs (available while the
  Prometheus container exists); a per-run export can be made from there.
- Single node, 4 CPUs, one data day (Monday, all benign), static model only, offline replay
  of a stored dataset, offered rate capped by the load generator.
- 3 repeats, not 5.