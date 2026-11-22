# Experiment 3: Kafka partitions x Flink parallelism

Single-node Docker (Kafka 3.9.1, Flink 1.20.5, 1 task manager with 4 slots) on a
Mac with 4 CPUs and 8 GiB for Docker. Static ONNX autoencoder (`static-v1`) scores
CIC-IDS2017 Monday flows. Produced with:

    python pipeline/run_experiment.py        # 18 runs
    python pipeline/summarize_exp3.py        # aggregate.csv

## Design
- Settings (partitions:parallelism): 1:1, 2:2, 4:4
- Offered rates: 800 and 1500 flows/s (about 60 s of sending per run)
- 3 repeats per setting, 18 runs in total
- Each run uses fresh topics and a fresh Flink job; the job manager and task
  manager are restarted before every run
- The runs for 1:1 and 2:2 (repeats 1 to 3) and the first 4:4 run were started
  with a restart of the task manager only; the job manager restart was added
  afterwards because 4:4 runs failed to obtain slots. This does not change the results.

## Measurement rules
- Latency (ms) = completion_ts_ms - produce_ts_ms. produce_ts_ms is stamped by the
  producer just before the flow is sent. completion_ts_ms is stamped inside Flink
  after scoring, before the write to the output topic.
- Throughput = unique flows scored / (last completion - first produce).
- A run counts only if the producer kept up (`producer_valid`). All 18 runs are valid.
- Offered rates are kept below the producer limit (about 2000 flows/s in total, about
  500 per worker process).

## Files
- `summary.csv`: one line per run (settings, git SHA, model version, latency
  percentiles, throughput, missing/duplicates/errors, producer validity)
- `aggregate.csv`: mean and spread over the 3 repeats per setting
- `<run_id>/meta.json`: full settings and times of one run
- `<run_id>/producer.log`: producer output of one run
- `<run_id>/scores.csv`: RAW per-flow results (flow_id, status, score, is_alert,
  produce_ts_ms, completion_ts_ms, ...). Not in Git because of size; shared as
  `exp3_raw.zip`.

## Findings
- Throughput equals the offered rate in every setting; no flows lost, no errors.
  Flink was not saturated, so more partitions/workers gave no throughput gain.
- Median latency rises with parallelism (800/s: 336 -> 581 -> 1149 ms for 1, 2, 4
  workers) and falls with the offered rate. Likely cause (not tested): PyFlink
  batches records before the Python scoring step.
- Limits: single node, 4 CPUs, one data day (Monday, benign), the load generator
  caps the offered rate, so the saturation point is not measured.