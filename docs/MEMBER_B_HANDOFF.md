# Member B hand-off for Member C

Everything here describes what was actually run for the streaming pipeline and Experiment 3. Nothing in the repository was changed to produce this folder.

## 1. What is in the hand-off folder

| Item | What it is |
|---|---|
| `results/exp3/summary.csv` | one row per finished run (18) |
| `results/exp3/aggregate.csv` | mean/std per setting (6 settings) |
| `results/exp3/verification.csv` | independent recomputation from raw scores: 18 of 18 match, min latency, negative latencies (0) |
| `results/exp3/<run_id>/meta.json` | start time, duration, topics, job id, git sha of each run |
| `exp3_raw.zip` | `scores.csv` and `producer.log` of every run (22 MB) |
| `results/exp3_metrics/metrics_per_run.csv` | Prometheus samples for all 18 runs, tagged with `run_id` |
| `results/exp3_metrics/run_windows.csv` | exact time window, topics and Flink job id of each run |

## 2. Matching Prometheus metrics to runs

- Every row in `metrics_per_run.csv` has `run_id` (for example `p2-par2-r800-rep1` = 2 partitions, parallelism 2, 800 flows/s, repeat 1).
- `seconds_from_run_start` counts from the moment the run began (it includes about 15-20 s of Flink restart and about 5 s of job warm-up before flows are sent, then roughly 60 s of sending, then collection). Use `producer.log` and `scores.csv` (`produce_ts_ms`) if you need the exact sending window.
- Kafka offset rows are also matched by topic name (`flows-<run_id>-<epoch>`, `scores-<run_id>-<epoch>`), which are listed in `run_windows.csv` and `meta.json`.
- Samples are taken every 5 seconds. There is one series per parallel worker (`subtask_index`), so parallelism 4 has four times the rows of parallelism 1.
- Metrics: `busy_ms_per_s`, `idle_ms_per_s`, `backpressured_ms_per_s` (milliseconds per second, 0-1000, per worker); `pending_records` (source backlog); `source_records_in_per_s`; `jvm_cpu_load`, `jvm_heap_used_bytes`, `jvm_metaspace_used_bytes` (task manager); `kafka_topic_offset` (messages written per partition, from the Kafka exporter).
- Busy time is reported for the whole chained task (source, scoring and sink in one task). Per-run means are about 250-370 ms/s, with short spikes to 1000 ms/s at job start-up (warm-up), so the workers were not saturated. Use it together with `pending_records` (flat backlog) and `backpressured_ms_per_s`.
- The metrics come from one Prometheus history and cover the 18 finished runs only. The earlier failed attempts are not included, and no run folder exists for them.

## 3. Configuration used in the experiments

Environment: single-node Docker on an Intel Mac, Docker VM with 8 GiB memory and 4 CPUs. Everything runs in `deploy/docker-compose.yml`.

Kafka:
- image `apache/kafka:3.9.1`, KRaft mode (no ZooKeeper), one broker
- listeners: `EXTERNAL` `localhost:9092` (for the host, producers, consumers), `INTERNAL` `kafka:19092` (for containers, Flink)
- experiment topics were created fresh for every run with partitions = 1, 2 or 4 (same as the Flink parallelism), replication factor 1

Flink:
- version 1.20.5, custom image `ids-flink:1.20.5` (deploy/flink/Dockerfile) with PyFlink, onnxruntime 1.23.2, Kafka connector 3.4.0-1.20
- one job manager (1600m process memory), one task manager (1728m process memory, 4 task slots, JVM metaspace 512m)
- job parallelism 1, 2 or 4 (`-p`); `parallelism.default` is 1
- Prometheus reporter on port 9249 of both Flink containers
- source reads from the earliest offset; sink delivery guarantee is at-least-once
- Flink and its task manager were restarted before every run to avoid stale slot state

Scoring:
- ONNX Runtime on CPU, one thread per worker (`intra_op_num_threads = 1`, `inter_op_num_threads = 1`), one flow per call
- static model `models/static_ae.onnx`, threshold 0.1953052431344986; input `x` float32 [N,67], output `score` float32 [N]
- features are preprocessed once in the producer; Flink does not preprocess again

Producer (`pipeline/replay_producer.py`):
- kafka-python, `acks=all`, `retries=5`, `linger_ms=20`, `batch_size=65536`
- replays the Monday segment (200,000 rows prepared by `pipeline/prepare_replay.py`)
- offered rates 800 and 1500 flows/s, 60 s of sending, workers = ceil(rate / 500)
- `produce_ts_ms` is stamped when the message is built; latency = `completion_ts_ms` - `produce_ts_ms`
- kafka-python delivery tops out near 1000 messages/s per process, hence the worker rule

Experiment 3 settings: partitions:parallelism = 1:1, 2:2, 4:4, rates 800 and 1500, 3 repeats = 18 runs.

Measured findings (for context, see summary.csv): Flink and Kafka were not saturated (throughput equals the offered rate, backlog flat). The load generator, not the pipeline, caps at about 2000 flows/s. Median latency at 800/s was 336, 581, 1149 ms for parallelism 1, 2, 4; at 1500/s it was 241, 384, 632 ms. Latency rose with parallelism on this single machine; the reason (for example PyFlink bundling between Python and Java) has not been tested.

## 4. Start, stop and monitor

From the repository root:

```
docker compose -f deploy/docker-compose.yml up -d        # start everything
docker ps --format '{{.Names}}\t{{.Status}}'             # ids-kafka, ids-jobmanager, ids-taskmanager, ids-kafka-exporter, ids-prometheus, ids-grafana
docker compose -f deploy/docker-compose.yml stop         # stop but keep Prometheus history
docker compose -f deploy/docker-compose.yml down         # removes containers; Prometheus history is lost
```

Monitor:
- Flink UI: http://localhost:8081 (jobs, backpressure, task managers)
- Prometheus: http://localhost:9090 (targets under Status > Targets: jobmanager:9249, taskmanager:9249, kafka-exporter:9308)
- Grafana: usually http://localhost:3000, dashboard "ids-pipeline" (6 panels)
- Flink jobs: `docker exec ids-jobmanager /opt/flink/bin/flink list`

Run the job by hand:

```
docker exec ids-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --create --topic flows-demo --partitions 1 --replication-factor 1
docker exec ids-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --create --topic scores-demo --partitions 1 --replication-factor 1
docker exec ids-jobmanager /opt/flink/bin/flink run -d -p 1 \
  -pyfs /opt/ids/pipeline/model_holder.py -py /opt/ids/pipeline/flink_score.py \
  --input flows-demo --output scores-demo --group ids-demo --unbounded
python pipeline/replay_producer.py --segment mon --topic flows-demo --rate 800 --workers 2 --max-rows 48000
python pipeline/measure_latency.py --help        # prints how to read a scores topic
docker exec ids-jobmanager /opt/flink/bin/flink cancel <JOB_ID>
```

(The `-pyfs model_holder.py` flag exists only after the safe-reload change is merged; with the original `flink_score.py` it is not needed.)

Full Experiment 3: `python pipeline/run_experiment.py` (defaults reproduce the 18 runs; skips finished runs).

## 5. Kafka schemas

Input topic (flows), value is JSON, key is the flow id:

```
{"flow_id": "...", "produce_ts_ms": 1760000000000, "label": "BENIGN",
 "features": {"<67 feature names in the exact order of models/preprocess_spec.json>": 0.0}}
```

Output topic (scores), one record per input flow:

```
ok:    {"flow_id","status":"ok","model_version","score","threshold","is_alert","label","produce_ts_ms","completion_ts_ms"}
error: {"flow_id","status":"error","reason","model_version","completion_ts_ms"}
```

- Feature names and order must match the spec exactly; values must be finite numbers (no booleans, nulls, NaN).
- A record that fails validation is not dropped: it produces an `error` record with a reason. Tested with 9 inputs, giving 2 ok and 7 error (`pipeline/send_bad_messages.py`).
- `label` is carried only for evaluation. `completion_ts_ms` is the Flink time in milliseconds when the record was scored; it is taken from the Docker VM clock, while `produce_ts_ms` comes from the Mac clock (checked: minimum latency in verification.csv is positive for every run).
- Note: this pipeline's output topic is `scores` with `completion_ts_ms`; the written contract used the names `alerts` and `alert_ts_ms`. Please confirm which names the report should use.

## 6. Known problems and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Producer prints `KafkaTimeoutError`, delivery stalls, latency of tens of seconds | one kafka-python process cannot deliver more than about 1000 msgs/s; queue wait counted as latency | at most 500 flows/s per worker; check "final flush wait" in the producer output (invalid at 1.0 s or more) |
| Task manager exits (code 239), log has `OutOfMemoryError: Metaspace` | PyFlink jobs overnight filled metaspace | `taskmanager.memory.jvm-metaspace.size: 512m`, restart before each run |
| `NoResourceAvailableException` for parallelism 4, job does not reach RUNNING | stale slot state in the job manager | `docker restart ids-jobmanager ids-taskmanager`, wait for one task manager with free slots |
| Job fails with `ModuleNotFoundError: No module named 'model_holder'` | Python worker cannot see the helper file | submit with `-pyfs /opt/ids/pipeline/model_holder.py` |
| Prometheus shows no Flink metrics | reporter not loaded | check `metrics.reporter.prom.*` lines in the compose file and `docker logs ids-jobmanager | grep -i prometheus` |
| Flink job cannot reach Kafka | wrong listener | inside containers use `kafka:19092`; on the host use `localhost:9092` |
| ONNX load or run errors | wrong input shape, name or type | model must take float32 `x` [N,67] and return `score` [N]; `pipeline/check_model.py` and `pipeline/check_parity.py` test this; parity with PyTorch is 6.3e-08 |
| `HTTP 409` when cancelling a job | the job already ended (often failed) | look at `flink list -a` and the task manager log first |

Limitations to state honestly: one machine, Docker only, CIC-IDS2017 only, latency measured with two clocks on one computer, load generator limits the top rate, and the cause of the latency rise with parallelism is not confirmed.
