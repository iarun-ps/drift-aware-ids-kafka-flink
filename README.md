# Drift-aware real-time intrusion detection on Kafka and Flink

Evaluating METER-style adaptation for accuracy, latency and scalability in the cloud, on CIC-IDS2017.

A network-flow intrusion detector is trained once on benign traffic (a static autoencoder). When traffic changes, an ADWIN drift monitor can trigger an offline fine-tuning step that produces a new model version. The models run in a real-time pipeline: a replay producer sends flows to Kafka, Apache Flink scores every flow with ONNX Runtime, and scores go to an output topic. Prometheus and Grafana monitor the pipeline.

## What is in this project

| Part | What it does | Where |
|---|---|---|
| Static detector | Autoencoder trained on benign Monday traffic; alert if reconstruction error exceeds a threshold (0.1953, 99th percentile of benign validation scores) | `src/ids/`, `scripts/train_static.py`, `models/static_ae.onnx` |
| Baselines | MemStream-style detector (simplified) and Half-Space Trees (used in place of Kitsune) | `src/ids/memstream.py`, `src/ids/half_space_trees.py` |
| Adaptive detector | Autoencoder + ADWIN drift monitor + fine-tuning on recent flows, with safeguards against training on attack traffic (METER-inspired, not a reproduction of METER) | `src/ids/adaptive_*.py`, `src/ids/adwin_monitor.py`, `docs/decisions.md` |
| Adaptive ONNX models | Three versions, `adaptive-v1` to `adaptive-v3`, exported from the offline adaptation | `models/adaptive/`, `docs/adaptive_model_handoff.md` |
| Streaming pipeline | Replay producer, Kafka, Flink job with ONNX scoring, safe model reload, error handling | `pipeline/`, `docs/streaming_pipeline.md` |
| Monitoring | Prometheus, Kafka exporter, Grafana dashboard | `deploy/` |
| Experiment 3 | Scaling: 1, 2 and 4 Kafka partitions / Flink workers at 800 and 1500 flows/s, 3 repeats each | `results/exp3/`, `docs/exp3_analysis.md` |

## Architecture

```
CIC-IDS2017 flows
      |  (preprocessed once, 67 features)
      v
replay producer  -->  Kafka topic `flows`  -->  Flink job (PyFlink + ONNX Runtime)  -->  Kafka topic `scores`
                                                       |                                     |
                                       models/active_model.json                     latency measurement
                                       (safe model reload)
Flink and Kafka metrics --> Prometheus --> Grafana
```

- Input record: `{"flow_id", "produce_ts_ms", "label", "features": {67 named values}}`.
- Output record: `{"flow_id", "status", "model_version", "score", "threshold", "is_alert", "label", "produce_ts_ms", "completion_ts_ms"}`. Invalid input produces an error record (`status: "error"` with a reason) instead of being dropped.
- Latency of a flow = `completion_ts_ms - produce_ts_ms`. See `docs/contracts.md` for the full contract.

## Setup

Requirements: Python 3.10, Docker Desktop (8 GiB memory and 4 CPUs for Docker were used in our runs).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The CIC-IDS2017 data is not stored in the repository. Place it where `src/ids/data.py` expects it; file checksums are in `scripts/cic2017_checksums.txt`.

## Run the streaming pipeline

```
docker compose -f deploy/docker-compose.yml up -d      # Kafka, Flink, Prometheus, Grafana
docker ps --format '{{.Names}}\t{{.Status}}'

# create topics and start the Flink job
docker exec ids-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --create --topic flows-demo --partitions 1 --replication-factor 1
docker exec ids-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --create --topic scores-demo --partitions 1 --replication-factor 1
docker exec ids-jobmanager /opt/flink/bin/flink run -d -p 1 \
  -pyfs /opt/ids/pipeline/model_holder.py -py /opt/ids/pipeline/flink_score.py \
  --input flows-demo --output scores-demo --group ids-demo --unbounded

# send flows and measure latency
python pipeline/prepare_replay.py                      # once: prepares the replay files
python pipeline/replay_producer.py --segment mon --topic flows-demo --rate 800 --workers 2 --max-rows 48000
python pipeline/measure_latency.py --topic scores-demo --expected 48000

docker compose -f deploy/docker-compose.yml stop       # stop (keeps Prometheus history)
```

Web pages: Flink http://localhost:8081, Prometheus http://localhost:9090, Grafana http://localhost:3000 (dashboard `ids-pipeline`).
More detail, troubleshooting and known problems: `docs/streaming_pipeline.md` and `docs/MEMBER_B_HANDOFF.md`.

### Switching models while the job runs

Write the version into `models/active_model.json`, for example `{"model_version": "adaptive-v1"}`. The job checks the file every 2 seconds and switches only after the new model passes validation; if it fails, the current model keeps scoring. Every output record carries the `model_version` that produced it. Tests: `python pipeline/test_model_reload.py` (offline) and `python pipeline/live_reload_test.py` (live, needs the Docker stack).

## Reproduce Experiment 3

```
python pipeline/run_experiment.py                      # 18 runs (about 1 hour), results in results/exp3/
python pipeline/summarize_exp3.py                      # aggregate.csv
python pipeline/export_prometheus.py                   # Prometheus metrics per run (needs Prometheus still running)
python pipeline/make_figures.py                        # figures in results/exp3_figures/
```

## Main results

- **Experiment 3 (static model, 18 runs, all verified):** throughput equals the offered rate in every setting; no missing, duplicate or error records. Median latency rises with parallelism (800 flows/s: 336, 581, 1149 ms for 1, 2, 4 workers) and falls with a higher rate. Flink showed no backpressure or backlog and low CPU, so CPU and memory were not bottlenecks. The cause of the latency increase is not confirmed. Details and the split between confirmed findings and hypotheses: `docs/exp3_analysis.md`.
- **Experiment 4 (safeguards for adaptation):** the score filter kept attack records out of the training data; without it, 57.4% of the training records were attacks (Wednesday buffer: 3389 of 4096). Details: `docs/experiment4_ablation.md`.
- Results of Experiments 1 and 2 (model comparison, adaptive pilot) are in `results/exp1_*.csv` and `results/exp2_*.csv`.

## Limitations

- Single-node Docker on one laptop; the load generator, Kafka, Flink and monitoring share the same CPUs. The Python load generator caps at about 2000 flows/s, so the maximum throughput of the pipeline was not measured.
- Only CIC-IDS2017 was used. Experiment 3 used the benign Monday segment and the static model only.
- Adaptation is offline: ADWIN and fine-tuning ran on recorded data and produced model versions that the stream can load. The live stream does not trigger adaptation by itself.
- The adaptive detector is METER-inspired (autoencoder + ADWIN + fine-tuning); METER's hypernetwork and evidential uncertainty are not implemented (`docs/decisions.md`, D1). MemStream is a simplified version, and Half-Space Trees replaced Kitsune.
- Pilot results are preliminary. Bursts of attack traffic are not proof of benign concept drift.
- Latency uses two clocks (producer on the host, Flink in the Docker VM) on one machine.

## Repository layout

```
src/ids/        models, adaptation and evaluation code
scripts/        training, export, evaluation and analysis scripts
pipeline/       producer, Flink job, model reload, latency measurement, experiment runner
deploy/         Docker Compose, Flink image, Prometheus and Grafana config
models/         static and adaptive ONNX models, preprocessing spec
results/        experiment outputs (exp3 holds the 18 streaming runs)
docs/           contracts, decisions, experiment notes, hand-off notes
tests/          unit tests
```

## Team roles

- Member A: models, drift adaptation (AE + ADWIN), Experiments 1, 2 and 4, ONNX export.
- Member B: streaming pipeline, monitoring, Experiment 3, safe model reload.
- Member C: integration (Hugging Face producer), benchmark methodology, testing, analysis, documentation and the final report.
