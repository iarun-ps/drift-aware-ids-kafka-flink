# Streaming pipeline (Member B): how to run it

Pipeline: CIC-IDS2017 flows -> replay producer -> Kafka topic -> PyFlink job
(static ONNX autoencoder, `static-v1`) -> scored output topic.
Kafka 3.9.1 (KRaft), Flink 1.20.5, single node in Docker. Not a real cluster.

## One-time setup
1. Python 3.10 virtual environment, then `pip install -e .`, plus
   `pip install kafka-python onnxruntime` for the streaming scripts.
2. Put the dataset under `data/cic2017/GeneratedLabelledFlows/TrafficLabelling/`
   (not in Git).
3. Build the Flink image: `docker build -t ids-flink:1.20.5 deploy/flink`
4. Start everything: `docker compose -f deploy/docker-compose.yml up -d`

URLs: Flink http://localhost:8081, Prometheus http://localhost:9090,
Grafana http://localhost:3000 (admin/admin; import `deploy/grafana/dashboard.json`).

## Checks
- Model parity offline: `python pipeline/check_parity.py`
- Model parity in the stream: create topics `parity-in` and `parity-out`, run
  `python pipeline/send_fixture.py`, submit the job (below) with
  `--input parity-in --output parity-out`, then `python pipeline/check_stream_parity.py`
- Broken-message handling: `python pipeline/send_bad_messages.py` into `errors-in`,
  job with `--input errors-in --output errors-out`; expect 2 ok and 7 error lines.

## Run a replay
1. Prepare the flows once: `python pipeline/prepare_replay.py --segment mon --max-rows 200000`
2. Create the input and output topics (kafka-topics.sh inside `ids-kafka`).
3. Submit the always-on job:
   `docker exec ids-jobmanager /opt/flink/bin/flink run -d -p 1 -py /opt/ids/pipeline/flink_score.py --input <in> --output <out> --group <group> --unbounded`
4. Send flows: `python pipeline/replay_producer.py --segment mon --topic <in> --rate 800 --workers 2 --max-rows 48000`
5. Measure: `python pipeline/measure_latency.py --topic <out> --expected <rows sent>`

## Experiment 3
`python pipeline/run_experiment.py` (18 runs, about 1 hour), then
`python pipeline/summarize_exp3.py`. See `results/exp3/README.md`.

## Message formats
- Input topic: `{"flow_id","produce_ts_ms","label","features":{<67 names in spec order>}}`
- Output topic: `{"flow_id","status":"ok","model_version","score","threshold","is_alert",
  "label","produce_ts_ms","completion_ts_ms"}` or
  `{"flow_id","status":"error","reason","model_version","completion_ts_ms"}`.
- Latency = `completion_ts_ms` - `produce_ts_ms`. `completion_ts_ms` is taken after scoring,
  before the write to the output topic.

## Known limits
- One producer process delivers at most about 1000 flows/s, about 2000 in total on a
  4-CPU Mac, so the saturation point of Flink was not measured.
- Keep each producer worker at about 500 flows/s or less; `replay_producer.py` prints a
  warning when it could not keep up, and such latency is not valid.
- Between experiment runs both Flink containers are restarted: otherwise the Metaspace
  limit fills up and the job manager keeps stale slot state (4-worker jobs then fail).
- Output topic is `scores` with `completion_ts_ms`; the contract proposed `alerts` with
  `alert_ts_ms`. Agree the final names with Member C.