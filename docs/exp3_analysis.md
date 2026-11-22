# Experiment 3 analysis: scaling the Kafka + Flink scoring pipeline

Setting: single-node Docker on one Intel Mac (Docker VM 4 CPUs, 8 GiB), Kafka 3.9.1, Flink 1.20.5 (PyFlink + ONNX Runtime), static autoencoder model, replay of the benign Monday segment of CIC-IDS2017. Configurations: Kafka partitions = Flink parallelism = 1, 2, 4; offered rates 800 and 1500 flows/s; 60 s of sending; 3 repeats each (18 runs). Latency = `completion_ts_ms` - `produce_ts_ms`.

## Results (mean of 3 runs)

| Setting | Rate | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (flows/s) |
|---|---|---|---|---|---|
| 1 / 1 | 800 | 336 | 567 | 600 | 796 |
| 1 / 1 | 1500 | 241 | 408 | 532 | 1465 |
| 2 / 2 | 800 | 581 | 1061 | 1146 | 792 |
| 2 / 2 | 1500 | 384 | 612 | 651 | 1466 |
| 4 / 4 | 800 | 1149 | 1979 | 2338 | 789 |
| 4 / 4 | 1500 | 632 | 1170 | 1346 | 1482 |

Resource use while flows were arriving (mean of 3 runs; see `metrics_summary.csv`):

| Setting | Rate | JVM CPU load mean (peak) | Heap used mean (peak) MB | Busy ms/s | Backpressured ms/s | Pending records |
|---|---|---|---|---|---|---|
| 1 / 1 | 800 | 4% (9%) | 162 (265) | 162 | 0 | 0 |
| 1 / 1 | 1500 | 6% (28%) | 139 (269) | 319 | 0 | 0 |
| 2 / 2 | 800 | 7% (36%) | 143 (271) | 290 | 0 | 0 |
| 2 / 2 | 1500 | 8% (39%) | 135 (262) | 296 | 0 | 0 |
| 4 / 4 | 800 | 8% (36%) | 165 (279) | 267 | 0 | 0 |
| 4 / 4 | 1500 | 10% (40%) | 157 (266) | 242 | 0 | 0 |

## Confirmed findings (measured, and independently recomputed from the raw scores: 18 of 18 runs match, no missing flows, no duplicates, no error records, no negative latencies)

1. **Throughput equals the offered rate** in every setting (about 790 for 800 offered, about 1470 for 1500 offered). No flows were lost.
2. **Median latency rises with parallelism** at a fixed rate: at 800 flows/s 336, 581, 1149 ms for parallelism 1, 2, 4; at 1500 flows/s 241, 384, 632 ms. The run-to-run spread is small (p50 standard deviation up to 49 ms), far smaller than the differences between settings.
3. **Latency falls when the rate rises** at the same parallelism (for example 336 ms to 241 ms at parallelism 1).
4. **No backpressure and no backlog:** backpressured time was 0 and pending records stayed at 0 in every setting, so Flink always kept up with Kafka.
5. **CPU and memory were not bottlenecks:** mean JVM CPU load stayed under 10% (peaks around 40%), heap stayed around 140-165 MB on average with peaks near 280 MB against roughly 384 MB of task heap, and workers were busy about 16-32% of the time.
6. **Scaling out added no throughput here,** because the offered rate was already served. Higher parallelism did not help at these rates; it only added latency.
7. **Alert rate on benign traffic** was roughly 1.2-1.9% of flows with the static model.

## Hypotheses (not tested, do not state as facts)

- **Why latency grows with parallelism and falls with rate.** One plausible cause is how PyFlink hands records to its Python process: records are collected into bundles that are flushed on a size or time limit, so when each parallel worker receives fewer records per second (lower rate, or the same rate split across more workers) a bundle takes longer to fill and records wait longer. This fits both observed trends, but we did not test it, for example by shortening the bundle time and repeating a run.
- Extra cost of running 2 or 4 Python workers on the same 4-CPU virtual machine (context switching, shared CPU for Kafka, Flink and the load generator) may also contribute. CPU averages were low, so this seems less likely, but it is not ruled out.
- The JVM CPU load figure is a process-level fraction as reported by Flink; we did not verify how it is normalised for a 4-CPU container, so treat the percentages as indicative.

## Limitations to state

- Single machine, Docker only; the load generator, Kafka, Flink and Prometheus share the same 4 CPUs.
- The Python load generator tops out at about 2000 flows/s in total, so we could not push Flink to saturation. We therefore cannot say what the maximum throughput of the pipeline is, only that it keeps up to 1500 flows/s with 1 to 4 workers.
- Two clocks are used (producer on the Mac, Flink in the Docker VM); minimum latency was positive in all 18 runs (verification.csv), so clock offset is not visible, but it is not measured separately.
- Only the Monday benign segment of CIC-IDS2017 and only the static model were used. Results say nothing about attack traffic or the adaptive models.
- Three repeats per setting; standard deviations are given but this is a small sample.
- Metrics are from one Prometheus history sampled every 5 seconds; averages were taken over the period when records were flowing.
