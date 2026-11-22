\# Streaming Latency Benchmark Methodology



\## 1. Objective

Measure the initial end-to-end processing latency of the Kafka and Flink streaming pipeline.



\## 2. Test Environment

\- Operating System: Windows 11

\- Container Platform: Docker Desktop

\- Kafka: Apache Kafka 3.9.1

\- Flink: Apache Flink 1.20.5

\- Flink Job: ONNX-based static scoring

\- Kafka input topic: `latency-in`

\- Kafka output topic: `latency-out`



\## 3. Experiment Procedure

1\. Created the Kafka input and output topics.

2\. Started the Flink scoring job in unbounded mode.

3\. Confirmed that the Flink job was running.

4\. Sent 100 fixture records to `latency-in`.

5\. Collected the corresponding scoring results from `latency-out`.

6\. Matched records using their unique flow IDs.

7\. Calculated latency for each valid record.



\## 4. Latency Calculation

Per-record latency was calculated as:



Latency (ms) = completion\_ts\_ms - produce\_ts\_ms



The measurement includes all successfully scored records, not only alerts.



\## 5. Initial Results



| Metric | Result |

|---|---:|

| Records sent | 100 |

| Successful records | 100 |

| Valid latency records | 100 |

| Negative latencies | 0 |

| Average latency | 827.77 ms |

| P50 latency | 825.50 ms |

| P95 latency | 910.05 ms |

| P99 latency | 915.09 ms |

| Maximum latency | 1023 ms |



\## 6. Observations

\- All 100 records were successfully processed.

\- No negative latency values were observed.

\- The P95 latency was approximately 910 ms.

\- The experiment demonstrates successful initial end-to-end streaming integration.



\## 7. Limitations

\- This was an initial test with only 100 records.

\- Results do not establish sustained throughput or scalability.

\- The latency calculation uses timestamps from the producer and Flink environments. It assumes their clocks were sufficiently synchronized.

\- A larger, repeated benchmark is required for reliable performance conclusions.

\- These measurements should not be treated as production performance guarantees.



\## 8. Result File

Raw summary metrics are stored in:



`results/streaming\_latency\_initial.csv`



This experiment is an initial integration benchmark, not a complete scalability evaluation.

