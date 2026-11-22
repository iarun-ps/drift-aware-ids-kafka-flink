"""Replay flows prepared by prepare_replay.py into Kafka, fast, with N workers.

The offered rate is the TOTAL rate across all workers. Worker k sends every
N-th flow of the file, so flow_ids never collide. The send time is stamped at
the moment of sending. The script reports how far behind its schedule it fell:
if that is large, the producer was the bottleneck and latency numbers are not valid.
"""
import argparse
import multiprocessing as mp
import time

from kafka import KafkaProducer

from ids.data import REPO_ROOT

PLACEHOLDER = b"@@TS@@"


def worker(k, n_workers, path, max_rows, topic, bootstrap, rate, barrier, results):
    items = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            if i % n_workers != k:
                continue
            key, body = line.rstrip(b"\n").split(b"\t", 1)
            head, tail = body.split(PLACEHOLDER, 1)
            items.append((key, head, tail))

    producer = KafkaProducer(bootstrap_servers=bootstrap, acks="all", retries=5,
                             linger_ms=20, batch_size=65536)
    failed = [0]

    def on_error(_exc):
        failed[0] += 1

    barrier.wait()  # every worker has loaded its share, start together
    start = time.time() + 0.5
    max_behind = 0.0
    for j, (key, head, tail) in enumerate(items):
        due = start + j / rate
        now = time.time()
        if due > now:
            time.sleep(due - now)
        else:
            max_behind = max(max_behind, now - due)
        stamp = str(int(time.time() * 1000)).encode()
        try:
            producer.send(topic, key=key, value=head + stamp + tail).add_errback(on_error)
        except Exception:
            failed[0] += 1
    sending_done = time.time()
    producer.flush()
    flush_wait = time.time() - sending_done
    results.put((k, len(items), failed[0], time.time() - start, max_behind, flush_wait))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", default="mon")
    parser.add_argument("--topic", default="flows")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--rate", type=float, default=1000.0, help="TOTAL flows per second")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=200000)
    args = parser.parse_args()

    path = REPO_ROOT / "data" / "replay" / f"{args.segment}.tsv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run prepare_replay.py first.")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.workers)
    results = ctx.Queue()
    per_worker_rate = args.rate / args.workers
    procs = [ctx.Process(target=worker,
                         args=(k, args.workers, path, args.max_rows, args.topic,
                               args.bootstrap, per_worker_rate, barrier, results))
             for k in range(args.workers)]
    print(f"Sending up to {args.max_rows} flows from '{args.segment}' to '{args.topic}' "
          f"at {args.rate} flows/s using {args.workers} worker(s)")
    for p in procs:
        p.start()
    rows = [results.get(timeout=3600) for _ in procs]
    for p in procs:
        p.join()

    total = sum(r[1] for r in rows)
    failed = sum(r[2] for r in rows)
    wall = max(r[3] for r in rows)
    behind = max(r[4] for r in rows)
    flush_wait = max(r[5] for r in rows)
    sent = total - failed
    print(f"Done: {sent} sent, {failed} failed, {wall:.1f}s, achieved {sent / wall:.1f} flows/s")
    print(f"Worst lag behind schedule: {behind:.2f}s, final flush wait: {flush_wait:.2f}s")
    if behind > 1.0 or flush_wait > 1.0:
        print("WARNING: the producer could not keep up. Latency from this run is NOT valid.")


if __name__ == "__main__":
    main()