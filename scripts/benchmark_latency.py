"""Benchmark end-to-end HTTP latency of a running inference API.

Sends real customer records (from the hold-out split when the dataset exists) over a
keep-alive connection and reports client-observed round-trip latency alongside the
server-side processing time from the ``X-Process-Time-Ms`` header.

Usage:
    uvicorn app.main:app --port 8000            # or: docker compose up -d
    python scripts/benchmark_latency.py --url http://localhost:8000 --requests 2000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import EXAMPLE_CUSTOMER, ChurnPredictionRequest  # noqa: E402
from src.config import CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, ArtifactPaths  # noqa: E402
from src.data import load_splits  # noqa: E402
from src.pipeline import normalise_category  # noqa: E402

INT_FIELDS = {
    name
    for name, field in ChurnPredictionRequest.model_fields.items()
    if "int" in str(field.annotation)
}


def load_payloads(n: int) -> list[dict]:
    """Valid API payloads built from hold-out rows (falls back to the schema example)."""
    if not DEFAULT_DATA_PATH.exists():
        return [dict(EXAMPLE_CUSTOMER) for _ in range(n)]
    splits = load_splits(DEFAULT_DATA_PATH)
    payloads = []
    for customer_id, row in zip(splits.ids_test, splits.X_test.to_dict("records")):
        record = {"customer_id": customer_id}
        for key, value in row.items():
            if isinstance(value, float) and np.isnan(value):
                value = None
            elif key in CATEGORICAL_FEATURES:
                value = normalise_category(value)
            elif key in INT_FIELDS:
                value = int(value)
            elif isinstance(value, (np.bool_, bool)):
                value = bool(value)
            elif isinstance(value, (int, float, np.number)):
                value = float(value)
            record[key] = value
        try:
            ChurnPredictionRequest.model_validate(record)
        except ValueError:
            continue
        payloads.append(record)
        if len(payloads) == n:
            break
    return (payloads * (n // len(payloads) + 1))[:n]


def percentiles(samples: list[float]) -> dict[str, float]:
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {
        "mean": round(statistics.fmean(samples), 3),
        "p50": round(float(p50), 3),
        "p95": round(float(p95), 3),
        "p99": round(float(p99), 3),
        "max": round(max(samples), 3),
    }


def run_single(url: str, payloads: list[dict], concurrency: int) -> dict:
    def worker(chunk: list[dict]) -> tuple[list[float], list[float]]:
        client_ms, server_ms = [], []
        with requests.Session() as session:
            for payload in chunk:
                started = time.perf_counter()
                response = session.post(f"{url}/predict", json=payload, timeout=10)
                client_ms.append((time.perf_counter() - started) * 1000)
                response.raise_for_status()
                server_ms.append(float(response.headers["x-process-time-ms"]))
        return client_ms, server_ms

    chunks = [payloads[i::concurrency] for i in range(concurrency)]
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(worker, chunks))
    elapsed = time.perf_counter() - started
    client = [x for c, _ in results for x in c]
    server = [x for _, s in results for x in s]
    return {
        "requests": len(client),
        "concurrency": concurrency,
        "throughput_rps": round(len(client) / elapsed, 1),
        "client_round_trip_ms": percentiles(client),
        "server_processing_ms": percentiles(server),
    }


def run_batch(url: str, payloads: list[dict], batch_size: int, rounds: int) -> dict:
    body = {"customers": payloads[:batch_size]}
    timings = []
    with requests.Session() as session:
        for _ in range(rounds):
            started = time.perf_counter()
            session.post(f"{url}/predict/batch", json=body, timeout=30).raise_for_status()
            timings.append((time.perf_counter() - started) * 1000)
    stats = percentiles(timings)
    return {
        "batch_size": batch_size,
        "rounds": rounds,
        "round_trip_ms": stats,
        "rows_per_second_at_p50": round(batch_size / (stats["p50"] / 1000)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--batch-rounds", type=int, default=20)
    parser.add_argument("--label", default="local", help="Environment label for the report")
    parser.add_argument(
        "--output", type=Path, default=ArtifactPaths.from_env().root / "latency_benchmark.json"
    )
    args = parser.parse_args()

    url = args.url.rstrip("/")
    health = requests.get(f"{url}/health", timeout=5).json()
    payloads = load_payloads(max(args.requests, args.batch_size))
    run_single(url, payloads[: args.warmup], 1)  # warm connections, caches, allocator

    report = {
        "label": args.label,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "model_version": health.get("model_version"),
        "client_host": {"platform": platform.platform(), "python": platform.python_version()},
        "single_prediction": run_single(url, payloads[: args.requests], args.concurrency),
        "batch_prediction": run_batch(url, payloads, args.batch_size, args.batch_rounds),
    }

    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    existing[args.label] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=2))

    single = report["single_prediction"]
    print(f"POST /predict x{single['requests']} (concurrency {single['concurrency']})")
    for key in ("client_round_trip_ms", "server_processing_ms"):
        s = single[key]
        print(f"  {key:<22} p50 {s['p50']:6.2f}  p95 {s['p95']:6.2f}  p99 {s['p99']:6.2f} ms")
    print(f"  throughput {single['throughput_rps']} req/s")
    batch = report["batch_prediction"]
    print(
        f"POST /predict/batch ({batch['batch_size']} rows): "
        f"p50 {batch['round_trip_ms']['p50']:.1f} ms"
        f" -> {batch['rows_per_second_at_p50']:,} rows/s"
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
