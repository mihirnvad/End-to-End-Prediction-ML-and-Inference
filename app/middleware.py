"""Pure-ASGI request timing middleware with rolling per-route p50/p95/p99.

Implemented against the raw ASGI interface rather than ``BaseHTTPMiddleware`` to avoid
the extra task/stream hop that middleware adds to every request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("churn.api.latency")

MAX_TRACKED_ROUTES = 32


class LatencyTracker:
    """Thread-safe rolling window of request durations per route."""

    def __init__(self, window_size: int = 2000):
        self.window_size = window_size
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, route: str, duration_ms: float) -> tuple[float, float]:
        """Store a sample and return the route's current (p50, p95)."""
        with self._lock:
            if route not in self._samples:
                if len(self._samples) >= MAX_TRACKED_ROUTES:
                    route = "other"
                self._samples.setdefault(route, deque(maxlen=self.window_size))
            window = self._samples[route]
            window.append(duration_ms)
            p50, p95 = np.percentile(np.fromiter(window, float, len(window)), [50, 95])
        return float(p50), float(p95)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            windows = {route: np.array(samples) for route, samples in self._samples.items()}
        out = {}
        for route, arr in windows.items():
            p50, p95, p99 = np.percentile(arr, [50, 95, 99])
            out[route] = {
                "count": int(arr.size),
                "mean_ms": round(float(arr.mean()), 3),
                "p50_ms": round(float(p50), 3),
                "p95_ms": round(float(p95), 3),
                "p99_ms": round(float(p99), 3),
            }
        return out


class LatencyMiddleware:
    """Adds ``X-Process-Time-Ms`` and rolling ``X-Latency-P50-Ms`` / ``X-Latency-P95-Ms``."""

    def __init__(self, app: ASGIApp, tracker: LatencyTracker):
        self.app = app
        self.tracker = tracker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        route = f"{scope['method']} {scope['path']}"

        async def send_with_timing(message: Message) -> None:
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - start) * 1000
                p50, p95 = self.tracker.record(route, elapsed_ms)
                headers = MutableHeaders(scope=message)
                headers.append("X-Process-Time-Ms", f"{elapsed_ms:.3f}")
                headers.append("X-Latency-P50-Ms", f"{p50:.3f}")
                headers.append("X-Latency-P95-Ms", f"{p95:.3f}")
                logger.info(
                    "%s %d %.2fms (rolling p50=%.2fms p95=%.2fms)",
                    route,
                    message["status"],
                    elapsed_ms,
                    p50,
                    p95,
                )
            await send(message)

        await self.app(scope, receive, send_with_timing)
