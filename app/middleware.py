"""Prometheus latency middleware.
Collects per-endpoint latency histograms and request counters.
Exposed separately from FastAPI so it can be re-used elsewhere if needed.
"""

import time
from typing import Any, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import Histogram, Counter

# Histogram buckets default to Prometheus defaults (powers of 10)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency by endpoint",
    ["endpoint"],
)

REQUEST_COUNT = Counter(
    "request_count_total",
    "Total request count by endpoint and status",
    ["endpoint", "status"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        REQUEST_COUNT.labels(endpoint, str(response.status_code)).inc()

        return response 