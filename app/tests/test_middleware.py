# app/tests/test_middleware.py
"""Tests for Prometheus metrics middleware."""

import re
import pytest


@pytest.mark.asyncio
async def test_metrics_middleware_counts(client):
    """Calling /healthz should increment Prometheus counters."""
    # Snapshot before
    r_before = await client.get("/metrics")
    before_cnt = _get_healthz_count(r_before.text)

    # Perform a health check
    resp = await client.get("/healthz")
    assert resp.status_code == 200

    # Snapshot after
    r_after = await client.get("/metrics")
    after_cnt = _get_healthz_count(r_after.text)

    assert after_cnt >= before_cnt + 1


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")


def _get_healthz_count(metrics_text: str) -> float:
    pattern = r'request_count_total\{endpoint="/healthz",status="200"\} ([0-9.]+)'
    m = re.search(pattern, metrics_text)
    return float(m.group(1)) if m else 0.0
