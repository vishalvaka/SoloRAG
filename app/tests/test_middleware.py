import re
import pytest
from httpx import AsyncClient

from app.main import app


def _get_healthz_count(metrics_text: str) -> float:
    """Parse request_count_total for /healthz 200"""
    pattern = r'request_count_total\{endpoint="/healthz",status="200"\} ([0-9.]+)'
    m = re.search(pattern, metrics_text)
    if m:
        return float(m.group(1))
    return 0.0


@pytest.mark.asyncio
async def test_metrics_middleware_counts():
    """Calling /healthz should increment Prometheus counters."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Snapshot before any request
        r_before = await ac.get("/metrics")
        before_cnt = _get_healthz_count(r_before.text)

        # Perform a health check request (this should be fast & deterministic)
        resp = await ac.get("/healthz")
        assert resp.status_code == 200

        # Snapshot after the request
        r_after = await ac.get("/metrics")
        after_cnt = _get_healthz_count(r_after.text)

    assert after_cnt >= before_cnt + 1, "Metrics counter did not increment as expected" 