import pytest, re, json
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_histograms():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/metrics")
    assert r.status_code == 200
    text = r.text
    # Basic check: default python metrics and our custom histogram name
    assert "python_gc_objects_collected_total" in text
    assert "request_latency_seconds_bucket" in text


@pytest.mark.asyncio
async def test_openapi_examples_present():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/openapi.json")
    schema = r.json()

    # Query schema should include example question
    query_schema = schema["components"]["schemas"]["Query"]
    examples = query_schema.get("examples", [])
    assert any("How do I issue a refund" in ex["question"] for ex in examples)

    # Health response example
    health_schema = schema["components"]["schemas"]["Health"]
    assert health_schema["examples"][0]["status"] == "ok" 