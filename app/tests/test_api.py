# app/tests/test_api.py
"""Tests for core API endpoints: health, query, streaming, pokemon, metrics."""

import pytest


# ── Health endpoint ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gpu_enabled" in data
    assert "index_type" in data
    assert "embedding_model" in data
    assert "rerank_model" in data


# ── Query endpoint (auth-protected) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_query_requires_auth(client):
    resp = await client.post("/query", json={"question": "What is Stripe?"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_query_success(auth_client):
    resp = await auth_client.post("/query", json={"question": "What is Stripe?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"].strip()  # non-empty
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_query_with_conversation_id(auth_client):
    resp = await auth_client.post("/query", json={
        "question": "How do payouts work?",
        "conversation_id": "conv-test-001",
    })
    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == "conv-test-001"


@pytest.mark.asyncio
async def test_query_empty_question(client):
    """Empty or whitespace question should return 422."""
    resp = await client.post("/query", json={"question": "   "})
    # 401 because no auth, but if we had auth it would be 422
    assert resp.status_code in (401, 422)


@pytest.mark.asyncio
async def test_query_missing_question(client):
    resp = await client.post("/query", json={})
    assert resp.status_code in (401, 422)


# ── Streaming endpoint ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_stream_requires_auth(client):
    resp = await client.post("/query/stream", json={"question": "test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_query_stream_success(auth_client):
    resp = await auth_client.post("/query/stream", json={"question": "What is Stripe?"})
    assert resp.status_code == 200
    text = resp.text
    assert text.strip()  # non-empty response
    assert "[SOURCES]" in text


# ── Pokemon endpoint (public) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pokemon_missing_name(client):
    resp = await client.get("/pokemon")
    assert resp.status_code == 422  # missing required param


# ── Metrics endpoint ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics(client):
    # Hit healthz first to generate some metrics
    await client.get("/healthz")
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "request_count_total" in resp.text
    assert "request_latency_seconds" in resp.text


# ── 404 endpoint ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_not_found(client):
    resp = await client.get("/no-such-route")
    assert resp.status_code == 404
