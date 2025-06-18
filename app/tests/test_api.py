# app/tests/test_api.py
import asyncio
import pytest
from httpx import AsyncClient

from app.main import app


# ---------- helpers --------------------------------------------------------
async def _post_query(question: str):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        return await ac.post("/query", json={"question": question})


# ---------- tests ----------------------------------------------------------
@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_basic_answer():
    """End-to-end: retrieval + Ollama + prompt."""
    r = await _post_query("When is my first payout on Stripe?")
    assert r.status_code == 200

    data = r.json()
    assert "answer" in data and data["answer"].strip()
    assert "sources" in data and len(data["sources"]) > 0

    # Every source should contain text & score
    for s in data["sources"]:
        assert "text" in s and "score" in s
        assert isinstance(s["score"], float)


# ---------- extra tests --------------------------------------------------
@pytest.mark.asyncio
async def test_validation_error():
    """POST /query without required 'question' field should return 422."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/query", json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_validation_error_whitespace_query():
    """POST /query with an empty or whitespace question should return 422."""
    r = await _post_query("   ")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_not_found():
    """Unknown path returns 404."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/no-such-route")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_mocked_answer(monkeypatch):
    """Patch the Ollama call so the test is fast and deterministic."""
    async def _fake_generate(prompt: str):
        return "This is a mocked answer."

    # Patch the generate helper used by retrieval
    from app import retrieval as retr
    monkeypatch.setattr(retr, "call_ollama", _fake_generate)

    r = await _post_query("How do I get paid on Stripe?")
    assert r.status_code == 200
    data = r.json()
    assert data["answer"].startswith("This is a mocked")
