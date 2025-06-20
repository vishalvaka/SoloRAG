# app/tests/test_api.py
import asyncio
import pytest
from httpx import AsyncClient

from app.main import app


# ---------- helpers --------------------------------------------------------
async def _post_query(question: str):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        return await ac.post("/query", json={"question": question})


async def _post_query_stream(question: str):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        return await ac.post("/query/stream", json={"question": question})


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


@pytest.mark.asyncio
async def test_streamlit_compatible_streaming(monkeypatch):
    """Test that the streaming endpoint works correctly for Streamlit app."""
    # Mock the streaming function to avoid hitting real LLM
    async def _fake_stream(prompt: str):
        yield "This is a test answer about Stripe."
        yield " It's a payment processor."
    
    # Patch the retrieval layer's Ollama streaming call
    import app.retrieval as retr
    monkeypatch.setattr(retr, "call_ollama_stream", _fake_stream, raising=True)
    
    r = await _post_query_stream("What is Stripe?")
    assert r.status_code == 200
    
    # The response should be text (not JSON) and contain the answer
    text_response = r.text
    assert text_response, "Streaming response should not be empty"
    
    # Should contain the SOURCES marker that Streamlit expects
    assert "[SOURCES]" in text_response, "Streaming response should contain [SOURCES] marker"
    
    # Should contain some answer content before the SOURCES marker
    sources_index = text_response.find("[SOURCES]")
    answer_part = text_response[:sources_index]
    assert answer_part.strip(), "Should have answer content before SOURCES marker"


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
async def test_streaming_validation_error():
    """POST /query/stream without required 'question' field should return 422."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/query/stream", json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_streaming_validation_error_whitespace_query():
    """POST /query/stream with an empty or whitespace question should return 422."""
    r = await _post_query_stream("   ")
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


@pytest.mark.asyncio
async def test_mocked_streaming_answer(monkeypatch):
    """Test streaming endpoint with mocked Ollama response."""
    async def _fake_stream(prompt: str):
        yield "This is a mocked streaming answer."
        yield " It works correctly."

    # Patch the streaming helper used by retrieval
    from app import retrieval as retr
    monkeypatch.setattr(retr, "call_ollama_stream", _fake_stream)

    r = await _post_query_stream("How do I get paid on Stripe?")
    assert r.status_code == 200
    text_response = r.text
    
    # Should contain the mocked answer
    assert "This is a mocked streaming answer" in text_response
    assert "It works correctly" in text_response
    assert "[SOURCES]" in text_response
