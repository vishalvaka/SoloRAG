import pytest, json
from httpx import AsyncClient

from app.main import app
from app import logger as app_logger

# ----------------- streaming endpoint -----------------

@pytest.mark.asyncio
async def test_stream_endpoint(monkeypatch):
    """Ensure /query/stream streams back content and ends with sources block."""

    # Fake streaming generator to avoid hitting real LLM
    async def _fake_stream(prompt: str):
        for chunk in ["Hello", " world"]:
            yield chunk
    # Patch the retrieval layer's Ollama streaming call
    import app.retrieval as retr
    monkeypatch.setattr(retr, "call_ollama_stream", _fake_stream, raising=True)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/query/stream", json={"question": "Test?"})

    assert r.status_code == 200
    body = r.text
    # Should contain the fake chunks joined together
    assert "Hello world" in body
    # Should contain the SOURCES marker appended by stream_answer
    assert "[SOURCES]" in body

# ----------------- logging -----------------

@pytest.mark.asyncio
async def test_query_logging(monkeypatch):
    """Verify that query_received log is emitted."""

    captured = {}

    def _fake_info(event: str, **kwargs):
        # Record the last log event so we can assert later
        captured["event"] = event
        captured["kwargs"] = kwargs

    monkeypatch.setattr(app_logger.logger, "info", _fake_info, raising=False)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/query", json={"question": "Ping"})

    assert captured.get("event") == "query_received"
    assert captured.get("kwargs", {}).get("question") == "Ping" 