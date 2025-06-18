# app/tests/test_retrieval.py
import pytest
from app import retrieval


@pytest.mark.asyncio
async def test_search_returns_ranked_snippets():
    """Vector + rerank pipeline should give k snippets in descending score."""
    k = 4
    query = "first payout on Stripe"

    # _search is sync; get_answer is async and hits Ollama. We only test retrieval.
    results = retrieval._search(query, k=k, overfetch=5)

    # basic shape checks
    assert isinstance(results, list) and len(results) == k
    for item in results:
        assert "text" in item and "score" in item
        assert isinstance(item["text"], str) and item["text"].strip()
        assert isinstance(item["score"], float)

    # ensure scores are sorted descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_get_answer_happy_path(monkeypatch):
    """Check that get_answer orchestrates search, prompt, and generation."""
    FAKE_QUESTION = "What is a banana?"
    FAKE_CONTEXT = [{"text": "a yellow fruit", "score": 0.9}]
    FAKE_PROMPT = "This is the prompt."
    FAKE_ANSWER = "It is a tasty fruit."

    called = {}
    def _fake_search(query, k=4, overfetch=5):
        called["_search"] = True
        assert query == FAKE_QUESTION
        return FAKE_CONTEXT

    def _fake_build_prompt(question, ctx):
        called["build_prompt"] = True
        assert question == FAKE_QUESTION
        assert ctx == FAKE_CONTEXT
        return FAKE_PROMPT

    async def _fake_call_ollama(prompt):
        called["call_ollama"] = True
        assert prompt == FAKE_PROMPT
        return FAKE_ANSWER

    monkeypatch.setattr(retrieval, "_search", _fake_search)
    monkeypatch.setattr(retrieval, "build_prompt", _fake_build_prompt)
    monkeypatch.setattr(retrieval, "call_ollama", _fake_call_ollama)

    result = await retrieval.get_answer(FAKE_QUESTION)

    assert called["_search"] and called["build_prompt"] and called["call_ollama"]
    answer, sources = result
    assert answer == FAKE_ANSWER
    assert sources == FAKE_CONTEXT


@pytest.mark.asyncio
async def test_get_answer_no_context(monkeypatch):
    """Check that get_answer works even if search returns no snippets."""
    FAKE_QUESTION = "query with no results"

    async def _fake_call_ollama(prompt: str) -> str:
        return "answer"

    # Ensure _search returns an empty list
    monkeypatch.setattr(retrieval, "_search", lambda query, k=4, overfetch=5: [])
    # Keep other mocks simple
    monkeypatch.setattr(retrieval, "build_prompt", lambda q, ctx: "prompt")
    monkeypatch.setattr(retrieval, "call_ollama", _fake_call_ollama)

    result = await retrieval.get_answer(FAKE_QUESTION)
    answer, sources = result
    assert answer == "answer"
    assert sources == []
