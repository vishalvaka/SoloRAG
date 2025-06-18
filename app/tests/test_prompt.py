# app/tests/test_prompt.py
import pytest

from app.prompt import build_prompt, SYSTEM_MSG


def _fake_ctx(n=2):
    return [
        {"text": f"This is paragraph {i} about payouts on Stripe.", "score": 0.9 - i * 0.1}
        for i in range(n)
    ]


def test_build_prompt_basic():
    q = "When is my first payout on Stripe?"
    ctx = _fake_ctx(3)
    prompt = build_prompt(q, ctx)

    # System instructions present
    assert SYSTEM_MSG.split(".")[0] in prompt
    # Question appears
    assert q in prompt
    # Every context snippet should appear shortened
    for snip in ctx:
        assert snip["text"].split()[0] in prompt  # first word preserved


def test_build_prompt_bullet_count():
    q = "How do refunds work?"
    ctx = _fake_ctx(5)
    prompt = build_prompt(q, ctx)
    # Should contain exactly len(ctx) dash bullets
    assert prompt.count("- ") == len(ctx)


@pytest.mark.asyncio
async def test_retrieval_uses_prompt(monkeypatch):
    """Patch build_prompt to ensure retrieval.get_answer calls it."""
    from app import retrieval as retr

    called = {}

    def _fake_build(question, ctx):
        called["question"] = question
        called["ctx"] = ctx
        return "dummy prompt"

    async def _fake_ollama(prompt):  # noqa: ARG001
        return "dummy answer"

    monkeypatch.setattr(retr, "build_prompt", _fake_build)
    monkeypatch.setattr(retr, "call_ollama", _fake_ollama)

    await retr.get_answer("Test question")

    # build_prompt must have been invoked with proper params
    assert called["question"] == "Test question"
    assert isinstance(called["ctx"], list)


def test_build_prompt_no_context():
    q = "A question with no context"
    prompt = build_prompt(q, [])

    # System instructions present
    assert SYSTEM_MSG.split(".")[0] in prompt
    # Question appears
    assert q in prompt
    # No "Based on these" section
    assert "Based on these" not in prompt 