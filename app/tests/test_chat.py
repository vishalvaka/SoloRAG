# app/tests/test_chat.py
"""Tests for chat history CRUD endpoints."""

import pytest


@pytest.mark.asyncio
async def test_chat_history_empty(auth_client):
    resp = await auth_client.get("/chat/history")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_chat_history_saved_via_query(auth_client):
    """POST /query should persist chat history."""
    await auth_client.post("/query", json={
        "question": "How does Stripe work?",
        "conversation_id": "conv-chat-test",
    })

    resp = await auth_client.get("/chat/history")
    assert resp.status_code == 200
    convos = resp.json()
    assert len(convos) >= 1
    assert any(c["conversation_id"] == "conv-chat-test" for c in convos)


@pytest.mark.asyncio
async def test_chat_history_detail(auth_client):
    """Retrieve messages for a specific conversation."""
    await auth_client.post("/query", json={
        "question": "First question",
        "conversation_id": "conv-detail-test",
    })
    await auth_client.post("/query", json={
        "question": "Second question",
        "conversation_id": "conv-detail-test",
    })

    resp = await auth_client.get("/chat/history/conv-detail-test")
    assert resp.status_code == 200
    messages = resp.json()
    # 2 questions + 2 answers = 4 messages
    assert len(messages) == 4
    roles = [m["role"] for m in messages]
    assert roles.count("user") == 2
    assert roles.count("assistant") == 2


@pytest.mark.asyncio
async def test_chat_history_delete(auth_client):
    """Delete a conversation."""
    await auth_client.post("/query", json={
        "question": "Delete me",
        "conversation_id": "conv-delete-test",
    })

    resp = await auth_client.delete("/chat/history/conv-delete-test")
    assert resp.status_code == 204

    # Verify it's gone
    resp = await auth_client.get("/chat/history/conv-delete-test")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_chat_save_manual(auth_client):
    """POST /chat/history to manually save a message."""
    resp = await auth_client.post("/chat/history", json={
        "conversation_id": "manual-save-test",
        "role": "user",
        "content": "Manually saved message",
    })
    assert resp.status_code == 201

    resp = await auth_client.get("/chat/history/manual-save-test")
    assert resp.status_code == 200
    messages = resp.json()
    assert len(messages) == 1
    assert messages[0]["content"] == "Manually saved message"
