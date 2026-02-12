# app/tests/test_session_store.py
"""Tests for the in-memory session store."""

import pytest
from app.session_store import MemorySessionStore


@pytest.mark.asyncio
async def test_put_and_get():
    store = MemorySessionStore()
    await store.put("token-123", "user-abc")
    result = await store.get("token-123")
    assert result == "user-abc"


@pytest.mark.asyncio
async def test_get_nonexistent():
    store = MemorySessionStore()
    result = await store.get("nonexistent-token")
    assert result is None


@pytest.mark.asyncio
async def test_delete():
    store = MemorySessionStore()
    await store.put("token-del", "user-xyz")
    await store.delete("token-del")
    result = await store.get("token-del")
    assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent():
    """Deleting a non-existent token should not raise."""
    store = MemorySessionStore()
    await store.delete("no-such-token")  # should not raise


@pytest.mark.asyncio
async def test_multiple_tokens():
    store = MemorySessionStore()
    await store.put("t1", "u1")
    await store.put("t2", "u2")
    await store.put("t3", "u1")  # same user, different token

    assert await store.get("t1") == "u1"
    assert await store.get("t2") == "u2"
    assert await store.get("t3") == "u1"
