# app/tests/test_cache.py
"""Tests for the cache module."""

import pytest
from app.cache import NoOpCache


@pytest.mark.asyncio
async def test_noop_cache_get():
    """NoOpCache.get should always return None."""
    cache = NoOpCache()
    result = await cache.get("any question")
    assert result is None


@pytest.mark.asyncio
async def test_noop_cache_put():
    """NoOpCache.put should not raise."""
    cache = NoOpCache()
    await cache.put("question", "answer", [{"text": "src", "score": 0.9}])
    # Still returns None after put
    result = await cache.get("question")
    assert result is None
