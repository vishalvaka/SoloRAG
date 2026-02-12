# app/cache.py
"""Redis semantic cache for RAG queries.

Caches ``(question_hash -> answer, sources)`` with a configurable TTL.
Falls back gracefully when ``CACHE_BACKEND=none``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from .config import get_settings
from .logger import logger


class SemanticCache:
    """Query-level cache backed by Redis."""

    def __init__(self) -> None:
        import redis  # type: ignore[import-untyped]

        settings = get_settings()
        self._ttl = settings.CACHE_TTL_SECONDS
        self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("redis_cache_init", url=settings.REDIS_URL, ttl=self._ttl)

    @staticmethod
    def _key(question: str) -> str:
        normalized = question.strip().lower()
        return "solorag:cache:" + hashlib.sha256(normalized.encode()).hexdigest()

    async def get(self, question: str) -> Optional[tuple[str, list]]:
        """Return cached ``(answer, sources)`` or ``None``."""
        import asyncio
        key = self._key(question)
        raw = await asyncio.to_thread(self._redis.get, key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            logger.info("cache_hit", question_hash=key)
            return data["answer"], data["sources"]
        except Exception:
            return None

    async def put(self, question: str, answer: str, sources: list) -> None:
        """Store an answer in the cache."""
        import asyncio
        key = self._key(question)
        value = json.dumps({"answer": answer, "sources": sources})
        await asyncio.to_thread(self._redis.setex, key, self._ttl, value)


class NoOpCache:
    """Dummy cache that always misses -- used when ``CACHE_BACKEND=none``."""

    async def get(self, question: str) -> None:
        return None

    async def put(self, question: str, answer: str, sources: list) -> None:
        pass


# ── Factory ───────────────────────────────────────────────────────────────

_cache_instance = None


def get_cache():
    """Return a singleton cache based on ``CACHE_BACKEND``."""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    settings = get_settings()
    if settings.CACHE_BACKEND.lower() == "redis":
        _cache_instance = SemanticCache()
    else:
        _cache_instance = NoOpCache()

    return _cache_instance
