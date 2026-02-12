# app/tests/test_config.py
"""Tests for configuration module."""

import os
import pytest
from app.config import Settings


class TestSettings:
    def test_defaults(self):
        """Settings should have sensible defaults for local dev."""
        s = Settings()
        assert s.LLM_BACKEND in ("ollama", "bedrock")
        assert s.VECTOR_BACKEND in ("faiss", "opensearch")
        assert s.SESSION_BACKEND in ("memory", "dynamodb")
        assert s.CACHE_BACKEND in ("none", "redis")
        assert s.FAISS_FORCE_CPU == "1"

    def test_env_override(self, monkeypatch):
        """Settings should read from environment variables."""
        monkeypatch.setenv("LLM_BACKEND", "bedrock")
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        s = Settings()
        assert s.LLM_BACKEND == "bedrock"
        assert s.CACHE_BACKEND == "redis"

    def test_proxy_settings_optional(self):
        """Proxy settings should default to None."""
        s = Settings()
        assert s.HTTP_PROXY is None
        assert s.HTTPS_PROXY is None
        assert s.NO_PROXY is None
        assert s.CUSTOM_CA_BUNDLE is None

    def test_database_url(self):
        s = Settings()
        assert "postgresql" in s.DATABASE_URL or "sqlite" in s.DATABASE_URL
