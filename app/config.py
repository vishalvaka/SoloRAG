# app/config.py
"""Centralized configuration via environment variables.

All settings are read once at import time via Pydantic Settings.
Local dev defaults allow running without any .env file.
Production values are injected via ECS task definition / Secrets Manager.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings -- every field maps to an env var of the same name."""

    # ── App ────────────────────────────────────────────────────────────────
    APP_ENV: str = "local"  # local | production
    LOG_LEVEL: str = "INFO"

    # ── Database (PostgreSQL / Aurora via RDS Proxy) ───────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://solorag:solorag@localhost:5432/solorag"

    # ── Auth ───────────────────────────────────────────────────────────────
    SESSION_SECRET: str = "change-me-in-production"
    SESSION_TTL_HOURS: int = 24

    # ── LLM backend ───────────────────────────────────────────────────────
    LLM_BACKEND: str = "ollama"  # ollama | bedrock
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b-instruct-q5_K_M"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    BEDROCK_REGION: str = "us-east-1"

    # ── Vector store ──────────────────────────────────────────────────────
    VECTOR_BACKEND: str = "faiss"  # faiss | opensearch
    OPENSEARCH_URL: str = "https://localhost:9200"
    OPENSEARCH_INDEX: str = "solorag-vectors"
    FAISS_FORCE_CPU: str = "1"

    # ── Session token store (auth tokens, NOT chat state) ─────────────────
    SESSION_BACKEND: str = "memory"  # memory | dynamodb
    DYNAMODB_TABLE: str = "solorag-sessions"
    DYNAMODB_REGION: str = "us-east-1"
    DYNAMODB_ENDPOINT: Optional[str] = None  # http://localhost:8100 for DynamoDB Local

    # ── Cache ─────────────────────────────────────────────────────────────
    CACHE_BACKEND: str = "none"  # none | redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    # ── Proxy / corporate compliance ──────────────────────────────────────
    HTTP_PROXY: Optional[str] = None
    HTTPS_PROXY: Optional[str] = None
    NO_PROXY: Optional[str] = None
    CUSTOM_CA_BUNDLE: Optional[str] = None  # path to CA cert file

    # ── Streamlit (frontend) ──────────────────────────────────────────────
    BACKEND_URL: str = "http://localhost:8000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance (cached after first call)."""
    return Settings()
