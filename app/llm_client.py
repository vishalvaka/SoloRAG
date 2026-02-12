# app/llm_client.py
"""Unified LLM interface.

Provides ``get_llm_client()`` which returns an object exposing:
  - ``async generate(prompt) -> str``
  - ``async stream_generate(prompt) -> AsyncGenerator[str, None]``

The backend is selected by the ``LLM_BACKEND`` setting (ollama | bedrock).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from .config import get_settings


class LLMClient(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    async def generate(self, prompt: str) -> str: ...

    @abstractmethod
    async def stream_generate(self, prompt: str) -> AsyncGenerator[str, None]: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a singleton LLM client for the configured backend."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    settings = get_settings()
    backend = settings.LLM_BACKEND.lower()

    if backend == "ollama":
        from .ollama_client import OllamaClient
        _client_instance = OllamaClient()
    elif backend == "bedrock":
        from .bedrock_client import BedrockClient
        _client_instance = BedrockClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}. Must be 'ollama' or 'bedrock'.")

    return _client_instance
