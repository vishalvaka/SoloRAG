# app/ollama_client.py
"""Ollama LLM backend.

Wraps the Ollama REST API with proxy-aware HTTP and the unified LLM interface.
Also exposes module-level ``generate`` / ``stream_generate`` helpers for
backward-compatibility with existing callers.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from .config import get_settings
from .http_client import get_http_client
from .llm_client import LLMClient
from .logger import logger


class OllamaClient(LLMClient):
    """LLM backend backed by a local or remote Ollama instance."""

    def __init__(self) -> None:
        settings = get_settings()
        self._url = settings.OLLAMA_URL
        self._model = settings.OLLAMA_MODEL
        self._timeout = 30.0
        logger.info("ollama_init", url=self._url, model=self._model)

    async def generate(self, prompt: str, retries: int = 3, delay_s: float = 0.5) -> str:
        """Send *prompt* to Ollama and return the generated response."""
        payload = {"model": self._model, "prompt": prompt, "stream": False}
        url = f"{self._url}/api/generate"
        last_exception: Exception | None = None

        async with get_http_client(timeout=self._timeout) as client:
            for attempt in range(retries):
                try:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    return r.json()["response"].strip()
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep(delay_s)
                    else:
                        break

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Exited retry loop unexpectedly")

    async def stream_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """Yield tokens as they arrive from the Ollama streaming API."""
        payload = {"model": self._model, "prompt": prompt, "stream": True}
        url = f"{self._url}/api/generate"

        async with get_http_client(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except Exception:
                        yield line
                        continue

                    token = data.get("response")
                    if token is not None:
                        yield token
                    if data.get("done") is True:
                        break


# ---------------------------------------------------------------------------
# Module-level helpers (backward-compat for retrieval.py imports)
# ---------------------------------------------------------------------------
_instance: OllamaClient | None = None


def _get_instance() -> OllamaClient:
    global _instance
    if _instance is None:
        _instance = OllamaClient()
    return _instance


async def generate(prompt: str) -> str:
    return await _get_instance().generate(prompt)


async def stream_generate(prompt: str) -> AsyncGenerator[str, None]:
    async for token in _get_instance().stream_generate(prompt):
        yield token
