# app/retrieval.py
"""
Vector search  ->  rerank  ->  build prompt  ->  call LLM

Exposes:
  - ``get_answer(question)``   -> (markdown, sources)
  - ``stream_answer(question)`` -> async generator of chunks
  - ``get_index_info()``        -> dict with backend metadata
  - ``_ensure_initialized()``   -> idempotent startup hook
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from .llm_client import get_llm_client
from .vectorstore import get_vector_store
from .prompt import build_prompt
from .logger import logger


# ─── initialization ──────────────────────────────────────────────────────

async def _ensure_initialized() -> None:
    """Initialize the vector store (and its models) on first use."""
    store = get_vector_store()
    await store.initialize()


# ─── public API ──────────────────────────────────────────────────────────

async def get_answer(question: str) -> tuple:
    """Return (markdown_answer, source_snippets)."""
    await _ensure_initialized()

    store = get_vector_store()
    ctx = store.search(question)

    prompt = build_prompt(question, ctx)
    llm = get_llm_client()
    answer = await llm.generate(prompt)

    return answer, ctx


async def stream_answer(question: str) -> AsyncGenerator[str, None]:
    """Async generator yielding answer chunks; yields sources at end as JSON."""
    await _ensure_initialized()

    store = get_vector_store()
    ctx = store.search(question)

    prompt = build_prompt(question, ctx)
    llm = get_llm_client()

    async for chunk in llm.stream_generate(prompt):
        yield chunk

    yield "\n\n[SOURCES] " + json.dumps(ctx)


# ─── utility ─────────────────────────────────────────────────────────────

def get_index_info() -> dict:
    """Return information about the current vector store."""
    store = get_vector_store()
    return store.get_info()
