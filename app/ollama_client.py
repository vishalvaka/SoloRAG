# app/ollama_client.py
"""Thin async wrapper around the Ollama REST API.

Call ``await generate(prompt)`` to obtain the model response.
The helper runs the synchronous ``requests`` call in a thread pool so it
plays nicely with the asyncio event-loop used by FastAPI.
"""

from __future__ import annotations

import os
import asyncio
import httpx
from typing import Final
import json

OLLAMA_URL: Final[str] = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: Final[str] = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M")
TIMEOUT: Final[float] = 30.0

async def generate(prompt: str, retries: int = 3, delay_s: float = 0.5) -> str:
    """Send *prompt* to Ollama and return the generated response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    url = f"{OLLAMA_URL}/api/generate"
    last_exception = None

    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                r = await client.post(url, json=payload, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()["response"].strip()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise # Don't retry on client errors (4xx)
                
                if attempt < retries - 1:
                    await asyncio.sleep(delay_s)
                else:
                    break
    
    # If we exited the loop without success, propagate the last captured exception
    if last_exception is not None:
        raise last_exception
    # Fallback safety net
    raise RuntimeError("Exited retry loop unexpectedly and no exception captured")

# ------------------------------------------------------------------------------------
# NEW: async generator for streaming tokens
async def stream_generate(prompt: str):
    """Yields chunks of the Ollama response as they arrive (server streaming)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }
    url = f"{OLLAMA_URL}/api/generate"

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except Exception:
                    # In case of malformed JSON just forward raw line
                    yield line
                    continue

                # Ollama streams each partial response under the 'response' key.
                token = data.get("response")
                if token is not None:
                    yield token
                # Check for end condition if API marks it.
                if data.get("done") is True:
                    break

# ------------------------------------------------------------------------------------
# End of public API – no additional helpers below to keep the surface minimal.
