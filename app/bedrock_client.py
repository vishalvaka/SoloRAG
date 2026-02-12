# app/bedrock_client.py
"""Amazon Bedrock LLM backend.

Uses ``boto3`` Bedrock Runtime to call models like Claude or Titan.
Supports both synchronous and streaming invocation.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import boto3  # type: ignore[import-untyped]
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]

from .config import get_settings
from .llm_client import LLMClient
from .logger import logger


class BedrockClient(LLMClient):
    """LLM backend backed by Amazon Bedrock Runtime."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model_id = settings.BEDROCK_MODEL_ID
        self._region = settings.BEDROCK_REGION

        boto_kwargs: dict = {"region_name": self._region}

        # Honor proxy settings for the boto3 client
        proxy_cfg: dict = {}
        if settings.HTTP_PROXY:
            proxy_cfg["http"] = settings.HTTP_PROXY
        if settings.HTTPS_PROXY:
            proxy_cfg["https"] = settings.HTTPS_PROXY
        if proxy_cfg:
            boto_kwargs["config"] = BotoConfig(proxies=proxy_cfg)

        self._client = boto3.client("bedrock-runtime", **boto_kwargs)
        logger.info("bedrock_init", model_id=self._model_id, region=self._region)

    # ── helpers ────────────────────────────────────────────────────────────
    def _build_body(self, prompt: str) -> str:
        """Build the request body depending on the model provider."""
        model = self._model_id.lower()

        if "anthropic" in model or "claude" in model:
            return json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            })
        # Default: Amazon Titan / generic
        return json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 2048,
                "temperature": 0.3,
            },
        })

    def _parse_response(self, body: dict) -> str:
        """Extract the text from a non-streaming response."""
        model = self._model_id.lower()

        if "anthropic" in model or "claude" in model:
            content = body.get("content", [])
            return "".join(block.get("text", "") for block in content).strip()
        # Titan
        results = body.get("results", [{}])
        return results[0].get("outputText", "").strip() if results else ""

    # ── public API ─────────────────────────────────────────────────────────
    async def generate(self, prompt: str) -> str:
        """Invoke the Bedrock model (non-streaming) and return the full response."""
        import asyncio

        def _invoke():
            resp = self._client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=self._build_body(prompt),
            )
            return json.loads(resp["body"].read())

        result = await asyncio.to_thread(_invoke)
        return self._parse_response(result)

    async def stream_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens from Bedrock using ``invoke_model_with_response_stream``."""
        import asyncio

        def _invoke_stream():
            return self._client.invoke_model_with_response_stream(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=self._build_body(prompt),
            )

        response = await asyncio.to_thread(_invoke_stream)
        stream = response.get("body", [])

        for event in stream:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"])

            model = self._model_id.lower()
            if "anthropic" in model or "claude" in model:
                # Anthropic Messages API streaming
                delta = data.get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text
                if data.get("type") == "message_stop":
                    break
            else:
                # Titan streaming
                text = data.get("outputText", "")
                if text:
                    yield text
