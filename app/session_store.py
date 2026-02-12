# app/session_store.py
"""Pluggable session-token store: in-memory (local dev) or DynamoDB (production).

Stores mapping of ``session_token -> user_id`` with optional TTL.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from .config import get_settings
from .logger import logger


class SessionStore(ABC):
    @abstractmethod
    async def put(self, token: str, user_id: str) -> None: ...

    @abstractmethod
    async def get(self, token: str) -> Optional[str]: ...

    @abstractmethod
    async def delete(self, token: str) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════
# In-memory (local development)
# ═══════════════════════════════════════════════════════════════════════════

class MemorySessionStore(SessionStore):
    """Simple dict-backed store -- data is lost on restart."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    async def put(self, token: str, user_id: str) -> None:
        settings = get_settings()
        ttl = settings.SESSION_TTL_HOURS * 3600
        self._store[token] = {"user_id": user_id, "expires": time.time() + ttl}

    async def get(self, token: str) -> Optional[str]:
        entry = self._store.get(token)
        if entry is None:
            return None
        if time.time() > entry["expires"]:
            del self._store[token]
            return None
        return entry["user_id"]

    async def delete(self, token: str) -> None:
        self._store.pop(token, None)


# ═══════════════════════════════════════════════════════════════════════════
# DynamoDB (production / DynamoDB Local)
# ═══════════════════════════════════════════════════════════════════════════

class DynamoDBSessionStore(SessionStore):
    """Session tokens stored in a DynamoDB table with TTL auto-expiry."""

    def __init__(self) -> None:
        import boto3  # type: ignore[import-untyped]

        settings = get_settings()
        self._table_name = settings.DYNAMODB_TABLE
        self._ttl_seconds = settings.SESSION_TTL_HOURS * 3600

        kwargs: dict = {"region_name": settings.DYNAMODB_REGION}
        if settings.DYNAMODB_ENDPOINT:
            kwargs["endpoint_url"] = settings.DYNAMODB_ENDPOINT
        self._dynamo = boto3.resource("dynamodb", **kwargs)
        self._table = self._dynamo.Table(self._table_name)

        # Ensure table exists (for DynamoDB Local; in prod the infra creates it)
        self._ensure_table()
        logger.info("dynamodb_session_store", table=self._table_name)

    def _ensure_table(self) -> None:
        """Create the table if it doesn't exist (DynamoDB Local convenience)."""
        try:
            self._table.table_status  # triggers DescribeTable
        except Exception:
            try:
                self._dynamo.create_table(
                    TableName=self._table_name,
                    KeySchema=[{"AttributeName": "session_token", "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": "session_token", "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST",
                )
                self._table.wait_until_exists()
                # Enable TTL
                import boto3
                client = boto3.client(
                    "dynamodb",
                    region_name=get_settings().DYNAMODB_REGION,
                    endpoint_url=get_settings().DYNAMODB_ENDPOINT,
                )
                try:
                    client.update_time_to_live(
                        TableName=self._table_name,
                        TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
                    )
                except Exception:
                    pass  # TTL may not be supported in DynamoDB Local
                logger.info("dynamodb_table_created", table=self._table_name)
            except Exception as e:
                logger.warning("dynamodb_table_create_failed", error=str(e))

    async def put(self, token: str, user_id: str) -> None:
        import asyncio
        ttl = int(time.time()) + self._ttl_seconds
        await asyncio.to_thread(
            self._table.put_item,
            Item={"session_token": token, "user_id": user_id, "ttl": ttl},
        )

    async def get(self, token: str) -> Optional[str]:
        import asyncio
        resp = await asyncio.to_thread(
            self._table.get_item, Key={"session_token": token}
        )
        item = resp.get("Item")
        if not item:
            return None
        # Check TTL manually (DynamoDB Local may not enforce TTL)
        if "ttl" in item and int(time.time()) > int(item["ttl"]):
            await self.delete(token)
            return None
        return item.get("user_id")

    async def delete(self, token: str) -> None:
        import asyncio
        await asyncio.to_thread(
            self._table.delete_item, Key={"session_token": token}
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

_store_instance: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return a singleton session store based on ``SESSION_BACKEND``."""
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    settings = get_settings()
    backend = settings.SESSION_BACKEND.lower()

    if backend == "memory":
        _store_instance = MemorySessionStore()
    elif backend == "dynamodb":
        _store_instance = DynamoDBSessionStore()
    else:
        raise ValueError(f"Unknown SESSION_BACKEND: {backend!r}")

    return _store_instance
