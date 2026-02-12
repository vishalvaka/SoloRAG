# app/tests/conftest.py
"""Shared fixtures for all SoloRAG tests.

Uses an in-memory SQLite database (async) so tests are fast and isolated.
Stubs out Ollama, DynamoDB, Redis, and the vector store.
"""

import os
import sys
import pathlib
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# ── Override env BEFORE any app module is imported ────────────────────────
os.environ.update({
    "APP_ENV": "test",
    "DATABASE_URL": "sqlite+aiosqlite:///",  # in-memory
    "LLM_BACKEND": "ollama",
    "OLLAMA_URL": "http://fake-ollama:11434",
    "OLLAMA_MODEL": "test-model",
    "VECTOR_BACKEND": "faiss",
    "FAISS_FORCE_CPU": "1",
    "SESSION_BACKEND": "memory",
    "CACHE_BACKEND": "none",
    "SESSION_SECRET": "test-secret",
})

# Clear cached settings singleton so test env vars are picked up
from app.config import get_settings  # noqa: E402
get_settings.cache_clear()


# ── Database fixtures ─────────────────────────────────────────────────────

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.models import Base  # noqa: E402

_test_engine = create_async_engine("sqlite+aiosqlite:///", echo=False)
_test_session_factory = async_sessionmaker(
    _test_engine, class_=AsyncSession, expire_on_commit=False,
)


@pytest_asyncio.fixture(autouse=True)
async def _setup_db():
    """Create all tables before each test, drop after."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _test_session_factory() as session:
        yield session


# ── App fixture with dependency overrides ─────────────────────────────────

from app.db import get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _override_deps():
    """Override FastAPI dependencies for testing."""
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


# ── Stub LLM client ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """Prevent real Ollama/Bedrock calls in tests."""
    from app import llm_client as lc

    class FakeLLMClient:
        async def generate(self, prompt: str) -> str:
            return "This is a test answer from the mocked LLM."

        async def stream_generate(self, prompt: str):
            yield "This is a streamed test answer."

    monkeypatch.setattr(lc, "_client_instance", FakeLLMClient())


# ── Stub vector store ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stub_vectorstore(monkeypatch):
    """Use a fake vector store so tests don't need FAISS artifacts."""
    from app import vectorstore as vs

    class FakeVectorStore:
        _initialized = True

        async def initialize(self):
            pass

        def search(self, query: str, k: int = 4, overfetch: int = 5):
            return [
                {"text": "Stripe processes payments globally.", "score": 0.95},
                {"text": "Payouts are sent on a rolling basis.", "score": 0.80},
            ]

        def get_info(self):
            return {
                "gpu_enabled": False,
                "index_type": "FakeFlat",
                "num_vectors": 42,
                "embedding_model": "test-model",
                "rerank_model": "test-reranker",
            }

    monkeypatch.setattr(vs, "_store_instance", FakeVectorStore())


# ── Stub session store ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stub_session_store(monkeypatch):
    """Use the in-memory session store."""
    from app import session_store as ss
    from app.session_store import MemorySessionStore

    monkeypatch.setattr(ss, "_store_instance", MemorySessionStore())


# ── Helper: authenticated client ─────────────────────────────────────────

from httpx import AsyncClient, ASGITransport  # noqa: E402


@pytest_asyncio.fixture
async def client():
    """Unauthenticated async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_client(client: AsyncClient):
    """Authenticated test client: registers a user and attaches the token."""
    resp = await client.post("/auth/register", json={
        "username": f"testuser_{uuid.uuid4().hex[:8]}",
        "password": "testpassword123",
    })
    assert resp.status_code == 201, f"Registration failed: {resp.text}"
    token = resp.json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"
    yield client
