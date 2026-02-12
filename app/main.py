# app/main.py
"""FastAPI application -- SoloRAG Enterprise Backend."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

# Local modules
from .config import get_settings
from .db import engine, async_session_factory
from .models import Base, User, ChatHistory
from .retrieval import get_answer, stream_answer, get_index_info, _ensure_initialized
from .logger import logger
from .pokeapi import get_pokemon
from .auth import get_current_user
from .db import get_db

# Route sub-routers
from .routes.auth import router as auth_router
from .routes.chat import router as chat_router
from .routes.favorites import router as favorites_router
from .routes.preferences import router as preferences_router

# Prometheus metrics
from .middleware import MetricsMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


# ── lifespan (startup / shutdown) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup; dispose engine on shutdown."""
    settings = get_settings()
    logger.info("startup", env=settings.APP_ENV, llm=settings.LLM_BACKEND, vector=settings.VECTOR_BACKEND)

    # Create tables (dev convenience -- production uses Alembic migrations)
    if settings.APP_ENV == "local":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("db_tables_created")

    yield

    await engine.dispose()
    logger.info("shutdown")


# ── app ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SoloRAG – Enterprise RAG Platform",
    version="0.2.0",
    description="Retrieval-Augmented Generation with pluggable backends (FAISS/OpenSearch, Ollama/Bedrock)",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(MetricsMiddleware)

# Mount sub-routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(favorites_router)
app.include_router(preferences_router)


# ── request / response models ─────────────────────────────────────────────

class Query(BaseModel):
    question: str
    conversation_id: str | None = None  # optional -- ties to chat history

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "How do I issue a refund on Stripe?"}]
        }
    }

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty.")
        return v


class Health(BaseModel):
    status: str = "ok"
    gpu_enabled: bool = False
    index_type: str = "unknown"
    num_vectors: str = "unknown"
    embedding_model: str = "unknown"
    rerank_model: str = "unknown"
    db: str = "unknown"
    cache: str = "unknown"
    session_store: str = "unknown"


# ── RAG endpoints (auth-protected) ────────────────────────────────────────

@app.post("/query")
async def query(
    q: Query,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """RAG query -- returns answer + sources. Persists to chat history."""
    logger.info("query_received", question=q.question, user=user.username)
    answer, sources = await get_answer(q.question)

    # Persist to chat history
    conv_id = q.conversation_id or uuid.uuid4().hex[:16]
    db.add(ChatHistory(user_id=user.id, conversation_id=conv_id, role="user", content=q.question))
    db.add(ChatHistory(user_id=user.id, conversation_id=conv_id, role="assistant", content=answer, sources=sources))
    await db.commit()

    return {"answer": answer, "sources": sources, "conversation_id": conv_id}


@app.post("/query/stream")
async def query_stream(
    q: Query,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream incremental answer tokens, then persist to chat history via background task."""
    logger.info("query_stream_received", question=q.question, user=user.username)
    conv_id = q.conversation_id or uuid.uuid4().hex[:16]
    user_id = user.id  # capture before dependency scope closes

    # Shared mutable container to collect streamed data
    collected: dict = {"answer": "", "sources": None}

    async def token_generator() -> AsyncGenerator[str, None]:
        buffer = ""
        async for chunk in stream_answer(q.question):
            yield chunk
            buffer += chunk

        # Parse the completed buffer to extract answer and sources
        if "[SOURCES]" in buffer:
            answer_part, src_part = buffer.split("[SOURCES]", 1)
            collected["answer"] = answer_part.strip()
            try:
                collected["sources"] = json.loads(src_part)
            except Exception:
                collected["sources"] = []
        else:
            collected["answer"] = buffer.strip()

    async def save_chat_history():
        """Background task: save the conversation after the response is fully sent."""
        try:
            async with async_session_factory() as save_db:
                save_db.add(ChatHistory(
                    user_id=user_id, conversation_id=conv_id,
                    role="user", content=q.question,
                ))
                save_db.add(ChatHistory(
                    user_id=user_id, conversation_id=conv_id,
                    role="assistant", content=collected["answer"],
                    sources=collected["sources"],
                ))
                await save_db.commit()
                logger.info("chat_history_saved", conversation_id=conv_id)
        except Exception as e:
            logger.error("chat_history_save_failed", error=str(e))

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        background=BackgroundTask(save_chat_history),
    )


# ── public endpoints (no auth) ────────────────────────────────────────────

@app.get("/healthz", response_model=Health)
async def health() -> Health:
    """Enhanced health probe with backend connectivity checks."""
    settings = get_settings()
    result = Health()
    try:
        import asyncio
        asyncio.create_task(_ensure_initialized())
    except Exception:
        pass

    # Vector store info
    try:
        info = get_index_info()
        result.gpu_enabled = info.get("gpu_enabled", False)
        result.index_type = info.get("index_type", "unknown")
        result.num_vectors = str(info.get("num_vectors", "unknown"))
        result.embedding_model = info.get("embedding_model", "unknown")
        result.rerank_model = info.get("rerank_model", "unknown")
    except Exception as e:
        logger.error("health_vectorstore_failed", error=str(e))

    # DB check
    try:
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        result.db = "ok"
    except Exception as e:
        result.db = f"error: {e}"
        result.status = "degraded"

    # Cache check
    result.cache = settings.CACHE_BACKEND
    if settings.CACHE_BACKEND == "redis":
        try:
            from .cache import get_cache
            cache = get_cache()
            if hasattr(cache, '_redis'):
                cache._redis.ping()
                result.cache = "redis:ok"
        except Exception as e:
            result.cache = f"redis:error: {e}"

    # Session store
    result.session_store = settings.SESSION_BACKEND

    return result


@app.get("/pokemon")
async def pokemon(name: str) -> dict:
    """Look up a Pokemon by name or Pokedex number (public, no auth)."""
    logger.info("pokemon_request", name=name)
    return await get_pokemon(name)


# ── metrics ───────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    responses={200: {"content": {"text/plain": {}}, "description": "Prometheus metrics"}},
)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
