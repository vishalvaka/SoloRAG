# app/db.py
"""Async SQLAlchemy engine and session factory.

Designed for PostgreSQL locally and Aurora PostgreSQL (via RDS Proxy) in production.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings

settings = get_settings()

_engine_kwargs: dict = {
    "echo": settings.APP_ENV == "local",
}

# Pool settings are only valid for PostgreSQL (not SQLite)
if "sqlite" not in settings.DATABASE_URL:
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
    )

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields a DB session and closes it after the request."""
    async with async_session_factory() as session:
        yield session
