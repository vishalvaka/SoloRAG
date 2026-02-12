"""Alembic env.py -- async-compatible for SQLAlchemy + asyncpg."""

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Ensure the project root is on sys.path so `from app.models import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Import your models so Alembic can see the metadata
from app.models import Base

# Alembic Config object
config = context.config

# Setup Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Read DATABASE_URL from environment (same as app/config.py)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://solorag:solorag@localhost:5432/solorag",
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode -- generates SQL without connecting."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    connectable = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
