# app/auth.py
"""Authentication helpers: password hashing, session tokens, FastAPI dependency."""

from __future__ import annotations

import uuid
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_db
from .models import User
from .session_store import get_session_store


# ── password helpers ──────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── session token helpers ─────────────────────────────────────────────────

async def create_session_token(user_id: str) -> str:
    """Generate a session token and persist it in the session store."""
    token = uuid.uuid4().hex
    store = get_session_store()
    await store.put(token, user_id)
    return token


async def invalidate_session_token(token: str) -> None:
    """Remove a session token from the store."""
    store = get_session_store()
    await store.delete(token)


# ── FastAPI dependency ────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract Bearer token from the request and return the corresponding user.

    Raises 401 if the token is missing, invalid, or expired.
    """
    auth_header: Optional[str] = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]
    store = get_session_store()
    user_id = await store.get(token)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
