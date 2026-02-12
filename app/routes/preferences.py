# app/routes/preferences.py
"""User preferences (model, theme, top_k, etc.)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import User, UserPreferences
from ..auth import get_current_user

router = APIRouter(prefix="/preferences", tags=["preferences"])


# ── schemas ───────────────────────────────────────────────────────────────

class PreferencesOut(BaseModel):
    preferred_model: str
    theme: str
    top_k: int
    extra: Optional[dict] = None


class PreferencesUpdate(BaseModel):
    preferred_model: Optional[str] = None
    theme: Optional[str] = None
    top_k: Optional[int] = None
    extra: Optional[dict] = None


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=PreferencesOut)
async def get_preferences(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(UserPreferences).where(UserPreferences.user_id == user.id)
    prefs = (await db.execute(stmt)).scalar_one_or_none()
    if prefs is None:
        # Create default preferences if missing
        prefs = UserPreferences(user_id=user.id)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
    return PreferencesOut(
        preferred_model=prefs.preferred_model,
        theme=prefs.theme,
        top_k=prefs.top_k,
        extra=prefs.extra,
    )


@router.put("", response_model=PreferencesOut)
async def update_preferences(
    body: PreferencesUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(UserPreferences).where(UserPreferences.user_id == user.id)
    prefs = (await db.execute(stmt)).scalar_one_or_none()
    if prefs is None:
        prefs = UserPreferences(user_id=user.id)
        db.add(prefs)

    if body.preferred_model is not None:
        prefs.preferred_model = body.preferred_model
    if body.theme is not None:
        prefs.theme = body.theme
    if body.top_k is not None:
        prefs.top_k = body.top_k
    if body.extra is not None:
        prefs.extra = body.extra

    await db.commit()
    await db.refresh(prefs)
    return PreferencesOut(
        preferred_model=prefs.preferred_model,
        theme=prefs.theme,
        top_k=prefs.top_k,
        extra=prefs.extra,
    )
