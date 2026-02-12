# app/routes/favorites.py
"""Saved / favorited Q&A pairs."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import User, Favorite
from ..auth import get_current_user

router = APIRouter(prefix="/favorites", tags=["favorites"])


# ── schemas ───────────────────────────────────────────────────────────────

class FavoriteOut(BaseModel):
    id: str
    question: str
    answer: str
    sources: Optional[list] = None
    created_at: str


class FavoriteCreate(BaseModel):
    question: str
    answer: str
    sources: Optional[list] = None


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=list[FavoriteOut])
async def list_favorites(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Favorite).where(Favorite.user_id == user.id).order_by(Favorite.created_at.desc())
    rows = (await db.execute(stmt)).scalars().all()
    return [
        FavoriteOut(
            id=str(r.id),
            question=r.question,
            answer=r.answer,
            sources=r.sources,
            created_at=str(r.created_at),
        )
        for r in rows
    ]


@router.post("", response_model=FavoriteOut, status_code=201)
async def create_favorite(
    body: FavoriteCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    fav = Favorite(
        user_id=user.id,
        question=body.question,
        answer=body.answer,
        sources=body.sources,
    )
    db.add(fav)
    await db.commit()
    await db.refresh(fav)
    return FavoriteOut(
        id=str(fav.id),
        question=fav.question,
        answer=fav.answer,
        sources=fav.sources,
        created_at=str(fav.created_at),
    )


@router.delete("/{favorite_id}", status_code=204)
async def delete_favorite(
    favorite_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Favorite).where(Favorite.id == uuid.UUID(favorite_id), Favorite.user_id == user.id)
    fav = (await db.execute(stmt)).scalar_one_or_none()
    if fav is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Favorite not found")
    await db.delete(fav)
    await db.commit()
