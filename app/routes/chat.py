# app/routes/chat.py
"""Chat history endpoints -- per-user conversation persistence."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models import User, ChatHistory
from ..auth import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


# ── schemas ───────────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    conversation_id: str
    message_count: int
    last_message_at: str


class ChatMessage(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[list] = None
    created_at: str


class SaveMessageRequest(BaseModel):
    conversation_id: str
    role: str
    content: str
    sources: Optional[list] = None


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("/history", response_model=list[ConversationSummary])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List distinct conversations for the current user."""
    stmt = (
        select(
            ChatHistory.conversation_id,
            func.count(ChatHistory.id).label("message_count"),
            func.max(ChatHistory.created_at).label("last_message_at"),
        )
        .where(ChatHistory.user_id == user.id)
        .group_by(ChatHistory.conversation_id)
        .order_by(func.max(ChatHistory.created_at).desc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        ConversationSummary(
            conversation_id=r.conversation_id,
            message_count=r.message_count,
            last_message_at=str(r.last_message_at),
        )
        for r in rows
    ]


@router.get("/history/{conversation_id}", response_model=list[ChatMessage])
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all messages in a conversation."""
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user.id, ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.created_at)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        ChatMessage(
            id=str(r.id),
            conversation_id=r.conversation_id,
            role=r.role,
            content=r.content,
            sources=r.sources,
            created_at=str(r.created_at),
        )
        for r in rows
    ]


@router.post("/history", response_model=ChatMessage, status_code=201)
async def save_message(
    body: SaveMessageRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Persist a single chat message."""
    msg = ChatHistory(
        user_id=user.id,
        conversation_id=body.conversation_id,
        role=body.role,
        content=body.content,
        sources=body.sources,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return ChatMessage(
        id=str(msg.id),
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        sources=msg.sources,
        created_at=str(msg.created_at),
    )


@router.delete("/history/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all messages in a conversation."""
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user.id, ChatHistory.conversation_id == conversation_id)
    )
    rows = (await db.execute(stmt)).scalars().all()
    for row in rows:
        await db.delete(row)
    await db.commit()
